"""Smooth frequency-envelope observation model and fixed-budget fitting."""
from copy import deepcopy
import time
import numpy as np
import torch

from cast.audio import shape
from cast.config import configuration
from cast.fit import Objective, initialize
from cast.renderer import Parameters, Renderer, interpolate, oscillator, seed_for, serialize, tensors, validate_controls


def model_config(variant="smooth8"):
    if variant not in ("smooth8",):
        raise ValueError("Unfrozen renderer variant")
    cfg = configuration()
    cfg["renderer"]["version"] = "cast_smooth_noise_v1_8"
    cfg["renderer"]["noise_filter"] = "log_power_linear_interpolation_at_original_band_centers"
    cfg["renderer"]["noise_weights"] = "simplex_point_power_times_original_band_width"
    cfg["seeds"]["sampling_realizations"] = 1
    return cfg


def linear_basis(grid, centers):
    """Constant endpoint extrapolation, linear interpolation, partition of one."""
    grid, centers = np.asarray(grid), np.asarray(centers)
    upper = np.searchsorted(centers, grid, side="right").clip(1, len(centers)-1)
    lower = upper-1
    frac = ((grid-centers[lower])/(centers[upper]-centers[lower])).clip(0, 1)
    result = np.zeros((len(grid), len(centers)), dtype=np.float32)
    result[np.arange(len(grid)), lower] = 1-frac
    result[np.arange(len(grid)), upper] += frac
    return torch.tensor(result)


class SmoothRenderer(Renderer):
    def __init__(self, cfg, input_id, sampling=False):
        super().__init__(cfg, input_id)
        edges = np.array(cfg["renderer"]["noise_edges_hz"], dtype=np.float64)
        freq = np.fft.rfftfreq(self.n, 1/self.sr)
        centers = (edges[:-1]+edges[1:])/2
        self.basis = linear_basis(freq, centers)
        upper = np.searchsorted(centers, freq, side="right").clip(1, len(centers)-1)
        self.lower, self.upper = torch.tensor(upper-1), torch.tensor(upper)
        self.fraction = torch.tensor(((freq-centers[upper-1])/(centers[upper]-centers[upper-1])).clip(0, 1), dtype=torch.float32)
        self.widths = torch.tensor(np.diff(edges), dtype=torch.float32)
        self.spectra = {}
        if sampling:
            g = torch.Generator().manual_seed(seed_for(cfg, input_id, "phase", "sampling"))
            self.phases = torch.rand(cfg["renderer"]["harmonics"], generator=g, dtype=torch.float64)*2*np.pi

    def spectral_gain(self, weights):
        log_density = (weights.clamp_min(1e-12)/self.widths).log()
        # Two-point interpolation avoids batch-size-dependent GEMM reductions.
        log_gain = log_density[:, self.lower]*(1-self.fraction)+log_density[:, self.upper]*self.fraction
        gain = torch.exp(.5*log_gain)
        # Exclude DC and exact Nyquist, preserving the parent's alias boundary.
        mask = torch.ones(gain.shape[-1]); mask[0] = 0; mask[-1] = 0
        return gain*mask

    def white_spectra(self, purpose):
        if purpose not in self.spectra:
            values = []
            for i in range(self.cfg["seeds"][purpose+"_realizations"]):
                g = torch.Generator().manual_seed(seed_for(self.cfg, self.input_id, "noise", purpose, i))
                values.append(torch.fft.rfft(torch.randn(self.n, generator=g)))
            self.spectra[purpose] = torch.stack(values)
        return self.spectra[purpose]

    def render(self, controls, purpose="fit", components=False):
        f0 = interpolate(controls["spacing_hz"], self.n)
        harmonic = shape(oscillator(f0, controls["harmonic_weights"], self.phases, self.sr))
        gain = self.spectral_gain(controls["noise_weights"])
        noise = shape(torch.fft.irfft(self.white_spectra(purpose)[None]*gain[:, None], n=self.n))
        envelope = interpolate(controls["envelope_knots"], self.n)
        envelope = envelope/envelope.square().mean(-1, keepdim=True).sqrt()
        mix = controls["harmonic_fraction"][:, None, None]
        hg = torch.where(mix > 0, mix.clamp_min(1e-12).sqrt(), 0)
        ng = torch.where(mix < 1, (1-mix).clamp_min(1e-12).sqrt(), 0)
        h = harmonic[:, None]*hg*envelope[:, None]
        n = noise*ng*envelope[:, None]
        wave = self.upsample(h+n)
        if components:
            return wave, {"harmonic": self.upsample(h.expand_as(n)), "noise": self.upsample(n),
                          "envelope_internal": envelope, "spacing_internal_hz": f0,
                          "noise_spectral_gain": gain}
        return wave


def fit(target, cfg, input_id, known_harmonic_weights=None):
    """Same starts/budget/objective as v0, with the separately versioned renderer."""
    started = time.perf_counter()
    initial, search = initialize(target, cfg, input_id)
    if known_harmonic_weights is not None:
        initial["harmonic_weights"] = [known_harmonic_weights for _ in cfg["seeds"]["starts"]]
    model, renderer = Parameters(initial, cfg), SmoothRenderer(cfg, input_id)
    if known_harmonic_weights is not None:
        model.harmonic_logits.requires_grad_(False)
    objective = Objective(target, cfg)
    opt = torch.optim.Adam([{"params": [model.spacing_raw], "lr": cfg["optimizer"]["spacing_lr"]},
                            {"params": [v for k, v in model.named_parameters() if k != "spacing_raw" and v.requires_grad]}],
                           lr=cfg["optimizer"]["lr"], betas=tuple(cfg["optimizer"]["betas"]), eps=cfg["optimizer"]["epsilon"])
    with torch.no_grad():
        base = {k: v.detach().clone() for k, v in model.controls().items()}
        initial_fit = objective(renderer.render(base)).mean(1)
        initial_check = torch.stack([objective(renderer.render({k: v[i:i+1] for k, v in base.items()}, "check")).mean() for i in range(len(initial_fit))])
    best_loss = initial_fit.clone()
    best = {k: v.clone() for k, v in base.items()}
    steps = torch.zeros(len(best_loss), dtype=torch.int64)
    history, max_grad = [], 0.
    for step in range(cfg["optimizer"]["steps"]+1):
        opt.zero_grad(set_to_none=True)
        controls = model.controls()
        loss = objective(renderer.render(controls)).mean(1)
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        history.append({"step": step, "fit_losses": loss.detach().tolist()})
        with torch.no_grad():
            improved = loss < best_loss
            for k, v in controls.items():
                best[k][improved] = v[improved]
            best_loss[improved] = loss[improved]
            steps[improved] = step
        if step == cfg["optimizer"]["steps"]:
            break
        loss.sum().backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
            raise FloatingPointError(f"Nonfinite gradient at step {step}")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optimizer"]["gradient_clip_norm"])
        max_grad = max(max_grad, float(norm))
        opt.step()
        model.project()
    winner = int(best_loss.argmin())
    with torch.no_grad():
        # Check each vector exactly as it will be rendered by the replay tool.
        check = torch.stack([objective(renderer.render({k: v[i:i+1] for k, v in best.items()}, "check")).mean() for i in range(len(best_loss))])
    starts = [{"seed": seed, "initial_parameters": serialize(base, i), "parameters": serialize(best, i),
               "best_step": int(steps[i]), "initial_fit_loss": float(initial_fit[i]), "initial_check_loss": float(initial_check[i]),
               "fit_loss": float(best_loss[i]), "check_loss": float(check[i])} for i, seed in enumerate(cfg["seeds"]["starts"])]
    return {"input_id": input_id, "parameters": starts[winner]["parameters"], "winner": winner,
            "fit_loss": float(best_loss[winner]), "check_loss": float(check[winner]),
            "baseline_fit_loss": float(initial_fit.min()), "baseline_check_loss": float(initial_check.min()),
            "baseline_check_winner": int(initial_check.argmin()),
            "selection": "fitted_winner_by_fit_stream_only_baseline_minimum_on_common_check_stream",
            "streams": renderer.streams(), "initialization": search, "starts": starts,
            "known_harmonic_weights": known_harmonic_weights, "history": history,
            "maximum_gradient_norm_before_clip": max_grad, "fit_seconds": time.perf_counter()-started}

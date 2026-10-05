"""Fixed-capacity revision: sixteen noise points and nine envelope knots."""
from copy import deepcopy
import time
import numpy as np
import torch

from cast.audio import shape
from cast.fit import Objective, initialize
from cast.renderer import Parameters, serialize, tensors
from .engine import SmoothRenderer, linear_basis, model_config as smooth_config


def model_config():
    cfg = deepcopy(smooth_config())
    edges = np.array(cfg["renderer"]["noise_edges_hz"], dtype=float)
    old = (edges[:-1]+edges[1:])/2
    centers = np.sort(np.concatenate([old, (old[:-1]+old[1:])/2, [4000.]]))
    limits = np.r_[0., (centers[:-1]+centers[1:])/2, 4000.]
    cfg["experiment_id"] = "cast_capacity_noise16_env9_v3"
    cfg["renderer"].update(version="cast_noise16_env9_v3", noise_control_centers_hz=centers.tolist(),
        noise_control_reference_widths_hz=np.diff(limits).tolist(), envelope_control_count=9,
        noise_filter="log_power_linear_at_sixteen_fixed_points", noise_weights="simplex_point_power_times_Voronoi_reference_width")
    return cfg


def validate_controls(values, cfg):
    expected = {"spacing_hz": (3,), "harmonic_weights": (8,), "noise_weights": (16,),
                "harmonic_fraction": (), "envelope_knots": (9,)}
    if set(values) != set(expected):
        raise ValueError("Capacity parameter schema mismatch")
    for key, dims in expected.items():
        a = np.asarray(values[key])
        if a.shape != dims or not np.isfinite(a).all():
            raise ValueError("Invalid capacity control tensor")
    for key, bounds in (("spacing_hz", cfg["renderer"]["spacing_hz_bounds"]),
                        ("envelope_knots", cfg["renderer"]["envelope_bounds"]), ("harmonic_fraction", [0, 1])):
        a = np.asarray(values[key])
        if np.any(a < bounds[0]-1e-6) or np.any(a > bounds[1]+1e-6):
            raise ValueError("Capacity control outside unchanged bounds")
    for key in ("harmonic_weights", "noise_weights"):
        a = np.asarray(values[key])
        if (a < 0).any() or abs(a.sum()-1) > 1e-6:
            raise ValueError("Invalid capacity simplex")
    return values


def lift_controls(values, cfg):
    p = deepcopy(values)
    old_edges = np.array(cfg["renderer"]["noise_edges_hz"], dtype=float)
    old_centers = (old_edges[:-1]+old_edges[1:])/2
    density = np.maximum(p["noise_weights"], 1e-12)/np.diff(old_edges)
    target = cfg["renderer"]["noise_control_centers_hz"]
    powers = np.exp(np.interp(target, old_centers, np.log(density)))*np.array(cfg["renderer"]["noise_control_reference_widths_hz"])
    p["noise_weights"] = (powers/powers.sum()).tolist()
    p["envelope_knots"] = np.interp(np.linspace(0, 1, 9), np.linspace(0, 1, 5), p["envelope_knots"]).tolist()
    return validate_controls(p, cfg)


class CapacityRenderer(SmoothRenderer):
    def __init__(self, cfg, input_id, sampling=False):
        super().__init__(cfg, input_id, sampling=sampling)
        centers = np.array(cfg["renderer"]["noise_control_centers_hz"], dtype=float)
        if centers.shape != (16,) or not np.all(np.diff(centers) > 0):
            raise ValueError("Invalid fixed spectral centers")
        freq = np.fft.rfftfreq(self.n, 1/self.sr)
        self.basis = linear_basis(freq, centers)
        upper = np.searchsorted(centers, freq, side="right").clip(1, len(centers)-1)
        self.lower, self.upper = torch.tensor(upper-1), torch.tensor(upper)
        self.fraction = torch.tensor(((freq-centers[upper-1])/(centers[upper]-centers[upper-1])).clip(0, 1), dtype=torch.float32)
        self.widths = torch.tensor(cfg["renderer"]["noise_control_reference_widths_hz"], dtype=torch.float32)
        if self.widths.shape != (16,) or not torch.all(self.widths > 0):
            raise ValueError("Invalid spectral reference widths")


def fit(target, cfg, input_id, known_harmonic_weights=None):
    started = time.perf_counter()
    old_initial, search = initialize(target, cfg, input_id)
    lifted = [lift_controls({k: v[i] for k, v in old_initial.items()}, cfg) for i in range(len(cfg["seeds"]["starts"]))]
    initial = {k: [row[k] for row in lifted] for k in old_initial}
    if known_harmonic_weights is not None:
        initial["harmonic_weights"] = [known_harmonic_weights for _ in cfg["seeds"]["starts"]]
    model, renderer = Parameters(initial, cfg), CapacityRenderer(cfg, input_id)
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
        check = torch.stack([objective(renderer.render({k: v[i:i+1] for k, v in best.items()}, "check")).mean() for i in range(len(best_loss))])
    starts = [{"seed": seed, "initial_parameters": serialize(base, i), "parameters": serialize(best, i),
               "best_step": int(steps[i]), "initial_fit_loss": float(initial_fit[i]), "initial_check_loss": float(initial_check[i]),
               "fit_loss": float(best_loss[i]), "check_loss": float(check[i])} for i, seed in enumerate(cfg["seeds"]["starts"])]
    return {"input_id": input_id, "parameters": starts[winner]["parameters"], "winner": winner,
            "fit_loss": float(best_loss[winner]), "check_loss": float(check[winner]),
            "baseline_fit_loss": float(initial_fit.min()), "baseline_check_loss": float(initial_check.min()),
            "baseline_check_winner": int(initial_check.argmin()),
            "selection": "fitted_winner_by_fit_stream_only_baseline_minimum_on_common_check_stream",
            "streams": renderer.streams(), "initialization": {**search, "resolution_lift": "noise16_env9_density_and_linear_envelope"},
            "starts": starts, "known_harmonic_weights": known_harmonic_weights, "history": history,
            "maximum_gradient_norm_before_clip": max_grad, "fit_seconds": time.perf_counter()-started}

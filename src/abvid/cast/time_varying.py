"""Time-varying spectral controls on the smooth8 renderer.

smooth8 holds its harmonic weights, noise-band weights and harmonic/noise mix
fixed for a whole clip; only pitch and overall loudness move. Here each of those
three has `renderer.weight_knots` knots, linearly interpolated over the clip.
Noise is the same white stream through one spectral filter per knot, crossfaded
with the knots' hat functions. With equal knots every component reduces to
SmoothRenderer up to float rounding (see `parity`).

Contract: experiments/cast_utility/time_varying/PROTOCOL.md.
"""
import math
import time

import numpy as np
import torch

from abvid.cast.audio import shape
from abvid.cast.fit import Objective, initialize
from abvid.cast.population.engine import SmoothRenderer
from abvid.cast.renderer import interpolate, serialize, tensors


def over_time(knots, n):
    """(batch, knots, *rest) -> (batch, *rest, n), linear and endpoint-inclusive."""
    b, k, rest = knots.shape[0], knots.shape[1], knots.shape[2:]
    flat = knots.reshape(b, k, -1).transpose(1, 2).reshape(-1, k)
    return interpolate(flat, n).reshape(b, *rest, n)


class TimeVaryingParameters(torch.nn.Module):
    """Static smooth8 starting values, copied to every spectral knot."""

    def __init__(self, values, cfg):
        super().__init__()
        self.cfg, k = cfg, cfg["renderer"]["weight_knots"]
        scale = cfg["optimizer"]["spacing_coordinate_scale_hz"]
        self.spacing_raw = torch.nn.Parameter(torch.tensor(values["spacing_hz"], dtype=torch.float32)/scale)
        for name in ("harmonic", "noise"):
            w = torch.tensor(values[name + "_weights"], dtype=torch.float32).clamp_min(1e-12).log()
            setattr(self, name + "_logits", torch.nn.Parameter(w[:, None].repeat(1, k, 1)))
        self.mix_raw = torch.nn.Parameter(torch.tensor(values["harmonic_fraction"], dtype=torch.float32)[:, None].repeat(1, k))
        lo, hi = cfg["renderer"]["envelope_bounds"]
        env = torch.tensor(values["envelope_knots"], dtype=torch.float32)
        self.envelope_raw = torch.nn.Parameter(torch.logit(((env-lo)/(hi-lo)).clamp(1e-6, 1-1e-6)))

    def controls(self):
        r = self.cfg["renderer"]
        lo, hi = r["envelope_bounds"]
        return {"spacing_hz": (self.spacing_raw*self.cfg["optimizer"]["spacing_coordinate_scale_hz"]).clamp(*r["spacing_hz_bounds"]),
                "harmonic_weights": self.harmonic_logits.softmax(-1),
                "noise_weights": self.noise_logits.softmax(-1),
                "harmonic_fraction": self.mix_raw.clamp(0, 1),
                "envelope_knots": lo+(hi-lo)*self.envelope_raw.sigmoid()}

    def project(self):
        with torch.no_grad():
            lo, hi = self.cfg["renderer"]["spacing_hz_bounds"]
            scale = self.cfg["optimizer"]["spacing_coordinate_scale_hz"]
            self.spacing_raw.clamp_(lo/scale, hi/scale)
            self.mix_raw.clamp_(0, 1)


class TimeVaryingRenderer(SmoothRenderer):
    def render(self, controls, purpose="fit", components=False):
        if components:
            raise NotImplementedError("Component outputs are not part of this renderer")
        f0 = interpolate(controls["spacing_hz"], self.n)
        weights = over_time(controls["harmonic_weights"], self.n)
        orders = torch.arange(1, weights.shape[1]+1, dtype=torch.float64)
        phase = 2*math.pi*torch.cumsum(f0.double(), -1)/self.sr
        wave = torch.sin(phase[:, None, :]*orders[None, :, None] + self.phases[None, :, None])
        mask = f0[:, None, :]*orders[None, :, None] < self.sr/2
        harmonic = shape((wave.to(weights.dtype)*mask*weights).sum(1))
        k = controls["noise_weights"].shape[1]
        hats = over_time(torch.eye(k)[None], self.n)[0]
        white = self.white_spectra(purpose)[None]
        filtered = torch.stack([torch.fft.irfft(white*self.spectral_gain(controls["noise_weights"][:, j])[:, None], n=self.n)
                                for j in range(k)], 1)
        noise = shape((filtered*hats[None, :, None, :]).sum(1))
        envelope = interpolate(controls["envelope_knots"], self.n)
        envelope = envelope/envelope.square().mean(-1, keepdim=True).sqrt()
        mix = over_time(controls["harmonic_fraction"], self.n)[:, None]
        hg = torch.where(mix > 0, mix.clamp_min(1e-12).sqrt(), 0)
        ng = torch.where(mix < 1, (1-mix).clamp_min(1e-12).sqrt(), 0)
        return self.upsample(harmonic[:, None]*hg*envelope[:, None] + noise*ng*envelope[:, None])


def expand(values, cfg):
    """One static smooth8 control vector -> the equal-knot time-varying vector."""
    k = cfg["renderer"]["weight_knots"]
    return {**values, **{key: [values[key]]*k for key in ("harmonic_weights", "noise_weights", "harmonic_fraction")}}


def validate(values, cfg):
    k = cfg["renderer"]["weight_knots"]
    expected = {"spacing_hz": (3,), "harmonic_weights": (k, 8), "noise_weights": (k, 8),
                "harmonic_fraction": (k,), "envelope_knots": (5,)}
    if set(values) != set(expected):
        raise ValueError("Parameter schema mismatch")
    for key, dims in expected.items():
        x = np.asarray(values[key])
        if x.shape != dims or not np.isfinite(x).all():
            raise ValueError(f"Invalid parameter tensor: {key}")
    for key in ("harmonic_weights", "noise_weights"):
        x = np.asarray(values[key])
        if (x < 0).any() or np.abs(x.sum(-1)-1).max() > 1e-5:
            raise ValueError("Invalid simplex weights")
    return values


def parity(static_values, cfg, input_id, purpose="check"):
    """Max |TimeVaryingRenderer(equal knots) - SmoothRenderer| for one control vector."""
    with torch.no_grad():
        a = SmoothRenderer(cfg, input_id).render(tensors(static_values), purpose)
        b = TimeVaryingRenderer(cfg, input_id).render(tensors(expand(static_values, cfg)), purpose)
    return float((a-b).abs().max())


def fit(target, cfg, input_id):
    """smooth8's starts, budget, objective and winner rule with time-varying spectral controls."""
    started = time.perf_counter()
    initial, search = initialize(target, cfg, input_id)
    model, renderer = TimeVaryingParameters(initial, cfg), TimeVaryingRenderer(cfg, input_id)
    objective = Objective(target, cfg)
    opt = torch.optim.Adam([{"params": [model.spacing_raw], "lr": cfg["optimizer"]["spacing_lr"]},
                            {"params": [v for k, v in model.named_parameters() if k != "spacing_raw"]}],
                           lr=cfg["optimizer"]["lr"], betas=tuple(cfg["optimizer"]["betas"]), eps=cfg["optimizer"]["epsilon"])
    with torch.no_grad():
        base = {k: v.detach().clone() for k, v in model.controls().items()}
        initial_fit = objective(renderer.render(base)).mean(1)
    best_loss, best = initial_fit.clone(), {k: v.clone() for k, v in base.items()}
    steps = torch.zeros(len(best_loss), dtype=torch.int64)
    history, max_grad = [], 0.
    for step in range(cfg["optimizer"]["steps"]+1):
        opt.zero_grad(set_to_none=True)
        controls = model.controls()
        loss = objective(renderer.render(controls)).mean(1)
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        history.append(loss.detach().tolist())
        with torch.no_grad():
            improved = loss < best_loss
            for k, v in controls.items():
                best[k][improved] = v[improved]
            best_loss[improved] = loss[improved]
            steps[improved] = step
        if step == cfg["optimizer"]["steps"]:
            break
        loss.sum().backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f"Nonfinite gradient at step {step}")
        max_grad = max(max_grad, float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optimizer"]["gradient_clip_norm"])))
        opt.step()
        model.project()
    winner = int(best_loss.argmin())
    with torch.no_grad():
        check = torch.stack([objective(renderer.render({k: v[i:i+1] for k, v in best.items()}, "check")).mean()
                             for i in range(len(best_loss))])
    starts = [{"parameters": serialize(best, i), "best_step": int(steps[i]), "initial_fit_loss": float(initial_fit[i]),
               "fit_loss": float(best_loss[i]), "check_loss": float(check[i])} for i in range(len(best_loss))]
    return {"input_id": input_id, "parameters": validate(starts[winner]["parameters"], cfg), "winner": winner,
            "fit_loss": float(best_loss[winner]), "check_loss": float(check[winner]),
            "initial_static_parameters": {k: v[winner] for k, v in initial.items()}, "initialization": search,
            "starts": starts, "history": history, "maximum_gradient_norm_before_clip": max_grad,
            "fit_seconds": time.perf_counter()-started}

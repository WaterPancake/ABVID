"""Faithful smooth8 v1 fitter extracted from the preserved CAST implementation.

This module provides numerical fitting only. Callers own role validation,
provenance, configuration freezing, and storage; no audio is opened here.

The numerical body of ``fit`` is unchanged from
``CAST/improvement/src/cast_improvement/engine.py`` (SHA-256
``78af0f34dd24e784601ac7ada639c9e88c69ec2457642cdcbc16e4adafa0a070``).
Only imports now use the packaged, parity-checked numerical components.
"""
import time

import torch

from abvid.cast.fit import Objective, initialize
from abvid.cast.population.engine import SmoothRenderer
from abvid.cast.renderer import Parameters, serialize


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

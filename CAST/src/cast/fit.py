"""Fixed-budget multistart Adam with independent stochastic checking streams."""
import time
import numpy as np
import torch

from .audio import shape
from .renderer import Parameters, Renderer, seed_for, serialize


class Objective:
    def __init__(self, target, cfg):
        self.cfg = cfg
        self.windows = {n: torch.hann_window(n) for n in cfg["loss"]["fft_sizes"]}
        self.targets = {n: self.magnitude(shape(target)[None], n) for n in self.windows}

    def magnitude(self, x, n):
        return torch.stft(x, n_fft=n, hop_length=n//4, win_length=n, window=self.windows[n],
                          center=False, normalized=False, onesided=True, return_complex=True).abs()

    def __call__(self, waves, detail=False):
        dims = waves.shape[:-1]
        flat = shape(waves).reshape(-1, waves.shape[-1])
        totals, terms = [], {}
        eps = self.cfg["loss"]["epsilon"]
        for n in self.windows:
            a, b = self.targets[n], self.magnitude(flat, n)
            rel = (a-b).abs().mean((-1, -2))/(a.mean()+eps)
            log = ((a+eps).log()-(b+eps).log()).abs().mean((-1, -2))
            terms[str(n)] = {"relative": rel.reshape(dims), "log": log.reshape(dims)}
            totals.append(rel+log)
        total = torch.stack(totals).mean(0).reshape(dims)
        return (total, terms) if detail else total


def initialize(target, cfg, input_id):
    init, r = cfg["initialization"], cfg["renderer"]
    y = target.detach().numpy()
    grid = np.arange(r["spacing_hz_bounds"][0], r["spacing_hz_bounds"][1]+0.1, init["grid_step_hz"])
    saliences, spectra = [], []
    for center in init["centers_samples"]:
        n = init["segment_samples"]
        segment = y[center-n//2:center+n//2]
        spectrum = np.abs(np.fft.rfft(segment*np.hanning(n), n=init["search_fft"]))
        frequencies = np.fft.rfftfreq(init["search_fft"], 1/cfg["audio"]["sample_rate_hz"])
        salience = sum(np.interp(grid*h, frequencies, spectrum)/h for h in range(1, r["harmonics"]+1))
        saliences.append(salience)
        spectra.append(spectrum)
    scores = saliences[1]
    candidates = []
    for i in np.argsort(-scores, kind="stable"):
        f = float(grid[i])
        if all(abs(f-v) >= init["candidate_separation_hz"] for v in candidates):
            candidates.append(f)
        if len(candidates) == len(cfg["seeds"]["starts"]):
            break
    values = {k: [] for k in ("spacing_hz", "harmonic_weights", "noise_weights", "harmonic_fraction", "envelope_knots")}
    for index, (f, seed) in enumerate(zip(candidates, cfg["seeds"]["starts"])):
        trajectory = []
        for salience in saliences:
            local = np.abs(grid-f) <= init["trajectory_radius_hz"]
            j = int(np.argmax(np.where(local,salience,-np.inf)))
            refined = float(grid[j])
            if 0 < j < len(grid)-1:
                left, center, right = np.log(salience[j-1:j+2]+1e-12)
                denom = left-2*center+right
                if denom < -1e-12:
                    refined += float(np.clip(.5*(left-right)/denom,-.5,.5))*init["grid_step_hz"]
            trajectory.append(refined)
        # The salience probes describe segment CENTERS, whereas renderer knots
        # sit at clip endpoints. Using the probes as endpoints biases chirps.
        times = np.asarray(init["centers_samples"])/cfg["audio"]["sample_rate_hz"]
        duration = cfg["audio"]["samples"]/cfg["audio"]["sample_rate_hz"]
        trajectory[0] -= times[0]*(trajectory[1]-trajectory[0])/(times[1]-times[0])
        trajectory[2] += (duration-times[2])*(trajectory[2]-trajectory[1])/(times[2]-times[1])
        trajectory = np.clip(trajectory, *r["spacing_hz_bounds"]).tolist()
        rng = np.random.default_rng(seed_for(cfg, input_id, "initialization", "start", seed))
        weights = np.interp(trajectory[1]*np.arange(1,9), frequencies, spectra[1])
        weights = np.maximum(weights,weights.max()*init["harmonic_weight_floor"])
        weights *= np.exp(rng.normal(0, init["weight_logit_jitter_std"], 8))
        values["spacing_hz"].append(trajectory)
        values["harmonic_weights"].append((weights/weights.sum()).tolist())
        values["noise_weights"].append([1/8]*8)
        values["harmonic_fraction"].append(init["mix"][index])
        values["envelope_knots"].append([1.0]*5)
    return values, {"candidates_hz": candidates, "trajectories_hz": values["spacing_hz"], "rule": init}


def fit(target, cfg, input_id, known_harmonic_weights=None):
    started = time.perf_counter()
    initial, search = initialize(target, cfg, input_id)
    if known_harmonic_weights is not None:
        initial["harmonic_weights"] = [known_harmonic_weights for _ in cfg["seeds"]["starts"]]
    model, renderer = Parameters(initial, cfg), Renderer(cfg, input_id)
    if known_harmonic_weights is not None:
        model.harmonic_logits.requires_grad_(False)
    objective = Objective(target, cfg)
    opt = torch.optim.Adam([{"params": [model.spacing_raw], "lr": cfg["optimizer"]["spacing_lr"]},
                            {"params": [v for k,v in model.named_parameters() if k != "spacing_raw" and v.requires_grad]}], lr=cfg["optimizer"]["lr"],
                           betas=tuple(cfg["optimizer"]["betas"]), eps=cfg["optimizer"]["epsilon"])
    with torch.no_grad():
        base_controls = {k: v.detach().clone() for k, v in model.controls().items()}
        baseline_fit = objective(renderer.render(base_controls)).mean(1)
        baseline_check = objective(renderer.render(base_controls, "check")).mean(1)
    best_loss = baseline_fit.clone()
    best_controls = {k: v.clone() for k, v in base_controls.items()}
    best_steps = torch.zeros(len(best_loss), dtype=torch.int64)
    history, max_grad = [], 0.0
    # Evaluate initial state and every post-update state; never lose the best state.
    for step in range(cfg["optimizer"]["steps"]+1):
        opt.zero_grad(set_to_none=True)
        controls = model.controls()
        loss = objective(renderer.render(controls)).mean(1)
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"Nonfinite objective at step {step}")
        history.append({"step": step, "fit_losses": loss.detach().tolist()})
        with torch.no_grad():
            improved = loss < best_loss
            for k, v in controls.items():
                best_controls[k][improved] = v[improved]
            best_loss[improved] = loss[improved]
            best_steps[improved] = step
        if step == cfg["optimizer"]["steps"]:
            break
        loss.sum().backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
            raise FloatingPointError(f"Nonfinite gradient at step {step}")
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optimizer"]["gradient_clip_norm"])
        max_grad = max(max_grad, float(grad))
        opt.step()
        model.project()
    winner = int(best_loss.argmin())
    with torch.no_grad():
        checking = objective(renderer.render(best_controls, "check")).mean(1)
    starts = [{"seed": seed, "initial_parameters": serialize(base_controls, i),
               "parameters": serialize(best_controls, i), "best_step": int(best_steps[i]),
               "initial_fit_loss": float(baseline_fit[i]), "initial_check_loss": float(baseline_check[i]),
               "fit_loss": float(best_loss[i]), "check_loss": float(checking[i])}
              for i, seed in enumerate(cfg["seeds"]["starts"])]
    return {"input_id": input_id, "parameters": starts[winner]["parameters"], "winner": winner,
            "fit_loss": float(best_loss[winner]), "check_loss": float(checking[winner]),
            "baseline_fit_loss": float(baseline_fit.min()),
            "baseline_check_loss": float(baseline_check.min()),
            "baseline_check_winner": int(baseline_check.argmin()),
            "selection": "fitted_winner_by_fit_stream_only_baseline_minimum_on_common_check_stream",
            "streams": renderer.streams(), "initialization": search, "starts": starts,
            "known_harmonic_weights": known_harmonic_weights,
            "history": history, "maximum_gradient_norm_before_clip": max_grad,
            "fit_seconds": time.perf_counter()-started}

"""Train-only empirical sampling and frozen marginal coverage metrics."""
import copy
import hashlib
import numpy as np
from scipy.stats import wasserstein_distance
import torch

from cast.artifacts import descriptors
from cast.audio import shape
from cast.config import canonical, digest
from cast.renderer import Renderer, seed_for, tensors, validate_controls

KEYS = ("spacing_hz", "harmonic_weights", "noise_weights", "harmonic_fraction", "envelope_knots")
SIZES = (3, 8, 8, 1, 5)


def pack(p):
    return np.concatenate([np.atleast_1d(p[k]).astype(np.float64) for k in KEYS])


def unpack(v):
    v = np.asarray(v, dtype=np.float64)
    if v.shape != (25,) or not np.isfinite(v).all():
        raise ValueError("Expected 25 finite control coordinates")
    p, start = {}, 0
    for k, n in zip(KEYS, SIZES):
        a = v[start:start+n].copy()
        if k in ("harmonic_weights", "noise_weights"):
            if (a < 0).any() or a.sum() <= 0:
                raise ValueError("Invalid sampled simplex")
            a /= a.sum()
        p[k] = float(a[0]) if n == 1 else a.tolist()
        start += n
    return p


def draw_seed(version, label, seed, index, arm):
    return int.from_bytes(hashlib.sha256(canonical([version, "selection", label, seed, index, arm])).digest()[:8], "big")


def sample(bank, cfg, label, seed, index, arm, renderer_cfg):
    if label not in cfg["class_order"] or seed not in cfg["seeds"] or arm not in cfg["arms"] or not 0 <= index < cfg["samples_per_class_per_seed_per_arm"]:
        raise ValueError("Outside frozen sampling schedule")
    rows = sorted((r for r in bank if r["class"] == label), key=lambda r: r["file_id"])
    groups = sorted({r["group"] for r in rows})
    if not groups or cfg["held_group"] in groups:
        raise ValueError("Empty or contaminated training bank")
    grouped = [[r for r in rows if r["group"] == g] for g in groups]
    rng_seed = draw_seed(cfg["version"], label, seed, index, arm)
    rng = np.random.default_rng(rng_seed)

    def parent():
        group = grouped[int(rng.integers(len(grouped)))]
        return group[int(rng.integers(len(group)))]

    if arm == "joint":
        chosen = parent()
        p = copy.deepcopy(chosen["parameters"])
        donors = [chosen["file_id"]] * 25
        parents = [chosen]
    elif arm == "prototype":
        p = unpack(np.mean([np.mean([pack(r["parameters"]) for r in group], axis=0) for group in grouped], axis=0))
        donors = {"rule": "equal_group_mean_then_class_mean", "coordinates": "all"}
        parents = rows
    else:
        choices = [parent() for _ in range(25)]
        p = unpack([pack(r["parameters"])[i] for i, r in enumerate(choices)])
        donors = [r["file_id"] for r in choices]
        parents = sorted({r["file_id"]: r for r in choices}.values(), key=lambda r: r["file_id"])
    validate_controls(p, renderer_cfg)
    return {"class": label, "seed": seed, "index": index, "arm": arm,
            "origin": "real_calibrated_observation_synthesis", "parameters": p,
            "input_id": f'{cfg["version"]}:sampling:{label}:{seed}:{index:04d}',
            "selection_seed": rng_seed, "coordinate_order": list(zip(KEYS, SIZES)),
            "coordinate_donors": donors,
            "parent_ids": [r["file_id"] for r in parents],
            "parent_groups": sorted({r["group"] for r in parents}),
            "parent_flags": {r["file_id"]: r["flags"] for r in parents}}


def render(record, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["seeds"]["sampling_realizations"] = 1
    r = Renderer(cfg, record["input_id"])
    phase = seed_for(cfg, record["input_id"], "phase", "sampling")
    noise = seed_for(cfg, record["input_id"], "noise", "sampling", 0)
    g = torch.Generator().manual_seed(phase)
    r.phases = torch.rand(cfg["renderer"]["harmonics"], generator=g, dtype=torch.float64)*2*np.pi
    with torch.no_grad():
        wave = shape(r.render(tensors(record["parameters"]), "sampling"))[0, 0].numpy()
    if wave.shape != (32000,) or not np.isfinite(wave).all():
        raise ValueError("Nonfinite or incorrect generated waveform")
    return wave, {"purpose": "sampling", "phase": phase, "noise": [noise], "render_config_sha256": digest(cfg)}


def describe(wave, base_cfg, cfg):
    d = descriptors(wave, base_cfg)
    edges = np.linspace(0, 4000, cfg["log_spectrum_bins"]+1)
    power = np.array([d["power"][(d["frequencies"] >= lo) & (d["frequencies"] < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])])
    power /= max(power.sum(), 1e-12)
    result = {"log_spectrum": np.log(np.maximum(power, cfg["spectral_probability_floor"])),
              "bands": d["band_proportions"], "envelope": d["envelope"],
              "modulation": d["modulation"]/len(d["envelope"])}
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError("Invalid acoustic descriptors")
    return result


def stack(rows, cfg):
    return {f: np.stack([r[f] for r in rows]) for f in cfg["primary_families"]}


def scales(train, cfg):
    return {f: np.maximum(np.std(train[f], axis=0), cfg["scale_floors"][f]).tolist() for f in cfg["primary_families"]}


def compare(generated, held, scale, cfg):
    families = {}
    for f in cfg["primary_families"]:
        a, b = np.asarray(generated[f]), np.asarray(held[f])
        s = np.asarray(scale[f])
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or s.shape != a.shape[1:] or len(a) == 0 or len(b) == 0 or not all(np.isfinite(x).all() for x in (a, b, s)) or (s <= 0).any():
            raise ValueError("Invalid descriptor distributions or scales")
        # These are marginal metrics: a canonical per-coordinate order also
        # makes floating-point SD reductions invariant to input row order.
        a, b = np.sort(a, axis=0), np.sort(b, axis=0)
        distances = np.array([wasserstein_distance(a[:, j], b[:, j])/s[j] for j in range(a.shape[1])])
        low, high = np.quantile(a, cfg["coverage_quantiles"], axis=0)
        below, above = (b < low).mean(0), (b > high).mean(0)
        denom = float(np.std(b/s, axis=0).mean())
        families[f] = {"W1": float(distances.mean()), "per_coordinate_W1": distances.tolist(),
                       "coverage": float((1-below-above).mean()), "lower_tail_miss": float(below.mean()), "upper_tail_miss": float(above.mean()),
                       "spread_ratio": float(np.std(a/s, axis=0).mean()/denom) if denom > 1e-12 else None,
                       "generated_quantiles_05_50_95": np.quantile(a, [.05, .5, .95], axis=0).tolist(),
                       "held_quantiles_05_50_95": np.quantile(b, [.05, .5, .95], axis=0).tolist(),
                       "zero_spread_coordinates": int((np.std(a, axis=0) < 1e-12).sum())}
    return {"W1": float(np.mean([v["W1"] for v in families.values()])),
            "coverage": float(np.mean([v["coverage"] for v in families.values()])),
            "families": families, "generated_count": len(a), "held_count": len(b)}

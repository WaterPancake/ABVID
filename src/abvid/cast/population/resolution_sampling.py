"""Numeric components extracted from CAST/improvement/src/cast_improvement/resolution_sampling.py."""
import copy


import hashlib


import numpy as np


from abvid.cast.config import canonical


from abvid.cast.population.resolution_model import validate_controls


KEYS = ("spacing_hz", "harmonic_weights", "noise_weights", "harmonic_fraction", "envelope_knots")


SIZES = (3, 8, 16, 1, 41)


def pack(p):
    return np.concatenate([np.atleast_1d(p[k]).astype(np.float64) for k in KEYS])


def unpack(v):
    v = np.asarray(v, dtype=np.float64)
    if v.shape != (69,) or not np.isfinite(v).all():
        raise ValueError("Expected 69 finite control coordinates")
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
        donors = [chosen["file_id"]] * 69
        parents = [chosen]
    elif arm == "prototype":
        p = unpack(np.mean([np.mean([pack(r["parameters"]) for r in group], axis=0) for group in grouped], axis=0))
        donors = {"rule": "equal_group_mean_then_class_mean", "coordinates": "all"}
        parents = rows
    else:
        choices = [parent() for _ in range(69)]
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

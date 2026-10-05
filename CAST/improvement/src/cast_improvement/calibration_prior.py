"""First-order source-only correction of persistent paired band-power bias."""
from copy import deepcopy
from itertools import product
import numpy as np

from cast.renderer import validate_controls
from .block_prior import transform_bank as temper, draw as base_draw
from .evaluate import aggregate

GRID = tuple(product((.5, .65, .8, 1.), (0., 1.)))


def method_id(spectral, strength):
    if (spectral, strength) not in GRID:
        raise ValueError("Undeclared calibration candidate")
    return f"smooth8_s{spectral:g}_bias{strength:g}"


def transform_bank(bank, cfg, spectral, strength, paired):
    method_id(spectral, strength)
    by_id = {r["file_id"]: r for r in paired}
    if len(by_id) != len(paired) or set(by_id) != {r["file_id"] for r in bank}:
        raise ValueError("Calibration parents must exactly equal source-training bank")
    for row in bank:
        p = by_id[row["file_id"]]
        if ((p["class"], p["group"]) != (row["class"], row["group"])
            or p.get("fit_sha256") != row.get("fit_sha256")):
            raise ValueError("Calibration ancestry mismatch")
        for field in ("observed_bands", "reconstructed_bands"):
            a = np.asarray(p[field])
            if a.shape != (8,) or not np.isfinite(a).all() or (a < 0).any() or abs(a.sum()-1) > 1e-6:
                raise ValueError("Invalid calibration power proportions")
    transformed, stats = temper(bank, cfg, spectral, 1.)
    for label in ("car", "truck"):
        group_ratios = {}
        for group in stats[label]["groups"]:
            rows = sorted((p for p in paired if p["class"] == label and p["group"] == group), key=lambda p:p["file_id"])
            observed = np.mean([p["observed_bands"] for p in rows], axis=0)
            reconstructed = np.mean([p["reconstructed_bands"] for p in rows], axis=0)
            group_ratios[group] = np.log(np.maximum(observed, 1e-8)/np.maximum(reconstructed, 1e-8)).tolist()
        correction = np.mean(list(group_ratios.values()), axis=0).clip(-np.log(2), np.log(2))
        stats[label].update(correction_strength=strength, group_log_band_ratios=group_ratios,
                            clipped_log_noise_correction=correction.tolist(),
                            correction_rule="equal_group_mean_log_observed_over_reconstructed_band_power; clip +/-ln2")
        for row in transformed:
            if row["class"] != label or strength == 0:
                continue
            p = deepcopy(row["parameters"])
            w = np.asarray(p["noise_weights"])*np.exp(strength*correction)
            p["noise_weights"] = (w/w.sum()).tolist()
            row["parameters"] = validate_controls(p, cfg)
    return transformed, stats


def draw(bank, statistics, cfg, metric_cfg, label, seed, index, arm, fold, spectral, strength):
    r = base_draw(bank, statistics, cfg, metric_cfg, label, seed, index, arm, fold, spectral, 1.)
    r.pop("envelope_temperature")
    r.update(method_id=method_id(spectral, strength), correction_strength=strength,
             calibration_role="all class-center and paired spectral-bias parents; retained also when correction is zero")
    return r


def summarize(records, metric_cfg):
    result = aggregate(records, metric_cfg)
    settings = {method_id(s, a):(s,a) for s,a in GRID}
    for row in result.values():
        spectral, strength = settings[row["variant"]]
        row.update(spectral_temperature=spectral, correction_strength=strength)
        row["selection_key"] = row["selection_key"][:3]+[abs(spectral-1)+strength, spectral, strength]
    return result

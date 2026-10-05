"""Versioned source-only prior with separate spectral/envelope temperatures."""
from copy import deepcopy
from itertools import product
import numpy as np

from cast.renderer import validate_controls
from cast_generalization.method import sample
from .evaluate import transformed, inverse, aggregate


SPECTRAL = (.5, .65, .8, 1.)
ENVELOPE = (1., 1.25, 1.5)
GRID = tuple(product(SPECTRAL, ENVELOPE))


def method_id(spectral, envelope):
    if (spectral, envelope) not in GRID:
        raise ValueError("Undeclared block temperature pair")
    return f"smooth8_s{spectral:g}_e{envelope:g}"


def transform_bank(bank, cfg, spectral, envelope):
    method_id(spectral, envelope)
    if len({r["file_id"] for r in bank}) != len(bank):
        raise ValueError("Duplicate prior parent")
    if set(r["class"] for r in bank) != {"car", "truck"}:
        raise ValueError("Both classes required, without undeclared labels")
    result, statistics = [], {}
    scale = np.array([spectral]*20+[envelope]*5, dtype=np.float64)
    for label in ("car", "truck"):
        rows = sorted((r for r in bank if r["class"] == label), key=lambda r: r["file_id"])
        groups = sorted({r["group"] for r in rows})
        for row in rows:
            validate_controls(row["parameters"], cfg)
        center = np.mean([np.mean([transformed(r["parameters"]) for r in rows if r["group"] == g], axis=0) for g in groups], axis=0)
        statistics[label] = {"center": center.tolist(), "parent_ids": [r["file_id"] for r in rows],
                             "groups": groups, "spectral_temperature": spectral,
                             "envelope_temperature": envelope, "center_rule": "equal_group_mean_in_transformed_coordinates"}
        for row in rows:
            p = deepcopy(row["parameters"]) if (spectral, envelope) == (1., 1.) else inverse(center+scale*(transformed(row["parameters"])-center), cfg)
            result.append({**row, "parameters": p})
    return sorted(result, key=lambda r: r["file_id"]), statistics


def draw(bank, statistics, cfg, metric_cfg, label, seed, index, arm, fold, spectral, envelope):
    sampling_cfg = deepcopy(metric_cfg)
    # Retain common random numbers from prior v1 in every matched comparison.
    sampling_cfg["version"] = f"cast_improvement_sampling_v1:{fold}"
    record = sample(bank, sampling_cfg, label, seed, index, arm, cfg)
    record.pop("parent_flags")
    calibration = statistics[label]
    forbidden = {metric_cfg["held_group"]}
    if fold != "outer":
        forbidden.add(fold)
    if forbidden.intersection(record["parent_groups"]) or forbidden.intersection(calibration["groups"]):
        raise ValueError("Direct or class-center ancestry crosses validation boundary")
    return {**record, "method_id": method_id(spectral, envelope), "fold": fold,
            "spectral_temperature": spectral, "envelope_temperature": envelope,
            "prior_calibration_parent_ids": calibration["parent_ids"],
            "prior_calibration_groups": calibration["groups"],
            "calibration_role": "all class-center inputs; conservative ancestry also retained at unit temperature"}


def summarize(records, metric_cfg):
    candidates = aggregate(records, metric_cfg)
    settings = {method_id(s, e): (s, e) for s, e in GRID}
    for result in candidates.values():
        spectral, envelope = settings[result["variant"]]
        result.update(spectral_temperature=spectral, envelope_temperature=envelope)
        result["selection_key"] = result["selection_key"][:3]+[abs(spectral-1)+abs(envelope-1), spectral, envelope]
    return candidates

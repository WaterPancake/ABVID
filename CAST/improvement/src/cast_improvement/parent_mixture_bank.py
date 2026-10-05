"""Per-parent calibration with explicit local ancestry and unchanged sampling."""
from copy import deepcopy
import numpy as np
from scipy.signal import welch

from cast.config import digest, sha
from cast.renderer import validate_controls
from cast_generalization.method import sample
from .parent_mixture_prior import calibrate

MODES = ("winner", "parent_calibrated")


def component_power(values, edges, sample_rate=16000):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (1, 2, 32000) or not np.isfinite(values).all():
        raise ValueError("Expected two finite saved checking components")
    f, power = welch(values[0], sample_rate, window="hann", nperseg=2048, noverlap=1024, axis=-1)
    power = power.mean(axis=0)
    return np.array([power[(f >= lo) & (f < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])])


def derived_row(row, harmonic, noise, target, input_hashes, cfg):
    fraction = row["parameters"]["harmonic_fraction"]
    h = component_power(harmonic, cfg["renderer"]["noise_edges_hz"])
    n = component_power(noise, cfg["renderer"]["noise_edges_hz"])
    # At a saved boundary the missing component is unrecoverable; calibrate
    # explicitly retains that parent and records an undefined-calibration flag.
    if 0 < fraction < 1:
        h /= fraction
        n /= 1-fraction
    record = {"file_id": row["file_id"], "class": row["class"], "group": row["group"],
              "input_hashes": input_hashes, "calibration_inputs": "same_parent_two_checking_components_and_observed_bands",
              **calibrate(fraction, h, n, target)}
    result = deepcopy(row)
    result["parameters"]["harmonic_fraction"] = record["harmonic_fraction"]
    validate_controls(result["parameters"], cfg)
    result["mixture_calibration"] = record
    result["mixture_calibration_sha256"] = digest(record)
    return result


def derive_bank(run, bank, observations, cfg, held_group):
    ids = [r["file_id"] for r in bank]
    targets = {r["file_id"]: r for r in observations}
    if (len(ids) != len(set(ids)) or len(targets) != len(observations)
        or set(ids) != set(targets) or any(r["group"] == held_group for r in bank)):
        raise ValueError("Calibration must use exactly the non-outer training parents")
    result = []
    for row in bank:
        obs = targets[row["file_id"]]
        if any(row[k] != obs[k] for k in ("file_id", "class", "group", "parent")):
            raise ValueError("Calibration observation ancestry differs")
        folder = run/"fits"/row["file_id"]
        inputs = {name: sha(folder/name) for name in ("fit.json", "components.npz", "original_shape.wav")}
        if inputs["fit.json"] != row["fit_sha256"]:
            raise ValueError("Calibration fit changed")
        with np.load(folder/"components.npz", allow_pickle=False) as components:
            result.append(derived_row(row, components["harmonic"], components["noise"],
                                      obs["descriptors"]["bands"], inputs, cfg))
    return result


def draw(bank, cfg, metric, label, seed, index, arm, variant, fold):
    if variant not in MODES or any(r["group"] in (fold, metric["held_group"]) for r in bank):
        raise ValueError("Undeclared bank or excluded source ancestor")
    scfg = {**metric, "version": f"cast_improvement_sampling_v1:{fold}"}
    row = sample(bank, scfg, label, seed, index, arm, cfg)
    row.pop("parent_flags")
    donors = {r["file_id"]: r for r in bank}
    ancestry = {}
    if variant == "parent_calibrated":
        for parent in row["parent_ids"]:
            donor = donors[parent]
            record = donor["mixture_calibration"]
            if (any(record[k] != donor[k] for k in ("file_id", "class", "group"))
                or digest(record) != donor["mixture_calibration_sha256"]
                or record["harmonic_fraction"] != donor["parameters"]["harmonic_fraction"]):
                raise ValueError("Invalid calibration ancestry or parameters")
            ancestry[parent] = donor["mixture_calibration_sha256"]
    return {**row, "calibration_ancestors": ancestry}

"""Empirical uncertainty over equally good numerical fits; parent weights fixed."""
from copy import deepcopy
import hashlib
import numpy as np

from cast.config import canonical
from cast.renderer import validate_controls
from cast_generalization.method import pack, unpack, sample as winner_sample

MODES = ("winner", "equivalent_fits")


def eligible_starts(result, cfg):
    limit = cfg["diagnostics"]["equally_good_relative_loss"]
    if limit != .05 or result["winner"] != int(np.argmin([s["fit_loss"] for s in result["starts"]])):
        raise ValueError("Changed ambiguity bound or winning start")
    answer = []
    for i, s in enumerate(result["starts"]):
        if s["fit_loss"] <= result["fit_loss"]*(1+limit):
            validate_controls(s["parameters"], cfg)
            answer.append({"start_index": i, "start_seed": s["seed"], "fit_loss": s["fit_loss"],
                           "parameters": deepcopy(s["parameters"])})
    if not answer or result["winner"] not in [s["start_index"] for s in answer]:
        raise ValueError("Missing best alternative")
    return answer


def alternative_seed(cfg, label, seed, index, arm, coordinate):
    key = [cfg["version"], "equivalent_fit_alternative_v4", label, seed, index, arm, coordinate]
    return int.from_bytes(hashlib.sha256(canonical(key)).digest()[:8], "big")


def sample(bank, cfg, label, seed, index, arm, renderer_cfg, mode):
    if mode not in MODES:
        raise ValueError("Undeclared alternative-fit mode")
    common = winner_sample(bank, cfg, label, seed, index, arm, renderer_cfg)
    rows = {r["file_id"]: r for r in bank}
    if len(rows) != len(bank):
        raise ValueError("Duplicate parent ID")
    choices, seeds = [], []

    def alternative(parent, coordinate):
        row = rows[parent]
        alternatives = row["alternatives"]
        if not alternatives:
            raise ValueError("Missing parent alternatives")
        if mode == "winner":
            result = next(a for a in alternatives if a["start_index"] == row["winning_start_index"])
        else:
            derived = alternative_seed(cfg, label, seed, index, arm, coordinate)
            seeds.append(derived)
            result = alternatives[int(np.random.default_rng(derived).integers(len(alternatives)))]
        choices.append({"parent_id": parent, "start_index": result["start_index"], "start_seed": result["start_seed"]})
        return result["parameters"]

    if arm == "joint":
        selected = alternative(common["parent_ids"][0], 0)
        coordinates = [choices[0]]*25
        if mode != "winner":
            common["parameters"] = deepcopy(selected)
    elif arm == "marginals":
        parameters = [alternative(parent, coordinate) for coordinate, parent in enumerate(common["coordinate_donors"])]
        coordinates = deepcopy(choices)
        if mode != "winner":
            common["parameters"] = unpack([pack(p)[i] for i, p in enumerate(parameters)])
    else:
        groups = sorted({rows[parent]["group"] for parent in common["parent_ids"]})
        parent_means = {}
        for parent in common["parent_ids"]:
            row = rows[parent]
            selected = row["alternatives"] if mode != "winner" else [
                a for a in row["alternatives"] if a["start_index"] == row["winning_start_index"]]
            parent_means[parent] = np.mean([pack(a["parameters"]) for a in selected], axis=0)
            choices.extend({"parent_id": parent, "start_index": a["start_index"], "start_seed": a["start_seed"]} for a in selected)
        if mode != "winner":
            common["parameters"] = unpack(np.mean([
                np.mean([parent_means[parent] for parent in common["parent_ids"] if rows[parent]["group"] == g], axis=0)
                for g in groups], axis=0))
        coordinates = {"rule": "equal_start_within_parent_then_equal_parent_within_group_then_equal_group", "coordinates": "all"}
    validate_controls(common["parameters"], renderer_cfg)
    unique = {(r["parent_id"], r["start_index"]): r for r in choices}
    return {**common, "prior_variant": mode, "coordinate_start_donors": coordinates,
            "alternative_ancestors": [unique[k] for k in sorted(unique)], "alternative_choice_seeds": seeds,
            "alternative_weighting": "uniform_starts_within_uniform_parent_within_uniform_group"}

"""Pure summaries for the frozen CAST-4A score schedule.

Rows have ``partition``, ``class``, ``seed``, ``arm``, ``score`` (the unchanged
``coverage.compare`` result), five explicit ``gates`` and a ``failures`` list.
This module fits nothing, reads no files and never drops a failed seed or fold.
The input partition list is the complete frozen nine-partition inventory.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from numbers import Real

import numpy as np


CLASSES = ("car", "truck")
SEEDS = (42, 123, 456, 789, 1024)
FAMILIES = ("log_spectrum", "bands", "envelope", "modulation")
WIDTHS = ("v14", "v15")
SAMPLERS = ("joint", "prototype", "marginals")
ARMS = tuple(f"{width}_{sampler}" for width in WIDTHS for sampler in SAMPLERS)
GATES = ("input", "completeness", "numerical", "ancestry", "replay")
FOLD_COUNTS = {"group": 6, "site": 3}
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 314159
METRICS = ("W1", "coverage", *(f"spread:{family}" for family in FAMILIES))


def _number(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def _safe(value):
    """JSON-safe preservation: invalid numeric values remain explicit nulls."""
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _mean(values):
    if not values or not all(_number(v) for v in values):
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.mean(values))
    return result if _number(result) else None


def _range(values):
    complete = bool(values) and all(_number(v) for v in values)
    return {"values": _safe(values), "min": float(min(values)) if complete else None,
            "max": float(max(values)) if complete else None, "all_seeds_numeric": complete}


def _configuration(cfg, partitions):
    required = {"class_order": list(CLASSES), "seeds": list(SEEDS),
                "primary_families": list(FAMILIES), "arms": list(SAMPLERS),
                "samples_per_class_per_seed_per_arm": 50, "coverage_quantiles": [.05, .95]}
    if not isinstance(cfg, Mapping) or any(cfg.get(key) != value for key, value in required.items()):
        raise ValueError("Metrics must use the frozen CAST-4A classes, families, seeds and budgets")
    # These are the protocol's effective criteria. A legacy 0.05 field may remain
    # historical metadata, but cannot override the frozen effective 0.025 margin.
    criteria = cfg.get("criteria", {})
    pinned = {"mean_marginal_coverage_each_class_each_fold": .8,
              "mean_seed_family_spread_ratio_bounds_each_class_each_fold": [.5, 2.],
              "relative_W1_gain_over_each_matched_width_control_each_class_each_fold": .025}
    if any(key in criteria and criteria[key] != value for key, value in pinned.items()):
        raise ValueError("Changed effective grouped-coverage acceptance criteria")
    if not isinstance(partitions, Sequence) or isinstance(partitions, (str, bytes)):
        raise ValueError("The complete frozen partition sequence is required")
    expected = {f"{analysis}_{fold}" for analysis, count in FOLD_COUNTS.items() for fold in range(count)}
    if (len(partitions) != len(expected) or any(not isinstance(part, Mapping) for part in partitions)
            or {part.get("id") for part in partitions} != expected):
        raise ValueError("All six group and three site partitions are required")
    for part in partitions:
        if (part.get("analysis") not in FOLD_COUNTS or type(part.get("fold")) is not int or
                part["id"] != f"{part['analysis']}_{part['fold']}" or
                not isinstance(part.get("held_counts"), Mapping) or
                any(type(part["held_counts"].get(label)) is not int or part["held_counts"][label] <= 0
                    for label in CLASSES)):
            raise ValueError("Invalid frozen partition identity or held support")
    return {part["id"]: part for part in partitions}


def _score_values(score, held_count):
    errors = []
    values = dict.fromkeys(METRICS)
    if not isinstance(score, Mapping):
        return values, ["score missing or malformed"]
    def inspect(value, path):
        if isinstance(value, Mapping):
            for key, item in value.items():
                inspect(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                inspect(item, f"{path}[{index}]")
        elif isinstance(value, Real) and not math.isfinite(value):
            errors.append(f"nonfinite {path}")
    inspect(score, "score")
    for key, count in (("generated_count", 50), ("held_count", held_count)):
        if type(score.get(key)) is not int or score[key] != count:
            errors.append(f"{key} does not match the complete frozen population")
    for key in ("W1", "coverage"):
        value = score.get(key)
        if not _number(value) or value < 0 or (key == "coverage" and value > 1):
            errors.append(f"invalid score.{key}")
        else:
            values[key] = float(value)
    families = score.get("families")
    if not isinstance(families, Mapping) or set(families) != set(FAMILIES):
        return values, errors + ["score must retain all four descriptor families"]
    for family in FAMILIES:
        item = families[family]
        if not isinstance(item, Mapping):
            errors.append(f"invalid score.families.{family}")
            continue
        for key in ("W1", "coverage", "spread_ratio"):
            value = item.get(key)
            if not _number(value) or value < 0 or (key == "coverage" and value > 1):
                errors.append(f"invalid score.families.{family}.{key}")
            elif key == "spread_ratio":
                values[f"spread:{family}"] = float(value)
    for key in ("W1", "coverage"):
        family_values = [families[family].get(key) if isinstance(families[family], Mapping) else None
                         for family in FAMILIES]
        mean = _mean(family_values)
        if mean is None or values[key] is None or not math.isclose(mean, values[key], rel_tol=1e-12, abs_tol=1e-12):
            errors.append(f"score.{key} is not the equal-family mean")
    return values, errors


def _row(row, part):
    errors = []
    gates = row.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != set(GATES):
        errors.append("all five explicit technical gates are required")
    else:
        errors.extend(f"technical gate failed or unknown: {key}" for key in GATES if gates[key] is not True)
    reported = row.get("failures")
    if not isinstance(reported, list):
        errors.append("explicit failures list required")
    elif reported:
        errors.append("runner reported failures")
    values, score_errors = _score_values(row.get("score"), part["held_counts"][row["class"]])
    errors.extend(score_errors)
    return {"record": _safe(row), "validation_failures": errors, "valid": not errors,
            "values": values}


def _arm_summary(slots, partition, label, arm):
    rows = [slots[(partition, label, seed, arm)] for seed in SEEDS]
    means = {metric: _mean([row["values"][metric] for row in rows]) for metric in METRICS}
    return {"means": means, "technical_complete": all(row["valid"] for row in rows),
            "seed_variation": {metric: _range([row["values"][metric] for row in rows]) for metric in METRICS},
            "failures": [{"seed": seed, "reasons": row["validation_failures"]}
                         for seed, row in zip(SEEDS, rows) if not row["valid"]]}


def _width_summary(arms, width):
    selected = {sampler: arms[f"{width}_{sampler}"] for sampler in SAMPLERS}
    joint = selected["joint"]["means"]
    gains = {}
    for control in ("prototype", "marginals"):
        denominator = selected[control]["means"]["W1"]
        value = (1. - joint["W1"] / denominator
                 if _number(joint["W1"]) and _number(denominator) and denominator > 0 else None)
        gains[control] = value if _number(value) else None
    technical = all(arm["technical_complete"] for arm in selected.values())
    passes = {"coverage": _number(joint["coverage"]) and joint["coverage"] >= .8,
              "gain": all(_number(value) and value >= .025 for value in gains.values()),
              "spread": all(_number(joint[f"spread:{family}"]) and .5 <= joint[f"spread:{family}"] <= 2.
                            for family in FAMILIES),
              "technical": technical}
    return {"temperature": 1. if width == "v14" else 1.1, "arms": selected,
            "relative_W1_gains": gains, "passes": passes, "passed": all(passes.values()),
            "technical_complete": technical,
            "criterion_failure_observed": technical and not all(passes[key] for key in ("coverage", "gain", "spread"))}


def _worst(values, partitions, *, largest):
    observed = [(part, value) for part, value in zip(partitions, values) if _number(value)]
    if not observed:
        return {"value": None, "partitions": [], "observed_folds": 0, "all_folds_numeric": False}
    value = (max if largest else min)(value for _, value in observed)
    return {"value": float(value), "partitions": [part for part, v in observed if v == value],
            "observed_folds": len(observed), "all_folds_numeric": len(observed) == len(partitions)}


def _contrasts(fold):
    result = {}
    def difference(a, b):
        value = float(a - b) if _number(a) and _number(b) else None
        return value if _number(value) else None
    a, b = (fold[width]["arms"]["joint"]["means"] for width in WIDTHS)
    result["coverage_v15_minus_v14"] = difference(b["coverage"], a["coverage"])
    result["W1_v14_minus_v15"] = difference(a["W1"], b["W1"])
    for family in FAMILIES:
        result[f"spread_{family}_v15_minus_v14"] = difference(b[f"spread:{family}"], a[f"spread:{family}"])
    for width in WIDTHS:
        for control in ("prototype", "marginals"):
            selected = fold[width]
            result[f"W1_{width}_{control}_minus_joint"] = difference(
                selected["arms"][control]["means"]["W1"], selected["arms"]["joint"]["means"]["W1"])
            result[f"relative_W1_gain_{width}_{control}"] = selected["relative_W1_gains"][control]
    return result


def _analysis(analysis, fold_summaries):
    folds = [row for row in fold_summaries if row["analysis"] == analysis]
    ids = [row["partition"] for row in folds]
    complete = all(row["classes"][label][width]["technical_complete"]
                   for row in folds for label in CLASSES for width in WIDTHS)
    # One joint resampling matrix is shared by every class/arm/contrast. Seeds
    # have already been averaged with all five retained; they are not resampled.
    indices = (np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, len(folds), size=(BOOTSTRAP_DRAWS, len(folds))) if complete else None)
    class_summaries = {}
    for label in CLASSES:
        widths = {}
        for width in WIDTHS:
            rows = [fold["classes"][label][width] for fold in folds]
            arms = {}
            for sampler in SAMPLERS:
                vectors = {metric: [row["arms"][sampler]["means"][metric] for row in rows] for metric in METRICS}
                arms[sampler] = {"equal_fold_means": {metric: _mean(values) for metric, values in vectors.items()},
                                 "worst_observed_fold": {metric: _worst(values, ids, largest=metric != "coverage")
                                                         for metric, values in vectors.items()}}
                # Spread has two adverse directions; never reduce it to only max.
                arms[sampler]["minimum_observed_family_spread"] = {
                    family: _worst(vectors[f"spread:{family}"], ids, largest=False) for family in FAMILIES}
            widths[width] = {"arms": arms,
                             "equal_fold_relative_W1_gains": {control: _mean([row["relative_W1_gains"][control] for row in rows])
                                                               for control in ("prototype", "marginals")},
                             "worst_observed_relative_W1_gains": {control: _worst(
                                 [row["relative_W1_gains"][control] for row in rows], ids, largest=False)
                                 for control in ("prototype", "marginals")},
                             "all_folds_passed": all(row["passed"] for row in rows)}
        class_summaries[label] = widths
    balanced = {}
    for width in WIDTHS:
        balanced[width] = {
            "arms": {sampler: {"equal_fold_equal_class_means": {
                metric: _mean([class_summaries[label][width]["arms"][sampler]["equal_fold_means"][metric]
                               for label in CLASSES]) for metric in METRICS}} for sampler in SAMPLERS},
            "equal_fold_equal_class_relative_W1_gains": {control: _mean([
                class_summaries[label][width]["equal_fold_relative_W1_gains"][control] for label in CLASSES])
                for control in ("prototype", "marginals")}}
    vectors = {label: [_contrasts(fold["classes"][label]) for fold in folds] for label in CLASSES}
    contrast_names = list(vectors["car"][0])
    contrasts = {}
    for name in contrast_names:
        per_class = {label: [record[name] for record in vectors[label]] for label in CLASSES}
        per_class["equal_class"] = [_mean([per_class[label][i] for label in CLASSES]) for i in range(len(folds))]
        contrasts[name] = {}
        for label, values in per_class.items():
            estimate = _mean(values)
            can_bootstrap = indices is not None and estimate is not None
            interval = (np.quantile(np.asarray(values)[indices].mean(axis=1), [.025, .975], method="linear").tolist()
                        if can_bootstrap else None)
            contrasts[name][label] = {"per_fold": _safe(values), "equal_fold_mean": estimate,
                                      "ci95": interval,
                                      "ci_status": "complete_schedule_paired_whole_fold" if can_bootstrap
                                      else ("unavailable_incomplete_analysis" if not complete else "undefined_nonfinite_contrast")}
    criterion_failures = []
    technical_failures = []
    for fold in folds:
        for label in CLASSES:
            for width in WIDTHS:
                item = fold["classes"][label][width]
                identity = {"partition": fold["partition"], "class": label, "width": width}
                if not item["technical_complete"]:
                    technical_failures.append(identity)
                elif item["criterion_failure_observed"]:
                    criterion_failures.append({**identity, "failed_criteria": [key for key in ("coverage", "gain", "spread")
                                                                             if not item["passes"][key]]})
    return {"analysis": analysis, "partitions": ids, "complete": complete,
            "status": "complete" if complete else "technical_incompletion",
            "all_folds_passed_by_width": {width: complete and all(class_summaries[label][width]["all_folds_passed"] for label in CLASSES)
                                          for width in WIDTHS},
            "criterion_failures": criterion_failures, "technical_failures": technical_failures,
            "classes": class_summaries, "equal_class": balanced, "contrasts": contrasts,
            "uncertainty": {"draws": BOOTSTRAP_DRAWS if complete else 0, "planned_draws": BOOTSTRAP_DRAWS,
                            "generator": "numpy.random.Generator(PCG64(314159))", "reset_per_analysis": True,
                            "folds_per_draw": len(folds), "paired": True, "seed_resampling": False,
                            "quantile_method": "linear", "interval": [.025, .975],
                            "interpretation": "descriptive conditional intervals; overlapping training sets and three sites; not independent confirmation"}}


def summarize(records, partitions, metric_cfg):
    """Aggregate the full declared score schedule, retaining incomplete outcomes.

    Unknown row identities/configuration are rejected. Missing or duplicated
    expected rows, failed gates and invalid scores are reported without filling,
    favorable imputation, selected-subset intervals or alternative seed draws.
    """
    parts = _configuration(metric_cfg, partitions)
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("Score records must be an explicit sequence")
    indexed = defaultdict(list)
    retained = []
    for row in records:
        if (not isinstance(row, Mapping) or row.get("partition") not in parts or row.get("class") not in CLASSES
                or type(row.get("seed")) is not int or row["seed"] not in SEEDS or row.get("arm") not in ARMS):
            raise ValueError("Score row is outside the frozen partition/class/seed/arm schedule")
        checked = _row(row, parts[row["partition"]])
        retained.append(checked)
        indexed[(row["partition"], row["class"], row["seed"], row["arm"])].append(checked)
    slots = {}
    schedule_failures = []
    for partition in sorted(parts):
        for label in CLASSES:
            for seed in SEEDS:
                for arm in ARMS:
                    key = (partition, label, seed, arm)
                    rows = indexed[key]
                    if len(rows) == 1:
                        slots[key] = rows[0]
                    else:
                        reason = "missing score row" if not rows else "duplicate score rows"
                        for row in rows:
                            row["valid"] = False
                            row["validation_failures"].append(reason)
                        identity = dict(zip(("partition", "class", "seed", "arm"), key))
                        schedule_failures.append({**identity, "failure": reason, "observed_rows": len(rows)})
                        slots[key] = {"valid": False, "values": dict.fromkeys(METRICS), "validation_failures": [reason]}
    fold_summaries = []
    for partition, part in sorted(parts.items()):
        classes = {}
        for label in CLASSES:
            arms = {arm: _arm_summary(slots, partition, label, arm) for arm in ARMS}
            classes[label] = {width: _width_summary(arms, width) for width in WIDTHS}
        fold_summaries.append({"partition": partition, "analysis": part["analysis"], "fold": part["fold"],
                               "held_counts": dict(part["held_counts"]), "classes": classes})
    analyses = {analysis: _analysis(analysis, fold_summaries) for analysis in FOLD_COUNTS}
    primary = analyses["group"]
    v15_failures = [row for row in primary["criterion_failures"] if row["width"] == "v15"]
    valid_primary = sum(fold["classes"][label]["v15"]["technical_complete"] for fold in fold_summaries
                        if fold["analysis"] == "group" for label in CLASSES)
    retained.sort(key=lambda item: (item["record"]["partition"], item["record"]["class"],
                                   item["record"]["seed"], ARMS.index(item["record"]["arm"]),
                                   json.dumps(item["record"], sort_keys=True, allow_nan=False)))
    return {"schema_version": 1, "scope": "source_descriptor_coverage_only_not_transfer",
            "status": "complete" if all(item["complete"] for item in analyses.values()) else "technical_incompletion",
            "primary_v15_all_groups_passed": primary["all_folds_passed_by_width"]["v15"],
            "primary_v15_falsified_on_complete_fold_class": bool(v15_failures) if valid_primary else None,
            "primary_v15_criterion_failures": v15_failures,
            "criteria": {"coverage_min": .8, "relative_W1_gain_min_each_matched_control": .025,
                         "mean_seed_family_spread_bounds": [.5, 2.], "all_technical_gates_required": list(GATES),
                         "legacy_0_05_threshold_active": False, "pooled_mean_can_rescue_fold": False},
            "schedule": {"expected_rows": 540, "observed_rows": len(records),
                         "valid_rows": sum(row["valid"] for row in retained), "seeds": list(SEEDS),
                         "failures": schedule_failures},
            "seed_records": retained, "folds": fold_summaries, "analyses": analyses}

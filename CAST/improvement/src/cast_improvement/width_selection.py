"""Original source criteria plus a preregistered minimum-fold coverage gate."""
from collections import Counter
from itertools import product
import numpy as np

from .evaluate import aggregate as original_aggregate
from .width_prior import VARIANT, TEMPERATURES


def aggregate(records, metric, folds):
    folds = tuple(folds)
    if len(folds) != 5 or len(set(folds)) != 5 or metric["held_group"] in folds:
        raise ValueError("Require five distinct admitted source folds")
    fields = ("fold", "variant", "temperature", "class", "arm", "seed")
    expected = set(product(folds, (VARIANT,), TEMPERATURES, metric["class_order"], metric["arms"], metric["seeds"]))
    actual = Counter(tuple(r[k] for k in fields) for r in records)
    if set(actual) != expected or any(n != 1 for n in actual.values()):
        raise ValueError("Incomplete, duplicated or undeclared source score schedule")
    for row in records:
        score = row["score"]
        if (not np.isfinite(score["W1"]) or score["W1"] < 0 or
                not np.isfinite(score["coverage"]) or not 0 <= score["coverage"] <= 1):
            raise ValueError("Invalid source score")
        for family in metric["primary_families"]:
            spread = score["families"][family]["spread_ratio"]
            if not np.isfinite(spread) or spread < 0:
                raise ValueError("Invalid source spread")
    candidates = original_aggregate(records, metric)
    for candidate in candidates.values():
        fold_coverage = {}
        for label in metric["class_order"]:
            fold_coverage[label] = {}
            for fold in folds:
                scores = [r["score"]["coverage"] for r in records
                          if (r["fold"], r["variant"], r["temperature"], r["class"], r["arm"]) ==
                          (fold, candidate["variant"], candidate["temperature"], label, "joint")]
                fold_coverage[label][fold] = float(np.mean(scores))
        minima = {c:min(values.values()) for c, values in fold_coverage.items()}
        fold_pass = all(value >= .8 for value in minima.values())
        original_pass = candidate["passes"]
        original_key = candidate["selection_key"]
        eligible = original_pass and fold_pass
        candidate.update(original_mean_criteria_passed=original_pass,
            original_selection_key=original_key, source_fold_coverage=fold_coverage,
            minimum_source_fold_coverage_by_class=minima,
            source_fold_coverage_passed=fold_pass, eligible=eligible, passes=eligible,
            selection_key=[not eligible] + original_key)
    return candidates

"""CAST-5A arm statistics: paired fold x selection bootstrap and decision flags.

Arm score = equal-weight mean over the 6 folds x 5 selections of each head's
held-fold metric. Draw 0 is the point estimate. Site sensitivity resamples the
three sites, each carrying both of its date folds, together with selections.
"""
from __future__ import annotations

import numpy as np

from abvid.evaluation.ranking import interval, selection_draws

SEEDS = (42, 123, 456, 789, 1024)
REPS = ("BEATs768", "MFCC26")
METRIC_KEYS = ("macro_f1", "balanced_accuracy", "auroc", "average_precision", "positive_recall", "negative_recall")
CONTRASTS = (("RC", "RA"), ("RC", "RR"), ("RC", "R"), ("RC", "RU"), ("C", "U"), ("RA", "R"))
MARGIN, RECALL_LIMIT = 0.03, 0.05


def grids(per_head, parts):
    """(rep, arm, metric) -> seeds x folds matrix; NaN marks a missing head."""
    folds = sorted({p["fold"] for p in parts})
    by_id = {p["id"]: p for p in parts}
    out = {}
    for h in per_head:
        part = by_id[h["partition"]]
        seed = SEEDS.index(int(part["analysis"].removeprefix("utility_s")))
        for key in METRIC_KEYS:
            out.setdefault((h["representation"], h["arm"], key),
                           np.full((len(SEEDS), len(folds)), np.nan))[seed, folds.index(part["fold"])] = h[key]
    return out, folds


def weighted(matrix, selection_weights, fold_weights):
    if np.isnan(matrix).any():
        return None
    mass = selection_weights[:, :, None] * fold_weights[:, None, :]
    return (mass * matrix[None]).sum(axis=(1, 2)) / mass.sum(axis=(1, 2))


def summary(samples):
    return None if samples is None else {"estimate": float(samples[0]), "ci95": interval(samples[1:])}


def statistics(per_head, parts, fold_site, draws=10000, seed=20261013):
    grid, folds = grids(per_head, parts)
    sites = sorted(set(fold_site.values()))
    rng = np.random.Generator(np.random.PCG64(seed))
    sw = np.vstack([np.ones(len(SEEDS), int), selection_draws(rng, draws, len(SEEDS))])
    fw = np.vstack([np.ones(len(folds), int), selection_draws(rng, draws, len(folds))])
    site = np.vstack([np.ones(len(sites), int), selection_draws(rng, draws, len(sites))])
    site_fw = site[:, [sites.index(fold_site[f]) for f in folds]]
    out = {"arms": {}, "contrasts": {}, "site_sensitivity": {}}
    for (rep, arm, key), matrix in grid.items():
        out["arms"].setdefault(rep, {}).setdefault(arm, {})[key] = summary(weighted(matrix, sw, fw))
    for rep in REPS:
        for a, b in CONTRASTS:
            for key in METRIC_KEYS:
                for weights, target in ((fw, out["contrasts"]), (site_fw, out["site_sensitivity"])):
                    sa, sb = (weighted(grid[(rep, arm, key)], sw, weights) for arm in (a, b))
                    target.setdefault(rep, {}).setdefault(f"{a}-{b}", {})[key] = (
                        None if sa is None or sb is None else summary(sa - sb))
    out["flags"] = flags(out["contrasts"]["BEATs768"]["RC-RA"])
    out["design"] = {"draws": draws, "seed": seed, "margin": MARGIN, "recall_limit": RECALL_LIMIT,
                     "folds": folds, "fold_site": {str(k): v for k, v in fold_site.items()}}
    return out


def flags(rc_ra):
    primary, recall, auroc = rc_ra["macro_f1"], rc_ra["positive_recall"], rc_ra["auroc"]
    if primary is None:
        return {"primary": "incomplete; no subset-only interval"}
    lo, hi = primary["ci95"]
    regression = recall["estimate"] < -RECALL_LIMIT
    return {"primary_supports_practical_gain": bool(lo > MARGIN and not regression),
            "primary_rules_out_practical_gain": bool(hi < MARGIN),
            "primary_inconclusive": bool(not lo > MARGIN and not hi < MARGIN),
            "any_gain_lower_bound_above_zero": bool(lo > 0),
            "truck_recall_regression_over_limit": bool(regression),
            "auroc_lower_bound_above_margin": bool(auroc["ci95"][0] > MARGIN),
            "auroc_upper_bound_below_margin": bool(auroc["ci95"][1] < MARGIN)}

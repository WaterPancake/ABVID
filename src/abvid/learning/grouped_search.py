"""Grouped configuration search with nested leave-one-group-out selection.

Outer held groups are never used to choose a configuration. Each configuration
gets an inner leave-one-group-out AUROC on the outer training groups; the best
inner configuration is then scored on the outer group.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from abvid.evaluation.ranking import interval, level_counts, selection_draws, weighted_metrics

ENCODERS = ("beats_l12", "beats_l6", "beats_avg", "cnn14", "mfcc26", "beats_l12+mfcc26")
ROUTES = ("core8", "full")
BUDGETS = ("b190", "ball")
CS = (0.001, 0.01, 0.1, 1.0)
REFERENCE = ("beats_l12", "core8", "b190", 1.0)


def grid():
    return list(itertools.product(ENCODERS, ROUTES, BUDGETS, CS))


def fit(x, y, c, balanced):
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=c, solver="lbfgs", tol=1e-6, max_iter=5000, class_weight="balanced" if balanced else None))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y)
    return model


def metrics(y, p):
    values = weighted_metrics(level_counts(p, y, np.zeros(len(y), int), 1), np.ones(1))
    return {k: float(values[k][0]) for k in ("auroc", "balanced_accuracy", "macro_f1", "average_precision")}


def nested_select(inner_scores, configs):
    """Pick the configuration with the highest inner score; ties -> grid order."""
    best = max(range(len(configs)), key=lambda i: (inner_scores[i], -i))
    return configs[best]


def fold_bootstrap(delta, fold_site, draws=10000, seed=20261014):
    """Percentile intervals for a mean over folds, by fold and by site block."""
    delta = np.asarray(delta, float)
    rng = np.random.Generator(np.random.PCG64(seed))
    w = np.vstack([np.ones(len(delta), int), selection_draws(rng, draws, len(delta))])
    sites = sorted(set(fold_site))
    sw = np.vstack([np.ones(len(sites), int), selection_draws(rng, draws, len(sites))])
    site_w = sw[:, [sites.index(s) for s in fold_site]]
    by_fold = (w * delta).sum(1) / w.sum(1)
    by_site = (site_w * delta).sum(1) / site_w.sum(1)
    return {"estimate": float(by_fold[0]), "ci95": interval(by_fold[1:]), "site_ci95": interval(by_site[1:])}

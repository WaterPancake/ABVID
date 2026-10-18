"""Prevalence-robust binary metrics with whole-group/selection resampling.

Scores are P(positive class), where positive is truck (or a declared proxy such
as cv). Ranking metrics do not depend on a decision threshold, so a uniform
score shift cannot change them. ``recall_at_fpr_*`` uses the evaluated
negatives to place its threshold: it is a ROC summary, not a deployable
operating point. The fixed rule ``score > 0.5`` matches a two-class argmax.

Group weights are integer multiplicities from resampling whole groups; events
inherit their group's weight. Metrics are exact for each weighted population.
"""
from __future__ import annotations

import itertools

import numpy as np

FPRS = (0.05, 0.10)
METRICS = ("auroc", "average_precision", "ap_lift", "prevalence",
           *(f"recall_at_fpr_{int(a * 100):02d}" for a in FPRS),
           "balanced_accuracy", "negative_recall", "positive_recall", "macro_f1")


def level_counts(score, y, group, n_groups):
    """Per-group cumulative counts at distinct score levels, highest first."""
    score, y, group = np.asarray(score, float), np.asarray(y), np.asarray(group)
    if score.ndim != 1 or score.shape != y.shape or y.shape != group.shape or not np.isfinite(score).all():
        raise ValueError("Expected matching finite score, label and group vectors")
    if not np.isin(y, (0, 1)).all() or group.min() < 0 or group.max() >= n_groups:
        raise ValueError("Labels must be 0/1 and groups must index n_groups")
    order = np.argsort(-score, kind="stable")
    s, yy, gg = score[order], y[order].astype(bool), group[order]
    last = np.r_[s[1:] != s[:-1], True]
    onehot = np.zeros((n_groups, s.size))
    onehot[gg, np.arange(s.size)] = 1
    positive = np.cumsum(onehot * yy, axis=1)[:, last]
    negative = np.cumsum(onehot * ~yy, axis=1)[:, last]
    above = s > 0.5
    return {"tp": positive, "fp": negative,
            "fixed_tp": np.bincount(gg[yy & above], minlength=n_groups).astype(float),
            "fixed_tn": np.bincount(gg[~yy & ~above], minlength=n_groups).astype(float)}


def weighted_metrics(counts, weights):
    """Metrics for each row of group multiplicities ``weights`` (draws x groups)."""
    w = np.atleast_2d(np.asarray(weights, float))
    tp, fp = w @ counts["tp"], w @ counts["fp"]
    p, n = tp[:, -1], fp[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        dtp = np.diff(tp, axis=1, prepend=0)
        dfp = np.diff(fp, axis=1, prepend=0)
        auroc = (dfp * (tp - dtp + dtp / 2)).sum(axis=1) / (p * n)
        precision = np.where(tp + fp > 0, tp / np.where(tp + fp > 0, tp + fp, 1), 0)
        ap = (dtp * precision).sum(axis=1) / p
        prevalence = p / (p + n)
        out = {"auroc": auroc, "average_precision": ap, "ap_lift": ap / prevalence,
               "prevalence": prevalence}
        for alpha in FPRS:
            allowed = (fp <= alpha * n[:, None]).sum(axis=1)
            reached = np.take_along_axis(tp, np.maximum(allowed - 1, 0)[:, None], axis=1)[:, 0]
            out[f"recall_at_fpr_{int(alpha * 100):02d}"] = np.where(allowed > 0, reached, 0) / p
        true_positive, true_negative = w @ counts["fixed_tp"], w @ counts["fixed_tn"]
        positive_recall, negative_recall = true_positive / p, true_negative / n
        errors = (p - true_positive) + (n - true_negative)
        f1 = [np.where(2 * t + errors > 0, 2 * t / (2 * t + errors), 0)
              for t in (true_positive, true_negative)]
    out.update(positive_recall=positive_recall, negative_recall=negative_recall,
               balanced_accuracy=(positive_recall + negative_recall) / 2,
               macro_f1=(f1[0] + f1[1]) / 2)
    invalid = (p == 0) | (n == 0)
    return {k: np.where(invalid, np.nan, v) for k, v in out.items()}


def group_draws(rng, draws, support):
    """Resample whole groups; redraw when either class is absent.

    ``support`` is a groups x 2 array of [negative, positive] counts. Returns a
    draws x groups multiplicity matrix and the number of rejected draws.
    """
    support = np.asarray(support)
    k = len(support)
    out, rejected = np.zeros((draws, k), int), 0
    for b in range(draws):
        while True:
            m = np.bincount(rng.integers(k, size=k), minlength=k)
            if (m @ support > 0).all():
                out[b] = m
                break
            rejected += 1
    return out, rejected


def selection_draws(rng, draws, size):
    """Multiplicities from resampling one selection dimension with replacement."""
    idx = rng.integers(size, size=(draws, size))
    return np.stack([np.bincount(row, minlength=size) for row in idx])


def model_metrics(counts, weights):
    """Evaluate only the distinct weight rows; few groups give few compositions."""
    unique, inverse = np.unique(weights, axis=0, return_inverse=True)
    values = weighted_metrics(counts, unique)
    return {k: v[inverse.reshape(-1)] for k, v in values.items()}


def evaluate(scores, y, group, n_groups, group_weights, selection_weights):
    """Mean over models of each model's pooled metric, for every weight draw.

    ``scores`` has shape (*selection_dims, events). ``selection_weights`` holds
    one draws x size multiplicity matrix per selection dimension, in order.
    Draw 0 conventionally carries all-ones weights for the point estimate.
    """
    scores = np.asarray(scores, float)
    dims = scores.shape[:-1]
    if len(dims) != len(selection_weights) or any(
            w.shape[1] != d for w, d in zip(selection_weights, dims)):
        raise ValueError("One selection weight matrix is required per model dimension")
    per_model = {k: np.empty((*dims, len(group_weights))) for k in METRICS}
    for index in itertools.product(*(range(d) for d in dims)):
        values = model_metrics(level_counts(scores[index], y, group, n_groups), group_weights)
        for k in METRICS:
            per_model[k][index] = values[k]
    mass = np.ones((len(group_weights),) + (1,) * len(dims))
    for axis, w in enumerate(selection_weights):
        shape = [len(group_weights)] + [1] * len(dims)
        shape[axis + 1] = dims[axis]
        mass = mass * w.reshape(shape)
    total = mass.reshape(len(group_weights), -1).sum(axis=1)
    out = {}
    for k, v in per_model.items():
        moved = np.moveaxis(v, -1, 0)
        out[k] = (moved * mass).reshape(len(group_weights), -1).sum(axis=1) / total
    return out


def interval(samples):
    finite = np.asarray(samples)[np.isfinite(samples)]
    return np.quantile(finite, [0.025, 0.975]).tolist() if finite.size else [None, None]

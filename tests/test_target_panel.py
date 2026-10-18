import numpy as np
import pytest

from abvid.evaluation.panel import Entry, markdown, validate
from abvid.evaluation.ranking import (evaluate, group_draws, level_counts, selection_draws,
                                      weighted_metrics)


def brute_auroc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    return ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (pos.size * neg.size)


def brute_ap(y, s):
    total, ap, previous = y.sum(), 0.0, 0.0
    for t in np.unique(s)[::-1]:
        chosen = s >= t
        recall = y[chosen].sum() / total
        ap += (recall - previous) * y[chosen].mean()
        previous = recall
    return ap


def fixture(n=600, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.integers(3, size=n)
    y = (rng.random(n) < 0.08 + 0.06 * g).astype(int)
    s = np.round(0.4 * rng.random(n) + 0.25 * y + 0.1 * g, 2)  # deliberate ties
    return y, s, g


def test_metrics_match_brute_force_with_ties():
    y, s, g = fixture()
    m = weighted_metrics(level_counts(s, y, g, 3), np.ones(3))
    assert m["auroc"][0] == pytest.approx(brute_auroc(y, s), abs=1e-12)
    assert m["average_precision"][0] == pytest.approx(brute_ap(y, s), abs=1e-12)
    assert m["prevalence"][0] == pytest.approx(y.mean())
    pred = s > 0.5
    assert m["positive_recall"][0] == pytest.approx(pred[y == 1].mean())
    assert m["negative_recall"][0] == pytest.approx((~pred[y == 0]).mean())


def test_group_weights_equal_replication():
    y, s, g = fixture(seed=1)
    w = np.array([2, 0, 3])
    keep = np.concatenate([np.flatnonzero(g == k).repeat(w[k]) for k in range(3)])
    m = weighted_metrics(level_counts(s, y, g, 3), w)
    assert m["auroc"][0] == pytest.approx(brute_auroc(y[keep], s[keep]), abs=1e-12)
    assert m["average_precision"][0] == pytest.approx(brute_ap(y[keep], s[keep]), abs=1e-12)


def test_ranking_metrics_ignore_monotone_shift_but_fixed_rule_does_not():
    y, s, g = fixture(seed=2)
    a = weighted_metrics(level_counts(s, y, g, 3), np.ones(3))
    b = weighted_metrics(level_counts(s * 0.1 + 0.8, y, g, 3), np.ones(3))
    for key in ("auroc", "average_precision", "recall_at_fpr_05", "recall_at_fpr_10"):
        assert a[key][0] == pytest.approx(b[key][0], abs=1e-12)
    assert b["positive_recall"][0] == 1 and b["negative_recall"][0] == 0


def test_recall_at_fpr_never_exceeds_negative_budget():
    y, s, g = fixture(seed=3)
    m = weighted_metrics(level_counts(s, y, g, 3), np.ones(3))
    for alpha, key in ((0.05, "recall_at_fpr_05"), (0.10, "recall_at_fpr_10")):
        feasible = [y[s >= t].sum() / y.sum() for t in np.unique(s) if (s[y == 0] >= t).mean() <= alpha]
        assert m[key][0] == pytest.approx(max(feasible, default=0.0), abs=1e-12)


def test_missing_class_is_nan_and_draws_reject_it():
    y, s, g = fixture(seed=4)
    y = np.where(g == 0, 0, y)
    m = weighted_metrics(level_counts(s, y, g, 3), np.array([3, 0, 0]))
    assert np.isnan(m["auroc"][0])
    support = np.stack([np.bincount(g[y == c], minlength=3) for c in (0, 1)], 1)
    draws, _ = group_draws(np.random.default_rng(0), 200, support)
    assert ((draws @ support) > 0).all()


def test_evaluate_point_is_model_mean_and_selection_weights_apply():
    y, s, g = fixture(seed=5)
    scores = np.stack([s, 1 - s, s])
    ones = np.ones((1, 3), int)
    point = evaluate(scores, y, g, 3, np.ones((1, 3), int), [ones])
    expected = np.mean([brute_auroc(y, x) for x in scores])
    assert point["auroc"][0] == pytest.approx(expected, abs=1e-12)
    only_first = evaluate(scores, y, g, 3, np.ones((1, 3), int), [np.array([[3, 0, 0]])])
    assert only_first["auroc"][0] == pytest.approx(brute_auroc(y, s), abs=1e-12)
    w = selection_draws(np.random.default_rng(1), 50, 4)
    assert (w.sum(axis=1) == 4).all()
    with pytest.raises(ValueError):
        evaluate(scores, y, g, 3, np.ones((1, 3), int), [])


def test_panel_rejects_pooling_and_scored_sealed_rows():
    metric = {"auroc": {"estimate": 0.6, "ci95": [0.5, 0.7]}}
    rows = [Entry("MELAUDIS_V1", "exposed_development", "car/truck", 4, {"negative": 7810, "positive": 256}, metric),
            Entry("DCASE2024_T10_DEV", "sealed_confirmation", "car/cv proxy", 6, {"negative": 1361, "positive": 228})]
    table = markdown(validate(rows), {"auroc": "AUROC"})
    assert "60.00 [50.00, 70.00]" in table and "sealed" in table
    with pytest.raises(ValueError, match="pooled"):
        validate([Entry("MELAUDIS+DCASE", "exposed_development", "car/truck", 10, {}, metric)])
    with pytest.raises(ValueError, match="scored roles"):
        validate([Entry("DCASE2024_T10_DEV", "sealed_confirmation", "car/cv proxy", 6, {}, metric)])
    with pytest.raises(ValueError, match="scored roles"):
        validate([Entry("MELAUDIS_V1", "exposed_development", "car/truck", 4, {})])

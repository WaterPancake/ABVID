"""Conditional precision planning from paired, already-computed score draws.

All score arguments use percentage points. These calculations do not estimate
an unobserved treatment's variance or validate small-cluster CI coverage.
"""
from statistics import NormalDist

import numpy as np


NORMAL = NormalDist()
Z95 = NORMAL.inv_cdf(.975)


def paired_summary(left, right, point, margin=3.0):
    """Preserve pairing and the empirical percentile interval's asymmetry."""
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or a.size < 2 or not np.isfinite([a, b]).all():
        raise ValueError("Expected matching finite paired score draws")
    d = a - b
    covariance = np.cov(a, b, ddof=1)
    variance = float(np.var(d, ddof=1))
    if not np.isclose(variance, covariance[0, 0] + covariance[1, 1] - 2*covariance[0, 1], atol=1e-10):
        raise ValueError("Paired variance identity failed")
    lo, hi = np.quantile(d, [.025, .975])
    se = variance**.5
    return {"estimate_pp": float(point), "ci95_pp": [float(lo), float(hi)],
            "width_pp": float(hi-lo), "paired_estimator_variance_pp2": variance,
            "paired_estimator_sd_pp": se,
            "arm_draw_covariance_pp2": float(covariance[0, 1]),
            "arm_draw_correlation": float(np.corrcoef(a, b)[0, 1]),
            "unpaired_variance_pp2": float(covariance[0, 0]+covariance[1, 1]),
            "lower_distance_pp": float(point-lo), "upper_distance_pp": float(hi-point),
            "observed_point_to_clear_margin_pp": float(margin+point-lo),
            "normal_expected_width_pp": float(2*Z95*se),
            "normal_true_effect_for_80pct_margin_support_pp": float(margin+(Z95+NORMAL.inv_cdf(.8))*se),
            "normal_true_effect_for_90pct_margin_support_pp": float(margin+(Z95+NORMAL.inv_cdf(.9))*se)}


def crossed_components(matrix):
    """Exact variance of an empirical crossed group/selection bootstrap mean.

    matrix[selection, cluster] must already average both dates for site blocks.
    Components describe resampling these observed cells, not population random
    effects or additional independent sites. More seeds only reduce two terms.
    """
    d = np.asarray(matrix, dtype=float)
    if d.ndim != 2 or min(d.shape) < 2 or not np.isfinite(d).all():
        raise ValueError("Expected a finite selection-by-cluster score matrix")
    ns, ng = d.shape
    grand = d.mean()
    cluster = d.mean(axis=0)-grand
    selection = d.mean(axis=1)-grand
    residual = d-grand-cluster[None, :]-selection[:, None]
    c, s, r = (float(np.mean(v*v)) for v in (cluster, selection, residual))
    variance = c/ng+s/ns+r/(ng*ns)
    return {"clusters": ng, "selections": ns, "cluster_term_pp2": c/ng,
            "selection_term_pp2": s/ns, "interaction_term_pp2": r/(ng*ns),
            "exact_bootstrap_variance_pp2": variance,
            "exact_bootstrap_sd_pp": variance**.5,
            "cluster_variance_fraction": (c/ng)/variance if variance else None,
            "normal_width_with_more_selections_pp": {
                str(n): float(2*Z95*np.sqrt(c/ng+s/n+r/(ng*n))) for n in (5, 10, 20, 100)},
            "normal_width_infinite_selections_pp": float(2*Z95*np.sqrt(c/ng)),
            "per_cluster_mean_difference_pp": d.mean(axis=0).tolist(),
            "per_selection_mean_difference_pp": d.mean(axis=1).tolist()}


def normal_scenarios(sd, margin=3.0):
    """Unvalidated normal location/scale scenarios; not a measured CAST power."""
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Expected positive finite standard deviation")
    rows = []
    for multiplier in (.5, 1., 1.5, 2.):
        se = sd*multiplier
        for effect in (0., 2., 3., 4., 5., 6., 7.):
            support = NORMAL.cdf((effect-margin)/se-Z95)
            rule_out = NORMAL.cdf((margin-effect)/se-Z95)
            rows.append({"sd_multiplier": multiplier, "true_effect_pp": effect,
                         "width_pp": 2*Z95*se,
                         "probability_lower95_above_margin": support,
                         "probability_upper95_below_margin": rule_out,
                         "probability_inconclusive_about_margin": 1-support-rule_out,
                         "probability_lower95_above_zero": NORMAL.cdf(effect/se-Z95)})
    return rows

"""CAST-4B joint descriptor diagnostic; no data access or significance test.

Freeze training scales, family weighting and bandwidth before scoring any
held group. The exhaustive biased MMD-squared estimator uses one fixed Gaussian
kernel, k(x, y) = exp(-||x-y||^2 / (2*sigma^2)), including diagonal terms.
It compares empirical distributions and supports singleton/degenerate samples.
It has finite-sample bias and supplies neither a p-value nor a universal pass
threshold, especially for correlated observations from recording groups.
"""
from collections.abc import Mapping

import numpy as np
from scipy.spatial.distance import cdist, pdist

from abvid.cast.grouped_metrics import FAMILIES


BANDWIDTH_RULE = (
    "median strictly positive unordered pairwise Euclidean distance on the "
    "training-scaled, family-weighted training vectors; data-derived at freeze "
    "time only; at most max_samples rows without replacement using seeded "
    "PCG64, retained in input order; reject an all-identical calibration sample"
)
FAMILY_WEIGHT_RULE = "equal family mean squared distance: coordinate weight 1/(F*d_family)"


def _matrix(values):
    result = np.asarray(values, dtype=np.float64)
    if (result.ndim != 2 or min(result.shape) == 0
            or not np.isfinite(result).all()):
        raise ValueError("Expected a nonempty finite observation-by-feature matrix")
    return result


def descriptor_vector(descriptors, scales):
    """Combine four families using caller-supplied, frozen training scales.

    Family f with d_f coordinates contributes z_f = x_f / (s_f*sqrt(F*d_f)).
    Thus squared Euclidean distance averages coordinate distances within each
    family, then averages the four families equally. Scales must come from
    training only (e.g. ``coverage.scales`` with its frozen floors); this function
    fits nothing. Centering by one shared training mean cancels in distances.
    """
    if (not isinstance(descriptors, Mapping) or not isinstance(scales, Mapping)
            or set(descriptors) != set(FAMILIES) or set(scales) != set(FAMILIES)):
        raise ValueError("Expected exactly the four CAST descriptor families and scales")
    blocks, count = [], None
    for family in FAMILIES:
        values = _matrix(descriptors[family])
        scale = np.asarray(scales[family], dtype=np.float64)
        if (scale.shape != values.shape[1:] or not np.isfinite(scale).all()
                or (scale <= 0).any()):
            raise ValueError("Expected positive finite training scales for every coordinate")
        if count is not None and len(values) != count:
            raise ValueError("Descriptor families must retain matching observation rows")
        count = len(values)
        blocks.append(values / scale / np.sqrt(len(FAMILIES) * values.shape[1]))
    return _matrix(np.concatenate(blocks, axis=1))


def median_bandwidth(training, *, seed, max_samples=1024):
    """Data-derived-at-freeze-time median heuristic, declared in BANDWIDTH_RULE.

    Only training vectors may enter here. Save the returned sigma, seed, row
    order/identity, scales and max_samples in a later experiment's frozen record;
    reuse that sigma for every comparison arm and held group in that role.
    Duplicate rows' zero distances are excluded explicitly. No arbitrary unit
    bandwidth is substituted when calibration is entirely collapsed.
    """
    values = _matrix(training)
    if len(values) < 2:
        raise ValueError("Bandwidth calibration needs at least two training observations")
    if type(seed) is not int or seed < 0:
        raise ValueError("Expected a nonnegative integer seed")
    if type(max_samples) is not int or max_samples < 2:
        raise ValueError("Expected an integer max_samples of at least two")
    if len(values) > max_samples:
        rng = np.random.Generator(np.random.PCG64(seed))
        indices = np.sort(rng.choice(len(values), size=max_samples, replace=False))
        values = values[indices]
    distances = pdist(values, metric="euclidean")
    positive = distances[distances > 0]
    if not np.isfinite(distances).all() or not len(positive):
        raise ValueError("Bandwidth calibration needs finite, positive training distances")
    return float(np.median(positive))


def gaussian_mmd_squared(left, right, *, bandwidth):
    """Biased empirical MMD^2 with an explicit, fixed Gaussian bandwidth sigma.

    mean(k(X,X)) + mean(k(Y,Y)) - 2*mean(k(X,Y)), with n*n, m*m and n*m
    denominators respectively. All pairs are used: scoring has no random draws,
    subsampling, bandwidth fitting or preprocessing fitting. Seeded calibration
    happens separately in ``median_bandwidth``. Time/memory are quadratic in
    sample count. Roundoff below zero is clipped to zero.
    """
    a, b = _matrix(left), _matrix(right)
    if a.shape[1] != b.shape[1]:
        raise ValueError("Expected matching feature dimensions")
    if not np.isscalar(bandwidth) or not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Expected a positive finite frozen bandwidth")

    def kernel_mean(x, y):
        with np.errstate(over="ignore"):
            squared = (cdist(x, y, metric="euclidean") / bandwidth)**2
        return float(np.exp(-.5 * squared).mean())

    value = kernel_mean(a, a) + kernel_mean(b, b) - 2 * kernel_mean(a, b)
    return max(0., value)

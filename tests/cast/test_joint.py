"""Fixed synthetic CAST-4B sensitivity controls; no real/archive inputs."""
from copy import deepcopy

import numpy as np
import pytest

from abvid.cast.coverage import compare, scales
from abvid.cast.joint import (
    FAMILIES, descriptor_vector, gaussian_mmd_squared, median_bandwidth,
)


WIDTHS = (3, 1, 2, 4)
METRIC = {"primary_families": list(FAMILIES), "coverage_quantiles": [.05, .95],
          "scale_floors": dict.fromkeys(FAMILIES, .01)}


def descriptors(columns):
    return {family: np.repeat(np.asarray(column, dtype=float)[:, None], width, axis=1)
            for family, column, width in zip(FAMILIES, columns, WIDTHS)}


@pytest.fixture(scope="module")
def frozen():
    # Separate, fixed synthetic training data; no compared distribution enters
    # scale/bandwidth calibration. Unequal family widths exercise weighting.
    rng = np.random.Generator(np.random.PCG64(4217))
    training = {family: rng.normal(size=(256, width))
                for family, width in zip(FAMILIES, WIDTHS)}
    scale = scales(training, METRIC)
    bandwidth = median_bandwidth(descriptor_vector(training, scale), seed=42)
    return scale, bandwidth


def score(left, right, frozen):
    scale, bandwidth = frozen
    return gaussian_mmd_squared(descriptor_vector(left, scale),
                                descriptor_vector(right, scale), bandwidth=bandwidth)


def test_mmd_detects_dependence_with_exactly_matched_family_marginals(frozen):
    # Dyadic values ensure sign reversal preserves every empirical marginal
    # exactly, while cross-family correlations change from +1 to -1.
    latent = (np.arange(-64, 64) + .5) / 32
    reference = descriptors([latent] * 4)
    changed = descriptors([latent, -latent, latent, -latent])
    marginal = compare(changed, reference, frozen[0], METRIC)
    for family in FAMILIES:
        np.testing.assert_array_equal(np.sort(reference[family], axis=0),
                                      np.sort(changed[family], axis=0))
        assert marginal["families"][family]["W1"] == pytest.approx(0., abs=1e-15)
    assert marginal["W1"] == pytest.approx(0., abs=1e-15)
    assert np.corrcoef(reference["bands"][:, 0], reference["envelope"][:, 0])[0, 1] > .99
    assert np.corrcoef(changed["bands"][:, 0], changed["envelope"][:, 0])[0, 1] < -.99
    # Fixture-specific separation margin, not a scientific MMD pass threshold.
    assert score(reference, changed, frozen) > .05


@pytest.mark.parametrize("collapse", ["point", "two_point_mixture"])
def test_mmd_detects_collapsed_samples_and_degenerate_mixtures(frozen, collapse):
    rng = np.random.Generator(np.random.PCG64(138))
    reference = {family: rng.normal(size=(256, width))
                 for family, width in zip(FAMILIES, WIDTHS)}
    latent = np.zeros(256) if collapse == "point" else np.tile([-1., 1.], 128)
    collapsed = descriptors([latent] * 4)
    assert all(len(np.unique(value, axis=0)) == (1 if collapse == "point" else 2)
               for value in collapsed.values())
    assert np.isfinite(score(reference, collapsed, frozen))
    assert score(reference, collapsed, frozen) > .02
    assert score(collapsed, collapsed, frozen) == pytest.approx(0., abs=1e-14)


def test_mmd_detects_excessive_spread_with_frozen_bandwidth(frozen):
    rng = np.random.Generator(np.random.PCG64(139))
    reference = {family: rng.normal(size=(256, width))
                 for family, width in zip(FAMILIES, WIDTHS)}
    wider = {family: 8 * value for family, value in reference.items()}
    marginal = compare(wider, reference, frozen[0], METRIC)
    assert all(row["spread_ratio"] == pytest.approx(8.) for row in marginal["families"].values())
    assert score(reference, wider, frozen) > .2


def test_identical_empirical_distributions_have_zero_mmd(frozen):
    rng = np.random.Generator(np.random.PCG64(140))
    reference = {family: rng.normal(size=(128, width))
                 for family, width in zip(FAMILIES, WIDTHS)}
    permutation = rng.permutation(128)
    reordered = {family: value[permutation] for family, value in reference.items()}
    assert score(reference, deepcopy(reference), frozen) == pytest.approx(0., abs=1e-14)
    assert score(reference, reordered, frozen) == pytest.approx(0., abs=1e-14)


def test_seeded_bandwidth_and_mmd_repeat_bit_for_bit():
    rng = np.random.Generator(np.random.PCG64(141))
    training, left, right = (rng.normal(size=(n, 5)) for n in (200, 80, 90))
    # Exercise actual seeded row subsampling during training-only calibration.
    first = median_bandwidth(training, seed=42, max_samples=32)
    again = median_bandwidth(training.copy(), seed=42, max_samples=32)
    assert first.hex() == again.hex()
    a = gaussian_mmd_squared(left, right, bandwidth=first)
    b = gaussian_mmd_squared(left.copy(), right.copy(), bandwidth=again)
    assert a.hex() == b.hex()
    assert first != median_bandwidth(training, seed=43, max_samples=32)


def test_marginal_w1_misses_missing_joint_modes_that_mmd_catches(frozen):
    # Four equally weighted corners versus only the diagonal corners. Every
    # one-dimensional histogram is identical, despite two missing joint modes.
    x = np.tile([-1., -1., 1., 1.], 32)
    y = np.tile([-1., 1., -1., 1.], 32)
    reference = descriptors([x, y, x, y])
    missing_modes = descriptors([x, x, x, x])
    marginal = compare(missing_modes, reference, frozen[0], METRIC)
    assert marginal["W1"] == 0.
    assert all(row["W1"] == 0. and row["spread_ratio"] == pytest.approx(1.)
               and row["coverage"] == 1. for row in marginal["families"].values())
    assert score(reference, reference, frozen) == pytest.approx(0., abs=1e-14)
    assert score(reference, missing_modes, frozen) > .01


def test_biased_estimator_matches_analytic_gaussian_with_unequal_counts():
    # Two point masses have MMD^2 = 2*(1-k(a,b)), irrespective of multiplicity.
    # This also verifies inclusion of diagonals and support for a singleton.
    actual = gaussian_mmd_squared([[0.], [0.]], [[3.]], bandwidth=2.)
    expected = 2 * (1 - np.exp(-9 / 8))
    assert actual == pytest.approx(expected, abs=1e-15)
    assert gaussian_mmd_squared([[3.]], [[0.], [0.]], bandwidth=2.) == actual


def test_descriptor_vector_uses_frozen_scales_and_equal_family_weights():
    scale = {family: np.arange(1, width + 1, dtype=float)
             for family, width in zip(FAMILIES, WIDTHS)}
    values = {family: np.stack([np.zeros_like(s), s]) for family, s in scale.items()}
    before = deepcopy((values, scale))
    vector = descriptor_vector(values, scale)
    start = 0
    for width in WIDTHS:
        block = vector[:, start:start + width]
        assert np.sum((block[1] - block[0])**2) == pytest.approx(.25)
        start += width
    # Mapping insertion order cannot change the frozen coordinate order.
    np.testing.assert_array_equal(vector, descriptor_vector(dict(reversed(list(values.items()))), scale))
    for family in FAMILIES:
        np.testing.assert_array_equal(values[family], before[0][family])
        np.testing.assert_array_equal(scale[family], before[1][family])


def test_median_bandwidth_uses_only_positive_unordered_training_distances():
    # Positive distances are [2,6,2,6,4]; their median sigma is exactly 4.
    assert median_bandwidth([[0.], [0.], [2.], [6.]], seed=42) == 4.
    for training in ([[0.]], np.zeros((8, 2))):
        with pytest.raises(ValueError, match="training"):
            median_bandwidth(training, seed=42)
    for options in ({"seed": -1}, {"seed": True}, {"seed": 42, "max_samples": 1}):
        with pytest.raises(ValueError):
            median_bandwidth([[0.], [1.]], **options)


def test_mmd_rejects_invalid_matrices_dimensions_and_bandwidths():
    for invalid in ([], [1., 2.], np.empty((2, 0)), [[np.nan]], [[np.inf]]):
        with pytest.raises(ValueError, match="matrix"):
            gaussian_mmd_squared(invalid, [[0.]], bandwidth=1.)
        with pytest.raises(ValueError, match="matrix"):
            gaussian_mmd_squared([[0.]], invalid, bandwidth=1.)
    with pytest.raises(ValueError, match="dimensions"):
        gaussian_mmd_squared([[0.]], [[0., 1.]], bandwidth=1.)
    for bandwidth in (0., -1., np.nan, np.inf, [1.]):
        with pytest.raises(ValueError, match="bandwidth"):
            gaussian_mmd_squared([[0.]], [[1.]], bandwidth=bandwidth)


def test_descriptor_vector_rejects_missing_misaligned_or_invalid_scales():
    good = descriptors([np.arange(4.)] * 4)
    scale = {family: np.ones(width) for family, width in zip(FAMILIES, WIDTHS)}
    bad = deepcopy(good)
    bad.pop("bands")
    with pytest.raises(ValueError, match="families"):
        descriptor_vector(bad, scale)
    bad = deepcopy(good)
    bad["bands"] = bad["bands"][:-1]
    with pytest.raises(ValueError, match="rows"):
        descriptor_vector(bad, scale)
    for invalid in ([0.], [-1.], [np.nan], [np.inf], [1., 2.]):
        bad_scale = {**scale, "bands": invalid}
        with pytest.raises(ValueError, match="scales"):
            descriptor_vector(good, bad_scale)

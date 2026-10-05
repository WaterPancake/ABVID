import numpy as np
import pytest
from cast_improvement.parent_mixture_prior import calibrate


@pytest.mark.parametrize("truth", [0., .1, .5, .9, 1.])
def test_identifiable_power_mixture_recovered_with_unequal_component_levels(truth):
    h = np.array([4.,3.,2.,1.,0.,0.,0.,0.])
    n = np.array([0.,0.,0.,0.,.1,.2,.3,.4])
    y = truth*h+(1-truth)*n; y /= y.sum()
    result = calibrate(.35, h, n, y)
    assert result["harmonic_fraction"] == pytest.approx(truth, abs=1e-12)
    assert result["calibrated_expected_band_squared_error"] < 1e-24
    assert result == calibrate(.35, h, n, y)


@pytest.mark.parametrize("boundary", [0., 1.])
def test_original_boundary_is_retained_and_flagged(boundary):
    r = calibrate(boundary, np.arange(8.), np.ones(8), np.ones(8)/8)
    assert r["harmonic_fraction"] == boundary and r["flags"]


def test_identical_spectral_shapes_do_not_invent_a_mixture():
    r = calibrate(.3, np.ones(8)*3, np.ones(8), np.ones(8)/8)
    assert r["harmonic_fraction"] == .3
    assert "mixture_calibration_not_identifiable" in r["flags"]


def test_impossible_target_stays_bounded_and_objective_cannot_increase():
    r = calibrate(.4, np.arange(1,9.), np.arange(8,0,-1.), np.array([0,0,0,1,0,0,0,0.]))
    assert 0 <= r["harmonic_fraction"] <= 1
    assert r["calibrated_expected_band_squared_error"] <= r["original_expected_band_squared_error"]+1e-12


def test_nonfinite_or_invalid_inputs_rejected():
    for bad in (np.ones(7), np.r_[np.ones(7),np.nan], -np.ones(8)):
        with pytest.raises(ValueError): calibrate(.4, bad, np.ones(8), np.ones(8)/8)
    with pytest.raises(ValueError): calibrate(1.1, np.ones(8), np.ones(8), np.ones(8)/8)

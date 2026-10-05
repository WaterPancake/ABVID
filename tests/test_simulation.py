"""Small synthetic regression fixtures; no research model or corpus run."""
import numpy as np
import pytest

from abvid.simulation.propagation import (
    build_plan, ground_filters, reference_manager, render_sources,
)


def test_ground_angles_match_scalar_reference():
    manager = reference_manager()
    angles = np.array([-95., -89., -30.51, 0., 30.49, 60., 89., 95.])
    distances = np.linspace(5., 135., len(angles))
    batched = ground_filters(angles, distances, manager)
    reference = np.stack([manager._get_asphalt_reflection_filter(a, d)
                          for a, d in zip(angles, distances)])
    np.testing.assert_allclose(batched, reference, rtol=1e-11, atol=1e-12)


def test_reflected_path_and_combination_are_finite_and_deterministic():
    n = 2048
    trajectory = np.tile([0., 10., .5], (n, 1))
    signal = np.sin(2*np.pi*123*np.arange(n)/8000)
    plan = build_plan(trajectory)
    first = render_sources(signal, plan, components=True)
    again = render_sources(signal, plan, components=True)
    for key in ("P0", "P1", "direct_filtered", "reflected"):
        assert np.isfinite(first[key]).all()
        np.testing.assert_array_equal(first[key], again[key])
    np.testing.assert_array_equal(first["P1"], first["direct_filtered"] + first["reflected"])
    assert np.any(first["reflected"])


def test_invalid_geometries_and_silent_inputs_fail():
    with pytest.raises(ValueError): build_plan(np.array([[0., 10., -1.]]))
    plan = build_plan(np.tile([0., 10., .5], (64, 1)))
    with pytest.raises(ValueError): render_sources(np.zeros(64), plan)

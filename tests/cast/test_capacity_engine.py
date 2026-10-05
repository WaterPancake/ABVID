from copy import deepcopy
import numpy as np
import pytest
import torch

from abvid.cast.audio import shape
from abvid.cast.renderer import Parameters, tensors
from abvid.cast.synthetic import fixtures
from abvid.cast.population.engine import SmoothRenderer, model_config as old_config
from abvid.cast.population.capacity_engine import CapacityRenderer, lift_controls, model_config, validate_controls


@pytest.fixture(autouse=True)
def cpu():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def test_capacity_configuration_preserves_objective_budget_and_metric_bands():
    old, cfg = old_config(), model_config()
    assert cfg["optimizer"] == old["optimizer"] and cfg["loss"] == old["loss"]
    assert cfg["acceptance"] == old["acceptance"] and cfg["audio"] == old["audio"]
    assert cfg["renderer"]["noise_edges_hz"] == old["renderer"]["noise_edges_hz"]
    centers = cfg["renderer"]["noise_control_centers_hz"]
    widths = cfg["renderer"]["noise_control_reference_widths_hz"]
    assert len(centers) == len(widths) == 16 and centers[-1] == 4000
    assert np.all(np.diff(centers) > 0) and np.all(np.array(widths) > 0)
    assert sum(widths) == 4000 and cfg["renderer"]["envelope_control_count"] == 9


@pytest.mark.parametrize("fixture", ["single_tone", "noise", "mixture"])
def test_lift_nests_existing_signal_family(fixture):
    p = fixtures()[fixture]; cfg = model_config(); q = lift_controls(p, cfg)
    validate_controls(q, cfg)
    with torch.no_grad():
        a = shape(SmoothRenderer(old_config(), "nested").render(tensors(p), "check"))
        b = shape(CapacityRenderer(cfg, "nested").render(tensors(q), "check"))
    torch.testing.assert_close(a, b, rtol=2e-5, atol=5e-6)


def test_capacity_determinism_components_and_boundary_mask():
    cfg = model_config(); p = tensors(lift_controls(fixtures()["mixture"], cfg))
    r = CapacityRenderer(cfg, "determinism")
    a, parts = r.render(p, "check", components=True)
    b = CapacityRenderer(cfg, "determinism").render(p, "check")
    assert torch.equal(a, b) and a.shape == (1, 2, 32000) and torch.isfinite(a).all()
    torch.testing.assert_close(a, parts["harmonic"]+parts["noise"], rtol=1e-5, atol=1e-6)
    gain = r.spectral_gain(p["noise_weights"])
    assert torch.all(gain[:, 0] == 0) and torch.all(gain[:, -1] == 0)
    assert not torch.equal(a, r.render(p, "fit"))


def test_capacity_finite_gradients_and_spectral_finite_difference():
    cfg = model_config(); p = lift_controls(fixtures()["mixture"], cfg)
    model = Parameters({k: [v] for k, v in p.items()}, cfg)
    renderer = CapacityRenderer(cfg, "gradients")
    loss = shape(renderer.render(model.controls(), "check"))[:, :, ::19].pow(4).mean()
    loss.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
    model.zero_grad()
    def scalar():
        return renderer.spectral_gain(model.controls()["noise_weights"])[:, 5700:6200].square().sum()
    scalar().backward(); analytical = float(model.noise_logits.grad[0, 13]); delta = .002
    with torch.no_grad():
        original = float(model.noise_logits[0, 13])
        model.noise_logits[0, 13] = original+delta; a = float(scalar())
        model.noise_logits[0, 13] = original-delta; b = float(scalar())
        model.noise_logits[0, 13] = original
    numerical = (a-b)/(2*delta)
    assert abs(analytical-numerical)/max(abs(analytical), abs(numerical), 1e-6) < .03


def test_old_schema_and_invalid_capacity_parameters_rejected():
    cfg = model_config()
    with pytest.raises(ValueError): validate_controls(fixtures()["mixture"], cfg)
    p = lift_controls(fixtures()["mixture"], cfg)
    for key, bad in (("spacing_hz", [401.]*3), ("envelope_knots", [0.]*9), ("noise_weights", [float("nan")]*16)):
        q = deepcopy(p); q[key] = bad
        with pytest.raises(ValueError): validate_controls(q, cfg)

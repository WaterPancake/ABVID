import copy
import numpy as np
import pytest
import torch

from abvid.cast.audio import shape
from abvid.cast.config import configuration
from abvid.cast.renderer import Parameters, tensors
from abvid.cast.synthetic import fixtures
from abvid.cast.population.engine import SmoothRenderer, linear_basis, model_config


@pytest.fixture(autouse=True)
def cpu():
    torch.set_num_threads(1)


def test_model_config_preserves_frozen_parent():
    old = configuration()
    cfg = model_config()
    assert old == configuration()
    assert cfg["optimizer"] == old["optimizer"] and cfg["loss"] == old["loss"]
    assert cfg["renderer"]["noise_filter"] != old["renderer"]["noise_filter"]


def test_interpolation_partition_and_endpoints():
    b = linear_basis([-5, 0, .5, 1, 2], [0, 1])
    np.testing.assert_array_equal(b.numpy(), [[1, 0], [1, 0], [.5, .5], [0, 1], [0, 1]])


def test_noise_spectrum_smooth_at_original_boundaries():
    cfg = model_config(); r = SmoothRenderer(cfg, "smooth_test")
    w = tensors(fixtures()["mixture"])["noise_weights"]
    gain = r.spectral_gain(w).numpy()[0]
    assert gain[0] == 0 and gain[-1] == 0
    for edge in cfg["renderer"]["noise_edges_hz"][1:-1]:
        j = int(edge*2)
        assert abs(np.log(gain[j+1]/gain[j-1])) < .04


def test_spectral_interpolation_batch_size_invariant():
    r = SmoothRenderer(model_config(), "batch_check")
    p = tensors(fixtures()["noise"])["noise_weights"]
    a = r.spectral_gain(p)
    b = r.spectral_gain(p.expand(4, -1).clone())
    assert torch.equal(a, b[:1])


def test_determinism_streams_shape_and_components():
    cfg = model_config(); p = tensors(fixtures()["mixture"])
    a, comps = SmoothRenderer(cfg, "same").render(p, "check", True)
    b = SmoothRenderer(cfg, "same").render(p, "check")
    assert torch.equal(a, b) and a.shape == (1, 2, 32000)
    assert torch.isfinite(a).all()
    torch.testing.assert_close(a, comps["harmonic"]+comps["noise"], rtol=1e-5, atol=1e-6)
    assert not torch.equal(a, SmoothRenderer(cfg, "same").render(p, "fit"))
    assert not torch.equal(a, SmoothRenderer(cfg, "different").render(p, "check"))


def test_sampling_phase_and_noise_separate_from_check():
    cfg = model_config(); p = tensors(fixtures()["mixture"])
    a = SmoothRenderer(cfg, "sampling_test", True).render(p, "sampling")
    b = SmoothRenderer(cfg, "sampling_test", True).render(p, "sampling")
    c = SmoothRenderer(cfg, "sampling_test").render(p, "check")
    assert a.shape == (1, 1, 32000) and torch.equal(a, b)
    assert not torch.equal(a[0, 0], c[0, 0])


def test_finite_gradients_all_controls_and_extreme_noise_weights():
    cfg = model_config(); init = {k: [v] for k, v in fixtures()["mixture"].items()}
    m = Parameters(init, cfg); r = SmoothRenderer(cfg, "grad")
    wave = r.render(m.controls())
    (shape(wave)[:, :, ::19]**4).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    for mix in [0, 1]:
        p = copy.deepcopy(fixtures()["noise"]); p["noise_weights"] = [1, 0, 0, 0, 0, 0, 0, 0];p["harmonic_fraction"] = mix
        assert torch.isfinite(r.render(tensors(p))).all()


def test_noise_weight_derivative_matches_finite_difference():
    cfg = model_config(); init = {k: [v] for k, v in fixtures()["noise"].items()}
    m = Parameters(init, cfg); r = SmoothRenderer(cfg, "finite_difference")
    def scalar():
        wave = r.render(m.controls(), "check")
        return wave[:, :, ::53].pow(2).mean()
    loss = scalar();loss.backward();g = float(m.noise_logits.grad[0, 3]); delta=.002
    with torch.no_grad():
        original=float(m.noise_logits[0, 3]);m.noise_logits[0, 3]=original+delta;a=float(scalar())
        m.noise_logits[0, 3]=original-delta;b=float(scalar());m.noise_logits[0, 3]=original
    fd=(a-b)/(2*delta)
    assert abs(fd-g)/max(abs(fd),abs(g),1e-6) < .03


def test_new_metadata_guard_uses_exact_existing_H1_ids_without_held():
    from archive_inputs import PILOT, PREVIOUS, read, readl, ROOT
    if not (PILOT/"config.resolved.json").is_file():
        pytest.skip("Local frozen CAST artifact bundle is not installed")
    cfg = read(PILOT/"config.resolved.json")
    parent=read(ROOT/cfg["data"]["parent_lock"]);fold=next(r for r in parent["folds"] if r["fold"]==0)
    train=set(fold["train_ids"]["42"]); held=set(fold["test_ids"])
    assert len(train)==380 and train.isdisjoint(held)
    assert set(read(PILOT/"lock.json")["selected_ids"]) <= train

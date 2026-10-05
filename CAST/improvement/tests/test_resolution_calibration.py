from copy import deepcopy
import numpy as np
import pytest
from scipy.signal import welch
import torch

from cast.renderer import tensors
from cast.synthetic import fixtures
from cast_improvement.resolution_model import model_config, ResolutionRenderer as SmoothRenderer, lift_controls
from cast_improvement.temporal_prior import lift as temporal_lift
from cast_improvement.resolution_calibration import ExpectedSpectrum,fit,welch_power


def setup():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg=model_config()
    return lift_controls(temporal_lift(deepcopy(fixtures()["mixture"])),cfg),cfg,{"spectral_probability_floor":1e-8}


def test_welch_matches_independent_scipy_for_mean_trend_and_tones():
    rng=np.random.default_rng(42);x=rng.normal(size=(2,32000))+3
    x+=np.sin(2*np.pi*125*np.arange(32000)/16000)
    expected=welch(x,16000,window="hann",nperseg=2048,noverlap=1024,axis=-1)[1]
    actual=welch_power(torch.tensor(x,dtype=torch.float64)).numpy()
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-14)


@pytest.mark.parametrize("mixture",[0.,1.])
def test_expected_component_endpoints_match_waveform_renderer(mixture):
    p,cfg,metric=setup();p["harmonic_fraction"]=mixture
    ex=ExpectedSpectrum(p,cfg,"component_fixture",metric["spectral_probability_floor"])
    with torch.no_grad():
        wave=SmoothRenderer(cfg,"component_fixture").render(tensors(p),"check")[0]
        power=welch_power(wave).mean(dim=0)
        expected=power[:512].reshape(64,8).sum(dim=1);expected/=expected.sum()
        actual=ex.probabilities(torch.tensor(p["noise_weights"]),torch.tensor(mixture))[0]
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-8)


@pytest.mark.parametrize("mixture",[0.,.35])
def test_known_expected_spectrum_recovery_retains_original_thresholds(mixture):
    truth,cfg,metric=setup();truth["harmonic_fraction"]=mixture
    weights=np.array([.01,.02,.03,.04,.06,.08,.13,.16,.14,.1,.08,.06,.04,.02,.02,.01])
    truth["noise_weights"]=(weights/weights.sum()).tolist()
    ex=ExpectedSpectrum(truth,cfg,"recovery_fixture",metric["spectral_probability_floor"])
    with torch.no_grad():
        spec,bands=ex.probabilities(torch.tensor(truth["noise_weights"]),torch.tensor(mixture))
    target={"log_spectrum":spec.clamp_min(1e-8).log().tolist(),"bands":bands.tolist()}
    initial=deepcopy(truth);initial.update(noise_weights=[1/16]*16,harmonic_fraction=.2)
    result=fit(initial,target,cfg,"recovery_fixture",metric)
    assert abs(result["parameters"]["harmonic_fraction"]-mixture)<.1
    assert np.abs(np.array(result["parameters"]["noise_weights"])-truth["noise_weights"]).sum()<.35
    assert result["best_objective"]<.02*result["initial_objective"]
    assert result==fit(initial,target,cfg,"recovery_fixture",metric)
    for key in ("spacing_hz","harmonic_weights","envelope_knots"):
        assert result["parameters"][key]==initial[key]


def test_expected_mixture_gradient_remains_finite_at_boundary():
    p,cfg,metric=setup();ex=ExpectedSpectrum(p,cfg,"gradient_fixture",1e-8)
    logits=torch.tensor(p["noise_weights"],requires_grad=True)
    mix=torch.tensor(0.,requires_grad=True)
    a,b=ex.logs(logits.softmax(0),mix)
    (a.square().mean()+b.square().mean()).backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(mix.grad)
    assert abs(float(mix.grad))>1e-8


def test_bad_targets_rejected_before_optimization():
    p,cfg,metric=setup()
    with pytest.raises(ValueError):fit(p,{"log_spectrum":[np.nan]*64,"bands":[.125]*8},cfg,"bad",metric)
    with pytest.raises(ValueError):welch_power(torch.ones(31999))


def test_nested_spectrum_lift_preserves_waveform_and_finite_noise_gradients():
    from cast_improvement.engine import SmoothRenderer as OldRenderer
    from cast_improvement.temporal_prior import model_config as old_config
    p=temporal_lift(deepcopy(fixtures()["mixture"]));cfg=model_config();lifted=lift_controls(p,cfg)
    torch.set_num_threads(1)
    with torch.no_grad():
        old=OldRenderer(old_config(),"nested_spectrum").render(tensors(p),"check")
        new=SmoothRenderer(cfg,"nested_spectrum").render(tensors(lifted),"check")
        replay=SmoothRenderer(cfg,"nested_spectrum").render(tensors(lifted),"check")
    torch.testing.assert_close(new,old,rtol=1e-5,atol=1e-5)
    assert torch.equal(new,replay)
    controls=tensors(lifted);controls["noise_weights"].requires_grad_(True)
    wave=SmoothRenderer(cfg,"gradient").render(controls,"check")
    wave[...,:1000].square().mean().backward()
    grad=controls["noise_weights"].grad
    assert grad.shape==(1,16) and torch.isfinite(grad).all() and grad.abs().sum()>0
    for key in ("spacing_hz","harmonic_weights","harmonic_fraction","envelope_knots"):
        assert lifted[key]==p[key]

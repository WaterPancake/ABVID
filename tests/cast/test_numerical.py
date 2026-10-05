from copy import deepcopy
import json
import numpy as np
import pytest
from scipy.signal import resample_poly
import torch

from abvid.cast.audio import normalize, observation, shape
from abvid.cast.config import configuration, validate
from abvid.cast.fit import Objective, fit
from abvid.cast.renderer import Parameters, Renderer, oscillator, seed_for, serialize, tensors, validate_controls
from abvid.cast.synthetic import fixtures


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def test_config_rejects_placeholders_and_omissions():
    for mode in ("missing", "null", "changed"):
        c = configuration()
        if mode == "missing": del c["renderer"]["noise_filter"]
        elif mode == "null": c["data"]["manifest_sha256"] = None
        else: c["optimizer"]["steps"] = 30
        with pytest.raises(ValueError): validate(c)


def test_renderer_replay_roundtrip_bounds_components_and_streams():
    c = configuration(); p = fixtures()["mixture"]
    decoded = json.loads(json.dumps(p)); validate_controls(decoded, c)
    a, parts = Renderer(c, "repeat").render(tensors(p), "check", True)
    b = Renderer(c, "repeat").render(tensors(decoded), "check")
    assert a.shape == (1, 2, 32000) and torch.isfinite(a).all()
    assert torch.equal(a, b)
    torch.testing.assert_close(a, parts["harmonic"]+parts["noise"], atol=1e-6, rtol=1e-6)
    assert not torch.equal(a, Renderer(c, "different").render(tensors(p), "check"))
    assert not torch.equal(a, Renderer(c, "repeat").render(tensors(p), "fit"))
    streams = Renderer(c, "repeat").streams()
    assert not set(streams["fit"]) & set(streams["check"])
    model = Parameters({k:[v] for k,v in p.items()}, c)
    for value in model.parameters():
        with torch.no_grad(): value.add_(100)
    validate_controls(serialize(model.controls()), c)


def test_polyphase_equivalence():
    c = configuration(); r = Renderer(c, "resample")
    rng = np.random.default_rng(42)
    for x in (rng.normal(size=16000).astype(np.float32), np.eye(1,16000,0,dtype=np.float32)[0], np.eye(1,16000,15999,dtype=np.float32)[0]):
        actual = r.upsample(torch.tensor(x)[None])[0].numpy()
        expected = resample_poly(x, 2, 1, window=("kaiser", 5))
        assert actual.shape == expected.shape == (32000,)
        assert np.max(np.abs(actual-expected)) < c["acceptance"]["resampler_max_abs_tolerance"]


def test_alias_suppression_and_noise_bands():
    c = configuration(); n = 16000
    # 1000 Hz spacing, only harmonic four: exactly Nyquist must be suppressed.
    w = torch.tensor([[0.,0,0,1,0,0,0,0]])
    assert torch.count_nonzero(oscillator(torch.full((1,n),1000.),w,torch.zeros(8),8000)) == 0
    w = torch.tensor([[0.,0,0,0,1,0,0,0]])
    assert torch.count_nonzero(oscillator(torch.full((1,n),1000.),w,torch.zeros(8),8000)) == 0
    r = Renderer(c, "bands"); bank = r.noise_bank("fit").numpy()
    freq = np.fft.rfftfreq(n, 1/8000)
    for i,(lo,hi) in enumerate(zip(c["renderer"]["noise_edges_hz"][:-1], c["renderer"]["noise_edges_hz"][1:])):
        power = np.abs(np.fft.rfft(bank[:,i], axis=-1))**2
        allowed = (freq>=lo)&(freq<hi)&(freq>0)
        assert power[:,~allowed].sum()/power.sum() < c["acceptance"]["alias_energy_fraction_tolerance"]
    np.testing.assert_allclose((bank**2).mean(-1),1,atol=1e-6)


def test_finite_gradients_and_frequency_finite_difference():
    c=configuration(); p=fixtures()["mixture"]
    model=Parameters({k:[v] for k,v in p.items()},c); r=Renderer(c,"grad")
    def objective():
        # Short prefix avoids oscillatory cancellation in finite differences.
        return (r.render(model.controls())[...,:64]*torch.linspace(.1,1,64)).mean()
    value=objective(); value.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
    analytical=float(model.spacing_raw.grad[0,0]); step=1e-4
    with torch.no_grad():
        old=float(model.spacing_raw[0,0]); model.spacing_raw[0,0]=old+step; high=float(objective())
        model.spacing_raw[0,0]=old-step; low=float(objective()); model.spacing_raw[0,0]=old
    fd=(high-low)/(2*step)
    assert abs(analytical-fd)/max(abs(fd),1e-6) < c["acceptance"]["finite_difference_relative_tolerance"]


def test_observation_channel_mean_resample_crop():
    c=configuration(); sr=48000; rng=np.random.default_rng(2); x=rng.normal(size=(144000,2))
    expected=resample_poly(x.mean(1),1,6,window=("kaiser",5))
    expected=resample_poly(expected,2,1,window=("kaiser",5))[8000:40000].astype(np.float32)
    np.testing.assert_array_equal(observation(x,sr,c),expected)
    # Averaging stored columns 0/1, not channel IDs 3/4, must cancel antiphase.
    cancelled=observation(np.stack((x[:96000,0],-x[:96000,0]),1),sr,c)
    assert not cancelled.any()
    with pytest.raises(ValueError,match="silent"):normalize(cancelled,c)


@pytest.mark.parametrize("bad", [np.zeros((100,2)),np.zeros((96000,1)),np.full((96000,2),np.nan),np.zeros(96000)])
def test_malformed_and_short_audio(bad):
    with pytest.raises(ValueError):observation(bad,48000,configuration())


def test_near_silence_not_amplified():
    with pytest.raises(ValueError):normalize(np.full(32000,1e-9,dtype=np.float32),configuration())


def test_objective_formula_and_zero_self_loss():
    c=configuration(); x=torch.tensor(np.random.default_rng(1).normal(size=32000),dtype=torch.float32)
    obj=Objective(x,c); y=x.roll(127)
    assert obj(x[None,None]).item() == 0
    total,terms=obj(y[None,None],True)
    manual=[]
    for n in c["loss"]["fft_sizes"]:
        a=obj.magnitude(shape(x)[None],n);b=obj.magnitude(shape(y)[None],n);eps=c["loss"]["epsilon"]
        manual.append((a-b).abs().mean()/(a.mean()+eps)+((a+eps).log()-(b+eps).log()).abs().mean())
    torch.testing.assert_close(total.squeeze(),torch.stack(manual).mean())


def test_optimizer_determinism_best_state_and_gain_ambiguity():
    c=configuration(); c["optimizer"]["steps"]=3
    p=fixtures()["mixture"]; r=Renderer(c,"det_target"); x=shape(r.render(tensors(p))[0,0]).detach()
    a,b=fit(x,c,"same"),fit(x,c,"same")
    for k in ("parameters","history","fit_loss","check_loss","starts"):assert a[k]==b[k]
    assert a["fit_loss"]<=a["baseline_fit_loss"]
    q=deepcopy(p);q["envelope_knots"]=[v*1.2 for v in p["envelope_knots"]]
    torch.testing.assert_close(shape(r.render(tensors(p))),shape(7*r.render(tensors(q))),atol=3e-6,rtol=1e-6)

from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from cast.renderer import tensors
from cast.synthetic import fixtures
from cast_improvement.engine import SmoothRenderer,model_config as old_config
from cast_improvement import temporal_prior as tp
from cast_improvement.temporal_sampling import pack,unpack,sample
from test_spectrum_bank import fixture


def test_dense_grid_nests_original_waveforms_and_preserves_bounds():
    torch.set_num_threads(1);p=deepcopy(fixtures()["mixture"]);p["envelope_knots"]=[.3,1.3,.7,1.8,.5]
    lifted=tp.lift(p);cfg=tp.model_config();tp.validate_controls(lifted,cfg)
    with torch.no_grad():
        old=SmoothRenderer(old_config(),"nested").render(tensors(p),"check")
        new=SmoothRenderer(cfg,"nested").render(tensors(lifted),"check")
        replay=SmoothRenderer(cfg,"nested").render(tensors(lifted),"check")
    torch.testing.assert_close(new,old,rtol=1e-5,atol=1e-5)
    assert torch.equal(new,replay)
    assert lifted["envelope_knots"][::10]==p["envelope_knots"]


@pytest.mark.parametrize("slope",[0.,.5,-.5])
def test_affine_envelope_recovery_with_same_amplitude_gauge(slope):
    truth=1+slope*np.linspace(-1,1,41);target=tp.frame_basis()@truth
    result=tp.calibrate(np.ones(41),target)
    np.testing.assert_allclose(tp.frame_basis()@result["envelope_knots"],target,atol=5e-6)
    assert result["calibrated_objective"]<1e-10
    assert abs(result["preserved_mean_amplitude"]-result["result_mean_amplitude"])<=1e-8
    assert result==tp.calibrate(np.ones(41),target)


def test_spiky_nonrepresentable_target_keeps_bounds_and_nonincrease():
    target=np.ones(40)*.1;target[20]=10
    r=tp.calibrate(np.ones(41),target)
    assert all(.1<=v<=3 for v in r["envelope_knots"])
    assert r["calibrated_objective"]<=r["initial_objective"]+1e-12
    assert "calibrated_temporal_boundary" in r["flags"]
    assert abs(r["result_mean_amplitude"]-r["preserved_mean_amplitude"])<1e-8


def test_solver_failure_retains_complete_lifted_trajectory(monkeypatch):
    original=np.linspace(.5,1.5,41)
    monkeypatch.setattr(tp,"minimize",lambda *a,**k:SimpleNamespace(x=np.ones(41),success=False,status=9,message="fixture",nit=300,nfev=301))
    r=tp.calibrate(original,np.ones(40))
    assert r["envelope_knots"]==original.tolist()
    assert "temporal_solver_failure_lifted_original_retained" in r["flags"]


def test_invalid_schema_targets_and_dense_gradient():
    cfg=tp.model_config();p=tp.lift(fixtures()["mixture"])
    with pytest.raises(ValueError):tp.validate_controls(fixtures()["mixture"],cfg)
    for target in (np.zeros(40),np.ones(39),np.r_[np.ones(39),np.nan]):
        with pytest.raises(ValueError):tp.calibrate(np.ones(41),target)
    controls=tensors(p);controls["envelope_knots"].requires_grad_(True)
    wave=SmoothRenderer(cfg,"gradient").render(controls,"check")
    wave[...,:800].square().mean().backward()
    grad=controls["envelope_knots"].grad
    assert grad.shape==(1,41) and torch.isfinite(grad).all() and grad.abs().sum()>0


@pytest.mark.parametrize("arm",["joint","prototype","marginals"])
def test_new_schema_sampler_and_ancestry_are_exact(arm):
    old,metric,bank,obs=fixture();cfg=tp.model_config(old);original=deepcopy(bank)
    derived=tp.derive_bank(bank,obs,cfg,"outer")
    assert bank==original
    row=tp.draw(derived,cfg,metric,"car",42,3,arm,"temporal41","held")
    assert row==tp.draw(list(reversed(derived)),cfg,metric,"car",42,3,arm,"temporal41","held")
    tp.validate_controls(row["parameters"],cfg)
    assert sum(n for _,n in row["coordinate_order"])==61
    assert set(row["calibration_ancestors"])==set(row["parent_ids"])
    if arm!="prototype":assert len(row["coordinate_donors"])==61
    p=derived[0]["parameters"]
    np.testing.assert_allclose(pack(unpack(pack(p))),pack(p),rtol=1e-15,atol=1e-15)
    for before,after in zip(bank,derived):
        for key in ("spacing_hz","harmonic_weights","noise_weights","harmonic_fraction"):
            assert before["parameters"][key]==after["parameters"][key]


def test_excluded_and_tampered_ancestry_fail_closed():
    old,metric,bank,obs=fixture();cfg=tp.model_config(old)
    with pytest.raises(ValueError):tp.derive_bank(bank,obs[:-1],cfg,"outer")
    derived=tp.derive_bank(bank,obs,cfg,"outer")
    with pytest.raises(ValueError,match="excluded"):tp.draw(derived,cfg,metric,"car",42,0,"joint","temporal41","a")
    derived[0]["temporal_calibration"]["group"]="other"
    with pytest.raises(ValueError,match="ancestry"):tp.draw(derived,cfg,metric,"car",42,0,"prototype","temporal41","held")
    with pytest.raises(ValueError):unpack(np.ones(60))

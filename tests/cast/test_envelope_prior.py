from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import abvid.cast.population.envelope_prior as ep
from test_parent_mixture_bank import fixture


@pytest.mark.parametrize("truth", [[1,1,1,1,1],[.2,.7,1.5,.9,.3],[2.5,1.8,1,.4,.2]])
def test_exact_representable_trajectory_recovered_with_preserved_gauge(truth):
    b = ep.frame_basis(); original = np.ones(5)
    result = ep.calibrate(original,b@truth)
    reconstructed = b@result["envelope_knots"]
    expected = b@truth; expected /= expected.mean()
    assert np.allclose(reconstructed,expected,atol=3e-6)
    assert result["calibrated_frame_objective"] < 1e-11
    assert result["result_mean_amplitude"] == pytest.approx(1,abs=1e-8)
    assert result == ep.calibrate(original,b@truth)


def test_impossible_spike_retained_with_bounds_and_nonincreasing_objective():
    target = np.zeros(40); target[20] = 6
    r = ep.calibrate(np.ones(5),target)
    assert all(.1<=x<=3 for x in r["envelope_knots"])
    assert r["result_mean_amplitude"] == pytest.approx(r["preserved_mean_amplitude"],abs=1e-8)
    assert r["calibrated_frame_objective"] <= r["original_frame_objective"]+1e-12
    assert "calibrated_envelope_boundary" in r["flags"]


def test_solver_failure_keeps_original_and_flags(monkeypatch):
    monkeypatch.setattr(ep,"minimize",lambda *a,**k:SimpleNamespace(x=np.ones(5),success=False,status=9,message="fixture failure",nit=300,nfev=301))
    original=np.array([.5,.7,1,1.3,1.5])
    r=ep.calibrate(original,np.ones(40))
    assert r["envelope_knots"]==original.tolist()
    assert "envelope_calibration_solver_failure_original_retained" in r["flags"]


def test_invalid_targets_fail_closed():
    for target in (np.zeros(40),-np.ones(40),np.ones(39),np.r_[np.ones(39),np.nan]):
        with pytest.raises(ValueError): ep.calibrate(np.ones(5),target)


def bank_fixture():
    cfg,metric,bank=fixture()
    for row in bank: row.update(parent={"id":row["file_id"]},fit_sha256="fixture")
    observations=[{k:r[k] for k in ("file_id","class","group","parent")} | {"descriptors":{"envelope":np.linspace(.5,1.5,40).tolist()}} for r in bank]
    return cfg,metric,bank,observations


@pytest.mark.parametrize("arm",["joint","prototype","marginals"])
def test_all_arms_preserve_mixture_and_envelope_ancestry(arm):
    cfg,metric,bank,obs=bank_fixture(); original=deepcopy(bank)
    derived=ep.derive_bank(bank,obs,cfg,"outer")
    assert bank==original
    for before,after in zip(bank,derived):
        for key in before["parameters"]:
            if key!="envelope_knots": assert before["parameters"][key]==after["parameters"][key]
    a=ep.draw(derived,cfg,metric,"car",42,4,arm,"mixture_envelope","held")
    assert set(a["calibration_ancestors"])==set(a["envelope_calibration_ancestors"])==set(a["parent_ids"])
    assert a==ep.draw(list(reversed(derived)),cfg,metric,"car",42,4,arm,"mixture_envelope","held")


def test_leakage_and_tampered_ancestry_rejected():
    cfg,metric,bank,obs=bank_fixture()
    with pytest.raises(ValueError): ep.derive_bank(bank,obs[:-1],cfg,"outer")
    derived=ep.derive_bank(bank,obs,cfg,"outer")
    with pytest.raises(ValueError): ep.draw(derived,cfg,metric,"car",42,0,"prototype","mixture_envelope","a")
    derived[0]["envelope_calibration"]["group"]="other"
    with pytest.raises(ValueError,match="ancestry"):
        ep.draw(derived,cfg,metric,"car",42,0,"prototype","mixture_envelope","held")

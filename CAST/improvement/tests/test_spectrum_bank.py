from copy import deepcopy
import pytest
from cast.config import digest
from cast_generalization.method import sample
from cast_improvement.spectrum_bank import draw,fit_record,validate_ancestry
from test_envelope_prior import bank_fixture
from cast_improvement.envelope_prior import derive_bank


def fixture():
    cfg,metric,bank,obs=bank_fixture();bank=derive_bank(bank,obs,cfg,"outer")
    for row in bank:
        result={"file_id":row["file_id"],"class_label":row["class"],"group":row["group"],
                "initial_parameters":deepcopy(row["parameters"]),"parameters":deepcopy(row["parameters"])}
        row["spectral_calibration"]=result;row["spectral_calibration_sha256"]=digest(result)
    return cfg,metric,bank,obs


@pytest.mark.parametrize("arm",["joint","prototype","marginals"])
def test_calibrated_arms_preserve_frozen_schedule_and_complete_ancestry(arm):
    cfg,metric,bank,_=fixture()
    expected=sample(bank,{**metric,"version":"cast_improvement_sampling_v1:held"},"car",42,5,arm,cfg)
    expected.pop("parent_flags")
    actual=draw(bank,cfg,metric,"car",42,5,arm,"spectrum_calibrated","held")
    assert all(actual[k]==v for k,v in expected.items())
    assert set(actual["calibration_ancestors"])==set(actual["parent_ids"])
    assert actual==draw(list(reversed(bank)),cfg,metric,"car",42,5,arm,"spectrum_calibrated","held")
    assert all(set(a)=={"spectral","earlier_mixture","envelope"} for a in actual["calibration_ancestors"].values())


@pytest.mark.parametrize("excluded",["a","outer"])
def test_excluded_ancestry_fails_before_sampling(excluded):
    cfg,metric,bank,_=fixture()
    if excluded=="outer":bank[0]["group"]="outer"
    with pytest.raises(ValueError,match="excluded"):
        draw(bank,cfg,metric,"car",42,0,"prototype","spectrum_calibrated",excluded)


def test_tampered_calibration_and_changed_unoptimized_controls_rejected():
    _,_,bank,_=fixture();row=bank[0]
    row["spectral_calibration"]["group"]="other"
    with pytest.raises(ValueError,match="ancestry"):validate_ancestry(row)
    row=fixture()[2][0]
    row["parameters"]["envelope_knots"][0]+=.01
    row["spectral_calibration"]["parameters"]=deepcopy(row["parameters"])
    row["spectral_calibration_sha256"]=digest(row["spectral_calibration"])
    with pytest.raises(ValueError,match="ancestry"):validate_ancestry(row)


def test_calibration_checks_parent_and_outer_boundary_before_optimizer():
    cfg,metric,bank,obs=fixture();wrong=deepcopy(obs[0]);wrong["group"]="other"
    with pytest.raises(ValueError,match="boundary"):fit_record(bank[0],wrong,cfg,metric,"fixture")
    row=deepcopy(bank[0]);row["group"]="outer";wrong=deepcopy(obs[0]);wrong["group"]="outer"
    with pytest.raises(ValueError,match="boundary"):fit_record(row,wrong,cfg,metric,"fixture")

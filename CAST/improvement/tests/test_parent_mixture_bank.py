from copy import deepcopy
import numpy as np
import pytest

from cast.config import digest
from cast.synthetic import fixtures
from cast_generalization.method import sample
from cast_improvement.engine import model_config
from cast_improvement.parent_mixture_bank import component_power, derived_row, derive_bank, draw


def fixture():
    cfg = model_config()
    metric = {"class_order": ["car", "truck"], "seeds": [42], "arms": ["joint", "prototype", "marginals"],
              "samples_per_class_per_seed_per_arm": 50, "held_group": "outer"}
    bank = []
    for group in ("a", "b"):
        for i in range(2):
            p = deepcopy(fixtures()["mixture"])
            p["harmonic_fraction"] = .2+.1*i
            record = {"file_id": f"{group}{i}", "class": "car", "group": group,
                      "harmonic_fraction": p["harmonic_fraction"]}
            bank.append({**record, "parameters": p, "flags": {}, "mixture_calibration": record,
                         "mixture_calibration_sha256": digest(record)})
    return cfg, metric, bank


@pytest.mark.parametrize("arm", ["joint", "prototype", "marginals"])
def test_original_schedule_and_complete_calibration_ancestry_are_retained(arm):
    cfg, metric, bank = fixture()
    original = sample(bank, {**metric, "version": "cast_improvement_sampling_v1:held"}, "car", 42, 3, arm, cfg)
    original.pop("parent_flags")
    a = draw(bank, cfg, metric, "car", 42, 3, arm, "parent_calibrated", "held")
    assert all(a[k] == v for k, v in original.items())
    assert set(a["calibration_ancestors"]) == set(a["parent_ids"])
    assert a == draw(list(reversed(bank)), cfg, metric, "car", 42, 3, arm, "parent_calibrated", "held")
    b = draw(bank, cfg, metric, "car", 42, 3, arm, "winner", "held")
    assert b["calibration_ancestors"] == {}
    assert all(b[k] == v for k, v in original.items())


@pytest.mark.parametrize("excluded", ["a", "outer"])
def test_excluded_parent_fails_before_sampling(excluded):
    cfg, metric, bank = fixture()
    if excluded == "outer": bank[0]["group"] = "outer"
    with pytest.raises(ValueError, match="excluded"):
        draw(bank, cfg, metric, "car", 42, 0, "prototype", "parent_calibrated", excluded)


def test_tampered_calibration_rejected():
    cfg, metric, bank = fixture()
    bank[0]["mixture_calibration"]["harmonic_fraction"] += .01
    with pytest.raises(ValueError, match="ancestry"):
        draw(bank, cfg, metric, "car", 42, 0, "prototype", "parent_calibrated", "held")


def test_derived_bank_changes_only_fraction_and_keeps_original_input():
    cfg, _, bank = fixture()
    row = bank[0]; before = deepcopy(row)
    t = np.arange(32000)/16000
    harmonic = np.tile(np.sin(2*np.pi*100*t), (1,2,1))
    noise = np.tile(np.sin(2*np.pi*2000*t), (1,2,1))
    target = component_power(noise, cfg["renderer"]["noise_edges_hz"]); target /= target.sum()
    result = derived_row(row, harmonic, noise, target, {"fixture": "synthetic"}, cfg)
    assert row == before
    assert result["parameters"]["harmonic_fraction"] == pytest.approx(0, abs=1e-12)
    for key in row["parameters"]:
        if key != "harmonic_fraction": assert result["parameters"][key] == row["parameters"][key]
    assert result["mixture_calibration_sha256"] == digest(result["mixture_calibration"])


def test_component_power_retains_amplitude_and_both_realizations():
    cfg, _, _ = fixture()
    t = np.arange(32000)/16000
    values = np.tile(np.sin(2*np.pi*100*t), (1,2,1))
    edges = cfg["renderer"]["noise_edges_hz"]
    assert np.allclose(component_power(values*2, edges), 4*component_power(values, edges))
    values[0,1] = 0
    assert np.allclose(component_power(values, edges), .5*component_power(np.tile(np.sin(2*np.pi*100*t), (1,2,1)), edges))
    with pytest.raises(ValueError): component_power(values[:, :1], edges)


def test_mismatched_calibration_inventory_rejected_before_file_access(tmp_path):
    cfg, _, bank = fixture()
    with pytest.raises(ValueError, match="exactly"):
        derive_bank(tmp_path, bank, [], cfg, "outer")

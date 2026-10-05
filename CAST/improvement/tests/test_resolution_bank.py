from copy import deepcopy
import numpy as np
import pytest

from cast.config import digest
from cast_improvement import temporal_prior as temporal
from cast_improvement.resolution_model import model_config, lift_controls, validate_controls
from cast_improvement.resolution_bank import draw, fit_record, validate_ancestry
from cast_improvement.resolution_sampling import pack, unpack
from cast_improvement.resolution_evaluation import render
from test_spectrum_bank import fixture as old_fixture


def fixture():
    old_cfg, metric, bank, obs = old_fixture()
    temporal_cfg = temporal.model_config(old_cfg)
    old_bank = temporal.derive_bank(bank, obs, temporal_cfg, "outer")
    cfg = model_config()
    derived = []
    for row in old_bank:
        p = lift_controls(row["parameters"], cfg)
        record = {"file_id": row["file_id"], "class_label": row["class"], "group": row["group"],
                  "initial_parameters": deepcopy(p), "parameters": deepcopy(p),
                  "initial_bank_row_sha256": digest(row)}
        copied = deepcopy(row)
        copied.update(parameters=p, resolution_calibration=record, resolution_calibration_sha256=digest(record))
        derived.append(copied)
    return cfg, metric, derived, obs, temporal_cfg, old_bank


@pytest.mark.parametrize("arm", ["joint", "prototype", "marginals"])
def test_69_coordinate_sampling_roundtrip_and_ancestry(arm):
    cfg, metric, bank, _, _, _ = fixture()
    row = draw(bank, cfg, metric, "car", 42, 3, arm, "spectrum16", "held")
    assert row == draw(list(reversed(bank)), cfg, metric, "car", 42, 3, arm, "spectrum16", "held")
    validate_controls(row["parameters"], cfg)
    assert sum(n for _, n in row["coordinate_order"]) == 69
    assert set(row["calibration_ancestors"]) == set(row["parent_ids"])
    if arm != "prototype":
        assert len(row["coordinate_donors"]) == 69
    p = bank[0]["parameters"]
    np.testing.assert_allclose(pack(unpack(pack(p))), pack(p), rtol=1e-15, atol=1e-15)
    a, streams = render(row, cfg, "spectrum16")
    b, replay_streams = render(row, cfg, "spectrum16")
    assert a.shape == (32000,) and np.isfinite(a).all() and np.array_equal(a, b)
    assert streams == replay_streams and streams["purpose"] == "sampling"


@pytest.mark.parametrize("arm", ["joint", "prototype", "marginals"])
def test_temporal_reference_sampling_is_exact(arm):
    _, metric, _, _, cfg, bank = fixture()
    actual = draw(bank, cfg, metric, "car", 42, 3, arm, "temporal41", "held")
    assert actual == temporal.draw(bank, cfg, metric, "car", 42, 3, arm, "temporal41", "held")


def test_changed_ancestry_unoptimized_controls_and_excluded_groups_fail():
    cfg, metric, bank, _, _, _ = fixture()
    with pytest.raises(ValueError, match="excluded"):
        draw(bank, cfg, metric, "car", 42, 0, "joint", "spectrum16", "a")
    bank[0]["resolution_calibration"]["group"] = "other"
    with pytest.raises(ValueError, match="ancestry"):
        draw(bank, cfg, metric, "car", 42, 0, "prototype", "spectrum16", "held")
    row = fixture()[2][0]
    row["parameters"]["envelope_knots"][0] += .01
    row["resolution_calibration"]["parameters"] = deepcopy(row["parameters"])
    row["resolution_calibration_sha256"] = digest(row["resolution_calibration"])
    with pytest.raises(ValueError, match="ancestry"):
        validate_ancestry(row)
    with pytest.raises(ValueError):
        unpack(np.ones(68))


def test_excluded_or_mismatched_parent_stops_before_calibration():
    cfg, metric, _, obs, _, bank = fixture()
    wrong = deepcopy(obs[0]); wrong["group"] = "other"
    with pytest.raises(ValueError, match="boundary"):
        fit_record(bank[0], wrong, cfg, metric, "fixture")
    row = deepcopy(bank[0]); row["group"] = "outer"
    wrong = deepcopy(obs[0]); wrong["group"] = "outer"
    with pytest.raises(ValueError, match="boundary"):
        fit_record(row, wrong, cfg, metric, "fixture")

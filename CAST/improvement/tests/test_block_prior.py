from copy import deepcopy
import numpy as np
import pytest

from cast.synthetic import fixtures
from cast.renderer import validate_controls
from cast_improvement.engine import model_config
from cast_improvement.block_prior import GRID, transform_bank, draw


def bank():
    rows = []
    for c in ("car", "truck"):
        for g in range(3):
            p = deepcopy(fixtures()["mixture"])
            p["spacing_hz"] = [80+50*g]*3
            p["envelope_knots"] = [.2+g*.15, .8, 2.4, .7, .15+g*.1]
            rows.append({"file_id": f"{c}_{g}", "class": c, "group": str(g), "parameters": p, "flags": {}})
    return rows


def test_identity_is_exact_and_envelope_expansion_preserves_spectral_controls():
    rows = bank(); cfg = model_config()
    actual, _ = transform_bank(rows, cfg, 1, 1)
    assert actual == rows
    # Envelope-only expansion does not change spectral controls after inversion.
    raised, _ = transform_bank(rows, cfg, 1, 1.25)
    for a, b in zip(rows, raised):
        for key in ("spacing_hz", "harmonic_weights", "noise_weights", "harmonic_fraction"):
            np.testing.assert_allclose(a["parameters"][key], b["parameters"][key], atol=1e-12, rtol=1e-12)


def test_every_declared_pair_retains_finite_bounded_controls_and_ancestry():
    for s, e in GRID:
        rows, stats = transform_bank(bank(), model_config(), s, e)
        for row in rows:
            validate_controls(row["parameters"], model_config())
        assert stats["car"]["parent_ids"] == ["car_0", "car_1", "car_2"]
        assert stats["truck"]["groups"] == ["0", "1", "2"]


def test_fold_exclusion_changes_center_without_using_validation_values():
    rows = bank()
    altered = deepcopy(rows)
    for r in altered:
        if r["group"] == "2": r["parameters"]["spacing_hz"] = [399]*3
    a = transform_bank([r for r in rows if r["group"] != "2"], model_config(), .65, 1.25)
    b = transform_bank([r for r in altered if r["group"] != "2"], model_config(), .65, 1.25)
    assert a == b


def test_class_center_is_isolated_and_group_weighted():
    rows = bank(); altered = deepcopy(rows)
    for r in altered:
        if r["class"] == "truck": r["parameters"]["spacing_hz"] = [390]*3
    a, sa = transform_bank(rows, model_config(), .65, 1.25)
    b, sb = transform_bank(altered, model_config(), .65, 1.25)
    assert [r for r in a if r["class"] == "car"] == [r for r in b if r["class"] == "car"]
    repeated = deepcopy(rows[0]); repeated["file_id"] = "car_0_extra"
    _, sc = transform_bank(rows+[repeated], model_config(), .65, 1.25)
    np.testing.assert_array_equal(sa["car"]["center"], sc["car"]["center"])


def test_direct_and_center_ancestry_are_checked_before_rendering():
    rows, stats = transform_bank(bank(), model_config(), .65, 1.25)
    metric = {"version": "unused", "class_order": ["car", "truck"], "arms": ["joint", "prototype", "marginals"],
              "seeds": [42], "samples_per_class_per_seed_per_arm": 2, "held_group": "held"}
    r = draw(rows, stats, model_config(), metric, "car", 42, 0, "joint", "valid", .65, 1.25)
    assert len(r["parent_ids"]) == 1 and len(r["prior_calibration_parent_ids"]) == 3
    assert r == draw(rows, stats, model_config(), metric, "car", 42, 0, "joint", "valid", .65, 1.25)
    with pytest.raises(ValueError, match="ancestry"):
        draw(rows, stats, model_config(), metric, "car", 42, 0, "joint", "2", .65, 1.25)


def test_undeclared_grid_missing_classes_and_duplicate_parents_rejected():
    with pytest.raises(ValueError): transform_bank(bank(), model_config(), .9, 1.)
    with pytest.raises(ValueError): transform_bank(bank()[:3], model_config(), .65, 1.25)
    with pytest.raises(ValueError): transform_bank(bank()+[bank()[0]], model_config(), .65, 1.25)

from copy import deepcopy
import numpy as np
import pytest

from cast.config import save, sha
from cast.synthetic import fixtures
from cast_improvement.capacity_engine import lift_controls, model_config, validate_controls
from cast_improvement.capacity_sampling import pack, unpack, sample
from cast_improvement.capacity_evaluation import generated_rows, require_audit


def bank():
    cfg = model_config()
    rows = []
    for label in ("car", "truck"):
        for g in ("a", "b"):
            for i in range(2):
                p = lift_controls(fixtures()["mixture"], cfg)
                p["spacing_hz"] = [80.+i*40+(g == "b")*30]*3
                rows.append({"class": label, "group": g, "file_id": f"{label}_{g}_{i}",
                             "parameters": p, "flags": {"ambiguity": True}})
    return rows, cfg


def schedule():
    return {"class_order": ["car", "truck"], "seeds": [42], "arms": ["joint", "prototype", "marginals"],
            "samples_per_class_per_seed_per_arm": 50, "held_group": "outer", "version": "test"}


def test_capacity_sampling_schema_roundtrip_and_invalid_sizes():
    rows, cfg = bank(); p = rows[0]["parameters"]
    assert pack(p).shape == (37,)
    np.testing.assert_allclose(pack(unpack(pack(p))), pack(p), rtol=1e-15, atol=1e-15)
    for a in (np.ones(25), np.ones(38), np.r_[np.ones(36), np.nan]):
        with pytest.raises(ValueError): unpack(a)


@pytest.mark.parametrize("arm", ["joint", "prototype", "marginals"])
def test_capacity_arms_deterministic_valid_and_complete_ancestry(arm):
    rows, cfg = bank()
    a = sample(rows, schedule(), "car", 42, 0, arm, cfg)
    assert a == sample(list(reversed(rows)), schedule(), "car", 42, 0, arm, cfg)
    validate_controls(a["parameters"], cfg)
    parents = {r["file_id"]: r for r in rows}
    assert all(parents[i]["class"] == "car" for i in a["parent_ids"])
    assert a["parent_groups"] == sorted({parents[i]["group"] for i in a["parent_ids"]})
    assert sum(n for _, n in a["coordinate_order"]) == 37
    if arm == "joint":
        assert len(a["parent_ids"]) == 1
        assert a["parameters"] == parents[a["parent_ids"][0]]["parameters"]
        assert len(a["coordinate_donors"]) == 37
    elif arm == "prototype":
        assert len(a["parent_ids"]) == 4
    else:
        assert len(a["coordinate_donors"]) == 37
        assert set(a["coordinate_donors"]) == set(a["parent_ids"])


def test_source_fold_guard_precedes_rendering():
    rows, cfg = bank()
    with pytest.raises(ValueError, match="boundary"):
        next(generated_rows(rows, cfg, schedule(), "capacity16x9", "a"))
    s = schedule(); s["held_group"] = "a"
    with pytest.raises(ValueError, match="contaminated"):
        sample(rows, s, "car", 42, 0, "joint", cfg)


def test_new_bank_cannot_bypass_fit_audit(tmp_path):
    save(tmp_path/"fit_pilot_summary.json", {"passed": True, "failures": [], "records": []})
    save(tmp_path/"fit_pilot_verification.json", {"passed": False})
    with pytest.raises(ValueError, match="audit"):
        require_audit(tmp_path, "pilot", {"missing"})

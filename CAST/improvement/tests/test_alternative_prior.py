from copy import deepcopy
import numpy as np
import pytest

from cast.synthetic import fixtures
from cast_generalization.method import pack, sample as original_sample
from cast_improvement.engine import model_config
from cast_improvement.alternative_prior import eligible_starts, sample


def setup():
    cfg = {"version": "test", "class_order": ["car", "truck"], "seeds": [42],
           "arms": ["joint", "prototype", "marginals"], "samples_per_class_per_seed_per_arm": 50,
           "held_group": "outer"}
    rows = []
    for g in ("a", "b"):
        for i in range(2):
            p = deepcopy(fixtures()["mixture"])
            p["spacing_hz"] = [50.+i*20+(g == "b")*50]*3
            q = deepcopy(p); q["spacing_hz"] = [v+20 for v in p["spacing_hz"]]
            alternatives = [{"start_index": 0, "start_seed": 42, "parameters": p, "fit_loss": 1.}]
            if i:
                alternatives.append({"start_index": 1, "start_seed": 123, "parameters": q, "fit_loss": 1.04})
            rows.append({"file_id": f"{g}_{i}", "class": "car", "group": g, "flags": {}, "parameters": p,
                         "alternatives": alternatives, "winning_start_index": 0})
    return rows, cfg, model_config()


@pytest.mark.parametrize("arm", ["joint", "prototype", "marginals"])
def test_winner_mode_preserves_every_original_sampling_field(arm):
    rows, cfg, renderer = setup()
    a = original_sample(rows, cfg, "car", 42, 0, arm, renderer)
    b = sample(rows, cfg, "car", 42, 0, arm, renderer, "winner")
    assert all(b[k] == v for k, v in a.items())
    assert all(r["start_index"] == 0 for r in b["alternative_ancestors"])


@pytest.mark.parametrize("arm", ["joint", "prototype", "marginals"])
def test_alternatives_replay_and_preserve_parent_schedule(arm):
    rows, cfg, renderer = setup()
    a = sample(rows, cfg, "car", 42, 3, arm, renderer, "equivalent_fits")
    assert a == sample(list(reversed(rows)), cfg, "car", 42, 3, arm, renderer, "equivalent_fits")
    b = original_sample(rows, cfg, "car", 42, 3, arm, renderer)
    assert a["parent_ids"] == b["parent_ids"] and a["coordinate_donors"] == b["coordinate_donors"]
    parents = {r["file_id"]: r for r in rows}
    for donor in a["alternative_ancestors"]:
        assert donor["start_index"] in [s["start_index"] for s in parents[donor["parent_id"]]["alternatives"]]
    if arm == "joint":
        donor = a["alternative_ancestors"][0]
        p = next(s["parameters"] for s in parents[donor["parent_id"]]["alternatives"] if s["start_index"] == donor["start_index"])
        assert a["parameters"] == p


def test_prototype_equal_parent_weight_despite_different_alternative_counts():
    rows, cfg, renderer = setup()
    a = sample(rows, cfg, "car", 42, 0, "prototype", renderer, "equivalent_fits")
    # Parent values are 50, mean(70,90)=80, 100, mean(120,140)=130.
    assert a["parameters"]["spacing_hz"] == [90.]*3


def test_eligibility_uses_fitting_loss_and_keeps_fixed_bound():
    rows, _, renderer = setup(); p = rows[0]["parameters"]
    r = {"winner": 0, "fit_loss": 1., "starts": [
        {"fit_loss": v, "seed": seed, "check_loss": check, "parameters": p}
        for v, seed, check in ((1.,42,2.),(1.05,123,3.),(1.05001,456,.1))]}
    a = eligible_starts(r, renderer)
    assert [s["start_index"] for s in a] == [0, 1]
    other = deepcopy(renderer); other["diagnostics"]["equally_good_relative_loss"] = .1
    with pytest.raises(ValueError): eligible_starts(r, other)


def test_contaminated_alternative_bank_rejected():
    rows, cfg, renderer = setup(); rows[0]["group"] = "outer"
    with pytest.raises(ValueError, match="contaminated"):
        sample(rows, cfg, "car", 42, 0, "joint", renderer, "equivalent_fits")

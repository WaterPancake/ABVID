"""Acceptance boundaries and fail-closed sample/selection provenance."""
from copy import deepcopy
import json

import pytest

from cast.config import sha
from cast_improvement.evaluate import aggregate
from cast_improvement.outer import require_source_pass, validate_schedule


CFG = {"class_order": ["car", "truck"], "arms": ["joint", "prototype", "marginals"],
       "seeds": [42, 123], "samples_per_class_per_seed_per_arm": 2,
       "primary_families": ["bands", "envelope"]}


def schedule():
    return [{"class": c, "arm": a, "seed": s, "index": i}
            for c in CFG["class_order"] for a in CFG["arms"] for s in CFG["seeds"] for i in range(2)]


def test_complete_schedule_rejects_duplicate_replacement_and_missing_seed():
    rows = schedule()
    validate_schedule(list(reversed(rows)), CFG)
    rows[-1] = deepcopy(rows[0])
    with pytest.raises(ValueError, match="schedule"):
        validate_schedule(rows, CFG)
    with pytest.raises(ValueError, match="schedule"):
        validate_schedule([r for r in schedule() if r["seed"] == 42], CFG)


def scores():
    return [{"class": c, "arm": a, "seed": s, "variant": "test", "temperature": 1.,
             "score": {"W1": .974 if a == "joint" else 1., "coverage": .801,
                       "families": {f: {"spread_ratio": 1.} for f in CFG["primary_families"]}}}
            for c in CFG["class_order"] for a in CFG["arms"] for s in CFG["seeds"]]


@pytest.mark.parametrize("failure", ["coverage", "margin", "spread"])
def test_one_class_cannot_hide_failure_in_pooled_average(failure):
    rows = scores()
    assert aggregate(rows, CFG)["test_T1"]["passes"]
    for r in rows:
        if r["class"] == "truck" and r["arm"] == "joint":
            if failure == "coverage": r["score"]["coverage"] = .799
            if failure == "margin": r["score"]["W1"] = .976
            if failure == "spread": r["score"]["families"]["envelope"]["spread_ratio"] = .499
    result = aggregate(rows, CFG)["test_T1"]
    assert not result["passes"]
    assert all(result["classes"]["car"]["passes"].values())


def test_failed_source_candidate_rejected_before_any_file_access(tmp_path):
    with pytest.raises(ValueError, match="fails declared criteria"):
        require_source_pass(tmp_path/"absent", {"passes": False})


def test_source_pass_requires_matching_replay_audit(tmp_path):
    (tmp_path/"summary.json").write_text('{"selected":"example"}')
    audit = {"passed": True, "summary_sha256": sha(tmp_path/"summary.json"), "outer_held_audio_access": False}
    (tmp_path/"audit_verification.json").write_text(json.dumps(audit))
    require_source_pass(tmp_path, {"passes": True})
    (tmp_path/"summary.json").write_text('{"selected":"changed"}')
    with pytest.raises(ValueError, match="audit"):
        require_source_pass(tmp_path, {"passes": True})

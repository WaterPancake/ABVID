from pathlib import Path
import pytest
from cast_improvement import width_outer as outer


def test_failed_group_source_stops_before_any_outer_access(monkeypatch, tmp_path):
    monkeypatch.setattr(outer, "setup_cpu", lambda: None)
    monkeypatch.setattr(outer, "check_source", lambda _: ({}, {}, {"passes": False}))
    def forbidden(*args, **kwargs):
        raise AssertionError("Outer prior/held inputs accessed despite failed source candidate")
    monkeypatch.setattr(outer, "prior_frozen", forbidden)
    destination = tmp_path/"outer"
    with pytest.raises(ValueError, match="fails declared criteria"):
        outer.run(tmp_path/"source", destination)
    assert not destination.exists()


def test_passing_candidate_still_requires_matching_audit_before_access(monkeypatch, tmp_path):
    monkeypatch.setattr(outer, "setup_cpu", lambda: None)
    monkeypatch.setattr(outer, "check_source", lambda _: ({}, {}, {"passes": True,"eligible":True,"original_mean_criteria_passed":True,"source_fold_coverage_passed":True}))
    def read_only_failed_audit(path):
        assert Path(path).name == "audit_verification.json"
        return {"passed": False}
    monkeypatch.setattr(outer, "read", read_only_failed_audit)
    monkeypatch.setattr(outer, "prior_frozen", lambda *_: pytest.fail("Outer access before valid audit"))
    with pytest.raises(ValueError, match="audit missing"):
        outer.run(tmp_path/"source", tmp_path/"outer")


def test_unchanged_reference_does_not_repeat_outer(monkeypatch,tmp_path):
    monkeypatch.setattr(outer,"setup_cpu",lambda:None)
    monkeypatch.setattr(outer,"check_source",lambda _:({}, {}, {"passes":True,"variant":"spectrotemporal_context","temperature":1.,"eligible":True,"original_mean_criteria_passed":True,"source_fold_coverage_passed":True}))
    monkeypatch.setattr(outer,"require_source_pass",lambda *_:None)
    monkeypatch.setattr(outer,"prior_frozen",lambda *_:pytest.fail("Reference winner must not reopen outer inputs"))
    with pytest.raises(ValueError,match="Reference width one retained"):
        outer.run(tmp_path/"source",tmp_path/"outer")


def test_original_mean_pass_cannot_bypass_stronger_source_fold_gate(monkeypatch,tmp_path):
    monkeypatch.setattr(outer,"setup_cpu",lambda:None)
    candidate={"passes":True,"eligible":False,"original_mean_criteria_passed":True,"source_fold_coverage_passed":False}
    monkeypatch.setattr(outer,"check_source",lambda _:({}, {}, candidate))
    monkeypatch.setattr(outer,"read",lambda *_:pytest.fail("Failed fold gate must stop before audit or outer access"))
    with pytest.raises(ValueError,match="fails declared criteria"):
        outer.run(tmp_path/"source",tmp_path/"outer")

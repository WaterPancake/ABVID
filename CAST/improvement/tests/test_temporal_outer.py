from pathlib import Path
import pytest
from cast_improvement import temporal_outer as outer


def test_failed_temporal_source_stops_before_any_outer_access(monkeypatch, tmp_path):
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
    monkeypatch.setattr(outer, "check_source", lambda _: ({}, {}, {"passes": True}))
    def read_only_failed_audit(path):
        assert Path(path).name == "audit_verification.json"
        return {"passed": False}
    monkeypatch.setattr(outer, "read", read_only_failed_audit)
    monkeypatch.setattr(outer, "prior_frozen", lambda *_: pytest.fail("Outer access before valid audit"))
    with pytest.raises(ValueError, match="audit missing"):
        outer.run(tmp_path/"source", tmp_path/"outer")

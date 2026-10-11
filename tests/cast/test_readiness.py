"""Readiness auditing is metadata/provenance work, not a new scientific run."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from abvid.cast import readiness
from abvid.provenance.hashing import sha256


@pytest.fixture
def storage(tmp_path, monkeypatch):
    locations = {name: tmp_path / name for name in ("repo", "data", "artifact", "archive")}
    for path in locations.values():
        path.mkdir()
    for name, path in locations.items():
        monkeypatch.setenv(f"ABVID_{name.upper()}_ROOT", str(path))
    return locations


def reference(storage, name, content):
    path = storage["repo"] / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    return {"path": f"repo:{name}", "sha256": sha256(path)}


@pytest.fixture
def passing_evidence(storage):
    junit = reference(storage, "evidence/tests.xml", """<testsuites><testsuite tests="2">
        <testcase name="roles"/><testcase name="smooth8_full_parity"/>
        </testsuite></testsuites>""")
    package_fit = reference(storage, "src/abvid/cast/smooth_fit.py", "# packaged fixture\n")
    parity_test = reference(storage, "tests/cast/test_smooth_fit.py", "# parity test fixture\n")
    archive_engine = storage["archive"] / "CAST/improvement/src/cast_improvement/engine.py"
    archive_engine.parent.mkdir(parents=True)
    archive_engine.write_text("# archived fixture\n")
    result = {"starts": list(range(4)), "history": list(range(301)), "final_loss": 0.1}
    parity = reference(storage, "evidence/full_parity.json", json.dumps({
        "exact_parity_excluding_elapsed_seconds": True, "real_audio_opened": False,
        "scope": "synthetic_fixture_packaging_parity_only",
        "archive": {"fit_seconds": 1.0, "result": result},
        "package": {"fit_seconds": 2.0, "result": result},
        "request": {"config": {"optimizer": {"steps": 300}}},
        "packaged_fit_sha256": package_fit["sha256"], "test_sha256": parity_test["sha256"],
        "archived_engine_sha256": sha256(archive_engine),
    }))
    return [junit, parity]


def test_validation_gate_requires_complete_pass_and_full_budget(passing_evidence):
    assert readiness.validation_gate(passing_evidence) == {
        "passed_tests": 2, "parity_cases": 1, "full_budget_parity": True,
        "scientific_experiment_run": False,
    }


@pytest.mark.parametrize("failure", ["failure", "error", "skipped"])
def test_validation_gate_rejects_failed_or_skipped_tests(storage, passing_evidence, failure):
    passing_evidence[0] = reference(storage, "evidence/tests.xml", f"""<testsuite>
        <testcase name="incomplete"><{failure}/></testcase></testsuite>""")
    with pytest.raises(ValueError, match="complete passing test schedule"):
        readiness.validation_gate(passing_evidence)


@pytest.mark.parametrize("xml", ["<testsuites/>", "<testsuites><testsuite/></testsuites>"])
def test_validation_gate_rejects_empty_schedule(storage, passing_evidence, xml):
    passing_evidence[0] = reference(storage, "evidence/tests.xml", xml)
    with pytest.raises(ValueError, match="complete passing test schedule"):
        readiness.validation_gate(passing_evidence)


@pytest.mark.parametrize("field,value", [
    ("steps", 2), ("steps", 299), ("starts", 1), ("starts", 3),
    ("history_states", 300),
])
def test_validation_gate_rejects_smoke_only_or_failed_parity(storage, passing_evidence, field, value):
    evidence = json.loads((storage["repo"] / "evidence/full_parity.json").read_text())
    if field == "steps":
        evidence["request"]["config"]["optimizer"]["steps"] = value
    else:
        for implementation in ("archive", "package"):
            evidence[implementation]["result"]["history" if field == "history_states" else field] = list(range(value))
    passing_evidence[1] = reference(storage, "evidence/full_parity.json", json.dumps(evidence))
    with pytest.raises(ValueError, match="Missing full-budget"):
        readiness.validation_gate(passing_evidence)


@pytest.mark.parametrize("remove", [0, 1])
def test_validation_gate_requires_both_test_report_and_numerical_evidence(passing_evidence, remove):
    passing_evidence.pop(remove)
    with pytest.raises(ValueError, match="evidence are required"):
        readiness.validation_gate(passing_evidence)


def test_validation_gate_rejects_a_failed_case_even_if_full_case_passes(storage, passing_evidence):
    passing_evidence.append(reference(storage, "evidence/failed_parity.json",
                                      json.dumps({"exact_parity_excluding_elapsed_seconds": False})))
    with pytest.raises(ValueError, match="failed or out-of-scope case"):
        readiness.validation_gate(passing_evidence)


@pytest.mark.parametrize("field,value", [
    ("exact_parity_excluding_elapsed_seconds", False), ("exact_parity_excluding_elapsed_seconds", 1),
    ("real_audio_opened", True), ("real_audio_opened", None), ("scope", "new_scientific_experiment"),
])
def test_parity_attestation_must_have_verified_packaging_scope(storage, passing_evidence, field, value):
    evidence = json.loads((storage["repo"] / "evidence/full_parity.json").read_text())
    evidence[field] = value
    passing_evidence[1] = reference(storage, "evidence/full_parity.json", json.dumps(evidence))
    with pytest.raises(ValueError, match="failed or out-of-scope case"):
        readiness.validation_gate(passing_evidence)


def test_exact_attestation_is_insufficient_when_saved_numerical_records_differ(storage, passing_evidence):
    evidence = json.loads((storage["repo"] / "evidence/full_parity.json").read_text())
    evidence["package"]["result"]["final_loss"] = 100.0
    passing_evidence[1] = reference(storage, "evidence/full_parity.json", json.dumps(evidence))
    with pytest.raises(ValueError, match="numerical evidence differs"):
        readiness.validation_gate(passing_evidence)


@pytest.mark.parametrize("root,path", [
    ("repo", "src/abvid/cast/smooth_fit.py"), ("repo", "tests/cast/test_smooth_fit.py"),
    ("archive", "CAST/improvement/src/cast_improvement/engine.py"),
])
def test_parity_proof_requires_the_same_implementation_and_test_bytes(storage, passing_evidence, root, path):
    (storage[root] / path).write_text("# changed after numerical parity\n")
    with pytest.raises(ValueError, match="Changed readiness input"):
        readiness.validation_gate(passing_evidence)


def test_validation_evidence_hash_must_still_match(storage, passing_evidence):
    (storage["repo"] / "evidence/full_parity.json").write_text("{}")
    with pytest.raises(ValueError, match="Changed readiness input"):
        readiness.validation_gate(passing_evidence)


@pytest.fixture
def contract(storage, passing_evidence, monkeypatch):
    entrypoint = reference(storage, "entry.py", "# Synthetic metadata audit entrypoint fixture\n")
    monkeypatch.setattr(readiness, "__file__", str(storage["repo"] / "entry.py"))
    cfg = {
        "schema_version": 1, "id": "cast4a_fit_readiness_v1",
        "scope": "existing_source_fit_audit_and_synthetic_parity_only",
        "design_config": reference(storage, "configs/experiments/cast_grouped_coverage.json", "{}"),
        "design_lock": reference(storage, "design_lock.json", "{}"),
        "chain_contract": reference(storage, "chain.json", json.dumps({"candidate_ids": ["parent_a", "parent_b"]})),
        "implementation_references": [reference(storage, "implementation.py", "# original numerical settings\n")],
        "validation_references": passing_evidence,
        "expected_environment": {"fixture": "pinned_environment"},
        "output": "artifact:cast/readiness_fixture", "entrypoint": entrypoint,
    }
    monkeypatch.setattr(readiness, "environment", lambda: {"fixture": "pinned_environment"})
    reference(storage, readiness.CONFIG.removeprefix("repo:"), json.dumps(cfg))
    return cfg


def test_fixed_contract_checks_its_dependencies_before_auditing(contract):
    assert readiness.load_contract() == contract
    with pytest.raises(ValueError, match="fixed readiness audit"):
        readiness.load_contract("repo:any_other_contract.json")


@pytest.mark.parametrize("field", ["design_config", "design_lock", "chain_contract", "entrypoint",
                                  "implementation_references", "validation_references"])
def test_changed_dependency_blocks_audit_before_loading_any_fits(storage, contract, field):
    ref = contract[field]
    if isinstance(ref, list):
        ref = ref[0]
    path = storage["repo"] / ref["path"].removeprefix("repo:")
    path.write_bytes(path.read_bytes() + b" changed")
    with pytest.raises(ValueError, match="Changed readiness input"):
        readiness.load_contract()


def test_numerical_environment_change_is_not_silently_accepted(contract, monkeypatch):
    monkeypatch.setattr(readiness, "environment", lambda: {"fixture": "different_environment"})
    with pytest.raises(ValueError, match="frozen numerical environment"):
        readiness.load_contract()


class FakeRole:
    def __init__(self, partition, train, held):
        self.partition_id = partition
        self.train_ids = tuple(train)
        self.held_groups = tuple(held)
        self.design_identity_sha256 = "verified_design_fixture"
        self.checked = []

    def source_metadata(self, parent):
        assert parent in self.train_ids
        return {"file_id": parent, "recording_session": None}

    def validate_parent(self, parent, *, ancestry):
        assert parent in self.train_ids
        assert ancestry == [self.source_metadata(parent)]
        self.checked.append(parent)


@pytest.fixture
def audit(contract, monkeypatch):
    declared = {
        "group_0": FakeRole("group_0", ["parent_a", "missing_a"], ["held_group_0"]),
        "site_0": FakeRole("site_0", ["parent_b", "missing_b"], ["held_date_0", "held_date_1"]),
    }
    records = [{"file_id": "parent_b", "eligible": True, "failures": []},
               {"file_id": "parent_a", "eligible": True, "failures": []}]
    shared = {"shared_failures": [], "scientific_experiment_run": False}
    module = ModuleType("abvid.cast.reuse_chain")
    def audit_saved_parents(chain):
        assert chain == {"candidate_ids": ["parent_a", "parent_b"]}
        return deepcopy(records), deepcopy(shared)
    module.audit_saved_parents = audit_saved_parents
    monkeypatch.setitem(sys.modules, "abvid.cast.reuse_chain", module)
    monkeypatch.setattr(readiness, "load_roles", lambda reference: declared)
    return SimpleNamespace(cfg=contract, roles=declared, records=records, shared=shared)


def test_eligible_parents_are_scoped_to_current_roles_without_claiming_complete_banks(audit):
    payload = readiness.evaluate(audit.cfg)
    summary = json.loads(payload["summary.json"])
    folds = json.loads(payload["fold_reuse.json"])
    assert summary["candidate_count"] == summary["eligible_candidate_count"] == 2
    assert summary["unique_selected_training_ids"] == 4
    assert summary["unique_selected_ids_without_eligible_cache"] == 2
    assert summary["new_real_data_fits"] == summary["generated_research_observations"] == 0
    assert summary["coverage_metrics_computed"] is False
    assert folds[0]["eligible_cached_ids"] == ["parent_a"]
    assert folds[0]["requires_new_fit_or_other_audited_cache"] == ["missing_a"]
    assert folds[1]["eligible_cached_ids"] == ["parent_b"]
    assert folds[1]["requires_new_fit_or_other_audited_cache"] == ["missing_b"]
    assert folds[1]["held_groups"] == ["held_date_0", "held_date_1"]
    assert all(fold["population_statistics_reusable"] is False for fold in folds)
    assert all(fold["role"] == "per_parent_cache_eligibility_only_no_bank_constructed" for fold in folds)
    assert audit.roles["group_0"].checked == ["parent_a"]
    assert audit.roles["site_0"].checked == ["parent_b"]


def test_audit_payload_is_deterministic_and_records_are_sorted(audit):
    first = readiness.evaluate(audit.cfg)
    audit.records.reverse()
    second = readiness.evaluate(audit.cfg)
    assert first == second
    assert [json.loads(line)["file_id"] for line in first["parents.jsonl"].splitlines()] == ["parent_a", "parent_b"]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "foreign", "eligible_failed", "shared_failure", "ambiguous_eligible"])
def test_incomplete_or_inconsistent_chain_audit_is_rejected(audit, defect):
    if defect == "missing":
        audit.records.pop()
    elif defect == "duplicate":
        audit.records[0] = deepcopy(audit.records[1])
    elif defect == "foreign":
        audit.records[0]["file_id"] = "outside_candidate_inventory"
    elif defect == "eligible_failed":
        audit.records[0]["failures"] = ["ancestry_mismatch"]
    elif defect == "shared_failure":
        audit.shared["shared_failures"] = ["numerical_settings_changed"]
    else:
        audit.records[0]["eligible"] = "unknown"
    with pytest.raises(ValueError):
        readiness.evaluate(audit.cfg)


def test_failed_candidates_remain_reported_and_are_never_reused(audit):
    audit.records[0].update(eligible=False, failures=["incomplete_same_parent_chain"])
    result = readiness.evaluate(audit.cfg)
    summary = json.loads(result["summary.json"])
    assert summary["candidate_count"] == 2
    assert summary["eligible_candidate_count"] == summary["failed_candidate_count"] == 1
    assert json.loads(result["fold_reuse.json"])[1]["eligible_cached_ids"] == []
    assert len(result["parents.jsonl"].splitlines()) == 2


@pytest.mark.parametrize("output", ["repo:readiness", "data:readiness", "archive:readiness", "artifact:."])
def test_destination_rejects_checkout_archive_and_root(storage, output):
    with pytest.raises(ValueError):
        readiness.destination({"output": output})


def test_destination_accepts_only_external_artifact_subdirectory(storage):
    assert readiness.destination({"output": "artifact:cast/readiness"}) == storage["artifact"] / "cast/readiness"
    for name in ("repo", "archive"):
        (storage["artifact"] / name).symlink_to(storage[name], target_is_directory=True)
        with pytest.raises(ValueError):
            readiness.destination({"output": f"artifact:{name}/write_attempt"})


@pytest.fixture
def saved(storage, monkeypatch):
    monkeypatch.setattr(readiness, "environment", lambda: {"fixture": "pinned_environment"})
    monkeypatch.setattr(readiness.subprocess, "check_output", lambda *args, **kwargs: "fixture_git_commit\n")
    payload = {"summary.json": readiness.encoded({"scientific_experiment_run": False}),
               "evidence/deep/proof.json": b'{"exact_parity":true}\n'}
    payload.update({f"evidence/snapshot/dependency_{index:02d}.py": f"# pinned dependency {index}\n".encode()
                    for index in range(20)})
    output = storage["artifact"] / "audit"
    lock = readiness.save(output, payload, 1.5)
    return output, payload, lock


def test_save_is_idempotent_and_preserves_existing_bytes_and_times(saved):
    output, payload, lock = saved
    before = {path.relative_to(output): (path.read_bytes(), path.stat().st_mtime_ns)
              for path in output.rglob("*") if path.is_file()}
    assert len(before) == 23  # Recursive proof/source inventory plus one lock.
    assert readiness.verify(output, payload) == lock
    assert readiness.save(output, payload, 999) == lock
    after = {path.relative_to(output): (path.read_bytes(), path.stat().st_mtime_ns)
             for path in output.rglob("*") if path.is_file()}
    assert before == after


def test_replay_rejects_tamper_even_when_local_lock_is_rewritten(saved):
    output, payload, lock = saved
    tampered = readiness.encoded({"scientific_experiment_run": True})
    (output / "summary.json").write_bytes(tampered)
    lock["payload_sha256"]["summary.json"] = hashlib.sha256(tampered).hexdigest()
    lock["payload_identity_sha256"] = readiness.digest(lock["payload_sha256"])
    (output / "lock.json").write_bytes(readiness.encoded(lock))
    with pytest.raises(ValueError, match="payload identity changed"):
        readiness.verify(output, payload)


@pytest.mark.parametrize("defect", ["missing", "extra", "symlink", "status"])
def test_replay_rejects_inventory_symlink_or_scope_changes(saved, tmp_path, defect):
    output, payload, lock = saved
    if defect == "missing":
        (output / "summary.json").unlink()
    elif defect == "extra":
        (output / "unexpected.json").write_text("{}")
    elif defect == "symlink":
        mirror = tmp_path / "untrusted.json"
        mirror.write_bytes(payload["summary.json"])
        (output / "summary.json").unlink()
        (output / "summary.json").symlink_to(mirror)
    else:
        lock["status"] = "scientific_experiment_complete"
        (output / "lock.json").write_bytes(readiness.encoded(lock))
    with pytest.raises(ValueError):
        readiness.verify(output, payload)

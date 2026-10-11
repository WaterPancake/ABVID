"""Metadata plan guards; no research audio or numerical arrays are opened."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from abvid.cast import grouped_execution_config as config
from abvid.cast import grouped_design
from abvid.provenance.hashing import canonical, sha256


@pytest.fixture
def storage(tmp_path, monkeypatch):
    locations = {name: (tmp_path/name).resolve() for name in ("repo", "artifact", "archive")}
    for path in locations.values():
        path.mkdir()
    monkeypatch.setattr(config, "roots", lambda: locations)
    def resolve(reference, must_exist=True):
        scheme, name = reference.split(":", 1)
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Path traversal")
        result = (locations[scheme]/name).resolve()
        if must_exist and not result.exists():
            raise FileNotFoundError(result)
        return result
    monkeypatch.setattr(config, "resolve", resolve)
    return locations


def bundle(storage):
    directory = storage["artifact"]/"receipt"
    directory.mkdir()
    payload = {"summary.json": canonical({"complete": True})+b"\n",
               "evidence/source.py": b"# metadata fixture\n"}
    hashes = {}
    for name, content in payload.items():
        target = directory/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[name] = sha256(target)
    lock = {"status": "fixture_complete", "payload_sha256": hashes,
            "payload_identity_sha256": grouped_design.digest(hashes)}
    path = directory/"lock.json"
    path.write_bytes(canonical(lock)+b"\n")
    return directory, {"path": "artifact:receipt/lock.json", "sha256": sha256(path)}, payload


def test_metadata_bundle_exact_lock_and_payload_identity(storage):
    _, ref, payload = bundle(storage)
    lock, observed = config.metadata_bundle(ref, "fixture_complete")
    assert observed == payload
    assert lock["payload_identity_sha256"] == grouped_design.digest(lock["payload_sha256"])


@pytest.mark.parametrize("change", ["changed_payload", "extra_array", "payload_symlink", "directory_symlink", "lock_symlink"])
def test_bundle_changes_fail_closed_before_unlisted_assets(storage, monkeypatch, change):
    directory, ref, _ = bundle(storage)
    if change == "changed_payload":
        (directory/"summary.json").write_text('{"changed":true}')
    elif change == "extra_array":
        (directory/"forbidden.npy").write_bytes(b"must not open")
    elif change == "payload_symlink":
        (directory/"summary.json").unlink()
        (directory/"summary.json").symlink_to(directory/"evidence/source.py")
    elif change == "directory_symlink":
        (directory/"evidence").rename(directory/"moved")
        (directory/"evidence").symlink_to(directory/"moved", target_is_directory=True)
    else:
        (directory/"lock.json").rename(directory/"real_lock.json")
        (directory/"lock.json").symlink_to(directory/"real_lock.json")
    opened = []
    original = config.sha256
    monkeypatch.setattr(config, "sha256", lambda path: (opened.append(path.name), original(path))[1])
    with pytest.raises(ValueError):
        config.metadata_bundle(ref, "fixture_complete")
    assert "forbidden.npy" not in opened


@pytest.mark.parametrize("reference", ["repo:result", "archive:result", "artifact:.", "artifact:../result"])
def test_unsafe_output_roots_rejected(storage, reference):
    with pytest.raises(ValueError):
        config.artifact_path(reference)


def test_output_symlink_escape_rejected_even_into_other_artifact_directory(storage):
    (storage["artifact"]/"real").mkdir()
    (storage["artifact"]/"alias").symlink_to(storage["artifact"]/"real", target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        config.artifact_path("artifact:alias/new_result")
    assert config.artifact_path("artifact:new/result") == storage["artifact"]/"new/result"
    assert not (storage["artifact"]/"new").exists()


def test_metadata_reference_rejects_array_before_hash(storage, monkeypatch):
    array = storage["artifact"]/"forbidden.npz"
    array.write_bytes(b"not a real array")
    monkeypatch.setattr(config, "sha256", lambda path: pytest.fail("Array was opened by plan"))
    with pytest.raises(ValueError, match="metadata only"):
        config.checked({"path": "artifact:forbidden.npz", "sha256": "0"*64})


def metadata_case():
    # The public loader separately fixes production counts. This small fixture
    # exercises joins/order/schedule checks without rebuilding the real freeze.
    sources = {p: {"file_id": p, "recording_session": None} for p in "abcd"}
    authority = deepcopy(sources)
    role = SimpleNamespace(partition_id="group_0", train_ids=("b", "a"), held_ids=("c", "d"),
                           held_groups=("held",), design_identity_sha256="fixture",
                           source_metadata=lambda p: deepcopy(authority[p]))
    roles = {"group_0": role}
    partitions = [{"id": "group_0", "train_ids": ["b", "a"], "held_ids": ["c", "d"], "held_groups": ["held"]}]
    planned = list(grouped_design.planned_schedule(partitions))
    counts = {"source_recordings": 4, "partitions": 1, "training_parents_per_partition": 2,
              "generated_waveforms": 3000, "eligible_cached_parents": 1,
              "unique_training_parents": 2, "new_parent_fits": 1}
    chain = {"candidate_ids": ["a"]}
    records = {"a": {"file_id": "a", "eligible": True, "failures": [], "reconstruction_reuse_allowed": False}}
    summary = {"status": "existing_fit_audit_complete", "validation": {"full_budget_parity": True},
               "eligible_candidate_count": 1, "failed_candidate_count": 0,
               "shared_audit": {"shared_failures": []}, "design_identity_sha256": "fixture"}
    return [{"expected_counts": counts}, roles, partitions, planned, sources, chain, records, summary]


def test_metadata_context_preserves_role_order_and_unknowns():
    values = metadata_case()
    assert config.validate_metadata(*values) == {"a", "b"}
    assert values[2][0]["train_ids"] == ["b", "a"]
    assert values[4]["a"]["recording_session"] is None


@pytest.mark.parametrize("change", ["training_order", "schedule_seed", "missing_schedule", "invented_session", "failed_cache", "waveform_reuse", "different_design", "shared_failure"])
def test_context_mismatch_rejected_without_execution(change):
    values = metadata_case()
    if change == "training_order": values[2][0]["train_ids"].reverse()
    elif change == "schedule_seed": values[3][0]["phase_seed"] = 0
    elif change == "missing_schedule": values[3].pop()
    elif change == "invented_session": values[4]["a"]["recording_session"] = "invented"
    elif change == "failed_cache": values[6]["a"]["failures"].append("failed")
    elif change == "waveform_reuse": values[6]["a"]["reconstruction_reuse_allowed"] = True
    elif change == "different_design": values[7]["design_identity_sha256"] = "other"
    else: values[7]["shared_audit"]["shared_failures"].append("failed")
    with pytest.raises(ValueError):
        config.validate_metadata(*values)


@pytest.mark.archive
def test_real_frozen_context_is_metadata_only(monkeypatch):
    import numpy as np
    import soundfile as sf
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("Plan opened an array"))
    monkeypatch.setattr(sf, "read", lambda *a, **kw: pytest.fail("Plan decoded audio"))
    monkeypatch.setattr(config.readiness, "evaluate", lambda *a, **kw: pytest.fail("Plan reran saved-fit audit"))
    context = config.load_context()
    assert context["plan"]["counts"] == config.COUNTS
    assert len(context["planned"]) == 27000 and len(context["readiness_records"]) == 380
    assert all(len(role.train_ids) == 380 for role in context["roles"].values())
    assert context["plan"]["boundaries"]["raw_audio_opened_by_plan"] is False

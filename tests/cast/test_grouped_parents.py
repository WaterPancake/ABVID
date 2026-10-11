"""Parent orchestration and IO guards with synthetic metadata and mocked fitting."""
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
import hashlib
import json
from types import MappingProxyType

import numpy as np
import pytest
import soundfile as sf

from abvid.cast import grouped_parents as parents
from abvid.cast.grouped_design import LINKS, known
from abvid.cast.grouped_parent import ParentCalibrationError
from abvid.cast.grouped_storage import read_bundle, write_bundle
from abvid.cast.roles import FoldRole
from abvid.paths import resolve
from abvid.provenance.hashing import canonical


def _context(tmp_path):
    source = {}
    for name, group in (("train_a", "group_a"), ("train_b", "group_b"), ("held", "group_h")):
        source[name] = {"file_id": name, "dataset_id": "IDMT", "release_id": "IDMT_V1",
                        "canonical_class": "car", "class_id": 0, "provenance_group_id": group,
                        "recording_session": "unknown", "site_id": group + "_site",
                        "source_path": f"dataset/IDMT_Traffic/audio/{name}.wav",
                        "provider_channel_pair": "CH34", "original_channel_ids": [3, 4]}
    def role(partition, train, held):
        return FoldRole(partition, "group", tuple(train), tuple(held),
                        tuple(source[i]["provenance_group_id"] for i in held),
                        tuple(source[i]["provenance_group_id"] for i in train),
                        MappingProxyType({i: canonical(r) for i, r in source.items()}),
                        MappingProxyType({key: frozenset(source[i][key] for i in held
                                                          if known(source[i].get(key))) for key in LINKS}),
                        MappingProxyType({"car": tuple(train), "truck": ()}), "synthetic_design")
    return {"config": {"max_workers": 1}, "config_reference": "repo:synthetic_config.json",
            "config_sha256": "a" * 64, "roles": {"group_0": role("group_0", ["train_a", "train_b"], ["held"]),
                                                   "group_1": role("group_1", ["train_b"], ["train_a"])},
            "numerical": {"synthetic_test_only": True}, "sources": source,
            "chain": {"candidate_ids": ["train_a"], "raw_source_root": "data:synthetic_raw"},
            "readiness_records": {"train_a": {"file_id": "train_a", "eligible": True,
                                                 "checked_hashes": {}, "failures": []}},
            "output": tmp_path / "output", "plan": {}, "planned": [], "partitions": []}


def _row(context, file_id):
    source = context["sources"][file_id]
    return {"file_id": file_id, "class": source["canonical_class"],
            "group": source["provenance_group_id"], "parent": source,
            "parameters": {"synthetic": True}}


def _payload(context, file_id):
    row = _row(context, file_id)
    observed = {**row, "descriptors": {"synthetic": [0.0]}}
    return {"fit.json": b'{"synthetic":true}\n', "v11.json": parents._json(row),
            "final_row.json": parents._json(row), "observation.json": parents._json(observed),
            "source.json": parents._json(context["sources"][file_id]), "origin.json": b"{}\n"}


@pytest.fixture
def mocked_execution(tmp_path, monkeypatch):
    context = _context(tmp_path)
    calls = {"audit": 0, "fit": [], "raw": []}
    def audit(chain):
        calls["audit"] += 1
        return list(context["readiness_records"].values()), {"shared_failures": [], "shared_checked_hashes": {}}
    monkeypatch.setattr(parents.reuse_chain, "audit_saved_parents", audit)
    monkeypatch.setattr(parents, "_configure_cpu", lambda: None)
    monkeypatch.setattr(parents, "_verified_raw_bytes", lambda c, i: calls["raw"].append(i))
    # Numeric-stage validation is independently tested by grouped_parent's full
    # production-budget parity suite; this fixture isolates orchestration.
    monkeypatch.setattr(parents, "_validate_complete", lambda *args: None)
    monkeypatch.setattr(parents, "_reuse_assets", lambda *args: ({}, {}))
    def reused(c, file_id, *args):
        write_bundle(parents._folder(c, file_id), _payload(c, file_id), parents._metadata(c, file_id, "complete"))
    monkeypatch.setattr(parents, "_save_reused", reused)
    monkeypatch.setattr(parents, "read_source", lambda *args: (np.zeros((8, 2)), 16000))
    monkeypatch.setattr(parents, "prepare_observation", lambda *args, **kwargs:
                        {"target": "synthetic", "original": "synthetic", "normalization": {}})
    def fit(target, source, numerical, **kwargs):
        calls["fit"].append(source["file_id"])
        payload = _payload(context, source["file_id"])
        return {"artifacts": payload, "final_row": json.loads(payload["final_row.json"]),
                "observation": json.loads(payload["observation.json"]), "stage_seconds": {"synthetic": 0.}}
    monkeypatch.setattr(parents, "calibrate_parent", fit)
    return context, calls


def test_prepare_resume_and_replay_never_refit_completed_parents(mocked_execution):
    context, calls = mocked_execution
    expected = {"complete_ids": ["train_a", "train_b"], "failed_ids": []}
    assert parents.ensure_parents(context) == expected
    assert calls["fit"] == ["train_b"] and calls["audit"] == 1
    assert parents.ensure_parents(context) == expected
    assert parents.ensure_parents(context, replay_only=True) == expected
    assert calls["fit"] == ["train_b"] and calls["audit"] == 3
    assert calls["raw"] == ["train_a", "train_b", "train_a", "train_b"]
    row, observed = parents.load_parent(context, "train_b", context["roles"]["group_1"])
    assert row["parent"]["recording_session"] == "unknown" and observed["file_id"] == "train_b"
    _, payload = read_bundle(parents._folder(context, "train_b"))
    runtime = json.loads(payload["worker_runtime.json"])
    assert runtime["elapsed_seconds"] >= 0 and runtime["peak_rss_bytes"] > 0
    assert runtime["device"] == "cpu" and runtime["peak_rss_scope"] == "worker_process_lifetime_high_water_mark"


def test_replay_missing_parent_reports_failure_without_fitting(mocked_execution):
    context, calls = mocked_execution
    parents._audit_candidates(context, False)
    parents._save_reused(context, "train_a", {}, {})
    assert parents.ensure_parents(context, replay_only=True) == {
        "complete_ids": ["train_a"], "failed_ids": ["train_b"]}
    assert calls["fit"] == []
    assert not parents._folder(context, "train_b").exists()


def test_terminal_failure_retains_partial_artifacts_and_is_never_replaced(mocked_execution, monkeypatch):
    context, calls = mocked_execution
    def failed(*args, **kwargs):
        calls["fit"].append("train_b")
        partial = {"artifacts": {"completed_fit.bin": b"preserve"}, "fit": {"history": [1, float("nan"), float("inf")]},
                   "stages": {"smooth8": {"synthetic": True}}, "flags": {"smooth8": ["retain"]},
                   "failed_stage": "v7"}
        raise ParentCalibrationError("v7", partial, FloatingPointError("injected synthetic failure"))
    monkeypatch.setattr(parents, "calibrate_parent", failed)
    expected = {"complete_ids": ["train_a"], "failed_ids": ["train_b"]}
    assert parents.ensure_parents(context) == expected
    assert parents.ensure_parents(context) == expected
    assert calls["fit"] == ["train_b"]
    metadata, payload = read_bundle(parents._folder(context, "train_b"))
    failure = json.loads(payload["failure_record.json"])
    assert metadata["status"] == "technical_failure"
    assert payload["completed_fit.bin"] == b"preserve"
    assert failure["fit"]["history"] == [1, {"nonfinite_float": "nan"}, {"nonfinite_float": "inf"}]
    assert failure["replacement_allowed"] is False
    with pytest.raises(ValueError, match="terminal"):
        parents.load_parent(context, "train_b", context["roles"]["group_0"])


def test_current_fold_role_rejects_held_cache_before_bundle_read(mocked_execution, monkeypatch):
    context, _ = mocked_execution
    monkeypatch.setattr(parents, "read_bundle", lambda *args: pytest.fail("Held cache read"))
    with pytest.raises(ValueError, match="Held ancestor|training role"):
        parents.load_parent(context, "train_a", context["roles"]["group_1"])


def test_changed_context_and_changed_audit_fail_before_new_fitting(mocked_execution, monkeypatch):
    context, calls = mocked_execution
    parents.ensure_parents(context)
    changed = {**context, "config_sha256": "b" * 64}
    with pytest.raises(ValueError, match="immutable|changed execution"):
        parents.ensure_parents(changed)
    monkeypatch.setattr(parents.reuse_chain, "audit_saved_parents", lambda *args:
                        ([{**context["readiness_records"]["train_a"], "eligible": False}],
                         {"shared_failures": [], "shared_checked_hashes": {}}))
    with pytest.raises(ValueError, match="frozen readiness"):
        parents.ensure_parents(context)
    assert calls["fit"] == ["train_b"]


def test_four_worker_spawn_schedule_uses_context_initializer(mocked_execution, monkeypatch):
    context, calls = mocked_execution
    context["config"]["max_workers"] = 4
    captured = {}
    class Pool:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def shutdown(self, **kwargs):
            captured["shutdown"] = kwargs
        def submit(self, function, file_id):
            assert function is parents._worker_fit
            future = Future()
            future.set_result(parents._fit_parent(context, file_id))
            return future
    monkeypatch.setattr(parents, "ProcessPoolExecutor", Pool)
    assert parents.ensure_parents(context)["complete_ids"] == ["train_a", "train_b"]
    assert captured["max_workers"] == 4
    assert captured["mp_context"].get_start_method() == "spawn"
    assert captured["initializer"] is parents._worker_initialize
    assert captured["initargs"] == (context["config_reference"], context["config_sha256"])
    assert captured["shutdown"] == {"wait": True, "cancel_futures": True}
    assert calls["fit"] == ["train_b"]


def test_pool_failure_preserves_unstarted_parents_for_unchanged_resume(mocked_execution, monkeypatch):
    context, calls = mocked_execution
    context["config"]["max_workers"] = 4
    closed = []
    class BrokenPool:
        def __init__(self, **kwargs):
            pass
        def submit(self, *args):
            future = Future()
            future.set_exception(BrokenProcessPool("injected infrastructure failure"))
            return future
        def shutdown(self, **kwargs):
            closed.append(kwargs)
    monkeypatch.setattr(parents, "ProcessPoolExecutor", BrokenPool)
    with pytest.raises(BrokenProcessPool):
        parents.ensure_parents(context)
    assert not parents._folder(context, "train_b").exists()
    assert calls["fit"] == []
    assert closed == [{"wait": True, "cancel_futures": True}]


def test_post_calibration_failure_retains_completed_result(mocked_execution, monkeypatch):
    context, _ = mocked_execution
    fit = parents.calibrate_parent
    def completed(*args, **kwargs):
        result = fit(*args, **kwargs)
        result.update(fit={"history": [1, 2]}, stages={"v11": result["final_row"]})
        return result
    monkeypatch.setattr(parents, "calibrate_parent", completed)
    monkeypatch.setattr(parents, "_validate_complete", lambda *args:
                        (_ for _ in ()).throw(ValueError("injected post-calibration failure")))
    assert parents._fit_parent(context, "train_b") == "technical_failure"
    _, payload = read_bundle(parents._folder(context, "train_b"))
    failure = json.loads(payload["failure_record.json"])
    assert failure["fit"]["history"] == [1, 2]
    assert failure["completed_stages"] == ["v11"]
    assert "v11.json" in payload


def test_reuse_reads_only_anchored_fit_records_and_retains_complete_history(tmp_path, monkeypatch):
    context = _context(tmp_path)
    source = context["sources"]["train_a"]
    fit = {"parameters": {"synthetic": True}, "history": [0, 1, 2]}
    flags = {"scientific_flags": ["retain_ambiguity"]}
    context["chain"]["stages"] = {"smooth8": "archive:synthetic_fit"}
    root = "archive:synthetic_fit/fits/train_a/"
    values = {root + "fit.json": parents._json(fit), root + "failure.json": parents._json(flags)}
    context["readiness_records"]["train_a"]["checked_hashes"] = {
        path: hashlib.sha256(value).hexdigest() for path, value in values.items()}
    observed = {**_row(context, "train_a"), "descriptors": {"synthetic": [0.]}}
    banks = {stage: {"train_a": _row(context, "train_a")} for stage in parents.reuse_chain.STAGES[1:]}
    reads = []
    def pinned(reference, expected):
        reads.append(reference)
        assert hashlib.sha256(values[reference]).hexdigest() == expected
        return values[reference]
    monkeypatch.setattr(parents, "_pinned_bytes", pinned)
    monkeypatch.setattr(parents, "_validate_complete", lambda *args: None)
    parents._save_reused(context, "train_a", banks, {"train_a": observed})
    _, payload = read_bundle(parents._folder(context, "train_a"))
    assert reads == [root + "fit.json", root + "failure.json"]
    assert json.loads(payload["fit.json"])["history"] == [0, 1, 2]
    assert not any(name.endswith(".wav") for name in payload)
    assert json.loads(payload["source.json"]) == source
    assert json.loads(payload["origin.json"])["saved_calibration_reconstruction_waveforms_reused"] is False


def test_complete_validator_rejects_cross_parent_observation_before_numeric_access(tmp_path):
    context = _context(tmp_path)
    payload = _payload(context, "train_a")
    wrong = json.loads(payload["observation.json"])
    wrong["parent"] = context["sources"]["train_b"]
    payload["observation.json"] = parents._json(wrong)
    with pytest.raises(ValueError, match="source ancestry"):
        parents._validate_complete(context, "train_a", payload, context["roles"]["group_0"])


def _raw_context(tmp_path, monkeypatch):
    context = _context(tmp_path)
    path = tmp_path / "synthetic_raw" / "train_a.wav"
    path.parent.mkdir()
    raw = np.random.default_rng(42).normal(scale=.1, size=(80, 2))
    sf.write(path, raw, 16000, subtype="FLOAT")
    data = path.read_bytes()
    context["sources"]["train_a"].update(
        source_file_bytes=len(data), source_file_sha256=hashlib.sha256(data).hexdigest(),
        native_sample_rate_hz=16000, native_channels=2, decoded_frames=80)
    monkeypatch.setattr(parents, "resolve", lambda reference: path)
    monkeypatch.setattr(parents, "roots", lambda: {"data": tmp_path})
    return context, path


def test_source_read_is_float64_two_columns_after_exact_hash_and_header_checks(tmp_path, monkeypatch):
    context, path = _raw_context(tmp_path, monkeypatch)
    raw, rate = parents.read_source(context, "train_a")
    expected, _ = sf.read(path, dtype="float64", always_2d=True)
    assert raw.dtype == np.float64 and raw.shape == (80, 2) and rate == 16000
    assert np.array_equal(raw, expected)


def test_raw_symlink_rejected_before_reading_linked_bytes(tmp_path, monkeypatch):
    context, path = _raw_context(tmp_path, monkeypatch)
    outside = tmp_path / "outside.wav"
    path.rename(outside)
    path.symlink_to(outside)
    monkeypatch.setattr(parents, "resolve", lambda *args: pytest.fail("Symlink reached resolve"))
    with pytest.raises(ValueError, match="Symlink"):
        parents.read_source(context, "train_a")


@pytest.mark.parametrize("change", ["hash", "header", "dataset", "class", "channel_pair"])
def test_raw_source_contract_rejected_before_decode(tmp_path, monkeypatch, change):
    context, _ = _raw_context(tmp_path, monkeypatch)
    source = context["sources"]["train_a"]
    if change == "hash":
        source["source_file_sha256"] = "0" * 64
    elif change == "header":
        source["decoded_frames"] += 1
    elif change == "dataset":
        source["dataset_id"] = "MELAUDIS"
    elif change == "class":
        source["canonical_class"] = "motorcycle"
    else:
        source["provider_channel_pair"] = "CH12"
    monkeypatch.setattr(parents.sf, "read", lambda *a, **k: pytest.fail("Invalid source reached decode"))
    with pytest.raises(ValueError):
        parents.read_source(context, "train_a")


@pytest.mark.archive
def test_admitted_archived_parent_per_class_round_trips_without_audio_or_fitting(tmp_path, monkeypatch):
    from abvid.cast.roles import load_roles
    execution = json.loads(resolve(
        "repo:experiments/cast_coverage/grouped_coverage/execution/config.json").read_text())
    reference = execution["readiness_lock"]
    try:
        lock_path = resolve(reference["path"])
    except FileNotFoundError:
        pytest.skip("Pinned local readiness audit is unavailable")
    assert hashlib.sha256(lock_path.read_bytes()).hexdigest() == reference["sha256"]
    lock = json.loads(lock_path.read_text())
    def saved(name):
        value = (lock_path.parent / name).read_bytes()
        assert hashlib.sha256(value).hexdigest() == lock["payload_sha256"][name]
        return value
    summary = json.loads(saved("summary.json"))
    records = {row["file_id"]: row for row in
               (json.loads(line) for line in saved("parents.jsonl").splitlines())}
    chain = json.loads(saved("chain_contract.json"))
    numerical_ref = chain["numerical_contract"]
    numerical_bytes = resolve(numerical_ref["path"]).read_bytes()
    assert hashlib.sha256(numerical_bytes).hexdigest() == numerical_ref["sha256"]
    roles = load_roles()
    first_role = next(iter(roles.values()))
    sources = {file_id: first_role.source_metadata(file_id) for file_id in records}
    context = {"config": {"max_workers": 1}, "config_reference": "synthetic_compatibility_test_only",
               "config_sha256": "d" * 64, "roles": roles, "numerical": json.loads(numerical_bytes),
               "sources": sources, "chain": chain, "readiness_records": records,
               "output": tmp_path / "archived_json_compatibility_only"}
    def forbidden(*args, **kwargs):
        pytest.fail("Metadata compatibility test reached audio, fitting or a fresh raw audit")
    monkeypatch.setattr(parents, "read_source", forbidden)
    monkeypatch.setattr(parents, "calibrate_parent", forbidden)
    monkeypatch.setattr(parents.reuse_chain, "audit_saved_parents", forbidden)
    monkeypatch.setattr(parents.sf, "read", forbidden)
    banks, observed = parents._reuse_assets(context, summary["shared_audit"])
    for label in ("car", "truck"):
        file_id = min(i for i, source in sources.items() if source["canonical_class"] == label)
        parents._save_reused(context, file_id, banks, observed)
        row, observation = parents.load_parent(context, file_id, parents._witness(context, file_id))
        assert row == banks["v11"][file_id] and observation == observed[file_id]
        _, payload = read_bundle(parents._folder(context, file_id))
        assert not any(name.endswith((".wav", ".npz")) for name in payload)
        assert len(json.loads(payload["fit.json"])["history"]) == 301

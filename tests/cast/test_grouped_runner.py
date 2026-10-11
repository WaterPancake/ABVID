"""Synthetic orchestration checks; no research audio, calibration or fitting.

Population/generation numerics are isolated behind deterministic fixtures. The
full 3,000-entry schedule is retained. Large sample bundles use a checked memory
store; population, generation, held and result bundles use the real immutable
storage implementation in pytest's temporary directory.
"""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import sys
from types import MappingProxyType, ModuleType, SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from abvid.cast import grouped_design, grouped_runner as runner, grouped_storage as storage
from abvid.cast.config import digest
from abvid.cast.roles import FoldRole
from abvid.provenance.hashing import canonical


FAMILIES = ("log_spectrum", "bands", "envelope", "modulation")


def descriptors(value):
    return {family: [float(value)] for family in FAMILIES}


@pytest.fixture
def context(tmp_path):
    sources, train, classes = {}, [], {label: [] for label in ("car", "truck")}
    for label in classes:
        for index in range(190):
            parent = f"synthetic_train_{label}_{index:03d}"
            train.append(parent)
            classes[label].append(parent)
            sources[parent] = {"file_id": parent, "dataset_id": "IDMT", "canonical_class": label,
                               "provenance_group_id": f"synthetic_train_group_{index % 5}",
                               "recording_session": None}
    partitions, held_by_partition = [], {}
    for analysis, count in (("group", 6), ("site", 3)):
        for fold in range(count):
            partition = f"{analysis}_{fold}"
            held = [f"synthetic_{partition}_{label}_{index}" for label in classes for index in range(2)]
            held_by_partition[partition] = held
            for parent in held:
                label = parent.split("_")[-2]
                sources[parent] = {"file_id": parent, "dataset_id": "IDMT", "canonical_class": label,
                                   "provenance_group_id": f"synthetic_held_{partition}", "recording_session": None}
            partitions.append({"id": partition, "analysis": analysis, "fold": fold,
                               "held_counts": {"car": 2, "truck": 2}})
    immutable = MappingProxyType({parent: canonical(value) for parent, value in sources.items()})
    roles = {}
    for part in partitions:
        held = held_by_partition[part["id"]]
        links = {key: frozenset(sources[parent][key] for parent in held if grouped_design.known(sources[parent].get(key)))
                 for key in grouped_design.LINKS}
        roles[part["id"]] = FoldRole(
            part["id"], part["analysis"], tuple(train), tuple(held),
            (f"synthetic_held_{part['id']}",), tuple(f"synthetic_train_group_{i}" for i in range(5)),
            immutable, MappingProxyType(links), MappingProxyType({k: tuple(v) for k, v in classes.items()}),
            "synthetic_design_identity")
    return {"output": tmp_path / "execution", "sources": sources, "roles": roles,
            "partitions": partitions,
            "planned": list(grouped_design.planned_schedule(partitions)),
            "config_sha256": "synthetic_config_identity", "config": {"implementation_references": []},
            "plan": {"scope": "synthetic orchestration fixture only"},
            "numerical": {"renderer_configs": {"smooth8_v1": {}}, "metric_config": {
                "class_order": ["car", "truck"], "arms": ["joint", "prototype", "marginals"],
                "primary_families": list(FAMILIES), "seeds": list(grouped_design.SEEDS),
                "samples_per_class_per_seed_per_arm": 50, "coverage_quantiles": [.05, .95]}}}


class MemorySample:
    def __init__(self, path, entries):
        self.path, self.entries = path, entries

    def exists(self):
        return str(self.path) in self.entries

    def __str__(self):
        return str(self.path)


@pytest.fixture
def harness(context, monkeypatch):
    events, entries, populations = [], {}, []
    originals = SimpleNamespace(read=runner.read_bundle, verify=runner.verify_bundle,
                                write=runner.write_bundle, sample_path=runner._sample_path,
                                held_observation=runner._held_observation)

    def read(path):
        if not isinstance(path, MemorySample):
            return originals.read(path)
        entry = entries[str(path)]
        found = storage._manifest(entry["metadata"], {name: runner.sha(value) for name, value in entry["payload"].items()})
        if found != entry["manifest"]:
            raise ValueError("Synthetic sample store detected changed bytes")
        return deepcopy(entry["metadata"]), deepcopy(entry["payload"])

    def verify(path):
        if not isinstance(path, MemorySample):
            return originals.verify(path)
        read(path)
        return deepcopy(entries[str(path)]["manifest"])

    def write(path, payload, metadata):
        if not isinstance(path, MemorySample):
            return originals.write(path, payload, metadata)
        manifest = storage._manifest(metadata, {name: runner.sha(value) for name, value in payload.items()})
        if str(path) in entries:
            if verify(path) != manifest:
                raise ValueError("Existing synthetic sample cannot be replaced")
        else:
            entries[str(path)] = {"metadata": deepcopy(metadata), "payload": deepcopy(payload), "manifest": manifest}
            events.append(("sample_write", metadata["planned_id"]))
        return deepcopy(manifest)

    monkeypatch.setattr(runner, "read_bundle", read)
    monkeypatch.setattr(runner, "verify_bundle", verify)
    monkeypatch.setattr(runner, "write_bundle", write)
    monkeypatch.setattr(runner, "_sample_path", lambda folder, planned: MemorySample(originals.sample_path(folder, planned), entries))

    class Population:
        def __init__(self, bank, observed, role, numerical):
            assert [row["file_id"] for row in bank] == list(role.train_ids)
            assert [row["file_id"] for row in observed] == list(role.train_ids)
            self.role = role
            self.statistics = {"partition": role.partition_id, "parents": list(role.train_ids),
                               "population_statistics_reused": False,
                               "bank_sha256": digest(sorted(bank, key=lambda row: row["file_id"])),
                               "observations_sha256": digest(sorted(observed, key=lambda row: row["file_id"]))}
            self.scales = {label: {family: [1.] for family in FAMILIES} for label in ("car", "truck")}
            self.fingerprint = digest(self.statistics)
            self.draws = 0
            populations.append(self)

    def sample_payload(population, planned):
        folder = context["output"] / "folds" / population.role.partition_id / "population"
        meta, payload = storage.read_bundle(folder)
        assert meta["fingerprint"] == population.fingerprint
        assert json.loads(payload["statistics.json"]) == population.statistics
        population.draws += 1
        events.append(("draw", planned["id"]))
        observed = descriptors(planned["index"] / 49)
        record = {"planned_sample_id": planned["id"], "population_fingerprint": population.fingerprint,
                  "fixture": "synthetic sample bytes; no waveform rendering"}
        payload = {"planned.json": runner.json_bytes(planned), "record.json": runner.json_bytes(record),
                   "descriptors.json": runner.json_bytes(observed), "waveform.wav": planned["id"].encode()}
        return payload, {"planned_id": planned["id"], "population_fingerprint": population.fingerprint,
                         "waveform_float32_sha256": runner.sha(payload["waveform.wav"])}

    def held_observation(ctx, role, parent):
        assert parent in role.held_ids and parent not in role.train_ids
        receipt = json.loads(storage.read_bundle(ctx["output"] / "folds" / role.partition_id / "generation")[1]["receipt.json"])
        assert receipt["count"] == 3000 and len(receipt["sample_identities"]) == 3000
        assert len([name for name in entries if f"/folds/{role.partition_id}/" in name]) == 3000
        events.append(("held", parent))
        return {"file_id": parent, "source_metadata": role.source_metadata(parent),
                "normalization": {"fixture": True}, "descriptors": descriptors(.3 if parent.endswith("_0") else .6)}

    helpers = ModuleType("abvid.cast.grouped_parents")
    def load_parent(ctx, parent, role):
        metadata = role.source_metadata(parent)
        role.validate_parent(metadata)
        events.append(("load_parent", parent))
        common = {"file_id": parent, "parent": metadata, "class": metadata["canonical_class"],
                  "group": metadata["provenance_group_id"]}
        return {**common, "parameters": {}}, {**common, "descriptors": descriptors(.5)}
    def ensure_parents(ctx, *, replay_only=False):
        events.append(("ensure_parents", replay_only))
        return {"scope": "synthetic mocked parents", "complete": True}
    helpers.load_parent, helpers.ensure_parents = load_parent, ensure_parents
    helpers.read_source = lambda *args, **kwargs: pytest.fail("Unexpected real source reader invocation")
    monkeypatch.setitem(sys.modules, "abvid.cast.grouped_parents", helpers)
    monkeypatch.setattr(runner, "build_population", Population)
    monkeypatch.setattr(runner, "sample_payload", sample_payload)
    monkeypatch.setattr(runner, "_held_observation", held_observation)
    monkeypatch.setattr(runner, "deterministic_cpu", lambda: None)
    monkeypatch.setattr(runner, "_runtime", lambda: {"scope": "synthetic test runtime"})
    monkeypatch.setattr(runner, "_git", lambda: {"commit": "synthetic_fixture"})
    return SimpleNamespace(context=context, entries=entries, events=events, populations=populations,
                           helpers=helpers, originals=originals, sample_payload=sample_payload,
                           held_observation=held_observation)


@pytest.fixture
def completed_fold(harness):
    role = harness.context["roles"]["group_0"]
    records = runner.run_fold(harness.context, role)
    return harness, role, records


def test_full_fold_freezes_bank_then_generates_replays_and_only_then_reads_held(completed_fold):
    h, role, records = completed_fold
    assert len(records) == 60
    assert {(row["class"], row["seed"], row["arm"]) for row in records} == {
        (label, seed, arm["id"]) for label in ("car", "truck") for seed in grouped_design.SEEDS for arm in grouped_design.ARMS}
    assert all(row["score"]["generated_count"] == 50 and row["score"]["held_count"] == 2 for row in records)
    assert all(all(row["gates"].values()) and not row["failures"] for row in records)
    assert h.populations[0].draws == 6000
    first_draw = next(i for i, event in enumerate(h.events) if event[0] == "draw")
    first_held = next(i for i, event in enumerate(h.events) if event[0] == "held")
    assert all(event[0] == "load_parent" for event in h.events[:first_draw])
    assert len(h.events[:first_draw]) == 380
    assert sum(event[0] == "sample_write" for event in h.events[:first_held]) == 3000
    assert sum(event[0] == "draw" for event in h.events[:first_held]) == 6000
    assert [event[1] for event in h.events if event[0] == "held"] == [parent for parent in role.held_ids for _ in range(2)]


def test_unchanged_fold_replay_recomputes_without_fitting_filling_or_rewriting(completed_fold):
    h, role, first = completed_fold
    before = {str(path): (path.read_bytes(), path.stat().st_mtime_ns)
              for path in h.context["output"].rglob("*") if path.is_file()}
    sample_before = deepcopy(h.entries)
    h.events.clear()
    second = runner.run_fold(h.context, role, replay_only=True)
    assert second == first and h.entries == sample_before
    assert h.populations[-1].draws == 3000
    assert not any(event[0] in {"sample_write", "ensure_parents"} for event in h.events)
    assert [event[1] for event in h.events if event[0] == "held"] == list(role.held_ids)
    assert before == {str(path): (path.read_bytes(), path.stat().st_mtime_ns)
                      for path in h.context["output"].rglob("*") if path.is_file()}


@pytest.mark.parametrize("defect", ["missing", "duplicate", "altered_seed"])
def test_generation_rejects_any_deviation_from_full_frozen_schedule_before_draw(harness, defect):
    ctx, role = harness.context, harness.context["roles"]["group_0"]
    pop = runner.prepare_population(ctx, role)
    item = next(row for row in ctx["planned"] if row["partition"] == role.partition_id)
    if defect == "missing":
        ctx["planned"].remove(item)
    elif defect == "duplicate":
        ctx["planned"].append(deepcopy(item))
    else:
        item["phase_seed"] += 1
    with pytest.raises(ValueError, match="exact frozen"):
        runner.generation_gate(ctx, role, pop)
    assert pop.draws == 0 and not harness.entries
    assert not any(event[0] == "held" for event in harness.events)


def test_missing_population_blocks_generation_before_draw(harness):
    role = harness.context["roles"]["group_0"]
    pop = SimpleNamespace(fingerprint="not_frozen")
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.generation_gate(harness.context, role, pop)
    assert not harness.entries


@pytest.mark.parametrize("defect", ["missing_sample", "changed_sample", "changed_draw"])
def test_replay_never_repairs_or_accepts_changed_samples(completed_fold, monkeypatch, defect):
    h, role, _ = completed_fold
    key = next(iter(h.entries))
    if defect == "missing_sample":
        del h.entries[key]
    elif defect == "changed_sample":
        h.entries[key]["payload"]["waveform.wav"] = b"changed bytes"
    else:
        def different(population, planned):
            payload, meta = h.sample_payload(population, planned)
            payload["descriptors.json"] = runner.json_bytes(descriptors(999))
            return payload, meta
        monkeypatch.setattr(runner, "sample_payload", different)
    before = deepcopy(h.entries)
    h.events.clear()
    with pytest.raises(runner.StageError) as exc:
        runner.run_fold(h.context, role, replay_only=True)
    assert exc.value.stage == "generation"
    assert h.entries == before
    assert not any(event[0] in {"sample_write", "held"} for event in h.events)


def rewrite_bundle(folder, name, content):
    """Simulate an attacker rewriting both data and its local hash manifest."""
    metadata, payload = storage.read_bundle(folder)
    payload[name] = content
    (folder / name).write_bytes(content)
    manifest = storage._manifest(metadata, {key: runner.sha(value) for key, value in payload.items()})
    (folder / storage.MANIFEST).write_bytes(storage._canonical(manifest) + b"\n")


@pytest.mark.parametrize("name", ["statistics.json", "scales.json", "bank.json", "observations.json"])
def test_generation_checks_frozen_population_content_before_drawing(harness, monkeypatch, name):
    ctx, role = harness.context, harness.context["roles"]["group_0"]
    population = runner.prepare_population(ctx, role)
    folder = ctx["output"] / "folds" / role.partition_id / "population"
    value = json.loads(storage.read_bundle(folder)[1][name])
    if isinstance(value, list):
        value[0]["undeclared_mutation"] = True
    else:
        value["undeclared_mutation"] = True
    rewrite_bundle(folder, name, runner.json_bytes(value))
    monkeypatch.setattr(runner, "sample_payload", lambda *a: pytest.fail("Changed frozen population reached a sample draw"))
    with pytest.raises(ValueError):
        runner.generation_gate(ctx, role, population)
    assert population.draws == 0 and not harness.entries


@pytest.mark.parametrize("defect", ["receipt", "population", "sample_self_consistent"])
def test_held_access_rejects_tamper_in_previously_frozen_generation_chain(completed_fold, defect):
    h, role, _ = completed_fold
    root = h.context["output"] / "folds" / role.partition_id
    receipt = json.loads(storage.read_bundle(root / "generation")[1]["receipt.json"])
    if defect == "receipt":
        receipt["count"] -= 1
    elif defect == "population":
        rewrite_bundle(root / "population", "statistics.json", b'{"changed":true}\n')
    else:
        key = next(iter(h.entries))
        entry = h.entries[key]
        entry["payload"]["waveform.wav"] = b"changed waveform with new local hashes"
        entry["manifest"] = storage._manifest(entry["metadata"], {k: runner.sha(v) for k, v in entry["payload"].items()})
    h.events.clear()
    with pytest.raises(ValueError):
        runner.held_descriptors(h.context, role, receipt, replay_only=True)
    assert not any(event[0] == "held" for event in h.events)


def test_held_replay_rejects_changed_observation_without_overwriting(completed_fold, monkeypatch):
    h, role, _ = completed_fold
    root = h.context["output"] / "folds" / role.partition_id
    receipt = json.loads(storage.read_bundle(root / "generation")[1]["receipt.json"])
    original = storage.read_bundle(root / "held" / role.held_ids[0])
    def changed(*args):
        row = h.held_observation(*args)
        row["descriptors"]["bands"][0] += 1
        return row
    monkeypatch.setattr(runner, "_held_observation", changed)
    with pytest.raises(ValueError, match="Replay differs"):
        runner.held_descriptors(h.context, role, receipt, replay_only=True)
    assert storage.read_bundle(root / "held" / role.held_ids[0]) == original


@pytest.mark.parametrize("defect", ["training_id", "other_fold_held_id", "changed_metadata"])
def test_held_role_guard_rejects_before_source_decode(harness, defect):
    ctx = harness.context
    role = ctx["roles"]["group_0"]
    parent = role.held_ids[0]
    if defect == "training_id":
        parent = role.train_ids[0]
    elif defect == "other_fold_held_id":
        parent = ctx["roles"]["group_1"].held_ids[0]
    else:
        ctx["sources"][parent]["recording_session"] = "invented_session"
    # The helper's source reader is a sentinel that fails on any access.
    with pytest.raises(ValueError, match="held role|metadata differs"):
        harness.originals.held_observation(ctx, role, parent)


def test_sample_payload_preserves_float32_waveform_and_all_frozen_ids():
    planned = next(grouped_design.planned_schedule([{"id": "group_0"}]))
    wave = np.linspace(-1, 1, 32000, dtype=np.float32)
    record = {"planned_sample_id": planned["id"], "parent_ids": ["synthetic_parent"]}
    population = SimpleNamespace(fingerprint="fixture", draw=lambda row: (record, wave, descriptors(.5)))
    payload, meta = runner.sample_payload(population, planned)
    restored, sr = sf.read(BytesIO(payload["waveform.wav"]), dtype="float32")
    assert sr == 16000 and np.array_equal(restored, wave)
    assert meta["waveform_float32_sha256"] == runner.sha(wave.tobytes())
    assert json.loads(payload["planned.json"]) == planned
    assert json.loads(payload["record.json"]) == record


@pytest.mark.parametrize("wave", [np.zeros(20, dtype=np.float32), np.zeros(32000, dtype=np.float64),
                                  np.full(32000, np.nan, dtype=np.float32)])
def test_sample_payload_rejects_invalid_waveform(wave):
    pop = SimpleNamespace(fingerprint="fixture", draw=lambda row: ({}, wave, descriptors(.5)))
    with pytest.raises(ValueError, match="finite float32 two-second"):
        runner.sample_payload(pop, {"id": "synthetic"})


def fabricated_scores(role):
    return [{"partition": role.partition_id, "class": label, "seed": seed, "arm": arm["id"],
             "score": {"W1": 1., "coverage": .9, "generated_count": 50, "held_count": 2,
                       "families": {family: {"W1": 1., "coverage": .9, "spread_ratio": 1.} for family in FAMILIES}},
             "gates": dict.fromkeys(("input", "completeness", "numerical", "ancestry", "replay"), True), "failures": []}
            for label in ("car", "truck") for seed in grouped_design.SEEDS for arm in grouped_design.ARMS]


def test_run_preserves_failed_fold_finishes_other_folds_and_replays_without_retry(harness, monkeypatch):
    calls = []
    def run_fold(ctx, role, *, replay_only=False):
        calls.append((role.partition_id, replay_only))
        if role.partition_id == "group_2":
            raise runner.StageError("population", ValueError("synthetic parent failure"))
        return fabricated_scores(role)
    monkeypatch.setattr(runner, "run_fold", run_fold)
    first = runner.run(harness.context)
    assert len(calls) == 9 and len(first["fold_failures"]) == 1
    assert first["fold_failures"][0]["partition"] == "group_2"
    assert first["summary"]["status"] == "technical_incompletion"
    assert first["summary"]["analyses"]["group"]["uncertainty"]["draws"] == 0
    assert first["summary"]["analyses"]["site"]["complete"]
    calls.clear()
    second = runner.run(harness.context, replay_only=True)
    assert second == first
    assert len(calls) == 8 and all(replay for _, replay in calls)
    assert not any(part == "group_2" for part, _ in calls)
    assert len(list((harness.context["output"] / "attempts").iterdir())) == 2


def test_replay_error_is_raised_and_cannot_create_a_new_terminal_failure(harness, monkeypatch):
    monkeypatch.setattr(runner, "run_fold", lambda ctx, role, **kw: fabricated_scores(role))
    runner.run(harness.context)
    before = storage.read_bundle(harness.context["output"] / "results")
    def failed(*args, **kwargs):
        raise runner.StageError("generation", ValueError("synthetic changed waveform"))
    monkeypatch.setattr(runner, "run_fold", failed)
    with pytest.raises(runner.StageError, match="changed waveform"):
        runner.run(harness.context, replay_only=True)
    assert not list((harness.context["output"] / "folds").glob("*/failure"))
    assert storage.read_bundle(harness.context["output"] / "results") == before


def test_results_anchor_scientific_bundle_identity_after_self_consistent_local_rewrite(harness, monkeypatch):
    monkeypatch.setattr(runner, "run_fold", lambda ctx, role, **kw: fabricated_scores(role))
    first = runner.run(harness.context)
    root = harness.context["output"]
    assert first["artifact_inventory"]["context"] == storage.verify_bundle(root / "context")["identity_sha256"]
    original_result = storage.read_bundle(root / "results")
    rewrite_bundle(root / "context", "environment.json", b'{"runtime_provenance":"altered"}\n')
    # The local context manifest is valid, but its separately recorded identity
    # cannot be revised by a replay of the original completed run.
    assert storage.verify_bundle(root / "context")["identity_sha256"] != first["artifact_inventory"]["context"]
    with pytest.raises(ValueError, match="Replay differs"):
        runner.run(harness.context, replay_only=True)
    assert storage.read_bundle(root / "results") == original_result


@pytest.mark.parametrize("changed", ["implementation_references", "test_references"])
def test_execution_snapshots_bind_source_and_test_bytes_before_parent_work(harness, monkeypatch, tmp_path, changed):
    import abvid.paths
    references = {}
    for field in ("implementation_references", "test_references"):
        path = tmp_path / (field + ".py")
        path.write_text("# synthetic frozen source fixture\n")
        ref = "repo:" + path.name
        references[ref] = path
        harness.context["config"][field] = [{"path": ref, "sha256": runner.sha(path.read_bytes())}]
    monkeypatch.setattr(abvid.paths, "resolve", lambda ref, **kwargs: references[ref])
    monkeypatch.setattr(runner, "run_fold", lambda ctx, role, **kw: fabricated_scores(role))
    runner.run(harness.context)
    payload = storage.read_bundle(harness.context["output"] / "context")[1]
    assert {key for key in payload if key.startswith("implementation/")} == {
        "implementation/implementation_references.py", "implementation/test_references.py"}
    references[harness.context["config"][changed][0]["path"]].write_text("# changed after execution\n")
    harness.events.clear()
    with pytest.raises(ValueError, match="immutable execution snapshot"):
        runner.run(harness.context, replay_only=True)
    assert not any(event[0] == "ensure_parents" for event in harness.events)


def test_interruption_records_attempt_and_exact_restart_does_not_mark_terminal_failure(harness, monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("synthetic interruption")
    monkeypatch.setattr(runner, "run_fold", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner.run(harness.context)
    assert not list((harness.context["output"] / "folds").glob("*/failure"))
    assert not (harness.context["output"] / "results").exists()
    attempts = list((harness.context["output"] / "attempts").iterdir())
    assert len(attempts) == 1
    status = json.loads(storage.read_bundle(attempts[0] / "finish")[1]["runtime.json"])["status"]
    assert status == "interrupted_or_failed"
    monkeypatch.setattr(runner, "run_fold", lambda ctx, role, **kw: fabricated_scores(role))
    result = runner.run(harness.context)
    assert result["summary"]["status"] == "complete"
    assert len(list((harness.context["output"] / "attempts").iterdir())) == 2


def test_replay_of_unstarted_run_does_not_create_output(context):
    with pytest.raises(FileNotFoundError):
        runner.run(context, replay_only=True)
    assert not context["output"].exists()


def test_default_cli_is_metadata_only_even_for_run_and_verify(context, monkeypatch, capsys):
    config = ModuleType("abvid.cast.grouped_execution_config")
    config.CONFIG = "repo:synthetic_configuration.json"
    config.load_context = lambda reference: context
    monkeypatch.setitem(sys.modules, "abvid.cast.grouped_execution_config", config)
    monkeypatch.setattr(runner, "run", lambda *a, **k: pytest.fail("Metadata command reached scientific execution"))
    for command in ("plan", "run", "verify"):
        assert runner.main([command]) == 0
        assert json.loads(capsys.readouterr().out)["executed"] is False
    assert not context["output"].exists()

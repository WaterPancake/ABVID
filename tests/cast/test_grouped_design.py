"""CAST-4A metadata contracts; these tests never open audio or scientific arrays."""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from abvid.cast import grouped_design as design
from abvid.data.splits import select
from abvid.paths import resolve


@pytest.fixture(scope="module")
def source():
    rows = []
    for group_number in range(6):
        group = f"group_{group_number}"
        for class_id, label in enumerate(("car", "truck")):
            for number in range(100):
                file_id = f"{group}_{label}_{number:03d}"
                rows.append({
                    "file_id": file_id, "dataset_id": "IDMT", "release_id": "IDMT_V1",
                    "canonical_class": label, "class_id": class_id,
                    "provider_channel_pair": "CH34", "original_channel_ids": [3, 4],
                    "intended_channel_operation": "mean(all_channels)",
                    "admitted_for_h1": True, "admitted_for_training": True,
                    "integrity_pass": True, "integrity_and_class_pass": True,
                    "metadata_parse_ok": True, "provenance_group_id": group,
                    "site_id": f"site_{group_number // 2}",
                    "calendar_date": f"2020-01-{group_number + 1:02d}",
                    "conservative_parent_proxy": group,
                    "candidate_recording_id": f"recording_{group_number}",
                    "source_time_axis": f"axis_{group_number}",
                    "paired_event_id": file_id, "duplicate_group_id": file_id,
                    "source_file_sha256": hashlib.sha256(file_id.encode()).hexdigest(),
                    "decoded_audio_sha256": hashlib.sha256((file_id + "decoded").encode()).hexdigest(),
                    "recording_session": None, "original_media_id": None,
                    "physical_vehicle_id": None, "vehicle_speed": None,
                    "source_template_id": None, "source_seed": None,
                    "unknown_reason": {"physical_vehicle_id": "unknown"},
                })
    return design.source_rows(rows)


@pytest.fixture(scope="module")
def h1_lock(source):
    folds = []
    for number, group in enumerate(sorted({r["provenance_group_id"] for r in source})):
        pool = [r for r in source if r["provenance_group_id"] != group]
        folds.append({
            "fold": number, "held_out_group": group,
            "train_ids": {"42": select(pool, 190, 42, f"synthetic_h1_fixture|{number}")},
            "test_ids": [r["file_id"] for r in source if r["provenance_group_id"] == group],
        })
    return {"folds": folds}


@pytest.fixture(scope="module")
def folds(source, h1_lock):
    return design.partitions(source, h1_lock)


def test_source_metadata_round_trip_preserves_every_null_and_unknown(source):
    original = deepcopy(source)
    non_source = {"dataset_id": "OTHER", "file_id": "excluded_domain"}
    selected = design.source_rows([non_source, *reversed(source)])
    assert selected == original
    restored = [json.loads(design.encoded(row)) for row in selected]
    assert restored == original
    for row in restored:
        assert row["recording_session"] is None
        assert row["original_media_id"] is None
        assert row["physical_vehicle_id"] is None
        assert row["vehicle_speed"] is None
        assert row["unknown_reason"] == {"physical_vehicle_id": "unknown"}
    assert source == original


@pytest.mark.parametrize("field,value", [
    ("canonical_class", "motorcycle"), ("canonical_class", "unknown"),
    ("class_id", 1), ("class_id", None),
    ("release_id", "unknown"), ("provider_channel_pair", "CH12"),
    ("original_channel_ids", [1, 2]), ("intended_channel_operation", "first_channel"),
    ("admitted_for_h1", False), ("admitted_for_training", None),
    ("integrity_pass", False), ("integrity_and_class_pass", False),
    ("metadata_parse_ok", False),
])
def test_unqualified_source_metadata_fails(source, field, value):
    row = deepcopy(next(row for row in source if row["canonical_class"] == "car"))
    row[field] = value
    with pytest.raises(ValueError, match="Unqualified"):
        design.source_rows([row])


def test_duplicate_source_ids_fail(source):
    with pytest.raises(ValueError, match="duplicate file IDs"):
        design.source_rows([source[0], deepcopy(source[0])])


@pytest.mark.parametrize("field", [
    "conservative_parent_proxy", "candidate_recording_id", "source_time_axis",
    "paired_event_id", "duplicate_group_id", "source_file_sha256",
    "decoded_audio_sha256", "recording_session", "original_media_id", "physical_vehicle_id",
])
def test_known_alias_cannot_span_groups_even_outside_selected_ids(source, field):
    first = deepcopy(source[0])
    second = deepcopy(next(row for row in source if row["provenance_group_id"] != first["provenance_group_id"]))
    first[field] = second[field] = "shared_known_ancestor"
    with pytest.raises(ValueError, match=f"Ancestor spans groups: {field}"):
        design.source_rows([first, second])


def test_all_nine_partitions_have_exact_selections_and_both_site_dates_excluded(source, h1_lock, folds):
    assert len(folds) == 9
    assert len({fold["id"] for fold in folds}) == 9
    index = {row["file_id"]: row for row in source}
    for original, frozen in zip(h1_lock["folds"], folds[:6]):
        assert frozen["train_ids"] == original["train_ids"]["42"]
        assert frozen["held_ids"] == original["test_ids"]
        assert frozen["held_groups"] == [original["held_out_group"]]
    for frozen in folds[6:]:
        site = frozen["held_label"]
        pool = [row for row in source if row["site_id"] != site]
        held = [row for row in source if row["site_id"] == site]
        assert frozen["train_ids"] == select(pool, 190, 42, f"cast4a_site_v1|{site}")
        assert frozen["held_ids"] == [row["file_id"] for row in held]
        assert len(frozen["held_groups"]) == 2
        assert len({row["calendar_date"] for row in held}) == 2
        assert all(index[file_id]["site_id"] != site for file_id in frozen["train_ids"])
    for frozen in folds:
        assert frozen["train_counts"] == {"car": 190, "truck": 190}
        assert len(frozen["train_ids"]) == len(set(frozen["train_ids"])) == 380
        assert frozen["train_ids_sha256"] == design.ids_digest(frozen["train_ids"])
        assert frozen["held_ids_sha256"] == design.ids_digest(frozen["held_ids"])
        assert frozen["validation_ids"] == []


@pytest.mark.archive
def test_current_h1_metadata_selections_are_copied_exactly():
    """Read only frozen JSON/JSONL metadata, never referenced audio or features."""
    manifest = resolve("archive:experiments/h1_diagnostics/frozen/source_diagnostics_20261003_v1/source_manifest.jsonl", must_exist=False)
    lock_path = resolve("archive:experiments/h1/frozen/h1_common_budget_v1.3/lock.json", must_exist=False)
    if not manifest.is_file() or not lock_path.is_file():
        pytest.skip("Preserved H1 metadata archive is unavailable")
    original_rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows = design.source_rows(original_rows)
    lock = json.loads(lock_path.read_text())
    frozen = design.partitions(rows, lock)
    assert len(rows) == 4413
    assert all(row["recording_session"] is None for row in rows)
    assert rows == sorted(original_rows, key=lambda row: row["file_id"])
    for original, output in zip(lock["folds"], frozen[:6]):
        assert output["train_ids"] == original["train_ids"]["42"]
        assert output["held_ids"] == original["test_ids"]
    for output in frozen[6:]:
        site = output["held_label"]
        pool = [row for row in rows if row["site_id"] != site]
        assert output["train_ids"] == select(pool, 190, 42, f"cast4a_site_v1|{site}")
        assert output["held_ids"] == [row["file_id"] for row in rows if row["site_id"] == site]
        assert len(output["held_groups"]) == 2
        assert output["train_counts"] == {"car": 190, "truck": 190}


@pytest.mark.parametrize("defect", ["unknown_id", "duplicate_id", "missing_held", "held_in_train", "short_budget"])
def test_partition_fails_closed_on_role_defects(source, h1_lock, defect):
    fold = deepcopy(h1_lock["folds"][0])
    train, held = fold["train_ids"]["42"], fold["test_ids"]
    if defect == "unknown_id":
        train[0] = "not_a_source_file"
    elif defect == "duplicate_id":
        train[0] = train[1]
    elif defect == "missing_held":
        held.pop()
    elif defect == "held_in_train":
        train[0] = held[0]
    else:
        train.pop()
    with pytest.raises(ValueError):
        design.partition(source, train, held, analysis="group", fold=0,
                         held_label=fold["held_out_group"], held_groups={fold["held_out_group"]})


def test_partition_checks_aliases_in_unselected_training_pool(source, h1_lock):
    rows = deepcopy(source)
    fold = h1_lock["folds"][0]
    train = set(fold["train_ids"]["42"])
    held = set(fold["test_ids"])
    unselected = next(row for row in rows if row["file_id"] not in train | held)
    held_row = next(row for row in rows if row["file_id"] in held)
    unselected["decoded_audio_sha256"] = held_row["decoded_audio_sha256"]
    with pytest.raises(ValueError, match="ancestry overlap: decoded_audio_sha256"):
        design.partition(rows, fold["train_ids"]["42"], fold["test_ids"], analysis="group", fold=0,
                         held_label=fold["held_out_group"], held_groups={fold["held_out_group"]})


def test_schedule_is_complete_unique_deterministic_and_paired(folds):
    schedule = list(design.planned_schedule(folds))
    assert len(schedule) == 27000
    assert len({row["id"] for row in schedule}) == 27000
    assert schedule == list(design.planned_schedule(folds))
    assert Counter(row["partition"].split("_")[0] for row in schedule) == {"group": 18000, "site": 9000}
    counts = Counter((row["partition"], row["class"], row["seed"], row["arm"]) for row in schedule)
    assert len(counts) == 9 * 2 * 5 * 6
    assert set(counts.values()) == {50}
    assert {row["arm"] for row in schedule} == {
        "v14_joint", "v14_prototype", "v14_marginals", "v15_joint", "v15_prototype", "v15_marginals"}
    assert {row["seed"] for row in schedule} == {42, 123, 456, 789, 1024}
    paired_temperatures = defaultdict(list)
    paired_arms = defaultdict(list)
    for row in schedule:
        paired_temperatures[row["partition"], row["class"], row["seed"], row["index"], row["sampler"]].append(row)
        paired_arms[row["partition"], row["class"], row["seed"], row["index"]].append(row)
        assert row["role"] == "planned_only_not_generated"
        assert 0 <= row["phase_seed"] < 2**63
        assert 0 <= row["noise_seed"] < 2**63
        assert 0 <= row["selection_seed"] < 2**64
        assert 0 <= row["child_seed"] < 2**64
    for pair in paired_temperatures.values():
        assert len(pair) == 2
        assert {row["temperature"] for row in pair} == {1.0, 1.1}
        for field in ("input_id", "selection_seed", "child_seed", "phase_seed", "noise_seed"):
            assert pair[0][field] == pair[1][field]
    for arms in paired_arms.values():
        assert len(arms) == 6
        assert len({(row["input_id"], row["phase_seed"], row["noise_seed"]) for row in arms}) == 1


def test_reuse_inventory_records_candidates_without_approving_reuse(folds):
    candidates = folds[0]["train_ids"][:12]
    inventory = design.reuse_inventory(folds, {"input_ids": {file_id: file_id for file_id in candidates}})
    assert inventory["status"] == "candidate_only_no_bank_opened_no_reuse_approved"
    assert inventory["candidate_ids"] == sorted(candidates)
    union = {file_id for fold in folds for file_id in fold["train_ids"]}
    assert inventory["unique_training_ids"] == sorted(union)
    assert inventory["missing_candidate_ids"] == sorted(union - set(candidates))
    for fold, record in zip(folds, inventory["folds"]):
        assert record["partition"] == fold["id"]
        assert record["candidate_ids_in_training"] == [file_id for file_id in fold["train_ids"] if file_id in candidates]
        assert record["requires_new_fit_or_other_audited_cache"] == [file_id for file_id in fold["train_ids"] if file_id not in candidates]
        assert not set(record["candidate_ids_in_training"]) & set(fold["held_ids"])


@pytest.mark.parametrize("entries", [["parent"], {"parent": "different_renderer_id"}])
def test_reuse_inventory_rejects_unmatched_parent_metadata(folds, entries):
    with pytest.raises(ValueError):
        design.reuse_inventory(folds, {"input_ids": entries})


@pytest.fixture
def frozen(tmp_path):
    directory = tmp_path / "frozen"
    expected = {
        "folds.json": design.encoded([{"train_ids": ["allowed_parent"], "held_ids": ["held_parent"]}]),
        "inputs/protocol.md": b"Metadata-only test protocol\n",
    }
    record = design.freeze(directory, expected)
    return directory, expected, record


def test_freeze_is_idempotent_and_does_not_rewrite(frozen):
    directory, expected, record = frozen
    before = {path.relative_to(directory): (path.read_bytes(), path.stat().st_mtime_ns)
              for path in directory.rglob("*") if path.is_file()}
    assert design.verify(directory, expected) == record
    assert design.freeze(directory, expected) == record
    after = {path.relative_to(directory): (path.read_bytes(), path.stat().st_mtime_ns)
             for path in directory.rglob("*") if path.is_file()}
    assert after == before


def test_verify_rejects_altered_output_even_with_updated_local_hash(frozen):
    directory, expected, _ = frozen
    altered = design.encoded([{"train_ids": ["held_parent"], "held_ids": ["allowed_parent"]}])
    (directory / "folds.json").write_bytes(altered)
    lock_path = directory / "lock.json"
    lock = json.loads(lock_path.read_text())
    lock["payload_sha256"]["folds.json"] = hashlib.sha256(altered).hexdigest()
    lock["payload_identity_sha256"] = design.digest(lock["payload_sha256"])
    lock_path.write_bytes(design.encoded(lock))
    with pytest.raises(ValueError, match="metadata replay"):
        design.verify(directory, expected)


@pytest.mark.parametrize("defect", ["extra", "missing_payload", "missing_hash", "bad_status", "bad_identity"])
def test_verify_rejects_incomplete_or_unexpected_artifacts(frozen, defect):
    directory, expected, _ = frozen
    lock_path = directory / "lock.json"
    lock = json.loads(lock_path.read_text())
    if defect == "extra":
        (directory / "unexpected.json").write_text("{}")
    elif defect == "missing_payload":
        (directory / "inputs/protocol.md").unlink()
    elif defect == "missing_hash":
        del lock["payload_sha256"]["inputs/protocol.md"]
        lock_path.write_bytes(design.encoded(lock))
    elif defect == "bad_status":
        lock["status"] = "scientific_execution_complete"
        lock_path.write_bytes(design.encoded(lock))
    else:
        lock["payload_identity_sha256"] = "0" * 64
        lock_path.write_bytes(design.encoded(lock))
    with pytest.raises(ValueError):
        design.verify(directory, expected)


def test_verify_rejects_symlink_payload_even_when_contents_match(frozen, tmp_path):
    directory, expected, _ = frozen
    outside = tmp_path / "outside.json"
    outside.write_bytes(expected["folds.json"])
    payload = directory / "folds.json"
    payload.unlink()
    payload.symlink_to(outside)
    with pytest.raises(ValueError, match="metadata replay"):
        design.verify(directory, expected)


def test_partial_directory_without_lock_is_not_overwritten(tmp_path):
    directory = tmp_path / "partial"
    directory.mkdir()
    retained = directory / "folds.json"
    retained.write_bytes(b"partial output")
    with pytest.raises(FileNotFoundError):
        design.freeze(directory, {"folds.json": b"replacement"})
    assert retained.read_bytes() == b"partial output"
    assert not (directory / "lock.json").exists()


def test_freeze_refuses_changed_design_at_existing_destination(frozen):
    directory, expected, _ = frozen
    before = (directory / "folds.json").read_bytes()
    changed = {**expected, "folds.json": design.encoded([{"new_design": True}])}
    with pytest.raises(ValueError, match="metadata replay"):
        design.freeze(directory, changed)
    assert (directory / "folds.json").read_bytes() == before


def test_failed_staging_does_not_publish_partial_freeze(tmp_path, monkeypatch):
    directory = tmp_path / "not_published"
    def fail_verification(*args):
        raise ValueError("injected metadata verification failure")
    monkeypatch.setattr(design, "verify", fail_verification)
    with pytest.raises(ValueError, match="injected"):
        design.freeze(directory, {"folds.json": b"[]\n"})
    assert not directory.exists()
    assert list(tmp_path.iterdir()) == []


def test_verified_metadata_rejects_changed_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("ABVID_REPO_ROOT", str(tmp_path))
    metadata = tmp_path / "protocol.md"
    metadata.write_text("Changed metadata contents\n")
    with pytest.raises(ValueError, match="Changed metadata: repo:protocol.md"):
        design.verified_metadata({"path": "repo:protocol.md", "sha256": "0" * 64})


@pytest.mark.parametrize("reference", ["data:metadata.json", "artifact:metadata.json", "repo:audio.wav", "repo:features.npy", "repo:fits.npz", "repo:model.pt"])
def test_verified_metadata_rejects_forbidden_source_before_open(tmp_path, monkeypatch, reference):
    path = tmp_path / reference.split(":", 1)[1]
    path.write_bytes(b"must not be opened")
    monkeypatch.setattr(design, "resolve", lambda value: path)
    def forbidden_open(*args, **kwargs):
        pytest.fail("A forbidden metadata source was opened before rejection")
    monkeypatch.setattr(Path, "open", forbidden_open)
    with pytest.raises(ValueError, match="Only named repository/archive metadata"):
        design.verified_metadata({"path": reference, "sha256": "0" * 64})


@pytest.mark.parametrize("dependency_kind", ["numerical", "implementation"])
def test_payloads_rejects_changed_dependency_before_reading_scientific_data(tmp_path, monkeypatch, dependency_kind):
    monkeypatch.setenv("ABVID_REPO_ROOT", str(tmp_path))
    dependency = tmp_path / "dependency.py"
    dependency.write_bytes(b"# frozen implementation\n")
    dependency_reference = {"path": "repo:dependency.py", "sha256": hashlib.sha256(dependency.read_bytes()).hexdigest()}
    inputs = {}
    for name in sorted(design.INPUT_NAMES):
        path = tmp_path / f"{name}.json"
        content = {"references": [dependency_reference] if dependency_kind == "numerical" else []} if name == "numerical_contract" else {}
        path.write_bytes(design.encoded(content))
        inputs[name] = {"path": f"repo:{path.name}", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    config_path = tmp_path / "canonical_config.json"
    config_path.write_text("{}")
    frozen_design = {
        "id": design.ID, "inputs": inputs, "budget_per_class": 190,
        "seeds": list(design.SEEDS), "arms": deepcopy(design.ARMS),
        "samples_per_class_seed_arm": 50, "execution_scope": "metadata_only",
        "selection_numpy_version": design.version("numpy"),
        "implementation_references": [dependency_reference] if dependency_kind == "implementation" else [],
    }
    dependency.write_bytes(b"# implementation changed after its hash was frozen\n")
    original_open = Path.open
    opened = []
    def guarded_open(path, *args, **kwargs):
        assert path.suffix.lower() not in {".wav", ".flac", ".npy", ".npz", ".pt", ".pth"}, "Scientific data must not be opened"
        opened.append(path)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    def unexpected_population_processing(*args, **kwargs):
        pytest.fail("Changed dependencies must fail before population processing")
    monkeypatch.setattr(design, "source_rows", unexpected_population_processing)
    with pytest.raises(ValueError, match="Changed metadata: repo:dependency.py"):
        design.payloads(frozen_design, config_path)
    assert dependency in opened
    assert config_path not in opened


def test_main_rejects_module_outside_hashed_repository_entry_point(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "canonical_config.json"
    repository_script = tmp_path / "repo/src/abvid/cast/grouped_design.py"
    installed_script = tmp_path / "site-packages/abvid/cast/grouped_design.py"
    script_reference = "repo:src/abvid/cast/grouped_design.py"
    cfg = {"id": "cast_grouped_coverage", "execution": {"mode": "metadata_only", "script": script_reference},
           "contract": {"path": "repo:design.json"}}
    def resolved(reference):
        return {design.CONFIG: config_path, script_reference: repository_script}[reference]
    monkeypatch.setattr(design, "resolve", resolved)
    monkeypatch.setattr(design, "load", lambda path, *, check_files: cfg)
    monkeypatch.setattr(design, "__file__", str(installed_script))
    def forbidden_payloads(*args, **kwargs):
        pytest.fail("A mismatched imported module cannot prepare payloads")
    monkeypatch.setattr(design, "payloads", forbidden_payloads)
    with pytest.raises(SystemExit) as exc:
        design.main(["freeze", "--config", design.CONFIG])
    assert exc.value.code == 2
    assert "Imported module is not the hashed repository entry point" in capsys.readouterr().err

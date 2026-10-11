"""Immutable artifact storage regression tests using temporary synthetic bytes."""
from copy import deepcopy
import json

import pytest

from abvid.cast import grouped_storage as storage


def test_binary_round_trip_nested_inventory_and_unchanged_idempotence(tmp_path):
    folder = tmp_path / "bundle"
    payload = {"observation.wav": bytes(range(256)) * 7,
               "components.npz": b"\x00\xff\x00\x12", "stages/v11.json": b'{"value":3}\n'}
    metadata = {"parent": "synthetic", "recording_session": "unknown", "seeds": [42, 123]}
    receipt = storage.write_bundle(folder, payload, metadata)
    assert storage.read_bundle(folder) == (metadata, payload)
    assert storage.verify_bundle(folder) == receipt
    before = {p: p.stat().st_mtime_ns for p in folder.rglob("*")}
    assert storage.write_bundle(folder, deepcopy(payload), deepcopy(metadata)) == receipt
    assert before == {p: p.stat().st_mtime_ns for p in folder.rglob("*")}
    assert not list(tmp_path.glob(".attempt-*"))
    assert set(receipt) == {"metadata", "payload_sha256", "identity_sha256"}


@pytest.mark.parametrize("change", ["payload", "metadata"])
def test_changed_existing_bundle_is_not_replaced(tmp_path, change):
    folder = tmp_path / "bundle"
    storage.write_bundle(folder, {"fit.json": b"original"}, {"seed": 42})
    with pytest.raises(ValueError, match="immutable"):
        storage.write_bundle(folder, {"fit.json": b"changed" if change == "payload" else b"original"},
                             {"seed": 123 if change == "metadata" else 42})
    assert storage.read_bundle(folder) == ({"seed": 42}, {"fit.json": b"original"})


@pytest.mark.parametrize("tamper", ["changed_bytes", "extra_file", "extra_empty_directory", "removed_file",
                                    "changed_manifest", "duplicate_manifest_key"])
def test_tampered_bundle_fails_closed(tmp_path, tamper):
    folder = tmp_path / "bundle"
    storage.write_bundle(folder, {"fit.json": b"original"}, {"seed": 42})
    if tamper == "changed_bytes":
        (folder / "fit.json").write_bytes(b"changed")
    elif tamper == "extra_file":
        (folder / "undeclared.txt").write_text("extra")
    elif tamper == "extra_empty_directory":
        (folder / "undeclared").mkdir()
    elif tamper == "removed_file":
        (folder / "fit.json").unlink()
    elif tamper == "changed_manifest":
        path = folder / storage.MANIFEST
        value = json.loads(path.read_text())
        value["metadata"]["seed"] = 123
        path.write_text(json.dumps(value))
    else:
        path = folder / storage.MANIFEST
        path.write_text(path.read_text().replace('{"identity_sha256":', '{"metadata":{},"identity_sha256":', 1))
    with pytest.raises(ValueError):
        storage.verify_bundle(folder)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../escape", "a//b", "./fit.json",
                                 "_bundle.json", "nested/_bundle.json", "a\\b", "", ".", "nul\0byte"])
def test_unsafe_payload_names_rejected_before_creating_output(tmp_path, name):
    folder = tmp_path / "bundle"
    with pytest.raises(ValueError):
        storage.write_bundle(folder, {name: b"value"}, {})
    assert not folder.exists()
    assert not list(tmp_path.iterdir())


def test_symlink_payload_rejected_before_any_file_read(tmp_path, monkeypatch):
    folder = tmp_path / "bundle"
    storage.write_bundle(folder, {"fit.json": b"original"}, {})
    outside = tmp_path / "outside"
    outside.write_bytes(b"original")
    (folder / "fit.json").unlink()
    (folder / "fit.json").symlink_to(outside)
    monkeypatch.setattr(storage, "_read_file", lambda path: pytest.fail("Symlink inventory reached a read"))
    with pytest.raises(ValueError, match="Symlink"):
        storage.read_bundle(folder)


def test_symlinked_output_ancestor_rejected_for_reads_and_writes(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        storage.write_bundle(link / "bundle", {"data": b"x"}, {})
    with pytest.raises(ValueError, match="Symlink"):
        storage.verify_bundle(link / "bundle")
    assert not list(actual.iterdir())


def test_incomplete_existing_directory_is_not_repaired_or_overwritten(tmp_path):
    folder = tmp_path / "bundle"
    folder.mkdir()
    (folder / "uncommitted.bin").write_bytes(b"preserve")
    with pytest.raises(ValueError, match="incomplete"):
        storage.write_bundle(folder, {"data": b"new"}, {})
    assert (folder / "uncommitted.bin").read_bytes() == b"preserve"
    assert sorted(p.name for p in folder.iterdir()) == ["uncommitted.bin"]


def test_failed_atomic_publication_preserves_complete_attempt(tmp_path, monkeypatch):
    folder = tmp_path / "bundle"
    def fail(*args):
        raise OSError("injected rename failure")
    monkeypatch.setattr(storage.os, "rename", fail)
    with pytest.raises(OSError, match="injected"):
        storage.write_bundle(folder, {"fit.json": b"recoverable"}, {"seed": 42})
    assert not folder.exists()
    attempts = list(tmp_path.glob(".attempt-*"))
    assert len(attempts) == 1
    assert storage.read_bundle(attempts[0]) == ({"seed": 42}, {"fit.json": b"recoverable"})
    assert not list(tmp_path.glob("*.lock"))


def test_failed_payload_write_preserves_partial_attempt(tmp_path, monkeypatch):
    write = storage._durable_write
    def fail_second(path, value):
        if path.name == "b":
            raise OSError("injected payload failure")
        write(path, value)
    monkeypatch.setattr(storage, "_durable_write", fail_second)
    with pytest.raises(OSError, match="injected"):
        storage.write_bundle(tmp_path / "bundle", {"a": b"first", "b": b"second"}, {})
    attempts = list(tmp_path.glob(".attempt-*"))
    assert len(attempts) == 1 and (attempts[0] / "a").read_bytes() == b"first"
    with pytest.raises(ValueError, match="incomplete"):
        storage.verify_bundle(attempts[0])

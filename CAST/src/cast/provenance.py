"""Fail closed on metadata and ancestry BEFORE reading any audio bytes."""
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from .config import ROOT, HERE, canonical, digest, save, sha, validate


def select(cfg, root=ROOT):
    validate(cfg)
    d = cfg["data"]
    for key in ("manifest", "source_lock", "parent_lock"):
        if sha(root / d[key]) != d[key + "_sha256"]:
            raise ValueError(f"Changed {key}")
    source = json.loads((root / d["source_lock"]).read_text())
    parent = json.loads((root / d["parent_lock"]).read_text())
    if source["artifacts_sha256"]["source_manifest.jsonl"] != d["manifest_sha256"]:
        raise ValueError("Source manifest not locked")
    if source["h1_lock_sha256"] != d["parent_lock_sha256"]:
        raise ValueError("Broken parent lock ancestry")
    fold = next(f for f in parent["folds"] if f["fold"] == d["parent_fold"])
    if fold["held_out_group"] != d["excluded_group"]:
        raise ValueError("Unexpected held-out group")
    ids = fold["train_ids"][str(d["selection_seed"])]
    if len(ids) != len(set(ids)) or set(ids) & set(fold["test_ids"]):
        raise ValueError("Duplicate or cross-split H1 IDs")
    rows = [json.loads(line) for line in (root / d["manifest"]).read_text().splitlines()]
    by_id = {r["file_id"]: r for r in rows}
    if len(by_id) != len(rows) or not set(ids) <= by_id.keys():
        raise ValueError("Invalid or missing manifest IDs")
    pool = [by_id[i] for i in ids]
    # Metadata validation for the entire permitted selection does not open audio.
    for row in pool:
        validate_row(row, cfg, set(ids), root)
    bins = sorted({(r["provenance_group_id"], r["canonical_class"]) for r in pool})
    selected, inventory = [], []
    for group, label in bins:
        available = sorted((r for r in pool if (r["provenance_group_id"], r["canonical_class"]) == (group, label)), key=lambda r: r["file_id"])
        chosen = available[:d["pilot_max_per_group_class"]]
        selected.extend(chosen)
        inventory.append({"group": group, "class": label, "available_in_H1_selection": len(available),
                          "selected": len(chosen), "shortage": max(0, 5-len(chosen))})
    if not selected or len(selected) > 50:
        raise ValueError("Pilot selection bound violated")
    held = [r for r in rows if r["provenance_group_id"] == d["excluded_group"]]
    for field in ("file_id", "source_file_sha256", "decoded_audio_sha256", "paired_event_id", "duplicate_group_id", "candidate_recording_id"):
        disallowed = {r[field] for r in held if r.get(field)}
        if any(r.get(field) in disallowed for r in selected):
            raise ValueError(f"Cross-split ancestry: {field}")
    return selected, inventory


def validate_row(row, cfg, allowed_ids, root=ROOT):
    d = cfg["data"]
    checks = [row.get("file_id") in allowed_ids,
              row.get("dataset_id") == "IDMT", row.get("release_id") == "IDMT_V1",
              row.get("canonical_class") in d["classes"],
              row.get("class_id") == d["classes"].get(row.get("canonical_class")),
              row.get("provenance_group_id") and row.get("provenance_group_id") != d["excluded_group"],
              row.get("split_role") == "source_logo_pool", row.get("admitted_for_training") is True,
              row.get("admitted_for_h1") is True, row.get("integrity_pass") is True,
              row.get("device_id") == "SE", row.get("provider_channel_pair") == "CH34",
              row.get("original_channel_ids") == [3, 4], row.get("native_channels") == 2,
              row.get("raw_audio_modified") is False]
    if not all(checks):
        raise ValueError("Disallowed data or ancestry")
    raw = Path(row["source_path"])
    base = root / d["allowed_audio_root"]
    if raw.is_absolute() or ".." in raw.parts or raw.suffix != ".wav" or not raw.name.endswith("_SE_CH34.wav"):
        raise ValueError("Disallowed audio path")
    path = root / raw
    # Never follow a source symlink into an excluded corpus, even while hashing.
    if base.resolve() != base.absolute() or path.resolve().parent != base.resolve() or path.is_symlink():
        raise ValueError("Audio path or symlink outside exact allowlisted root")
    for key in ("source_file_sha256", "decoded_audio_sha256", "file_id"):
        value = row.get(key, "")
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Missing or invalid {key}")
    return path


def checked_path(row, cfg, allowed_rows, root=ROOT):
    rows = {r["file_id"]: r for r in allowed_rows}
    path = validate_row(row, cfg, rows.keys(), root)
    if row != rows[row["file_id"]]:
        raise ValueError("Ancestry differs from immutable selected manifest")
    if sha(path) != row["source_file_sha256"]:
        raise ValueError("Audio hash mismatch")
    return path


def environment():
    names = ["numpy", "scipy", "torch", "soundfile", "matplotlib", "pytest"]
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor() or "unknown",
            "versions": {n: importlib.metadata.version(n) for n in names}}


def source_hashes():
    files = list((HERE / "src").rglob("*.py")) + list((HERE / "tests").glob("*.py"))
    files += [HERE / "run.sh", HERE / "configs/pilot_v0.json", HERE / "requirements-lock.txt"]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files) if p.is_file()}


def snapshot():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT).decode()
    return {"git_commit": git("rev-parse", "HEAD").strip(), "dirty_status": git("status", "--short").splitlines(),
            "tracked_diff_sha256": __import__("hashlib").sha256(git("diff", "--binary").encode()).hexdigest(),
            "source_hashes": source_hashes(), "environment": environment()}


def freeze(out, cfg):
    selected, inventory = select(cfg)
    # Check all selected bytes after metadata/split checks; never decode here.
    for row in selected:
        checked_path(row, cfg, selected)
    out.mkdir(parents=True, exist_ok=False)
    save(out / "config.resolved.json", cfg)
    manifest = out / cfg["data"]["selected_ids_file"]
    manifest.write_bytes(b"".join(canonical(r)+b"\n" for r in selected))
    save(out / "inventory.json", {"recordings": len(selected), "groups_classes": inventory,
                                  "audio_decoded": False, "excluded_group": cfg["data"]["excluded_group"],
                                  "preselected_example_ids": [next(r["file_id"] for r in selected if r["canonical_class"] == c) for c in ("car", "truck")]})
    state = snapshot()
    save(out / "snapshot.json", state)
    for path in state["source_hashes"]:
        dest = out / "source_snapshot" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT/path).read_bytes())
    save(out / "lock.json", {"config_sha256": digest(cfg), "manifest_sha256": sha(manifest),
                             "snapshot_sha256": sha(out/"snapshot.json"), "inventory_sha256": sha(out/"inventory.json"),
                             "parent_lock_sha256": cfg["data"]["parent_lock_sha256"],
                             "selected_ids": [r["file_id"] for r in selected],
                             "source_hashes": state["source_hashes"]})
    return selected


def read_frozen(out, require_current_code=True):
    lock = json.loads((out/"lock.json").read_text())
    cfg = validate(json.loads((out/"config.resolved.json").read_text()))
    if digest(cfg) != lock["config_sha256"]:
        raise ValueError("Changed resolved configuration")
    for name, key in ((cfg["data"]["selected_ids_file"], "manifest_sha256"), ("snapshot.json", "snapshot_sha256"), ("inventory.json", "inventory_sha256")):
        if sha(out/name) != lock[key]:
            raise ValueError(f"Changed frozen {name}")
    if require_current_code and source_hashes() != lock["source_hashes"]:
        raise ValueError("Implementation changed since freeze; use a new run")
    rows = [json.loads(s) for s in (out/cfg["data"]["selected_ids_file"]).read_text().splitlines()]
    expected, _ = select(cfg)
    if rows != expected or [r["file_id"] for r in rows] != lock["selected_ids"]:
        raise ValueError("Selection changed")
    return cfg, rows, lock

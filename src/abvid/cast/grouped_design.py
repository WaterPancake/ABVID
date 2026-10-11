"""Freeze CAST-4A metadata and planned streams, without accessing audio or fits.

This module deliberately has no fitting, rendering, descriptor or classifier
entry point. The numerical recipe describes a later, separately authorized run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

from abvid.data.splits import select
from abvid.paths import resolve, roots
from abvid.protocol import load
from abvid.provenance.hashing import canonical, sha256

CONFIG = "repo:configs/experiments/cast_grouped_coverage.json"
ID = "cast4a_grouped_design_v1"
SEEDS = [42, 123, 456, 789, 1024]
ARMS = [{"id": f"{name}_{arm}", "temperature": temperature, "sampler": arm}
        for name, temperature in (("v14", 1.0), ("v15", 1.1))
        for arm in ("joint", "prototype", "marginals")]
LINKS = ("file_id", "provenance_group_id", "conservative_parent_proxy",
         "candidate_recording_id", "source_time_axis", "paired_event_id",
         "duplicate_group_id", "source_file_sha256", "decoded_audio_sha256",
         "recording_session", "original_media_id", "physical_vehicle_id")
INPUT_NAMES = {"admission", "h1_lock", "h1_config", "candidate_fit_protocol",
               "candidate_fit_lock", "candidate_fit_verification",
               "numerical_contract", "protocol"}
META_SUFFIXES = {".json", ".jsonl", ".md", ".py", ".toml", ".lock", ".txt", ".sh"}


def encoded(value):
    return canonical(value) + b"\n"


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def ids_digest(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def known(value):
    return value is not None and value != "" and value != "unknown"


def count(rows):
    c = Counter(r["canonical_class"] for r in rows)
    return {name: c[name] for name in ("car", "truck")}


def assert_disjoint(train, held):
    for field in LINKS:
        a = {r[field] for r in train if known(r.get(field))}
        b = {r[field] for r in held if known(r.get(field))}
        if a & b:
            raise ValueError(f"Train/held ancestry overlap: {field}")


def source_rows(rows):
    """Retain exact admitted metadata; unknown sessions are never synthesized."""
    selected = sorted((r for r in rows if r.get("dataset_id") == "IDMT"),
                      key=lambda r: r["file_id"])
    if not selected or len({r["file_id"] for r in selected}) != len(selected):
        raise ValueError("Empty source inventory or duplicate file IDs")
    for r in selected:
        if (r.get("release_id") != "IDMT_V1" or
                r.get("canonical_class") not in ("car", "truck") or
                r.get("class_id") != {"car": 0, "truck": 1}.get(r.get("canonical_class")) or
                r.get("provider_channel_pair") != "CH34" or
                r.get("original_channel_ids") != [3, 4] or
                r.get("intended_channel_operation") != "mean(all_channels)" or
                any(r.get(k) is not True for k in (
                    "admitted_for_h1", "admitted_for_training", "integrity_pass",
                    "integrity_and_class_pass", "metadata_parse_ok"))):
            raise ValueError(f"Unqualified source row: {r['file_id']}")
    # A known shared ancestor must not straddle conservative groups, even if
    # a particular training subsample happens not to include that ancestor.
    for field in LINKS:
        groups = {}
        for r in selected:
            value = r.get(field)
            if known(value):
                old = groups.setdefault(value, r["provenance_group_id"])
                if old != r["provenance_group_id"]:
                    raise ValueError(f"Ancestor spans groups: {field}")
    return selected


def partition(rows, train_ids, held_ids, *, analysis, fold, held_label, held_groups):
    index = {r["file_id"]: r for r in rows}
    if any(len(ids) != len(set(ids)) for ids in (train_ids, held_ids)):
        raise ValueError("Repeated role IDs")
    if not set(train_ids + held_ids) <= set(index):
        raise ValueError("Role contains non-source ID")
    train, held = ([index[i] for i in ids] for ids in (train_ids, held_ids))
    remaining = [r for r in rows if r["provenance_group_id"] not in held_groups]
    expected_held = {r["file_id"] for r in rows if r["provenance_group_id"] in held_groups}
    if set(held_ids) != expected_held or not set(train_ids) <= {r["file_id"] for r in remaining}:
        raise ValueError("Incomplete holdout or contaminated training role")
    assert_disjoint(remaining, held)
    assert_disjoint(train, held)
    if count(train) != {"car": 190, "truck": 190}:
        raise ValueError("Training budget must be exactly 190/class")
    for label in ("car", "truck"):
        if len({r["provenance_group_id"] for r in train if r["canonical_class"] == label}) < 3:
            raise ValueError("Unchanged group prior requires three groups/class")
        if count(held)[label] == 0:
            raise ValueError("Missing held class")
    return {"id": f"{analysis}_{fold}", "analysis": analysis, "fold": fold,
            "held_label": held_label, "held_groups": sorted(held_groups),
            "train_ids": train_ids, "held_ids": held_ids,
            "train_ids_sha256": ids_digest(train_ids), "held_ids_sha256": ids_digest(held_ids),
            "train_counts": count(train), "held_counts": count(held),
            "available_train_counts": count(remaining),
            "training_group_counts": {g: count([r for r in train if r["provenance_group_id"] == g])
                                      for g in sorted({r["provenance_group_id"] for r in train})},
            "selection": ("exact H1 lock train_ids[42] in original order" if analysis == "group"
                          else f"abvid.data.splits.select(pool,190,42,'cast4a_site_v1|{held_label}')"),
            "validation_ids": [], "validation_reason": "fixed method; no selection or tuning",
            "exposure": "previously exposed source-domain development evidence",
            "ancestry_checks": list(LINKS), "ancestry_disjoint": True}


def partitions(rows, h1_lock):
    result = []
    folds = h1_lock["folds"]
    groups = {r["provenance_group_id"] for r in rows}
    if len(folds) != 6 or {f["held_out_group"] for f in folds} != groups:
        raise ValueError("Expected all six frozen H1 conservative groups")
    for f in folds:
        result.append(partition(rows, f["train_ids"]["42"], f["test_ids"],
                                analysis="group", fold=f["fold"], held_label=f["held_out_group"],
                                held_groups={f["held_out_group"]}))
    sites = sorted({r["site_id"] for r in rows})
    if len(sites) != 3:
        raise ValueError("Expected three IDMT sites")
    for i, site in enumerate(sites):
        pool = [r for r in rows if r["site_id"] != site]
        held = [r for r in rows if r["site_id"] == site]
        result.append(partition(rows, select(pool, 190, 42, f"cast4a_site_v1|{site}"),
                                [r["file_id"] for r in held], analysis="site", fold=i,
                                held_label=site, held_groups={r["provenance_group_id"] for r in held}))
    return result


def hash_seed(parts, *, signed=False):
    value = int.from_bytes(hashlib.sha256(canonical(parts)).digest()[:8], "big")
    return value % (2**63) if signed else value


def planned_schedule(folds):
    """Only seed values and role IDs; no parameter draws or waveform generation."""
    namespace = "cast_improvement_sampling_v1:outer"
    for fold in folds:
        for label in ("car", "truck"):
            for seed in SEEDS:
                for index in range(50):
                    input_id = f"{namespace}:sampling:{label}:{seed}:{index:04d}"
                    for arm in ARMS:
                        sampler = arm["sampler"]
                        yield {"id": f"{fold['id']}:{arm['id']}:{label}:{seed}:{index:04d}",
                               "partition": fold["id"], "class": label, "seed": seed,
                               "index": index, "arm": arm["id"], "sampler": sampler,
                               "temperature": arm["temperature"], "input_id": input_id,
                               "selection_seed": hash_seed([namespace, "selection", label, seed, index, sampler]),
                               "child_seed": hash_seed([namespace, "group_effect_v12_children", label, seed, index, sampler]),
                               "phase_seed": hash_seed([42, input_id, "phase", "sampling", 0], signed=True),
                               "noise_seed": hash_seed([42, input_id, "noise", "sampling", 0], signed=True),
                               "role": "planned_only_not_generated"}


def reuse_inventory(folds, candidate_protocol):
    entries = candidate_protocol["input_ids"]
    # Archived protocol maps each original file ID to its renderer input ID.
    if not isinstance(entries, dict):
        raise ValueError("Unexpected candidate protocol metadata shape")
    candidates = set()
    for file_id, input_id in entries.items():
        if input_id != file_id:
            raise ValueError("Candidate fit ID is not the original parent ID")
        candidates.add(file_id)
    union = sorted({i for f in folds for i in f["train_ids"]})
    return {"status": "candidate_only_no_bank_opened_no_reuse_approved",
            "candidate_ids": sorted(candidates), "unique_training_ids": union,
            "missing_candidate_ids": sorted(set(union) - candidates),
            "folds": [{"partition": f["id"],
                       "candidate_ids_in_training": [i for i in f["train_ids"] if i in candidates],
                       "requires_new_fit_or_other_audited_cache": [i for i in f["train_ids"] if i not in candidates]}
                      for f in folds],
            "requirements": ["complete same-parent chain and hashes", "identical numerical settings and streams",
                             "identical observation profile and implementation/environment",
                             "current fold training role only", "no reuse of population statistics"]}


def verified_metadata(reference):
    path = resolve(reference["path"])
    if reference["path"].split(":", 1)[0] not in {"repo", "archive"} or path.suffix not in META_SUFFIXES:
        raise ValueError("Only named repository/archive metadata may be read")
    if sha256(path) != reference["sha256"]:
        raise ValueError(f"Changed metadata: {reference['path']}")
    return path


def payloads(design, config_path):
    if set(design["inputs"]) != INPUT_NAMES or design["id"] != ID:
        raise ValueError("Unexpected grouped-design inputs")
    if (design["budget_per_class"] != 190 or design["seeds"] != SEEDS or design["arms"] != ARMS
            or design["samples_per_class_seed_arm"] != 50 or design["execution_scope"] != "metadata_only"):
        raise ValueError("Changed frozen design dimensions")
    if version("numpy") != design["selection_numpy_version"]:
        raise ValueError("Metadata selection requires the frozen NumPy version")
    files = {k: verified_metadata(v) for k, v in design["inputs"].items()}
    numerical = json.loads(files["numerical_contract"].read_text())
    for reference in numerical["references"]:
        verified_metadata(reference)
    for reference in design["implementation_references"]:
        verified_metadata(reference)
    lock = json.loads(files["h1_lock"].read_text())
    if lock["artifacts_sha256"]["admitted.jsonl"] != design["inputs"]["admission"]["sha256"]:
        raise ValueError("Admission manifest does not belong to H1 lock")
    if lock["artifacts_sha256"]["config.json"] != design["inputs"]["h1_config"]["sha256"]:
        raise ValueError("H1 configuration does not belong to lock")
    rows = source_rows([json.loads(line) for line in files["admission"].read_text().splitlines()])
    if count(rows) != {"car": 3902, "truck": 511}:
        raise ValueError("Changed source population")
    folds = partitions(rows, lock)
    reuse = reuse_inventory(folds, json.loads(files["candidate_fit_protocol"].read_text()))
    if len(reuse["candidate_ids"]) != 380 or set(reuse["candidate_ids"]) != set(folds[0]["train_ids"]):
        raise ValueError("Named saved-fit candidates must match the original fold-0 parents")
    summary = {"design_id": ID, "status": "design_frozen_experiment_unrun",
               "source_counts": count(rows), "source_ids_sha256": ids_digest([r["file_id"] for r in rows]),
               "partitions": [{k: v for k, v in f.items() if k not in {"train_ids", "held_ids"}}
                              for f in folds],
               "planned_waveforms": {"group": 18000, "site": 9000},
               "planned_score_rows": {"group": 360, "site": 180},
               "unique_training_ids": len(reuse["unique_training_ids"]),
               "candidate_fit_ids": len(reuse["candidate_ids"]),
               "training_ids_without_candidate_fit": len(reuse["missing_candidate_ids"]),
               "raw_audio_read": False, "feature_arrays_read": False, "fitted_banks_read": False,
               "fits_run": 0, "waveforms_generated": 0, "metrics": None,
               "raw_audio_hashes": "carried from pinned admission metadata; not reverified against audio",
               "unknown_metadata": {field: sum(not known(r.get(field)) for r in rows)
                                    for field in ("recording_session", "original_media_id", "physical_vehicle_id", "vehicle_speed")},
               "limitations": ["three sites; groups are conservative proxies, not verified sessions",
                               "held IDMT groups are exposed development data",
                               "source coverage is not classifier or transfer evidence",
                               "future packaged smooth8 fitting adapter and parity checks required",
                               "candidate saved fits require transitive provenance audit"]}
    result = {"sources.jsonl": b"".join(encoded(r) for r in rows),
              "folds.json": encoded(folds), "summary.json": encoded(summary),
              "candidate_reuse.json": encoded(reuse),
              "planned_generation.jsonl": b"".join(encoded(r) for r in planned_schedule(folds)),
              "canonical_config.json": config_path.read_bytes(), "design.json": encoded(design)}
    for name, path in files.items():
        if name != "admission":  # Export only IDMT metadata, not other-domain inventory.
            result[f"inputs/{name}{path.suffix}"] = path.read_bytes()
    for reference in design["implementation_references"]:
        path = verified_metadata(reference)
        result["source_snapshot/" + reference["path"].removeprefix("repo:")] = path.read_bytes()
    return result


def output_path(design):
    reference = design["output"]
    if not reference.startswith("artifact:"):
        raise ValueError("Design artifacts must be external")
    path = resolve(reference, must_exist=False)
    if path == roots()["artifact"] or path.is_relative_to(roots()["archive"]) or path.is_relative_to(roots()["repo"]):
        raise ValueError("Cannot write artifact root, archive or checkout")
    return path


def verify(directory, expected):
    record = json.loads((directory / "lock.json").read_text())
    observed = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if observed != set(expected) | {"lock.json"} or set(record["payload_sha256"]) != set(expected):
        raise ValueError("Frozen artifact file inventory changed")
    for name, content in expected.items():
        actual = directory / name
        if actual.is_symlink() or actual.read_bytes() != content or record["payload_sha256"][name] != hashlib.sha256(content).hexdigest():
            raise ValueError(f"Frozen artifact differs from metadata replay: {name}")
    if record["status"] != "metadata_only_complete_scientific_execution_unrun":
        raise ValueError("Invalid frozen status")
    if record["payload_identity_sha256"] != digest(record["payload_sha256"]):
        raise ValueError("Frozen payload identity changed")
    return record


def freeze(directory, expected):
    if directory.exists():
        return verify(directory, expected)  # Idempotent verification; never overwrite.
    directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".cast4a-metadata-", dir=directory.parent))
    try:
        for name, content in expected.items():
            dest = stage / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
        repo = roots()["repo"]
        git = lambda *args: subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
        record = {"status": "metadata_only_complete_scientific_execution_unrun",
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "git_commit": git("rev-parse", "HEAD"), "dirty_status": git("status", "--porcelain").splitlines(),
                  "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                  "numpy": version("numpy"), "scipy": version("scipy"),
                                  "torch": version("torch"), "soundfile": version("soundfile")},
                  "payload_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in expected.items()},
                  "payload_identity_sha256": digest({name: hashlib.sha256(content).hexdigest() for name, content in expected.items()})}
        (stage / "lock.json").write_bytes(encoded(record))
        verify(stage, expected)
        os.rename(stage, directory)
        return record
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify"))
    parser.add_argument("--config", required=True, choices=(CONFIG,))
    args = parser.parse_args(argv)
    try:
        path = resolve(args.config)
        cfg = load(path, check_files=True)
        if cfg["id"] != "cast_grouped_coverage" or cfg["execution"]["mode"] != "metadata_only":
            raise ValueError("This entry point only prepares the CAST-4A design")
        if Path(__file__).resolve() != resolve(cfg["execution"]["script"]):
            raise ValueError("Imported module is not the hashed repository entry point")
        design = json.loads(resolve(cfg["contract"]["path"]).read_text())
        expected = payloads(design, path)
        directory = output_path(design)
        record = freeze(directory, expected) if args.action == "freeze" else verify(directory, expected)
        print(json.dumps({"status": record["status"], "output": str(directory),
                          "payload_identity_sha256": record["payload_identity_sha256"],
                          "files_verified": len(expected)}))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

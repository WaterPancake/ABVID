"""Pinned CAST-4A execution context and metadata-only plan.

This loader never audits raw audio, opens parameter banks or arrays, fits,
generates, or scores observations. Existing audit receipts authorize only the
named same-parent parameter reuse, subject to each current training role.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import stat

from abvid.cast import grouped_design, readiness, roles as role_api
from abvid.paths import resolve, roots
from abvid.provenance.hashing import canonical, sha256


CONFIG = "repo:experiments/cast_coverage/grouped_coverage/execution/config.json"
OUTPUT = "artifact:cast/grouped_coverage/cast4a_run_20261008_v1"
COUNTS = {"source_recordings": 4413, "partitions": 9, "unique_training_parents": 1864,
          "eligible_cached_parents": 380, "new_parent_fits": 1484,
          "training_parents_per_partition": 380, "generated_waveforms": 27000}
FIELDS = {"schema_version", "id", "scope", "design_config", "design_lock", "readiness_config",
          "readiness_lock", "numerical_contract", "chain_contract", "implementation_references",
          "test_references", "expected_environment", "expected_counts", "output", "entrypoint", "max_workers"}
METADATA_SUFFIXES = {".json", ".jsonl", ".py", ".md", ".xml", ".log", ".toml", ".lock", ".txt", ".sh"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _no_symlinks(path):
    for ancestor in reversed((path, *path.parents)):
        try:
            mode = ancestor.lstat().st_mode
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(mode), f"Symlink in execution path: {ancestor}")


def artifact_path(reference):
    require(isinstance(reference, str) and reference.startswith("artifact:"), "Execution output must use artifact storage")
    resolved = resolve(reference, must_exist=False)
    locations = roots()
    lexical = locations["artifact"]/reference.split(":", 1)[1]
    _no_symlinks(lexical)
    require(lexical.resolve() == resolved, "Execution artifact path changed during resolution")
    require(resolved != locations["artifact"] and resolved.is_relative_to(locations["artifact"])
            and not resolved.is_relative_to(locations["archive"])
            and not resolved.is_relative_to(locations["repo"]), "Execution cannot write checkout, archive or storage root")
    return resolved


def checked(reference):
    require(isinstance(reference, dict) and set(reference) == {"path", "sha256"}, "Execution inputs require a path and SHA256")
    path = resolve(reference["path"])
    require(path.suffix in METADATA_SUFFIXES, "Execution plan may open metadata only")
    require(sha256(path) == reference["sha256"], f"Changed execution reference: {reference['path']}")
    return path


def _unique(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _json(path):
    return json.loads(path.read_text(), object_pairs_hook=_unique)


def _jsonl_bytes(value):
    return [json.loads(line, object_pairs_hook=_unique) for line in value.splitlines()]


def reference_inventory():
    repo = roots()["repo"]
    modules = sorted("repo:"+p.relative_to(repo).as_posix() for p in (repo/"src/abvid/cast").glob("grouped_*.py"))
    tests = sorted("repo:"+p.relative_to(repo).as_posix() for p in (repo/"tests/cast").glob("test_grouped_*.py"))
    return modules, tests


def load_contract(reference=CONFIG):
    require(reference == CONFIG, "Use the exact additive CAST-4A execution configuration")
    cfg = _json(resolve(reference))
    require(set(cfg) == FIELDS and cfg["schema_version"] == 1 and cfg["id"] == "cast4a_grouped_execution_v1"
            and cfg["scope"] == "source_domain_effective_observation_coverage_only", "Unexpected grouped execution contract")
    require(cfg["output"] == OUTPUT and cfg["expected_counts"] == COUNTS, "Changed frozen output or study budget")
    require(type(cfg["max_workers"]) is int and cfg["max_workers"] == 4, "Exactly four bounded parent-fit workers are declared")
    require(cfg["design_config"]["path"] == grouped_design.CONFIG
            and cfg["readiness_config"]["path"] == readiness.CONFIG, "Frozen design/readiness identity changed")
    require(cfg["entrypoint"] == {"module": "abvid.cast.grouped_runner", "script": "repo:src/abvid/cast/grouped_runner.py"},
            "Unknown grouped runner entrypoint")
    require(readiness.environment() == cfg["expected_environment"], "Execution requires its exact pinned CAST environment")
    modules, tests = reference_inventory()
    require(cfg["entrypoint"]["script"] in modules and "repo:src/abvid/cast/grouped_execution_config.py" in modules,
            "Grouped runner/config modules are missing")
    require([r["path"] for r in cfg["implementation_references"]] == modules
            and [r["path"] for r in cfg["test_references"]] == tests,
            "Grouped implementation/test pin inventory changed")
    require(Path(__file__).resolve() == resolve("repo:src/abvid/cast/grouped_execution_config.py"), "Loader is not the pinned repository module")
    for ref in [cfg[k] for k in ("design_config", "design_lock", "readiness_config", "readiness_lock", "numerical_contract", "chain_contract")]:
        checked(ref)
    for ref in [*cfg["implementation_references"], *cfg["test_references"]]:
        checked(ref)
    artifact_path(cfg["output"])
    return cfg


def metadata_bundle(lock_reference, expected_status):
    """Check an existing pinned metadata bundle without following extra assets."""
    require(lock_reference["path"].endswith("/lock.json"), "Expected a pinned metadata lock")
    directory = artifact_path(lock_reference["path"].rsplit("/", 1)[0])
    _no_symlinks(directory/"lock.json")
    lock = _json(checked(lock_reference))
    require(lock["status"] == expected_status, "Unexpected existing metadata completion status")
    hashes = lock["payload_sha256"]
    require(isinstance(hashes, dict) and lock["payload_identity_sha256"] == grouped_design.digest(hashes), "Existing metadata lock identity differs")
    for name, value in hashes.items():
        part = PurePosixPath(name)
        require(str(part) == name and not part.is_absolute() and ".." not in part.parts
                and "\\" not in name and part.suffix in METADATA_SUFFIXES, "Unsafe or nonmetadata payload in execution plan")
        require(isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef"), "Invalid payload hash")
    # Inventory first, before any payload read. Directory symlinks are rejected
    # too; Path.rglob does not provide permission to traverse an unlisted tree.
    actual = set()
    for path in directory.rglob("*"):
        _no_symlinks(path)
        mode = path.lstat().st_mode
        require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), "Nonregular metadata bundle entry")
        if stat.S_ISREG(mode):
            actual.add(path.relative_to(directory).as_posix())
    require(actual == set(hashes) | {"lock.json"}, "Existing metadata inventory changed")
    for name, expected in hashes.items():
        require(sha256(directory/name) == expected, f"Existing metadata payload changed: {name}")
    return lock, {name: (directory/name).read_bytes() for name in hashes}


def validate_metadata(cfg, roles, partitions, planned, sources, chain, records, summary):
    expected = cfg["expected_counts"]
    require(len(sources) == expected["source_recordings"], "Source inventory count changed")
    require(len(partitions) == expected["partitions"] and {p["id"] for p in partitions} == set(roles), "Partition inventory changed")
    for part in partitions:
        role = roles[part["id"]]
        require(part["train_ids"] == list(role.train_ids) and part["held_ids"] == list(role.held_ids)
                and part["held_groups"] == list(role.held_groups), "Fold roles or original training order changed")
        require(len(role.train_ids) == expected["training_parents_per_partition"], "Fold budget changed")
        for parent in role.train_ids + role.held_ids:
            require(sources[parent] == role.source_metadata(parent), "Frozen source metadata changed")
    require(planned == list(grouped_design.planned_schedule(partitions))
            and len(planned) == expected["generated_waveforms"], "Frozen planned sampling schedule changed")
    require(set(records) == set(chain["candidate_ids"]) and len(records) == expected["eligible_cached_parents"], "Readiness candidate set changed")
    for row in records.values():
        require(row["eligible"] is True and row["failures"] == [] and row["reconstruction_reuse_allowed"] is False,
                "Readiness does not authorize this parent-parameter reuse")
    require(summary["status"] == "existing_fit_audit_complete" and summary["validation"]["full_budget_parity"] is True,
            "Completed full-budget parity and saved-parent audit are required")
    require(summary["eligible_candidate_count"] == len(records)
            and summary["failed_candidate_count"] == 0 and not summary["shared_audit"]["shared_failures"], "Readiness summary is inconsistent")
    require(summary["design_identity_sha256"] == next(iter(roles.values())).design_identity_sha256,
            "Readiness belongs to another grouped design")
    union = {p for role in roles.values() for p in role.train_ids}
    require(set(records) <= union and len(union) == expected["unique_training_parents"]
            and len(union-set(records)) == expected["new_parent_fits"], "Union fitting/reuse budget changed")
    return union


def load_context(reference=CONFIG):
    cfg = load_contract(reference)
    role_map = role_api.load_roles(cfg["design_config"]["path"])
    audit_cfg = readiness.load_contract(cfg["readiness_config"]["path"])
    require(audit_cfg["design_config"] == cfg["design_config"] and audit_cfg["design_lock"] == cfg["design_lock"]
            and audit_cfg["chain_contract"] == cfg["chain_contract"], "Execution/readiness input pins differ")
    require(cfg["readiness_lock"]["path"] == audit_cfg["output"]+"/lock.json", "Readiness artifact location differs")
    design_lock, design_payload = metadata_bundle(cfg["design_lock"], "metadata_only_complete_scientific_execution_unrun")
    audit_lock, audit_payload = metadata_bundle(cfg["readiness_lock"], "saved_fit_audit_no_new_scientific_run")
    require(audit_lock["environment"] == cfg["expected_environment"], "Stored readiness environment differs")
    require(audit_payload["audit_config.json"] == checked(cfg["readiness_config"]).read_bytes()
            and audit_payload["chain_contract.json"] == checked(cfg["chain_contract"]).read_bytes(), "Stored readiness contracts differ")
    numerical_path = checked(cfg["numerical_contract"])
    require(design_payload["inputs/numerical_contract.json"] == numerical_path.read_bytes(), "Numerical recipe differs from frozen design")
    require(design_payload["canonical_config.json"] == checked(cfg["design_config"]).read_bytes(), "Canonical design config differs")
    source_rows = _jsonl_bytes(design_payload["sources.jsonl"])
    sources = {r["file_id"]: r for r in source_rows}
    require(len(sources) == len(source_rows), "Duplicate source rows")
    partitions = json.loads(design_payload["folds.json"])
    planned = _jsonl_bytes(design_payload["planned_generation.jsonl"])
    chain = _json(checked(cfg["chain_contract"]))
    rows = _jsonl_bytes(audit_payload["parents.jsonl"])
    records = {r["file_id"]: r for r in rows}
    require(len(records) == len(rows), "Duplicate readiness parent rows")
    summary = json.loads(audit_payload["summary.json"])
    union = validate_metadata(cfg, role_map, partitions, planned, sources, chain, records, summary)
    output = artifact_path(cfg["output"])
    plan = {"schema_version": 1, "id": cfg["id"], "status": "metadata_plan_only_no_execution",
            "counts": dict(cfg["expected_counts"]), "output": cfg["output"],
            "configuration": {"path": reference, "sha256": sha256(resolve(reference))},
            "design_identity_sha256": design_lock["payload_identity_sha256"],
            "readiness_identity_sha256": audit_lock["payload_identity_sha256"],
            "references": {key: cfg[key] for key in ("design_config", "design_lock", "readiness_config", "readiness_lock", "numerical_contract", "chain_contract")},
            "partitions": [{"id": p["id"], "held_groups": p["held_groups"],
                            "training_count": len(p["train_ids"]), "held_count": len(p["held_ids"]),
                            "cached_training_parents": len(set(p["train_ids"]) & set(records))}
                           for p in partitions],
            "boundaries": {"datasets": ["IDMT_V1"], "classes": ["car", "truck"],
                           "raw_audio_opened_by_plan": False, "parameter_banks_opened_by_plan": False,
                           "arrays_opened_by_plan": False, "fitting_performed": False,
                           "generation_performed": False, "scoring_performed": False,
                           "target_domain_access": False, "classifier_training": False,
                           "reconstruction_waveform_reuse": False,
                           "population_statistics_reuse": False,
                           "future_parent_access": "exact selected training union; each population uses its own complete role",
                           "future_held_access": "source-domain scoring only after immutable complete generation"},
            "missing_parent_ids_sha256": grouped_design.ids_digest(sorted(union-set(records)))}
    return {"config": cfg, "config_reference": reference, "config_sha256": plan["configuration"]["sha256"],
            "roles": role_map, "numerical": _json(numerical_path), "partitions": partitions,
            "planned": planned, "sources": sources, "chain": chain,
            "readiness_records": records, "output": output, "plan": plan}

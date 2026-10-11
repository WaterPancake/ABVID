"""Audit existing CAST fits for frozen roles; never fit, render or score data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

from abvid.cast.grouped_design import CONFIG as DESIGN_CONFIG, digest, encoded
from abvid.cast.roles import load_roles
from abvid.paths import resolve, roots
from abvid.provenance.hashing import sha256


CONFIG = "repo:experiments/cast_coverage/grouped_coverage/readiness/audit.json"
FIELDS = {"schema_version", "id", "scope", "design_config", "design_lock",
          "chain_contract", "implementation_references", "validation_references",
          "expected_environment", "output", "entrypoint"}


def checked(reference):
    if set(reference) != {"path", "sha256"}:
        raise ValueError("Audit inputs require an exact path and hash")
    path = resolve(reference["path"])
    if sha256(path) != reference["sha256"]:
        raise ValueError(f"Changed readiness input: {reference['path']}")
    return path


def environment():
    return {"python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "scipy", "torch", "soundfile")}}


def load_contract(reference=CONFIG):
    if reference != CONFIG:
        raise ValueError("Use the fixed readiness audit configuration")
    path = resolve(reference)
    cfg = json.loads(path.read_text())
    if (set(cfg) != FIELDS or cfg["schema_version"] != 1 or
            cfg["id"] != "cast4a_fit_readiness_v1" or
            cfg["scope"] != "existing_source_fit_audit_and_synthetic_parity_only"):
        raise ValueError("Unexpected readiness contract")
    if cfg["design_config"]["path"] != DESIGN_CONFIG:
        raise ValueError("Audit must use the exact frozen grouped design")
    if Path(__file__).resolve() != checked(cfg["entrypoint"]):
        raise ValueError("Readiness module does not match the pinned entry point")
    for ref in [cfg["design_config"], cfg["design_lock"], cfg["chain_contract"],
                *cfg["implementation_references"], *cfg["validation_references"]]:
        checked(ref)
    if environment() != cfg["expected_environment"]:
        raise ValueError("Audit requires its frozen numerical environment")
    return cfg


def validation_gate(references):
    """Require completed focused tests and actual original/package parity evidence."""
    junit = [checked(r) for r in references if r["path"].endswith(".xml")]
    evidence = [checked(r) for r in references if r["path"].endswith(".json")]
    if len(junit) != 1 or not evidence:
        raise ValueError("Focused test and numerical parity evidence are required")
    root = ET.parse(junit[0]).getroot()
    cases = list(root.iter("testcase"))
    if not cases or any(any(c.tag in {"failure", "error", "skipped"} for c in case) for case in cases):
        raise ValueError("Readiness requires a complete passing test schedule")
    parity = [json.loads(p.read_text()) for p in evidence]
    # Evidence is exported by the isolated original-layout parity test. It must
    # include the complete production iteration budget, not only smoke checks.
    complete = []
    for proof in parity:
        if (proof.get("exact_parity_excluding_elapsed_seconds") is not True or
                proof.get("real_audio_opened") is not False or
                proof.get("scope") != "synthetic_fixture_packaging_parity_only"):
            raise ValueError("Parity evidence includes a failed or out-of-scope case")
        old, new = dict(proof["archive"]), dict(proof["package"])
        old.pop("fit_seconds")
        new.pop("fit_seconds")
        if old != new:
            raise ValueError("Archived and packaged numerical evidence differs")
        for field, path in (
            ("packaged_fit_sha256", "repo:src/abvid/cast/smooth_fit.py"),
            ("test_sha256", "repo:tests/cast/test_smooth_fit.py"),
            ("archived_engine_sha256", "archive:CAST/improvement/src/cast_improvement/engine.py"),
        ):
            checked({"path": path, "sha256": proof[field]})
        steps = proof["request"]["config"]["optimizer"]["steps"]
        complete.append(steps == 300 and len(new["result"]["starts"]) == 4
                        and len(new["result"]["history"]) == 301)
    if not any(complete):
        raise ValueError("Missing full-budget smooth8 parity evidence")
    return {"passed_tests": len(cases), "parity_cases": len(parity),
            "full_budget_parity": True, "scientific_experiment_run": False}


def evaluate(cfg):
    validation = validation_gate(cfg["validation_references"])
    roles = load_roles(cfg["design_config"]["path"])
    # Imported only after config, design, source snapshot and validation gates.
    from abvid.cast.reuse_chain import audit_saved_parents
    chain = json.loads(checked(cfg["chain_contract"]).read_text())
    records, shared = audit_saved_parents(chain)
    expected = set(chain["candidate_ids"])
    if len(records) != len(expected) or {r["file_id"] for r in records} != expected:
        raise ValueError("Saved-fit audit did not retain every candidate exactly once")
    if any(type(r.get("eligible")) is not bool or not isinstance(r.get("failures"), list) for r in records):
        raise ValueError("Audit records must report explicit eligibility and failures")
    if any(r["eligible"] and r["failures"] for r in records):
        raise ValueError("Failed candidate cannot be eligible")
    if shared.get("shared_failures") and any(r["eligible"] for r in records):
        raise ValueError("Shared provenance failure must block reuse")
    eligible = {r["file_id"] for r in records if r["eligible"]}
    fold_records = []
    for role in roles.values():
        reusable = []
        for parent in role.train_ids:
            if parent in eligible:
                role.validate_parent(parent, ancestry=[role.source_metadata(parent)])
                reusable.append(parent)
        fold_records.append({"partition": role.partition_id, "held_groups": list(role.held_groups),
                             "training_count": len(role.train_ids), "eligible_cached_ids": reusable,
                             "requires_new_fit_or_other_audited_cache": [i for i in role.train_ids if i not in eligible],
                             "population_statistics_reusable": False,
                             "later_calibration_waveforms_reusable": False,
                             "role": "per_parent_cache_eligibility_only_no_bank_constructed"})
    union = {i for role in roles.values() for i in role.train_ids}
    summary = {"id": cfg["id"], "status": "existing_fit_audit_complete",
               "validation": validation, "candidate_count": len(records),
               "eligible_candidate_count": len(eligible), "failed_candidate_count": len(records) - len(eligible),
               "unique_selected_training_ids": len(union),
               "unique_selected_ids_without_eligible_cache": len(union - eligible),
               "fold_counts": [{"partition": f["partition"], "eligible_cached_parents": len(f["eligible_cached_ids"]),
                                "remaining_parents": len(f["requires_new_fit_or_other_audited_cache"])} for f in fold_records],
               "shared_audit": shared,
               "design_identity_sha256": next(iter(roles.values())).design_identity_sha256,
               "interpretation": "same-parent saved computation eligibility; not physical identity, population coverage or transfer",
               "reuse_scope": "anchored per-parent fitted/calibrated parameter records; later cached reconstruction waveforms excluded",
               "new_real_data_fits": 0, "generated_research_observations": 0, "coverage_metrics_computed": False,
               "remaining_work": ["new selected-parent fitting under a separately scoped scientific run",
                                  "every fold rebuilds its own bank statistics, scales, children and prototypes",
                                  "the full grouped generation/evaluation/replay experiment remains unrun"]}
    payload = {"summary.json": encoded(summary), "parents.jsonl": b"".join(encoded(r) for r in sorted(records, key=lambda x: x["file_id"])),
               "fold_reuse.json": encoded(fold_records), "audit_config.json": resolve(CONFIG).read_bytes(),
               "chain_contract.json": checked(cfg["chain_contract"]).read_bytes()}
    for ref in [cfg["entrypoint"], *cfg["implementation_references"], *cfg["validation_references"]]:
        path = checked(ref)
        label = ref["path"].replace(":", "/", 1)
        payload["evidence/" + label] = path.read_bytes()
    return payload


def destination(cfg):
    if not cfg["output"].startswith("artifact:"):
        raise ValueError("Readiness artifacts must remain outside the checkout")
    path = resolve(cfg["output"], must_exist=False)
    if path == roots()["artifact"] or path.is_relative_to(roots()["repo"]) or path.is_relative_to(roots()["archive"]):
        raise ValueError("Cannot write checkout, archive or artifact root")
    return path


def verify(directory, payload):
    lock = json.loads((directory / "lock.json").read_text())
    if lock["status"] != "saved_fit_audit_no_new_scientific_run":
        raise ValueError("Readiness lock has an invalid completion status")
    files = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if files != set(payload) | {"lock.json"}:
        raise ValueError("Readiness artifact inventory changed")
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in payload.items()}
    if lock["payload_sha256"] != hashes or lock["payload_identity_sha256"] != digest(hashes):
        raise ValueError("Readiness payload identity changed")
    for name, content in payload.items():
        if (directory / name).is_symlink() or (directory / name).read_bytes() != content:
            raise ValueError(f"Audit replay mismatch: {name}")
    return lock


def save(directory, payload, seconds):
    if directory.exists():
        return verify(directory, payload)
    directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".cast4a-audit-", dir=directory.parent))
    try:
        for name, content in payload.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        git = lambda *args: subprocess.check_output(["git", *args], cwd=roots()["repo"], text=True).strip()
        hashes = {name: hashlib.sha256(content).hexdigest() for name, content in payload.items()}
        lock = {"status": "saved_fit_audit_no_new_scientific_run", "created_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": git("rev-parse", "HEAD"), "dirty_status": git("status", "--porcelain").splitlines(),
                "environment": environment(), "elapsed_seconds": seconds,
                "payload_sha256": hashes, "payload_identity_sha256": digest(hashes)}
        (stage / "lock.json").write_bytes(encoded(lock))
        verify(stage, payload)
        os.rename(stage, directory)
        return lock
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "verify"))
    parser.add_argument("--config", required=True, choices=(CONFIG,))
    args = parser.parse_args(argv)
    try:
        started = time.perf_counter()
        cfg = load_contract(args.config)
        payload = evaluate(cfg)
        output = destination(cfg)
        lock = save(output, payload, time.perf_counter() - started) if args.action == "audit" else verify(output, payload)
        summary = json.loads(payload["summary.json"])
        print(json.dumps({"status": summary["status"], "output": str(output),
                          "eligible": summary["eligible_candidate_count"], "candidates": summary["candidate_count"],
                          "payload_identity_sha256": lock["payload_identity_sha256"]}))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

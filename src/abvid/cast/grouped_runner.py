"""Resumable execution of the frozen CAST-4A source-descriptor study.

The CLI is metadata-only unless --execute is present. Each numerical stage is
create-only and hash-verified; replay never fits parents or fills missing work.
No target dataset, audio embedding, classifier, or parameter search is used.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
import traceback
import uuid

import numpy as np
import soundfile as sf
import torch

from abvid.cast import coverage
from abvid.cast.audio import normalize, observation
from abvid.cast.config import digest
from abvid.cast.grouped_design import planned_schedule
from abvid.cast.grouped_metrics import summarize
from abvid.cast.grouped_population import build_population
from abvid.cast.grouped_storage import read_bundle, verify_bundle, write_bundle


class StageError(RuntimeError):
    """A terminal fold failure; unchanged work in other folds can still finish."""

    def __init__(self, stage, cause):
        super().__init__(f"{stage}: {type(cause).__name__}: {cause}")
        self.stage = stage


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def identity(value):
    return sha(json_bytes(value))


def _objects(folder):
    metadata, payload = read_bundle(folder)
    return metadata, {k: json.loads(v) for k, v in payload.items() if k.endswith('.json')}


def _checkpoint(folder, payload, metadata, *, replay_only):
    """Compare exact bytes on replay; immutable publication otherwise."""
    if replay_only:
        actual_meta, actual_payload = read_bundle(folder)
        if actual_meta != metadata or actual_payload != payload:
            raise ValueError(f"Replay differs: {folder}")
        return verify_bundle(folder)
    return write_bundle(folder, payload, metadata)


def _metadata(context, role, stage, **extra):
    return {"kind": "cast_grouped_coverage_v1", "stage": stage,
            "config_sha256": context["config_sha256"],
            "partition": role.partition_id,
            "design_identity_sha256": role.design_identity_sha256, **extra}


def _schedule(context, role):
    records = [r for r in context["planned"] if r["partition"] == role.partition_id]
    if records != list(planned_schedule([{"id": role.partition_id}])):
        raise ValueError("Generation requires the exact frozen 3,000-sample schedule")
    return records


def _sample_path(folder, planned):
    # Hashing the complete planned identifier avoids unsafe file-name characters.
    return folder / "samples" / sha(planned["id"].encode())


def sample_payload(population, planned):
    """Render one exact schedule entry with complete lineage and descriptors."""
    record, wave, descriptors = population.draw(planned)
    wave = np.asarray(wave)
    if wave.dtype != np.float32 or wave.shape != (32000,) or not np.isfinite(wave).all():
        raise ValueError("Expected finite float32 two-second generated waveform")
    descriptors = {k: np.asarray(v).tolist() for k, v in descriptors.items()}
    stream = BytesIO()
    sf.write(stream, wave, 16000, format="WAV", subtype="FLOAT")
    audio = stream.getvalue()
    restored, sr = sf.read(BytesIO(audio), dtype="float32")
    if sr != 16000 or not np.array_equal(restored, wave):
        raise ValueError("Generated FLOAT WAV round trip differs")
    payload = {"record.json": json_bytes(record), "descriptors.json": json_bytes(descriptors),
               "waveform.wav": audio, "planned.json": json_bytes(planned)}
    return payload, {"planned_id": planned["id"], "population_fingerprint": population.fingerprint,
                     "waveform_float32_sha256": sha(wave.tobytes())}


def prepare_population(context, role, *, replay_only=False):
    """Rebuild all class/group statistics using only the literal training IDs."""
    from abvid.cast.grouped_parents import load_parent

    bank, observed = [], []
    for file_id in role.train_ids:
        row, obs = load_parent(context, file_id, role)
        bank.append(row)
        observed.append(obs)
    role.validate_bank([r["parent"] for r in bank])
    population = build_population(bank, observed, role, context["numerical"])
    folder = context["output"] / "folds" / role.partition_id
    _checkpoint(folder / "population", {
        "bank.json": json_bytes(bank), "observations.json": json_bytes(observed),
        "statistics.json": json_bytes(population.statistics),
        "scales.json": json_bytes(population.scales)},
        _metadata(context, role, "training_population", fingerprint=population.fingerprint),
        replay_only=replay_only)
    return population


def generation_gate(context, role, population, *, replay_only=False):
    """Finish and replay every planned waveform before admitting held access."""
    folder = context["output"] / "folds" / role.partition_id
    meta, saved = _objects(folder / "population")
    if meta != _metadata(context, role, "training_population", fingerprint=population.fingerprint):
        raise ValueError("Population must be frozen before generation")
    statistics = population.statistics
    if (saved.get("statistics.json") != statistics or saved.get("scales.json") != population.scales or
            digest(statistics) != population.fingerprint or
            digest(sorted(saved["bank.json"], key=lambda r: r["file_id"])) != statistics["bank_sha256"] or
            digest(sorted(saved["observations.json"], key=lambda r: r["file_id"])) != statistics["observations_sha256"]):
        raise ValueError("Frozen population statistics or training inputs differ from the rebuilt population")
    planned = _schedule(context, role)
    if not replay_only:
        for item in planned:
            path = _sample_path(folder, item)
            if path.exists():
                # Existing samples are regenerated and compared in the replay below.
                verify_bundle(path)
                continue
            payload, details = sample_payload(population, item)
            _checkpoint(path, payload, _metadata(context, role, "sample", **details), replay_only=False)
    identities = {}
    for item in planned:
        # A missing sample fails here during replay, never silently regenerates.
        payload, details = sample_payload(population, item)
        result = _checkpoint(_sample_path(folder, item), payload,
                             _metadata(context, role, "sample", **details), replay_only=True)
        identities[item["id"]] = result["identity_sha256"]
    receipt = {"population_fingerprint": population.fingerprint,
               "population_bundle_identity": verify_bundle(folder / "population")["identity_sha256"],
               "schedule_sha256": identity(planned), "sample_identities": identities,
               "count": len(planned), "complete": True, "replay": "exact_all_samples"}
    _checkpoint(folder / "generation", {"receipt.json": json_bytes(receipt)},
                _metadata(context, role, "generation_and_replay"), replay_only=replay_only)
    return receipt


def validate_generation_receipt(context, role, receipt):
    """Check persisted evidence and every sample before opening any held source."""
    folder = context["output"] / "folds" / role.partition_id
    metadata, payload = _objects(folder / "generation")
    if (metadata != _metadata(context, role, "generation_and_replay") or
            payload != {"receipt.json": receipt}):
        raise ValueError("Held access requires the persisted generation replay receipt")
    pop_meta, _ = _objects(folder / "population")
    planned = _schedule(context, role)
    if (pop_meta != _metadata(context, role, "training_population",
                             fingerprint=receipt.get("population_fingerprint")) or
            receipt.get("population_bundle_identity") != verify_bundle(folder / "population")["identity_sha256"] or
            receipt.get("schedule_sha256") != identity(planned) or
            receipt.get("count") != len(planned) or receipt.get("complete") is not True or
            receipt.get("replay") != "exact_all_samples" or
            set(receipt.get("sample_identities", {})) != {r["id"] for r in planned}):
        raise ValueError("Incomplete or mismatched generation gate")
    for item in planned:
        result = verify_bundle(_sample_path(folder, item))
        if result["identity_sha256"] != receipt["sample_identities"][item["id"]]:
            raise ValueError("Generation artifact changed after replay")
    return identity(receipt)


def _held_observation(context, role, file_id):
    from abvid.cast.grouped_parents import read_source

    if file_id not in role.held_ids or file_id in role.train_ids:
        raise ValueError("Source is not in the exact held role")
    source = role.source_metadata(file_id)
    if source != context["sources"][file_id] or source["provenance_group_id"] not in role.held_groups:
        raise ValueError("Held metadata differs from frozen role")
    raw, sample_rate = read_source(context, file_id)
    cfg = context["numerical"]["renderer_configs"]["smooth8_v1"]
    original = observation(raw, sample_rate, cfg)
    target, normalization = normalize(original, cfg)
    descriptors = coverage.describe(target.numpy(), cfg, context["numerical"]["metric_config"])
    return {"file_id": file_id, "source_metadata": source, "normalization": normalization,
            "descriptors": {k: v.tolist() for k, v in descriptors.items()}}


def held_descriptors(context, role, receipt, *, replay_only=False):
    gate_sha = validate_generation_receipt(context, role, receipt)
    folder = context["output"] / "folds" / role.partition_id
    result, identities = {}, {}
    for file_id in role.held_ids:
        path = folder / "held" / file_id
        meta = _metadata(context, role, "held_descriptor", generation_receipt_sha256=gate_sha,
                         file_id=file_id)
        if not replay_only and not path.exists():
            value = _held_observation(context, role, file_id)
            _checkpoint(path, {"observation.json": json_bytes(value)}, meta, replay_only=False)
        # Independent decode and descriptor recomputation, including on resume.
        value = _held_observation(context, role, file_id)
        manifest = _checkpoint(path, {"observation.json": json_bytes(value)}, meta, replay_only=True)
        identities[file_id] = manifest["identity_sha256"]
        result[file_id] = value
    _checkpoint(folder / "held_replay", {"receipt.json": json_bytes({
        "generation_receipt_sha256": gate_sha, "held_identities": identities,
        "complete": True, "replay": "exact_all_held_descriptors"})},
        _metadata(context, role, "held_descriptor_replay"), replay_only=replay_only)
    return result


def score_fold(context, role, population, held, receipt):
    metric = context["numerical"]["metric_config"]
    if set(held) != set(role.held_ids):
        raise ValueError("Scoring requires the complete held role")
    folder = context["output"] / "folds" / role.partition_id
    samples = {}
    for item in _schedule(context, role):
        manifest = verify_bundle(_sample_path(folder, item))
        if manifest["identity_sha256"] != receipt["sample_identities"][item["id"]]:
            raise ValueError("Sample changed before scoring")
        _, payload = _objects(_sample_path(folder, item))
        key = item["class"], item["seed"], item["arm"]
        samples.setdefault(key, []).append(payload["descriptors.json"])
    records = []
    for (label, seed, arm), generated in sorted(samples.items()):
        observed = [held[file_id]["descriptors"] for file_id in role.held_ids
                    if context["sources"][file_id]["canonical_class"] == label]
        score = coverage.compare(coverage.stack(generated, metric), coverage.stack(observed, metric),
                                 population.scales[label], metric)
        finite_spread = all(v["spread_ratio"] is not None for v in score["families"].values())
        records.append({"partition": role.partition_id, "class": label, "seed": seed, "arm": arm,
                        "score": score, "gates": {"input": True, "completeness": True,
                        "numerical": finite_spread, "ancestry": True, "replay": True},
                        "failures": [] if finite_spread else ["undefined_held_spread_ratio"]})
    return records


def run_fold(context, role, *, replay_only=False):
    stage = "population"
    try:
        population = prepare_population(context, role, replay_only=replay_only)
        stage = "generation"
        receipt = generation_gate(context, role, population, replay_only=replay_only)
        stage = "held_descriptors"
        held = held_descriptors(context, role, receipt, replay_only=replay_only)
        stage = "scoring"
        records = score_fold(context, role, population, held, receipt)
        _checkpoint(context["output"] / "folds" / role.partition_id / "scores",
                    {"scores.json": json_bytes(records)}, _metadata(context, role, "scores"),
                    replay_only=replay_only)
        return records
    except Exception as exc:
        raise StageError(stage, exc) from exc


def deterministic_cpu():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def _runtime():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor() or "unknown", "cpu_count": os.cpu_count(),
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "soundfile": sf.__version__, "device": "cpu", "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "parent_process_peak_rss_bytes": peak if platform.system() == "Darwin" else peak * 1024,
            "worker_memory": "per-parent receipts; parent RSS excludes child processes"}


def _git():
    def command(*args):
        return subprocess.check_output(["git", *args], text=True).strip()
    try:
        return {"commit": command("rev-parse", "HEAD"),
                "dirty_status": command("status", "--porcelain"),
                "diff_sha256": sha(subprocess.check_output(["git", "diff", "--binary", "HEAD"]))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty_status": "unknown", "diff_sha256": "unknown"}


def artifact_inventory(root):
    """Anchor all scientific bundles; operational attempt logs remain separate."""
    result = {}
    for manifest in sorted(root.rglob("_bundle.json")):
        relative = manifest.parent.relative_to(root)
        if relative.parts[0] in {"attempts", "results"} or any(
                part.startswith(".attempt-") for part in relative.parts):
            continue
        result[relative.as_posix()] = verify_bundle(manifest.parent)["identity_sha256"]
    return result


@contextmanager
def exclusive_run(root):
    """OS lock releases on interruption; no stale-PID recovery or deletion needed."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".execution.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another runner owns this output directory") from exc
        yield


def _run_unlocked(context, *, replay_only):
    from abvid.cast.grouped_parents import ensure_parents
    from abvid.paths import resolve

    deterministic_cpu()
    root = context["output"]
    contract = {"config_sha256": context["config_sha256"], "kind": "cast4a_execution_context_v1"}
    snapshot = root / "context"
    source_snapshot = {
        "implementation/" + ref["path"].split(":", 1)[1]: resolve(ref["path"]).read_bytes()
        for ref in [*context["config"].get("implementation_references", []),
                    *context["config"].get("test_references", [])]}
    if snapshot.exists():
        metadata, payload = read_bundle(snapshot)
        if (metadata != contract or payload["config.json"] != json_bytes(context["config"]) or
                payload["plan.json"] != json_bytes(context["plan"]) or
                {k: v for k, v in payload.items() if k.startswith("implementation/")} != source_snapshot):
            raise ValueError("Run context differs from immutable execution snapshot")
    elif replay_only:
        raise FileNotFoundError("Replay requires an existing run context")
    else:
        payload = {"config.json": json_bytes(context["config"]), "plan.json": json_bytes(context["plan"]),
                   "environment.json": json_bytes(_runtime()), "git.json": json_bytes(_git())}
        # Exact current implementation bytes are saved alongside their pinned hashes.
        payload.update(source_snapshot)
        write_bundle(snapshot, payload, contract)
    parents = ensure_parents(context, replay_only=replay_only)
    records, failures = [], []
    for role in context["roles"].values():
        failure_path = root / "folds" / role.partition_id / "failure"
        if failure_path.exists():
            metadata, payload = _objects(failure_path)
            if metadata != _metadata(context, role, "terminal_failure"):
                raise ValueError("Failure belongs to another execution context")
            failures.append(payload["failure.json"])
            continue
        try:
            records.extend(run_fold(context, role, replay_only=replay_only))
        except StageError as exc:
            if replay_only:
                # A broken replay must fail, never revise the historical result.
                raise
            value = {"partition": role.partition_id, "stage": exc.stage,
                     "error": str(exc), "traceback": traceback.format_exc(),
                     "status": "technical_failure", "replacement": False}
            write_bundle(failure_path, {"failure.json": json_bytes(value)},
                         _metadata(context, role, "terminal_failure"))
            failures.append(value)
    summary = summarize(records, context["partitions"], context["numerical"]["metric_config"])
    result = {"summary": summary, "fold_failures": failures, "parents": parents,
              "artifact_inventory": artifact_inventory(root),
              "interpretation": "source-descriptor coverage on exposed IDMT development groups; not classifier transfer"}
    _checkpoint(root / "results", {"results.json": json_bytes(result), "scores.json": json_bytes(records)},
                contract, replay_only=replay_only)
    return result


def run(context, *, replay_only=False):
    """Execute the entire fixed schedule, or verify it without filling omissions."""
    started = time.perf_counter()
    root = context["output"]
    if replay_only and not root.is_dir():
        raise FileNotFoundError("No execution artifacts exist to replay")
    with exclusive_run(root):
        attempt_id = str(uuid.uuid4())
        attempt = {"attempt_id": attempt_id, "replay_only": replay_only,
                   "config_sha256": context["config_sha256"], "started_unix": time.time(),
                   "existing_attempts": sorted(p.name for p in (root / "attempts").glob("*"))}
        write_bundle(root / "attempts" / attempt_id / "start", {"attempt.json": json_bytes(attempt)},
                     {"kind": "cast4a_attempt_v1", "attempt_id": attempt_id})
        status = "interrupted_or_failed"
        try:
            result = _run_unlocked(context, replay_only=replay_only)
            status = result["summary"]["status"]
            return result
        finally:
            elapsed = time.perf_counter() - started
            write_bundle(root / "attempts" / attempt_id / "finish", {"runtime.json": json_bytes({
                **_runtime(), "elapsed_seconds": elapsed, "status": status,
                "abandoned_staging_directories": sorted(str(p.relative_to(root))
                    for p in root.rglob(".attempt-*"))})},
                {"kind": "cast4a_attempt_v1", "attempt_id": attempt_id})


def main(argv=None):
    from abvid.cast.grouped_execution_config import CONFIG, load_context

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "verify"))
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--execute", action="store_true", help="perform the declared scientific run or replay")
    args = parser.parse_args(argv)
    if args.command == "plan" and args.execute:
        parser.error("plan is metadata-only and cannot use --execute")
    context = load_context(args.config)
    if not args.execute:
        print(json_bytes({**context["plan"], "requested_command": args.command,
                          "executed": False}).decode(), end="")
        return 0
    result = run(context, replay_only=args.command == "verify")
    print(json_bytes({"output": str(context["output"]), "status": result["summary"]["status"],
                      "fold_failures": result["fold_failures"]}).decode(), end="")
    return 0 if result["summary"]["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

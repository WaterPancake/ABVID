"""Guarded, immutable per-parent preparation for grouped CAST execution.

This module is callable infrastructure, not authorization to run an experiment.
The validated execution context controls every input, training role and output.
Only same-parent artifacts are cached; no population statistics are reused.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import math
import multiprocessing
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import soundfile as sf
import torch

from abvid.cast import reuse_chain
from abvid.cast.grouped_parent import ParentCalibrationError, calibrate_parent, prepare_observation
from abvid.cast.grouped_storage import read_bundle, write_bundle
from abvid.paths import resolve, roots
from abvid.provenance.hashing import canonical


_WORKER_CONTEXT = None


def _digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _id(file_id):
    if (not isinstance(file_id, str) or not file_id or file_id in (".", "..") or
            Path(file_id).name != file_id or "\\" in file_id or "\0" in file_id):
        raise ValueError("Unsafe parent identifier")
    return file_id


def _folder(context, file_id):
    return Path(context["output"]) / "parents" / _id(file_id)


def _witness(context, file_id):
    candidates = [role for _, role in sorted(context["roles"].items()) if file_id in role.train_ids]
    if not candidates:
        raise ValueError("Parent has no selected training role")
    role = candidates[0]
    role.validate_parent(context["sources"][file_id])
    return role


def _metadata(context, file_id, status):
    role = _witness(context, file_id)
    prior = context["readiness_records"].get(file_id)
    return {"kind": "cast_grouped_parent_v1", "status": status, "file_id": file_id,
            "config_sha256": context["config_sha256"],
            "numerical_sha256": _digest(context["numerical"]),
            "source_metadata_sha256": _digest(context["sources"][file_id]),
            "design_identity_sha256": role.design_identity_sha256,
            "witness_partition": role.partition_id,
            "origin": "audited_historical_v11" if prior is not None else "new_same_parent_fit",
            "readiness_record_sha256": _digest(prior) if prior is not None else None}


def _reference_path(reference):
    """Screen lexical descendants before resolve dereferences any path link."""
    scheme, separator, relative = reference.partition(":")
    locations = roots()
    if (not separator or scheme not in locations or not relative or
            Path(relative).is_absolute() or ".." in Path(relative).parts):
        raise ValueError("Invalid source storage reference")
    lexical = locations[scheme]
    for part in Path(relative).parts:
        lexical = lexical / part
        if lexical.is_symlink():
            raise ValueError("Symlink in source artifact path")
    resolved = resolve(reference)
    if lexical.resolve() != resolved or not resolved.is_file():
        raise ValueError("Source artifact escaped its configured storage path")
    return resolved


def _verified_raw_bytes(context, file_id):
    source = context["sources"][_id(file_id)]
    if (source.get("dataset_id") != "IDMT" or source.get("release_id") != "IDMT_V1" or
            source.get("canonical_class") not in ("car", "truck") or
            source.get("provider_channel_pair") != "CH34" or source.get("original_channel_ids") != [3, 4]):
        raise ValueError("Raw source is outside the admitted IDMT car/truck channel contract")
    reference = reuse_chain.raw_reference(source, context["chain"]["raw_source_root"])
    path = _reference_path(reference)
    data = path.read_bytes()
    if len(data) != source["source_file_bytes"] or hashlib.sha256(data).hexdigest() != source["source_file_sha256"]:
        raise ValueError("Raw source bytes changed")
    info = sf.info(BytesIO(data))
    if ((info.samplerate, info.channels, info.frames) !=
            (source["native_sample_rate_hz"], source["native_channels"], source["decoded_frames"]) or
            info.channels != 2):
        raise ValueError("Raw source header changed")
    return data, source


def read_source(context, file_id):
    """Hash/header-check, then decode the exact bytes in float64 with two columns.

    The caller must enforce the current training role or the frozen-generation
    held-read gate before calling. This routine admits only the context's IDMT
    car/truck source rows; it never accepts an arbitrary filesystem filename.
    """
    data, source = _verified_raw_bytes(context, file_id)
    raw, sample_rate = sf.read(BytesIO(data), dtype="float64", always_2d=True)
    if (raw.shape != (source["decoded_frames"], 2) or sample_rate != source["native_sample_rate_hz"] or
            not np.isfinite(raw).all()):
        raise ValueError("Raw source decode differs or contains nonfinite samples")
    return raw, int(sample_rate)


def _validate_complete(context, file_id, payload, role):
    source = context["sources"][file_id]
    role.validate_parent(source)
    if json.loads(payload["source.json"]) != source:
        raise ValueError("Parent source metadata changed")
    row, observed = json.loads(payload["final_row.json"]), json.loads(payload["observation.json"])
    expected = {"file_id": file_id, "class": source["canonical_class"],
                "group": source["provenance_group_id"], "parent": source}
    if any(row.get(key) != value or observed.get(key) != value for key, value in expected.items()):
        raise ValueError("Parent row or observation crosses source ancestry")
    if row != json.loads(payload["v11.json"]):
        raise ValueError("Final parent row differs from v11")
    original_fit = json.loads(payload["fit.json"])
    cfg = context["numerical"]["renderer_configs"]["smooth8_v1"]
    reuse_chain.validate_fit(original_fit, file_id, cfg)
    fit_hash = hashlib.sha256(payload["fit.json"]).hexdigest()
    before = json.loads(payload["smooth8.json"])
    if (before["fit_sha256"] != fit_hash or before["parameters"] != original_fit["parameters"] or
            any(before.get(key) != value for key, value in expected.items())):
        raise ValueError("Original fit or smooth8 ancestry changed")
    settings = {key: context["numerical"]["expected_spectrum_calibration"][key]
                for key in ("betas", "epsilon", "gradient_clip", "lr", "steps")}
    for stage in reuse_chain.STAGES[1:]:
        after = json.loads(payload[stage + ".json"])
        stage_cfg = context["numerical"]["renderer_configs"][
            "spectrum16_temporal41_v11" if stage == "v11" else "smooth8_v1"]
        reuse_chain.validate_transition(stage, before, after, observed, fit_hash, stage_cfg, settings)
        before = after
    prior = context["readiness_records"].get(file_id)
    if prior is not None:
        if not prior["eligible"] or _digest(row) != prior["checked_hashes"][f"canonical:v11:{file_id}"]:
            raise ValueError("Reused parent differs from pinned readiness evidence")
    return row, observed


def _read_parent(context, file_id, role):
    role.validate_parent(context["sources"][file_id])
    metadata, payload = read_bundle(_folder(context, file_id))
    if metadata.get("status") not in ("complete", "technical_failure"):
        raise ValueError("Unknown parent completion status")
    if metadata != _metadata(context, file_id, metadata["status"]):
        raise ValueError("Existing parent belongs to a changed execution context")
    if json.loads(payload["source.json"]) != context["sources"][file_id]:
        raise ValueError("Cached source metadata changed")
    if metadata["status"] == "complete":
        _validate_complete(context, file_id, payload, role)
    elif "failure_record.json" not in payload:
        raise ValueError("Terminal parent failure record is missing")
    return metadata, payload


def load_parent(context, file_id, role):
    """Return a complete cached parent only after current-fold role validation."""
    metadata, payload = _read_parent(context, file_id, role)
    if metadata["status"] != "complete":
        raise ValueError("Parent has a terminal technical failure")
    return json.loads(payload["final_row.json"]), json.loads(payload["observation.json"])


def _pinned_bytes(reference, expected):
    path = _reference_path(reference)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Audited saved artifact changed before reuse")
    return data


def _audit_candidates(context, replay_only):
    expected_ids = sorted(context["readiness_records"])
    if context["chain"]["candidate_ids"] != expected_ids:
        raise ValueError("Readiness and chain candidate identities differ")
    for file_id in expected_ids:
        _witness(context, file_id).validate_parent(context["sources"][file_id])
    records, summary = reuse_chain.audit_saved_parents(context["chain"])
    if {record["file_id"]: record for record in records} != context["readiness_records"]:
        raise ValueError("Fresh saved-parent audit differs from frozen readiness records")
    if summary["shared_failures"] or any(not record["eligible"] for record in records):
        raise ValueError("Saved-parent candidate audit failed; replacement fitting is forbidden")
    payload = {"records.json": _json(records), "summary.json": _json(summary)}
    metadata = {"kind": "cast_grouped_parent_reuse_audit_v1", "config_sha256": context["config_sha256"],
                "chain_sha256": _digest(context["chain"]),
                "readiness_records_sha256": _digest(context["readiness_records"])}
    destination = Path(context["output"]) / "parent_reuse_audit"
    if replay_only:
        if read_bundle(destination) != (metadata, payload):
            raise ValueError("Recorded parent audit differs on replay")
    else:
        write_bundle(destination, payload, metadata)
    return records, summary


def _reuse_assets(context, summary):
    checked = summary["shared_checked_hashes"]
    banks = {}
    for stage, name in reuse_chain.BANKS.items():
        reference = context["chain"]["stages"][stage] + "/" + name
        rows = [json.loads(line) for line in _pinned_bytes(reference, checked[reference]).splitlines()]
        banks[stage] = {row["file_id"]: row for row in rows}
        if len(rows) != len(banks[stage]) or set(banks[stage]) != set(context["readiness_records"]):
            raise ValueError("Audited bank candidate inventory changed")
    reference = context["chain"]["stages"]["v11"] + "/observations.jsonl"
    values = [json.loads(line) for line in _pinned_bytes(reference, checked[reference]).splitlines()]
    observations = {row["file_id"]: row for row in values}
    if len(values) != len(observations) or set(observations) != set(context["readiness_records"]):
        raise ValueError("Audited observation inventory changed")
    return banks, observations


def _save_reused(context, file_id, banks, observations):
    role = _witness(context, file_id)
    role.validate_parent(context["sources"][file_id])
    audit = context["readiness_records"][file_id]
    root = context["chain"]["stages"]["smooth8"] + "/fits/" + file_id
    payload = {}
    for name in ("fit.json", "failure.json"):
        reference = root + "/" + name
        payload[name] = _pinned_bytes(reference, audit["checked_hashes"][reference])
    fit, flags = json.loads(payload["fit.json"]), json.loads(payload["failure.json"])
    source = context["sources"][file_id]
    base = {"file_id": file_id, "class": source["canonical_class"],
            "group": source["provenance_group_id"], "parent": source,
            "parameters": fit["parameters"], "flags": flags,
            "fit_sha256": hashlib.sha256(payload["fit.json"]).hexdigest()}
    payload["smooth8.json"] = _json(base)
    for stage in reuse_chain.STAGES[1:]:
        payload[stage + ".json"] = _json(banks[stage][file_id])
    payload.update({"final_row.json": _json(banks["v11"][file_id]),
                    "observation.json": _json(observations[file_id]), "source.json": _json(source),
                    "origin.json": _json({"origin": "audited_historical_v11", "readiness_record": audit,
                                          "saved_calibration_reconstruction_waveforms_reused": False,
                                          "population_statistics_reused": False})})
    _validate_complete(context, file_id, payload, role)
    write_bundle(_folder(context, file_id), payload, _metadata(context, file_id, "complete"))


def _diagnostic_json(value):
    """Preserve invalid numerical diagnostic values explicitly, never as JSON NaN."""
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else {"nonfinite_float": str(value)}
    if isinstance(value, np.ndarray):
        return {"array_dtype": str(value.dtype), "array_shape": list(value.shape),
                "values": _diagnostic_json(value.tolist())}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {key: _diagnostic_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_json(item) for item in value]
    return value


def _worker_runtime(started):
    native_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"elapsed_seconds": time.perf_counter() - started,
            "elapsed_scope": "source_validation_decode_and_calibration_before_bundle_serialization",
            "peak_rss_bytes": int(native_rss if sys.platform == "darwin" else native_rss * 1024),
            "peak_rss_scope": "worker_process_lifetime_high_water_mark",
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor() or "unknown", "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def _save_failure(context, file_id, error, partial=None):
    partial = {} if partial is None else partial
    payload = dict(partial.get("artifacts", {}))
    payload["source.json"] = _json(context["sources"][file_id])
    payload["failure_record.json"] = _json(_diagnostic_json({
        "file_id": file_id, "status": "terminal_technical_failure", "replacement_allowed": False,
        "error": {"type": type(error).__name__, "message": str(error)},
        "failed_stage": partial.get("failed_stage", "raw_source_or_worker"),
        "completed_stages": list(partial.get("stages", {})),
        "fit": partial.get("fit"), "stages": partial.get("stages", {}),
        "flags": partial.get("flags", {}), "stage_seconds": partial.get("stage_seconds", {}),
        "worker_runtime": partial.get("worker_runtime"),
        "nonfinite_diagnostic_encoding": "tagged_nonfinite_float_values_not_numeric_results"}))
    payload["origin.json"] = _json({"origin": "new_same_parent_fit", "population_statistics_reused": False})
    write_bundle(_folder(context, file_id), payload, _metadata(context, file_id, "technical_failure"))


def _configure_cpu():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def _fit_parent(context, file_id):
    role = _witness(context, file_id)
    source = context["sources"][file_id]
    role.validate_parent(source)
    result = None
    started = time.perf_counter()
    try:
        raw, sample_rate = read_source(context, file_id)
        prepared = prepare_observation(raw, sample_rate, source, context["numerical"], role=role)
        result = calibrate_parent(prepared["target"], source, context["numerical"], role=role,
                                  original=prepared["original"], normalization=prepared["normalization"])
        result["worker_runtime"] = _worker_runtime(started)
        payload = dict(result["artifacts"])
        payload.update({"final_row.json": _json(result["final_row"]),
                        "observation.json": _json(result["observation"]), "source.json": _json(source),
                        "stage_seconds.json": _json(result["stage_seconds"]),
                        "worker_runtime.json": _json(result["worker_runtime"]),
                        "origin.json": _json({"origin": "new_same_parent_fit",
                                              "population_statistics_reused": False,
                                              "all_same_parent_stages_recomputed": True})})
        _validate_complete(context, file_id, payload, role)
        write_bundle(_folder(context, file_id), payload, _metadata(context, file_id, "complete"))
        return "complete"
    except ParentCalibrationError as error:
        error.partial["worker_runtime"] = _worker_runtime(started)
        _save_failure(context, file_id, error, error.partial)
    except Exception as error:
        if _folder(context, file_id).exists():
            raise  # Never replace a published or incomplete existing directory.
        partial = {} if result is None else result
        partial["worker_runtime"] = _worker_runtime(started)
        _save_failure(context, file_id, error, partial)
    return "technical_failure"


def _worker_initialize(reference, expected_config_sha256):
    global _WORKER_CONTEXT
    from abvid.cast.grouped_execution_config import load_context
    context = load_context(reference)
    if context["config_sha256"] != expected_config_sha256:
        raise ValueError("Worker execution configuration changed")
    _configure_cpu()
    _WORKER_CONTEXT = context


def _worker_fit(file_id):
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Parent worker is uninitialized")
    return _fit_parent(_WORKER_CONTEXT, file_id)


def ensure_parents(context, *, replay_only=False):
    """Prepare each unique selected parent once; failures remain terminal.

    Replay re-audits historical candidates and verifies every existing bundle
    and source header/hash. It does not decode new audio or run optimization.
    Missing replay parents are returned in ``failed_ids`` without writing data.
    """
    ids = sorted({file_id for role in context["roles"].values() for file_id in role.train_ids})
    for file_id in ids:
        _id(file_id)
        _witness(context, file_id)
    workers = context["config"].get("max_workers", 4)
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Parent workers must be an integer between one and four")
    _, summary = _audit_candidates(context, replay_only)
    complete, failed, missing_reuse, missing_fit = [], [], [], []
    for file_id in ids:
        folder = _folder(context, file_id)
        if folder.exists() or folder.is_symlink():
            metadata, _ = _read_parent(context, file_id, _witness(context, file_id))
            _verified_raw_bytes(context, file_id)
            (complete if metadata["status"] == "complete" else failed).append(file_id)
        elif replay_only:
            failed.append(file_id)
        elif file_id in context["readiness_records"]:
            missing_reuse.append(file_id)
        else:
            missing_fit.append(file_id)
    if missing_reuse:
        banks, observations = _reuse_assets(context, summary)
        for file_id in missing_reuse:
            _save_reused(context, file_id, banks, observations)
            complete.append(file_id)
    if missing_fit:
        if workers == 1:
            _configure_cpu()
            for file_id in missing_fit:
                status = _fit_parent(context, file_id)
                (complete if status == "complete" else failed).append(file_id)
        else:
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                       initializer=_worker_initialize,
                                       initargs=(context["config_reference"], context["config_sha256"]))
            try:
                futures = {pool.submit(_worker_fit, file_id): file_id for file_id in missing_fit}
                for future in as_completed(futures):
                    file_id = futures[future]
                    # Expected scientific/numerical failures are persisted by
                    # _fit_parent and returned as status. A transport, worker
                    # initialization, cancellation or interrupted-pool error is
                    # recoverable infrastructure failure, never evidence that
                    # an unstarted parent failed numerically.
                    status = future.result()
                    (complete if status == "complete" else failed).append(file_id)
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
    return {"complete_ids": sorted(complete), "failed_ids": sorted(failed)}

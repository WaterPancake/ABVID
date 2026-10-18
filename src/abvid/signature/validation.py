"""Reproducible P0 preparation checks, using artificial fixtures only."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import soundfile as sf

from abvid.paths import resolve, roots
from abvid.provenance.hashing import canonical, sha256

from . import PROTOCOL_ID, TOOL_VERSION
from .audio import read_asset
from .config import DEFAULT_CONFIG, external_path, load
from .dedup import audit_manifest, with_reuse_links
from .fixtures import create_bundle
from .manifest import connected_groups, readiness, validate
from .trim import _write_json_new, export_segments, import_annotations, make_packet


def _digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def run(fixture_reference, output_reference, *, config_reference=DEFAULT_CONFIG):
    """Run the synthetic oracle through the same public preparation operations.

    Interval locations are known from fixture generation, not an audio detector
    or a human-reviewed scientific observation. No representation is fitted.
    """
    config = load(config_reference)
    output = external_path(output_reference)
    if output.exists():
        raise FileExistsError("Validation output is immutable; choose a new directory")
    if external_path(fixture_reference).exists():
        raise FileExistsError("Validation fixtures are immutable; choose a new directory")
    manifest = create_bundle(fixture_reference, config)
    packet = make_packet(manifest, config, output_reference + "/blind",
                         output_reference + "/private/mapping.json", seed=17)
    batch = json.loads((output / "blind/review_batch.json").read_text())
    private = json.loads((output / "private/mapping.json").read_text())
    recipes = {
        "fixture-a": [(2.00003, 6.00003, "foreground", None), (0, 2, "background", None)],
        "fixture-b": [(2.00003, 6.00003, "foreground", None), (0, 2, "background", None)],
        "fixture-crop": [(1.00003, 5.00003, "foreground", None), (7, 9, "background", None)],
        "fixture-resampled": [(2.00003, 6.00003, "foreground", None), (0, 2, "background", None)],
        "fixture-silence": [(0, 5, "rejected", "no_vehicle")],
        "fixture-short": [(0, 1, "rejected", "too_short")],
    }
    intervals = []
    for blind_id, aid in private["mapping"].items():
        for start, end, kind, reason in recipes[aid]:
            intervals.append({"blind_id": blind_id, "start_seconds": start, "end_seconds": end,
                "kind": kind, "audible_vehicle_count": "unknown" if reason else int(kind == "foreground"),
                "motion_state": "unknown", "contaminants": [],
                "rejection_reasons": [reason] if reason else []})
    annotations = {"batch_sha256": packet["batch_sha256"], "tool_version": TOOL_VERSION,
                   "trimmer_id": "synthetic-oracle", "intervals": intervals}
    _write_json_new(output / "oracle_annotations.json", annotations)
    imported = import_annotations(manifest, config, batch, private, annotations)
    # These arbitrary names test manifest bookkeeping, not vehicle identity.
    for segment in imported["segments"]:
        if segment["kind"] == "foreground":
            label = "artificial-b" if segment["asset_id"] == "fixture-b" else "artificial-a"
            imported["labels"].append({"segment_id": segment["segment_id"], "caption_model": label,
                "caption_variant": "unknown", "confidence": "confirmed", "L1": "unknown",
                "L2": "unknown", "L3": "unknown", "label_evidence": "Fixture generator seed; no vehicle claim"})
    validate(imported, config)
    reuse = audit_manifest(imported, config)
    grouped = with_reuse_links(imported, reuse, config)
    _write_json_new(output / "reuse_audit.json", reuse)
    _write_json_new(output / "manifest.json", grouped)
    exported = export_segments(grouped, config, output_reference + "/native_clips")
    by_id = {a["asset_id"]: a for a in grouped["assets"]}
    by_segment = {s["segment_id"]: s for s in grouped["segments"]}
    frame_checks = []
    for clip in exported["clips"]:
        raw, rate = read_asset(by_id[clip["asset_id"]], config)
        segment = by_segment[clip["segment_id"]]
        expected_start = math.floor(segment["start_seconds"] * rate)
        expected_end = math.ceil(segment["end_seconds"] * rate)
        actual, actual_rate = sf.read(resolve(clip["clip_ref"]), dtype="float32", always_2d=True)
        passed = (actual_rate == rate and clip["start_frame"] == expected_start and
                  clip["end_frame"] == expected_end and np.array_equal(actual, raw[expected_start:expected_end]))
        if not passed:
            raise AssertionError("Native samples/rounding were not preserved")
        frame_checks.append({"segment_id": clip["segment_id"], "start_frame": expected_start,
                             "end_frame": expected_end, "native_samples_equal": passed})
    pairs = {frozenset((p["left"], p["right"])) for p in reuse["pairs"]}
    required = [frozenset(("fixture-a", "fixture-crop")), frozenset(("fixture-a", "fixture-resampled"))]
    if not all(p in pairs for p in required) or any("fixture-b" in p for p in pairs):
        raise AssertionError("Synthetic reuse positive or independent negative failed")
    changed = deepcopy(imported)
    for asset in changed["assets"]:
        asset["title"] = "Deliberately changed hidden model name"
    if audit_manifest(changed, config)["pairs"] != reuse["pairs"]:
        raise AssertionError("Caption changed label-blind reuse matching")
    groups = connected_groups(grouped)
    reused_ids = ["fixture-a", "fixture-crop", "fixture-resampled", "fixture-short"]
    if len({groups[aid] for aid in reused_ids}) != 1:
        raise AssertionError("Declared or detected reuse was split")
    if len(exported["rejections"]) != 2:
        raise AssertionError("Rejected intervals were lost")
    blind_text = (output / "blind/index.html").read_text() + (output / "blind/review_batch.json").read_text()
    if any(a[k] in blind_text for a in manifest["assets"] for k in ["asset_id", "title", "recording_session"]):
        raise AssertionError("Source metadata leaked into the reviewer packet")
    version = subprocess.run(["git", "rev-parse", "HEAD"], cwd=roots()["repo"],
                             check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=roots()["repo"],
                            check=True, capture_output=True, text=True).stdout
    code_hashes = {"repo:" + str(p.relative_to(roots()["repo"])): sha256(p)
                   for p in sorted((roots()["repo"] / "src/abvid/signature").glob("*.py"))}
    report = {"protocol_id": PROTOCOL_ID, "tool_version": TOOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(), "scope": "synthetic_preparation_validation_only",
        "fixture_ref": fixture_reference, "output_ref": output_reference,
        "seed": 17, "independent_seed": 29, "commit": version, "working_tree_dirty": bool(status),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        **{name: importlib.metadata.version(name) for name in ["numpy", "scipy", "soundfile"]}},
        "configuration": config, "config_file_sha256": sha256(resolve(config_reference)),
        "schema_file_sha256": sha256(resolve(config["storage"]["manifest_schema"])),
        "implementation_sha256": code_hashes,
        "raw_hashes": {a["asset_id"]: a["raw_sha256"] for a in manifest["assets"]},
        "manifest_sha256": _digest(grouped), "native_export_checks": frame_checks,
        "checks": {"native_samples_and_rounding": True, "caption_hidden": True,
                   "label_blind_reuse": True, "crop_gain_polarity_reuse": True,
                   "resampling_reuse": True, "independent_negative": True,
                   "rejection_log_retained": True, "shared_origin_group_preserved": True},
        "readiness": readiness(grouped, config),
        "limitations": ["Synthetic oracle intervals do not validate human trimming accuracy",
                        "No real audio, representations, model fits or scientific evaluation",
                        "Browser interaction is a separately recorded functional check",
                        "Negative reuse detection does not certify recording independence"]}
    _write_json_new(output / "validation.json", report)
    return {"validation_ref": output_reference + "/validation.json", "passed": True,
            "assets": len(manifest["assets"]), "exported_clips": len(exported["clips"]),
            "rejections": len(exported["rejections"]), "groups": len(set(groups.values()))}

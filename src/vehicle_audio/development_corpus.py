"""Fail-closed access to the frozen non-startup development corpus."""

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from vehicle_audio.benchmark_followup import sha256
from vehicle_audio.real_corpus import audit_real_manifest

MANIFEST = Path("runs/preprocessing_grid_v1/dataset/real_manifest.jsonl")
MANIFEST_SHA256 = "c705cc95ef506cbd0ad4d504e53a5cc8fcae71baca63355d9b6b54c24ba81359"
CONTROL = Path("runs/preprocessing_grid_v1/variants/dc0_hp0_rmsoff")


def validate_records(records, catalog, policy, archive):
    sources = {s["id"]: s for s in catalog["sources"]}
    forbidden = set(
        policy["consumed_locked_source_ids"] + policy["future_confirmation_source_ids"]
    )
    sessions = {sources[s]["recording_session"] for s in forbidden}
    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("duplicate sample IDs")
    for r in records:
        s = sources[r["source_id"]]
        if (
            s.get("admitted_to_corpus") is not True
            or r["source_id"] in forbidden
            or r["recording_session"] in sessions
        ):
            raise ValueError("protected or unadmitted recording")
        if (
            s["recording_session"] != r["recording_session"]
            or s["vehicle_class"] != r["vehicle_class"]
        ):
            raise ValueError("source identity mismatch")
        start, end = r["window_start_seconds"], r["window_end_seconds"]
        if end - start != 2 or r["sample_rate"] != 16000 or r["num_channels"] != 1:
            raise ValueError("unexpected window geometry")
        if r["operating_condition"] == "startup" or not any(
            q["start_seconds"] <= start < end <= q["end_seconds"]
            and q["operating_condition"] == r["operating_condition"]
            for q in s.get("condition_segments", [])
        ):
            raise ValueError("unreviewed/startup window")
        if any(
            a["source_id"] == r["source_id"]
            and start < a["end_seconds"]
            and a["start_seconds"] < end
            for a in archive["intervals"]
        ):
            raise ValueError("archived startup overlap")
    return sources


def load_development():
    if sha256(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("development manifest hash mismatch")
    records = [json.loads(x) for x in MANIFEST.read_text().splitlines()]
    catalog = yaml.safe_load(Path("configs/audio_sources.yaml").read_text())
    policy = yaml.safe_load(Path("configs/benchmark_v0_1.yaml").read_text())
    archive = yaml.safe_load(
        Path("configs/archived_startup_intervals.yaml").read_text()
    )
    sources = validate_records(records, catalog, policy, archive)
    audit = audit_real_manifest(records, minimum_sessions_per_class=5)
    if not audit["full_protocol_ready"] or audit["recording_session_counts"] != {
        "tracked": 7,
        "wheeled": 5,
    }:
        raise ValueError("development audit failed")
    checked = set()
    diagnostics = json.loads((CONTROL / "waveform_diagnostics.json").read_text())
    hashes = {d["sample_id"]: d["waveform_sha256"] for d in diagnostics}
    waves = []
    for r in records:
        if r["source_id"] not in checked:
            source = Path("data") / sources[r["source_id"]]["output_path"]
            if sha256(source) != r["normalized_source_sha256"]:
                raise ValueError("source audio changed")
            duration = sf.info(source).duration
            for segment in sources[r["source_id"]]["condition_segments"]:
                start, end = segment["start_seconds"], segment["end_seconds"]
                if (
                    not 0 <= start < end <= duration
                    or segment["operating_condition"] == "startup"
                ):
                    raise ValueError("invalid active review bounds/condition")
                if any(
                    a["source_id"] == r["source_id"]
                    and start < a["end_seconds"]
                    and a["start_seconds"] < end
                    for a in archive["intervals"]
                ):
                    raise ValueError("active interval overlaps archived startup")
            sidecar = json.loads(source.with_suffix(".json").read_text())
            if (
                sidecar["condition_segments"]
                != sources[r["source_id"]]["condition_segments"]
            ):
                raise ValueError("sidecar/catalog mismatch")
            checked.add(r["source_id"])
        wave, sr = sf.read(MANIFEST.parent / r["audio_path"], dtype="float32")
        if sr != 16000 or wave.shape != (32000,) or not np.isfinite(wave).all():
            raise ValueError("invalid waveform")
        if hashlib.sha256(wave.tobytes()).hexdigest() != hashes[r["sample_id"]]:
            raise ValueError("window waveform changed")
        waves.append(wave)
    return records, sources, np.stack(waves), audit

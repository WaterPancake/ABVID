"""Deterministic artificial signals for preparation-tool validation only."""
from __future__ import annotations

import json
from copy import deepcopy
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from abvid.provenance.hashing import sha256

from . import PROTOCOL_ID, TOOL_VERSION
from .config import external_path, validate as validate_config
from .manifest import validate

GENERATOR_ID = "signature_p0_artificial_chirp_noise_v1"


def signal(seed=17, *, rate=16000, seconds=12):
    """A changing chirp/noise fixture, deliberately not a vehicle simulator."""
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * rate)) / rate
    noise = rng.normal(0, 0.003, len(t))
    active = (t >= 2) & (t < 8)
    frequency = 120 + (seed % 7) * 90
    phase = 2 * np.pi * (frequency * t + 13 * t * t + 0.9 * np.sin(2 * np.pi * 0.8 * t))
    envelope = 0.55 + 0.3 * np.sin(2 * np.pi * 1.3 * t)
    x = noise + active * (0.15 * envelope * np.sin(phase) + rng.normal(0, 0.008, len(t)))
    return np.asarray(x, dtype=np.float32)


def asset_record(asset_id, reference, path, *, session=None):
    info = sf.info(path)
    return {
        "asset_id": asset_id, "url": "fixture:" + asset_id,
        "title": "Hidden fixture label " + asset_id,
        "date_taken": "2026-10-11", "date_posted": "unknown", "unit": "synthetic_fixture",
        "location": "synthetic_fixture", "outlet": "synthetic_fixture", "producer": GENERATOR_ID,
        "license_status": "permitted", "license_note": "Artificial local fixture; no real media",
        "license_url": "unknown", "retrieved_at": "2026-10-11T00:00:00Z",
        "raw_sha256": sha256(path), "media_ref": reference,
        "media_kind": "synthetic_fixture", "duration_seconds": info.duration,
        "native_sample_rate_hz": info.samplerate, "channels": info.channels,
        "recording_session": session or "synthetic-session-" + asset_id,
        "session_evidence": GENERATOR_ID + ": artificial independent RNG seed or declared reuse",
        "independence_verified": True,
        "physical_vehicle_id": "unknown", "vehicle_identity_evidence": "unknown",
    }


def create_bundle(reference, config):
    validate_config(config)
    if not reference.startswith(config["storage"]["fixture_root"] + "/"):
        raise ValueError("Fixture output must use the configured fixture subtree")
    output = external_path(reference)
    if output.exists():
        raise FileExistsError("Fixture outputs are immutable; choose a new directory")
    output.mkdir(parents=True)
    base = signal()
    factor = gcd(16000, 11025)
    cases = [
        ("fixture-a", base, 16000, "synthetic-origin-a"),
        ("fixture-b", signal(29), 16000, "synthetic-origin-b"),
        ("fixture-crop", -0.45 * base[16000:160000], 16000, "synthetic-origin-a"),
        ("fixture-resampled", resample_poly(base, 11025 // factor, 16000 // factor), 11025, "synthetic-origin-a"),
        ("fixture-silence", np.zeros(80000), 16000, "synthetic-silence"),
        ("fixture-short", base[:16000], 16000, "synthetic-origin-a"),
    ]
    assets = []
    for aid, x, rate, session in cases:
        path = output / (aid + ".wav")
        sf.write(path, x, rate, subtype="FLOAT")
        assets.append(asset_record(aid, reference + "/" + path.name, path, session=session))
    manifest = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "assets": assets,
                "segments": [], "labels": [], "reuse_links": []}
    validate(manifest, config)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "generation.json").write_text(json.dumps({"generator_id": GENERATOR_ID,
        "seed": 17, "independent_seed": 29, "scope": "preparation_tool_fixture_only",
        "raw_hashes": {a["asset_id"]: a["raw_sha256"] for a in assets}}, indent=2) + "\n")
    return manifest


def sample_segment(asset_id, *, segment_id="fixture-segment", kind="foreground", start=2, end=6):
    return {"segment_id": segment_id, "asset_id": asset_id, "start_seconds": start, "end_seconds": end,
            "kind": kind, "role": "quarantine" if kind == "rejected" else "development",
            "rejection_reasons": ["uncertain_audibility"] if kind == "rejected" else [],
            "trimmer_id": "fixture-reviewer", "tool_version": TOOL_VERSION, "blind_review": True,
            "audible_vehicle_count": "unknown" if kind == "rejected" else int(kind == "foreground"),
            "motion_state": "unknown", "distance_m": "unknown", "mic_type": "unknown", "contaminants": []}


def sample_annotations(batch, batch_sha256):
    intervals = []
    for recording in batch["recordings"]:
        duration = recording["duration_seconds"]
        # This is synthetic tooling validation, not an automated real trimmer.
        row = {"blind_id": recording["blind_id"], "start_seconds": 0,
               "end_seconds": min(duration, 2), "kind": "rejected",
               "audible_vehicle_count": "unknown", "motion_state": "unknown",
               "contaminants": [], "rejection_reasons": ["uncertain_audibility"]}
        intervals.append(row)
    return {"batch_sha256": batch_sha256, "tool_version": TOOL_VERSION,
            "trimmer_id": "fixture-reviewer", "intervals": deepcopy(intervals)}

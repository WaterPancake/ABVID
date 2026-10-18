"""P1 label-blind audio reuse audit; no historical/protected audio is loaded."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from abvid.data.overlap import HOP, RATE, Index, landmarks
from abvid.paths import resolve
from abvid.provenance.hashing import sha256
from . import admission, admission_review
from .config import configuration, external_path
from .admission_reuse_votes import match


def audit(config_ref=admission.CONFIG_REF):
    cfg = admission.load_config(config_ref)
    rows = admission_review.records(cfg)
    spec = configuration()["fingerprint"]
    settings = spec["settings"]
    index = Index(settings)
    windows, raw, statuses, best = {}, {}, [], {}
    width = round(spec["window_seconds"] * RATE)
    stride = round(spec["hop_seconds"] * RATE)
    for row in sorted(rows, key=lambda r: r["asset_id"]):
        aid = row["asset_id"]
        audio = row["acquisition"]
        path = admission_review.validate_review_audio(row)
        key = audio["audio_sha256"]
        if key in raw:
            pair = tuple(sorted((raw[key], aid)))
            best[pair] = {"left": pair[0], "right": pair[1], "kind": "exact_retained_audio_hash",
                          "correlation": 1.0, "overlap_seconds": None}
        else:
            raw[key] = aid
        x = admission_review.display_audio(path, cfg["collection"]["maximum_asset_seconds"]).ravel()
        if len(x) < width or not np.any(x):
            statuses.append({"asset_id": aid, "status": "short_or_silent_unresolved", "windows": 0})
            continue
        if index.parts:
            index.freeze()
            for result in match(index, aid, x, RATE):
                if not result["accepted"]:
                    continue
                previous, previous_start = windows[result["reference_id"]]
                pair = tuple(sorted((previous, aid)))
                candidate = {
                    "left": pair[0], "right": pair[1], "kind": "waveform_reuse_suggestion",
                    "correlation": result["correlation"], "overlap_seconds": spec["window_seconds"],
                    "reference_asset_id": previous, "reference_start_seconds": previous_start,
                    "query_asset_id": aid, "query_start_seconds": result["offset_s"],
                    "landmark_votes": result["votes"],
                }
                if pair not in best or (best[pair]["kind"] != "exact_retained_audio_hash" and
                                        candidate["correlation"] > best[pair]["correlation"]):
                    best[pair] = candidate
        starts = list(range(0, len(x) - width + 1, stride))
        if len(starts) > spec["maximum_windows_per_asset"]:
            raise ValueError("Fingerprint window budget exceeded")
        count = 0
        for start in starts:
            window = x[start:start + width]
            hashes, _ = landmarks(window, settings)
            if not len(hashes):
                continue
            key = f"{aid}::{start}"
            index.add(key, window, RATE)
            windows[key] = (aid, start / RATE)
            count += 1
        statuses.append({"asset_id": aid, "status": "fingerprinted" if count else "no_landmarks_unresolved",
                         "windows": count, "unindexed_windows": len(starts) - count})
        print(json.dumps({"fingerprinted_assets": len(statuses), "total_assets": len(rows),
                          "indexed_windows": len(windows), "reuse_suggestions": len(best)}), flush=True)
    result = {
        "recorded_at": admission.now(), "protocol_id": cfg["id"],
        "scope": "P1 label-blind reuse suggestions only; no model or CAST fitting",
        "tool_version": "signature-admission-reuse-3",
        "input_audio_sha256": {r["asset_id"]: r["acquisition"]["audio_sha256"] for r in rows},
        "pairs": [best[k] for k in sorted(best)], "assets": statuses,
        "settings": deepcopy(spec), "implementation_rate_hz": RATE, "implementation_hop": HOP,
        "resampling": "ffmpeg display/fingerprint route to mono 8000 Hz float32; native audio retained",
        "source_code_sha256": sha256(Path(__file__)),
        "overlap_implementation_sha256": sha256(resolve("repo:src/abvid/data/overlap.py")),
        "vote_implementation_sha256": sha256(resolve("repo:src/abvid/signature/admission_reuse_votes.py")),
        "correlation_implementation_sha256": sha256(resolve("repo:src/abvid/signature/admission_reuse_correlation.py")),
        "voting": "exact packed integer counts; parity with the original reference-ID/offset sort verified",
        "correlation": "FFT winner nomination with original direct-dot scores; exact-score parity verified; ill-conditioned cases use original full lag scan",
        "independence_certified": False, "verified_independent_recording_groups": 0,
        "protected_audio_accessed": False,
        "limitations": [
            "No-match is not evidence of independent recording sessions or physical vehicles",
            "Periodic signals may conservatively link distinct recordings",
            "Codec, edits, mixtures, time stretch and overlaps shorter than four seconds may be missed",
            "Suggested reuse must be retained in admission grouping before any experimental role allocation",
        ],
    }
    output = external_path(cfg["storage"]["artifact_root"]) / "reuse_audit.json"
    encoded = json.dumps(result, indent=2, allow_nan=False).encode()
    if len(encoded) > admission.available_bytes(cfg):
        raise OSError("Storage budget cannot fit reuse record")
    admission.write_new(output, result)
    return result


if __name__ == "__main__":
    result = audit()
    print(json.dumps({"assets": len(result["assets"]), "reuse_suggestions": len(result["pairs"]),
                      "verified_independent_recording_groups": 0}, indent=2))

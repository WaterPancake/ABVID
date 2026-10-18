"""Label-blind reuse suggestions, reusing the existing landmark/correlation audit.

Absence of a match never certifies independence. Short, silent and otherwise
unfingerprintable assets are retained as unresolved, not dropped from the audit.
"""
from __future__ import annotations

import hashlib
from copy import deepcopy

import numpy as np

from abvid.data.overlap import HOP, RATE, Index, landmarks, match, mono8k
from abvid.provenance.hashing import canonical

from .audio import read_asset
from .config import validate as validate_config
from .manifest import validate as validate_manifest


def audit(assets, loader, config):
    """Audit (asset_id, raw_sha256) inputs with an explicit audio loader.

    The callable loader returns (waveform, native_rate). No class, caption,
    filename or representation is passed to the matching algorithm.
    """
    validate_config(config)
    spec = config["fingerprint"]
    settings = spec["settings"]
    index = Index(settings)
    windows, raw, statuses, best = {}, {}, [], {}
    width = round(spec["window_seconds"] * RATE)
    stride = round(spec["hop_seconds"] * RATE)
    ids = [a["asset_id"] for a in assets]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate asset IDs in fingerprint input")
    for a in sorted(assets, key=lambda a: a["asset_id"]):
        aid, digest = a["asset_id"], a["raw_sha256"]
        if digest in raw:
            pair = tuple(sorted((raw[digest], aid)))
            best[pair] = {"left": pair[0], "right": pair[1], "kind": "exact_raw_hash",
                          "correlation": 1.0, "overlap_seconds": None}
        else:
            raw[digest] = aid
        x, rate = loader(aid)
        x = mono8k(x, rate)
        if len(x) < width:
            statuses.append({"asset_id": aid, "status": "too_short_for_fingerprint", "windows": 0})
            continue
        if not np.any(x):
            statuses.append({"asset_id": aid, "status": "silent_unresolved", "windows": 0})
            continue
        if index.parts:
            index.freeze()
            for row in match(index, aid, x, RATE):
                if not row["accepted"]:
                    continue
                prior, prior_start = windows[row["reference_id"]]
                pair = tuple(sorted((prior, aid)))
                candidate = {"left": pair[0], "right": pair[1], "kind": "waveform_reuse",
                             "correlation": row["correlation"],
                             "overlap_seconds": spec["window_seconds"],
                             "reference_asset_id": prior, "reference_start_seconds": prior_start,
                             "query_asset_id": aid, "query_start_seconds": row["offset_s"],
                             "landmark_votes": row["votes"]}
                if pair not in best or (best[pair]["kind"] != "exact_raw_hash" and
                                        candidate["correlation"] > best[pair]["correlation"]):
                    best[pair] = candidate
        count, starts = 0, list(range(0, len(x) - width + 1, stride))
        if len(starts) > spec["maximum_windows_per_asset"]:
            raise ValueError("Fingerprint window budget exceeded; shorten/version the input protocol")
        for start in starts:
            window = x[start:start + width]
            # A time-varying snippet with no landmarks cannot be declared clear.
            hashes, _ = landmarks(window, settings)
            if not len(hashes):
                continue
            key = f"{aid}::{start}"
            index.add(key, window, RATE)
            windows[key] = (aid, start / RATE)
            count += 1
        statuses.append({"asset_id": aid, "status": "fingerprinted" if count else "no_landmarks_unresolved",
                         "windows": count, "unindexed_windows": len(starts) - count})
    return {"pairs": [best[k] for k in sorted(best)], "assets": statuses,
            "settings": deepcopy(spec), "implementation_rate_hz": RATE, "implementation_hop": HOP,
            "independence_certified": False,
            "limitations": ["No-match results do not prove independent recordings",
                            "Codec, editing, mixtures and time-stretch reuse may be missed",
                            "Shared periodic waveforms may conservatively overmerge distinct recordings",
                            "Overlap shorter than four seconds is not fingerprint-verified"]}


def audit_manifest(manifest, config):
    validate_manifest(manifest, config)
    by_id = {a["asset_id"]: a for a in manifest["assets"]}
    result = audit(manifest["assets"], lambda aid: read_asset(by_id[aid], config), config)
    result["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
    result["protocol_sha256"] = hashlib.sha256(canonical(config)).hexdigest()
    return result


def with_reuse_links(manifest, result, config):
    """Bind a reuse audit to its inputs before conservatively joining groups."""
    validate_manifest(manifest, config)
    if result.get("manifest_sha256") != hashlib.sha256(canonical(manifest)).hexdigest():
        raise ValueError("Reuse audit belongs to a different manifest")
    if result.get("protocol_sha256") != hashlib.sha256(canonical(config)).hexdigest():
        raise ValueError("Reuse audit belongs to a different protocol")
    updated = deepcopy(manifest)
    assets = {a["asset_id"]: a for a in manifest["assets"]}
    for pair in result["pairs"]:
        if pair["kind"] not in {"exact_raw_hash", "waveform_reuse"}:
            raise ValueError("Unsupported reuse evidence")
        if pair["left"] not in assets or pair["right"] not in assets or pair["left"] == pair["right"]:
            raise ValueError("Invalid reuse evidence identities")
        if pair["kind"] == "exact_raw_hash":
            if assets[pair["left"]]["raw_sha256"] != assets[pair["right"]]["raw_sha256"]:
                raise ValueError("Exact-hash reuse evidence does not match the source hashes")
        elif (not np.isfinite(pair["correlation"]) or
              pair["correlation"] < config["fingerprint"]["settings"]["min_correlation"] or
              pair["correlation"] > 1.0 + 1e-12 or
              pair["overlap_seconds"] != config["fingerprint"]["window_seconds"]):
            raise ValueError("Waveform reuse evidence violates the frozen acceptance rule")
        updated["reuse_links"].append({"left": pair["left"], "right": pair["right"],
            "kind": "waveform_reuse" if pair["kind"] == "waveform_reuse" else "provenance",
            "evidence": f"P0 audit {pair['kind']}; correlation={pair['correlation']:.8f}; "
                        f"input={result['manifest_sha256']}"})
    validate_manifest(updated, config)
    return updated

"""Strict manifest structure, conservative grouping, and readiness checks.

An asset ID is an identity, never evidence of a new recording session.
Unknown sessions stay unknown and share a conservative quarantine component.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

from abvid.data.grouping import Union

from . import PROTOCOL_ID, TOOL_VERSION
from .config import external_path, validate as validate_config

UNKNOWN = "unknown"
CONTAMINANTS = ["music", "narration", "gunfire", "aircraft", "wind", "crowd", "agc_pumping", UNKNOWN]
REASONS = ["music", "narration", "gunfire", "aircraft", "wind_masking", "multiple_vehicles",
           "no_vehicle", "no_background", "too_short", "clipping", "editing_or_codec_artifact",
           "uncertain_audibility", "access_failure", "rights_unresolved", "other"]
MOTION = ["idle", "rev", "accelerate", "constant", "pass_by", "brake", UNKNOWN]


def _text(**extra):
    return {"type": "string", "minLength": 1, **extra}


def _unknown_or(spec):
    return {"anyOf": [{"const": UNKNOWN}, spec]}


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def schema():
    identifier = _text(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    number = {"type": "number", "minimum": 0}
    asset = _object({
        "asset_id": identifier, "url": _text(), "title": _text(),
        "date_taken": _text(pattern=r"^(unknown|\d{4}-\d{2}-\d{2})$"),
        "date_posted": _text(), "unit": _text(),
        "location": _text(), "outlet": _text(), "producer": _text(),
        "license_status": {"enum": ["public_domain", "permitted", "restricted", UNKNOWN]},
        "license_note": _text(), "license_url": _text(), "retrieved_at": _text(),
        "raw_sha256": _text(pattern=r"^[0-9a-f]{64}$"),
        "media_ref": _text(pattern=r"^data:military_signature/[^:]+$"),
        "media_kind": {"enum": ["synthetic_fixture", "real"]},
        "duration_seconds": {"type": "number", "exclusiveMinimum": 0},
        "native_sample_rate_hz": {"type": "integer", "minimum": 1000},
        "channels": {"type": "integer", "minimum": 1, "maximum": 8},
        "recording_session": _text(), "session_evidence": _text(),
        "independence_verified": {"type": "boolean"},
        "physical_vehicle_id": _text(), "vehicle_identity_evidence": _text(),
    })
    segment = _object({
        "segment_id": identifier, "asset_id": identifier,
        "start_seconds": number, "end_seconds": {"type": "number", "exclusiveMinimum": 0},
        "kind": {"enum": ["foreground", "background", "rejected"]},
        "role": {"enum": ["development", "validation", "evaluation", "quarantine"]},
        "rejection_reasons": {"type": "array", "items": {"enum": REASONS}, "uniqueItems": True},
        "trimmer_id": identifier, "tool_version": {"const": TOOL_VERSION},
        "blind_review": {"const": True},
        "audible_vehicle_count": _unknown_or({"type": "integer", "minimum": 0}),
        "motion_state": {"enum": MOTION}, "distance_m": _unknown_or(number),
        "mic_type": _text(),
        "contaminants": {"type": "array", "items": {"enum": CONTAMINANTS}, "uniqueItems": True},
    })
    label = _object({
        "segment_id": identifier, "caption_model": _text(), "caption_variant": _text(),
        "confidence": {"enum": ["confirmed", "caption", UNKNOWN]},
        "L1": {"enum": ["tracked", "wheeled", UNKNOWN]},
        "L2": {"enum": ["turbine", "diesel", "other", UNKNOWN]},
        "L3": _text(), "label_evidence": _text(),
    })
    link = _object({
        "left": identifier, "right": identifier,
        "kind": {"enum": ["provenance", "suspected_reuse", "waveform_reuse"]},
        "evidence": _text(),
    })
    result = _object({
        "schema_version": {"const": 1}, "protocol_id": {"const": PROTOCOL_ID},
        "assets": {"type": "array", "items": {"$ref": "#/$defs/asset"}},
        "segments": {"type": "array", "items": {"$ref": "#/$defs/segment"}},
        "labels": {"type": "array", "items": {"$ref": "#/$defs/label"}},
        "reuse_links": {"type": "array", "items": {"$ref": "#/$defs/link"}},
    })
    result.update({"$schema": "https://json-schema.org/draft/2020-12/schema",
                   "$id": "urn:abvid:military-signature:manifest:1",
                   "$defs": {"asset": asset, "segment": segment, "label": label, "link": link}})
    return result


def _check(value, rule, root, where):
    """Validate only the vocabulary emitted by schema(), without a new dependency."""
    if "$ref" in rule:
        return _check(value, root["$defs"][rule["$ref"].rsplit("/", 1)[1]], root, where)
    if "anyOf" in rule:
        for option in rule["anyOf"]:
            try:
                _check(value, option, root, where)
                return
            except ValueError:
                pass
        raise ValueError(f"{where}: invalid value")
    if "const" in rule and (type(value) is not type(rule["const"]) or value != rule["const"]):
        raise ValueError(f"{where}: incorrect constant")
    if "enum" in rule and not any(type(value) is type(v) and value == v for v in rule["enum"]):
        raise ValueError(f"{where}: invalid enum")
    kind = rule.get("type")
    matches = {"string": isinstance(value, str), "object": isinstance(value, dict),
               "array": isinstance(value, list), "boolean": type(value) is bool,
               "integer": type(value) is int, "number": type(value) in (int, float)}
    if kind and not matches[kind]:
        raise ValueError(f"{where}: expected {kind}")
    if type(value) in (int, float):
        if not math.isfinite(value):
            raise ValueError(f"{where}: nonfinite value")
        if value < rule.get("minimum", -math.inf) or value > rule.get("maximum", math.inf):
            raise ValueError(f"{where}: outside range")
        if "exclusiveMinimum" in rule and value <= rule["exclusiveMinimum"]:
            raise ValueError(f"{where}: outside exclusive range")
    if isinstance(value, str):
        if len(value) < rule.get("minLength", 0) or ("pattern" in rule and not re.search(rule["pattern"], value)):
            raise ValueError(f"{where}: invalid string")
    if kind == "object":
        if set(value) != set(rule["required"]):
            raise ValueError(f"{where}: missing or unexpected fields")
        for key, item in value.items():
            _check(item, rule["properties"][key], root, f"{where}.{key}")
    if kind == "array":
        for i, item in enumerate(value):
            _check(item, rule["items"], root, f"{where}[{i}]")
        if rule.get("uniqueItems") and any(value[i] == value[j] for i in range(len(value)) for j in range(i)):
            raise ValueError(f"{where}: duplicate items")


def connected_groups(manifest):
    assets = manifest["assets"]
    union = Union(a["asset_id"] for a in assets)
    by_session, by_hash, by_vehicle = defaultdict(list), defaultdict(list), defaultdict(list)
    for a in assets:
        # Unknown-session assets are conservatively joined. This does not turn
        # them into a known recording_session or establish independence.
        by_session[a["recording_session"]].append(a["asset_id"])
        by_hash[a["raw_sha256"]].append(a["asset_id"])
        if a["physical_vehicle_id"] != UNKNOWN:
            by_vehicle[a["physical_vehicle_id"]].append(a["asset_id"])
    for members in [*by_session.values(), *by_hash.values(), *by_vehicle.values()]:
        for key in members[1:]:
            union.join(members[0], key)
    for link in manifest["reuse_links"]:
        union.join(link["left"], link["right"])
    return {a["asset_id"]: "group-" + union.find(a["asset_id"]) for a in assets}


def validate(manifest, config):
    validate_config(config)
    rules = schema()
    _check(manifest, rules, rules, "manifest")
    assets = {a["asset_id"]: a for a in manifest["assets"]}
    if len(assets) != len(manifest["assets"]):
        raise ValueError("Duplicate asset_id")
    for a in assets.values():
        external_path(a["media_ref"])
        if a["media_kind"] != "synthetic_fixture":
            raise ValueError("P0 is synthetic-only; real-media processing requires separate authorization")
        if not a["media_ref"].startswith(config["storage"]["fixture_root"] + "/"):
            raise ValueError("P0 input must be inside the configured fixture subtree")
        if a["duration_seconds"] > config["trimming"]["maximum_asset_seconds"]:
            raise ValueError("Asset exceeds the bounded P0 tooling duration")
        if a["independence_verified"] and (a["recording_session"] == UNKNOWN or a["session_evidence"] == UNKNOWN):
            raise ValueError("Independence requires a known session and recorded evidence")
        if a["physical_vehicle_id"] != UNKNOWN and a["vehicle_identity_evidence"] == UNKNOWN:
            raise ValueError("A physical-vehicle identity requires recorded evidence")
    for link in manifest["reuse_links"]:
        if link["left"] not in assets or link["right"] not in assets or link["left"] == link["right"]:
            raise ValueError("Invalid reuse link")
    groups = connected_groups(manifest)
    ids, roles, intervals = set(), defaultdict(set), defaultdict(list)
    for s in manifest["segments"]:
        if s["segment_id"] in ids or s["asset_id"] not in assets:
            raise ValueError("Duplicate segment_id or missing asset")
        ids.add(s["segment_id"])
        a = assets[s["asset_id"]]
        lo, hi = s["start_seconds"], s["end_seconds"]
        if hi <= lo or hi > a["duration_seconds"]:
            raise ValueError("Segment interval outside asset")
        if s["kind"] == "rejected":
            if not s["rejection_reasons"] or s["role"] != "quarantine":
                raise ValueError("Rejected segments require reasons and quarantine")
            continue
        if s["rejection_reasons"] or s["role"] == "quarantine":
            raise ValueError("Accepted segments cannot have rejection reasons or quarantine role")
        minimum = config["trimming"][f"minimum_{s['kind']}_seconds"]
        if not minimum <= hi - lo <= config["trimming"]["maximum_segment_seconds"]:
            raise ValueError("Accepted segment violates duration rules")
        expected_count = 1 if s["kind"] == "foreground" else 0
        if s["audible_vehicle_count"] != expected_count:
            raise ValueError("Foreground requires one audible vehicle; background requires zero")
        forbidden = config["trimming"]["forbidden_foreground_contaminants"]
        if s["kind"] == "foreground" and set(s["contaminants"]) & set(forbidden):
            raise ValueError("Foreground has a forbidden contaminant; record a rejection instead")
        if s["role"] in {"validation", "evaluation"} and not a["independence_verified"]:
            raise ValueError("Unverified sessions cannot enter held-out roles")
        roles[groups[s["asset_id"]]].add(s["role"])
        intervals[s["asset_id"]].append((lo, hi))
    if any(len(r) > 1 for r in roles.values()):
        raise ValueError("Connected recording/reuse group split across roles")
    for spans in intervals.values():
        ordered = sorted(spans)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise ValueError("Accepted intervals overlap within an asset")
    labelled = set()
    for label in manifest["labels"]:
        if label["segment_id"] not in ids or label["segment_id"] in labelled:
            raise ValueError("Label references a missing or duplicate segment")
        labelled.add(label["segment_id"])
        if label["confidence"] != UNKNOWN and label["label_evidence"] == UNKNOWN:
            raise ValueError("Caption/confirmed labels require evidence")
    return manifest


def readiness(manifest, config):
    validate(manifest, config)
    assets = {a["asset_id"]: a for a in manifest["assets"]}
    groups = connected_groups(manifest)
    foreground = {s["asset_id"] for s in manifest["segments"] if s["kind"] == "foreground"}
    background = {s["asset_id"] for s in manifest["segments"] if s["kind"] == "background"}
    label_by_segment = {s["segment_id"]: s for s in manifest["labels"]}
    per_model = defaultdict(set)
    unresolved = []
    for aid in sorted(foreground):
        a = assets[aid]
        if aid not in background:
            unresolved.append(f"{aid}: no reviewed same-asset background")
        if not a["independence_verified"]:
            unresolved.append(f"{aid}: session independence unverified")
        if a["license_status"] not in {"public_domain", "permitted"}:
            unresolved.append(f"{aid}: rights unresolved or restricted")
    for s in manifest["segments"]:
        if s["kind"] != "foreground":
            continue
        label = label_by_segment.get(s["segment_id"])
        if not label or label["confidence"] == UNKNOWN or label["caption_model"] == UNKNOWN:
            unresolved.append(f"{s['segment_id']}: model label unknown")
        elif assets[s["asset_id"]]["independence_verified"]:
            per_model[label["caption_model"]].add(groups[s["asset_id"]])
        if UNKNOWN in s["contaminants"]:
            unresolved.append(f"{s['segment_id']}: contamination unreviewed")
        backgrounds = [b for b in manifest["segments"] if b["asset_id"] == s["asset_id"] and b["kind"] == "background"]
        if backgrounds and all(b["contaminants"] for b in backgrounds):
            unresolved.append(f"{s['asset_id']}: no uncontaminated background reference")
    counts = {model: len(ids) for model, ids in sorted(per_model.items())}
    return {"scope": "synthetic_fixture_readiness_only", "real_collection_authorized": False,
            "asset_groups": groups, "verified_groups_per_fixture_label": counts,
            "unresolved": sorted(set(unresolved)), "model_claim_eligible": False,
            "outlet_invariance": UNKNOWN,
            "note": "Fixture support is not real-data admission or recognition evidence"}

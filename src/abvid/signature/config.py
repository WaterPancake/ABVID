"""Validate the canonical P0 contract before any preparation operation."""
from __future__ import annotations

import json
from copy import deepcopy

from abvid.data.overlap import SETTINGS
from abvid.paths import resolve, roots

from . import PROTOCOL_ID

DEFAULT_CONFIG = "repo:experiments/military_signature/p0_protocol.json"

SPEC = {
    "schema_version": 1,
    "id": PROTOCOL_ID,
    "phase": "P0",
    "authorization": {
        "literature_and_public_metadata": True,
        "synthetic_fixture_processing": True,
        "real_media_collection": False,
        "real_media_processing": False,
        "model_fitting": False,
        "protected_source_evaluation": False,
    },
    "storage": {
        "fixture_root": "data:military_signature/p0_fixtures",
        "artifact_root": "artifact:military_signature/p0",
        "manifest_schema": "repo:experiments/military_signature/manifest_schema.json",
    },
    "trimming": {
        "minimum_foreground_seconds": 2.0,
        "minimum_background_seconds": 2.0,
        "maximum_segment_seconds": 60.0,
        "maximum_asset_seconds": 1200.0,
        "display_sample_rate_hz": 8000,
        "caption_hidden": True,
        "rounding": "floor_start_ceil_end_at_native_rate",
        "forbidden_foreground_contaminants": ["music", "narration", "gunfire", "aircraft"],
    },
    "fingerprint": {
        "algorithm": "abvid.data.overlap.landmarks_plus_waveform_correlation",
        "window_seconds": 4.0,
        "hop_seconds": 2.0,
        "maximum_windows_per_asset": 600,
        "settings": deepcopy(SETTINGS),
        "short_asset_policy": "exact_hash_only_unresolved_fingerprint",
        "no_match_means": "no_detected_reuse_not_proven_independence",
    },
    "evaluation_design": {
        "pilot_models_target": 6,
        "pilot_sessions_per_model_target": 8,
        "model_claim_minimum_verified_groups_per_model": 10,
        "seeds": [17, 29, 43, 71, 101],
        "permutations": 1999,
        "group_bootstrap_replicates": 2000,
        "confidence_level": 0.95,
        "hierarchy_gate": "L1_and_L2_group_macro_mAP_minus_session_null_lower_CI_above_zero",
        "minimum_session_probe_excess": 0.10,
        "vehicle_labels_in_representation_learning_or_hyperparameter_selection": False,
        "phase_gates_use_exposed_development_labels": True,
        "single_outlet_claim": "outlet_invariance_unknown",
        "background_interpretation": "noise_reference_not_identified_channel_inverse",
    },
}


def configuration():
    return deepcopy(SPEC)


def validate(config):
    # Canonical JSON bytes are hashed separately by the P0 freeze ledger. Here
    # exact values protect scope and numerical settings, including JSON types.
    actual = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    expected = json.dumps(SPEC, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if actual != expected:
        raise ValueError("Changed P0 contract: version a new protocol; P0 permits synthetic fixtures only")
    for ref in config["storage"].values():
        resolve(ref, must_exist=False)
    return config


def load(reference=DEFAULT_CONFIG):
    config = validate(json.loads(resolve(reference).read_text()))
    from .manifest import schema
    if json.loads(resolve(config["storage"]["manifest_schema"]).read_text()) != schema():
        raise ValueError("Manifest schema artifact differs from the runtime contract")
    return config


def external_path(reference, *, must_exist=False):
    """Require explicitly configured external storage, including symlink checks."""
    if not isinstance(reference, str) or reference.partition(":")[0] not in {"data", "artifact"}:
        raise ValueError("Preparation media/results must use data: or artifact: storage")
    path = resolve(reference, must_exist=must_exist)
    if path.is_relative_to(roots()["repo"]):
        raise ValueError("Preparation media/results cannot live inside the checkout")
    return path

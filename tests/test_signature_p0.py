"""Synthetic-only P0 contracts; no real media, CAST fits, or learned models."""
import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest
import soundfile as sf

from abvid.provenance.hashing import canonical
from abvid.signature.audio import read_asset
from abvid.signature.config import configuration, external_path, validate as validate_config
from abvid.signature.dedup import audit_manifest, with_reuse_links
from abvid.signature.fixtures import create_bundle, sample_annotations, sample_segment
from abvid.signature.manifest import connected_groups, readiness, schema, validate
from abvid.signature.trim import export_segments, import_annotations, make_packet


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("ABVID_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ABVID_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    cfg = configuration()
    manifest = create_bundle("data:military_signature/p0_fixtures/validation", cfg)
    return manifest, cfg


def _asset(manifest, aid="fixture-a"):
    return next(a for a in manifest["assets"] if a["asset_id"] == aid)


def _packet(manifest, config):
    result = make_packet(manifest, config, "artifact:military_signature/blind",
                         "artifact:military_signature/private/map.json")
    blind = external_path(result["blind_reference"])
    batch = json.loads((blind / "review_batch.json").read_text())
    private = json.loads(external_path(result["private_reference"]).read_text())
    annotations = sample_annotations(batch, result["batch_sha256"])
    return result, batch, private, annotations


def test_canonical_config_and_schema_artifacts_match_runtime():
    from abvid.paths import resolve
    assert json.loads(resolve("repo:experiments/military_signature/p0_protocol.json").read_text()) == json.loads(json.dumps(configuration()))
    assert json.loads(resolve("repo:experiments/military_signature/manifest_schema.json").read_text()) == schema()


@pytest.mark.parametrize("field", ["real_media_collection", "real_media_processing", "model_fitting", "protected_source_evaluation"])
def test_p0_cannot_expand_authorization(field):
    cfg = configuration()
    cfg["authorization"][field] = True
    with pytest.raises(ValueError, match="Changed P0"):
        validate_config(cfg)


def test_strict_manifest_rejects_extra_fields_and_boolean_durations(bundle):
    m, cfg = bundle
    bad = deepcopy(m)
    bad["assets"][0]["invented_session"] = "session-999"
    with pytest.raises(ValueError, match="unexpected fields"):
        validate(bad, cfg)
    bad = deepcopy(m)
    bad["assets"][0]["duration_seconds"] = True
    with pytest.raises(ValueError, match="expected number"):
        validate(bad, cfg)


def test_unknown_session_preserved_and_not_counted_as_independent(bundle):
    m, cfg = bundle
    for a in m["assets"][:2]:
        a.update(recording_session="unknown", session_evidence="unknown", independence_verified=False)
    m["segments"] = [sample_segment("fixture-a"), sample_segment("fixture-b", segment_id="fixture-segment-b")]
    validate(m, cfg)
    assert connected_groups(m)["fixture-a"] == connected_groups(m)["fixture-b"]
    assert all(a["recording_session"] == "unknown" for a in m["assets"][:2])
    assert readiness(m, cfg)["verified_groups_per_fixture_label"] == {}


def test_asset_id_does_not_authorize_independence(bundle):
    m, cfg = bundle
    m["assets"][0].update(recording_session="unknown", session_evidence="unknown")
    with pytest.raises(ValueError, match="Independence requires"):
        validate(m, cfg)


def test_same_session_and_transitive_reuse_cannot_cross_roles(bundle):
    m, cfg = bundle
    a = sample_segment("fixture-a")
    b = sample_segment("fixture-b", segment_id="fixture-segment-b")
    b["role"] = "evaluation"
    m["segments"] = [a, b]
    validate(m, cfg)
    m["reuse_links"] = [
        {"left": "fixture-a", "right": "fixture-short", "kind": "suspected_reuse", "evidence": "fixture linkage"},
        {"left": "fixture-short", "right": "fixture-b", "kind": "provenance", "evidence": "fixture linkage"},
    ]
    with pytest.raises(ValueError, match="group split across roles"):
        validate(m, cfg)


def test_exact_raw_hash_groups_assets_even_if_session_claims_differ(bundle):
    m, cfg = bundle
    duplicate = deepcopy(m["assets"][0])
    duplicate.update(asset_id="fixture-copy", recording_session="different-claimed-session")
    m["assets"].append(duplicate)
    validate(m, cfg)
    groups = connected_groups(m)
    assert groups["fixture-copy"] == groups["fixture-a"]


@pytest.mark.parametrize("kind,field,value", [
    ("foreground", "audible_vehicle_count", 2),
    ("background", "audible_vehicle_count", 1),
    ("foreground", "contaminants", ["music"]),
    ("foreground", "start_seconds", float("nan")),
    ("foreground", "end_seconds", 99),
    ("foreground", "blind_review", False),
    ("rejected", "rejection_reasons", []),
])
def test_invalid_trim_decisions_are_rejected(bundle, kind, field, value):
    m, cfg = bundle
    row = sample_segment("fixture-a", kind=kind)
    row[field] = value
    m["segments"] = [row]
    with pytest.raises(ValueError):
        validate(m, cfg)


def test_overlapping_accepted_background_and_foreground_fail(bundle):
    m, cfg = bundle
    m["segments"] = [sample_segment("fixture-a"),
                     sample_segment("fixture-a", segment_id="background", kind="background", start=3, end=5)]
    with pytest.raises(ValueError, match="overlap"):
        validate(m, cfg)


def test_nonindependent_source_cannot_enter_evaluation(bundle):
    m, cfg = bundle
    m["assets"][0]["independence_verified"] = False
    m["segments"] = [sample_segment("fixture-a")]
    m["segments"][0]["role"] = "evaluation"
    with pytest.raises(ValueError, match="Unverified sessions"):
        validate(m, cfg)


def test_missing_background_is_readiness_failure(bundle):
    m, cfg = bundle
    m["segments"] = [sample_segment("fixture-a")]
    result = readiness(m, cfg)
    assert "fixture-a: no reviewed same-asset background" in result["unresolved"]
    assert not result["model_claim_eligible"] and not result["real_collection_authorized"]


def test_dedup_finds_crop_gain_polarity_and_resampling_without_class_labels(bundle):
    m, cfg = bundle
    result = audit_manifest(m, cfg)
    pairs = {frozenset((p["left"], p["right"])) for p in result["pairs"]}
    assert frozenset(("fixture-a", "fixture-crop")) in pairs
    assert frozenset(("fixture-a", "fixture-resampled")) in pairs
    assert frozenset(("fixture-a", "fixture-b")) not in pairs
    statuses = {r["asset_id"]: r["status"] for r in result["assets"]}
    assert statuses["fixture-short"] == "too_short_for_fingerprint"
    assert statuses["fixture-silence"] == "silent_unresolved"
    assert result["independence_certified"] is False
    altered = deepcopy(m)
    for a in altered["assets"]:
        a["title"] = "Conflicting hidden label"
    again = audit_manifest(altered, cfg)
    assert again["pairs"] == result["pairs"]


def test_stale_dedup_audit_and_role_leakage_fail(bundle):
    m, cfg = bundle
    result = audit_manifest(m, cfg)
    stale = deepcopy(m)
    stale["assets"][0]["title"] = "Changed metadata"
    with pytest.raises(ValueError, match="different manifest"):
        with_reuse_links(stale, result, cfg)
    # Deliberately conflicting synthetic claims: detected waveform reuse must
    # override the claimed independent session IDs.
    m["assets"][2]["recording_session"] = "claimed-independent-crop"
    m["segments"] = [sample_segment("fixture-a"), sample_segment("fixture-crop", segment_id="crop")]
    m["segments"][1]["role"] = "evaluation"
    validate(m, cfg)
    result = audit_manifest(m, cfg)
    with pytest.raises(ValueError, match="group split across roles"):
        with_reuse_links(m, result, cfg)


def test_corrupt_reuse_scores_cannot_create_links(bundle):
    m, cfg = bundle
    result = audit_manifest(m, cfg)
    result["pairs"][0]["correlation"] = 0.1
    with pytest.raises(ValueError, match="acceptance rule"):
        with_reuse_links(m, result, cfg)


def test_known_physical_vehicle_cannot_cross_roles(bundle):
    m, cfg = bundle
    for asset in m["assets"][:2]:
        asset["physical_vehicle_id"] = "known-artificial-individual"
        asset["vehicle_identity_evidence"] = "Synthetic fixture identity declaration"
    m["segments"] = [sample_segment("fixture-a"), sample_segment("fixture-b", segment_id="b")]
    m["segments"][1]["role"] = "evaluation"
    with pytest.raises(ValueError, match="group split across roles"):
        validate(m, cfg)


def test_preparation_workflow_retains_rejections_and_native_samples(bundle):
    from abvid.signature.validation import run
    result = run("data:military_signature/p0_fixtures/workflow",
                 "artifact:military_signature/p0/workflow")
    evidence = json.loads(external_path(result["validation_ref"], must_exist=True).read_text())
    assert result["exported_clips"] == 8 and result["rejections"] == 2
    assert all(evidence["checks"].values())
    assert len(evidence["native_export_checks"]) == 8
    assert evidence["readiness"]["model_claim_eligible"] is False


def test_blind_packet_hides_metadata_and_original_filenames(bundle):
    m, cfg = bundle
    result, batch, private, annotations = _packet(m, cfg)
    blind = external_path(result["blind_reference"])
    text = (blind / "index.html").read_text() + (blind / "review_batch.json").read_text()
    for a in m["assets"]:
        assert a["asset_id"] not in text
        assert a["title"] not in text
        assert a["recording_session"] not in text
    assert len(batch["recordings"]) == len(m["assets"])
    assert not external_path(result["private_reference"]).is_relative_to(blind)
    imported = import_annotations(m, cfg, batch, private, annotations)
    assert len(imported["segments"]) == len(m["assets"])
    assert all(s["kind"] == "rejected" and s["rejection_reasons"] for s in imported["segments"])


def test_private_mapping_cannot_be_inside_blind_directory(bundle):
    m, cfg = bundle
    with pytest.raises(ValueError, match="outside the blind"):
        make_packet(m, cfg, "artifact:blind", "artifact:blind/private.json")


def test_wrong_batch_and_unreviewed_recordings_fail(bundle):
    m, cfg = bundle
    _, batch, private, annotations = _packet(m, cfg)
    annotations["batch_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="batch hash mismatch"):
        import_annotations(m, cfg, batch, private, annotations)
    annotations["batch_sha256"] = hashlib.sha256(canonical(batch)).hexdigest()
    annotations["intervals"].pop()
    with pytest.raises(ValueError, match="Every recording"):
        import_annotations(m, cfg, batch, private, annotations)


def test_altered_private_mapping_is_detected(bundle):
    m, cfg = bundle
    _, batch, private, annotations = _packet(m, cfg)
    keys = list(private["mapping"])
    private["mapping"][keys[0]], private["mapping"][keys[1]] = private["mapping"][keys[1]], private["mapping"][keys[0]]
    with pytest.raises(ValueError, match="commitment mismatch"):
        import_annotations(m, cfg, batch, private, annotations)


def test_native_trim_rounding_preserves_samples_channels_and_rejections(bundle):
    m, cfg = bundle
    stereo, rate = read_asset(_asset(m), cfg)
    stereo = np.column_stack((stereo[:, 0], 0.3 * stereo[:, 0]))
    path = external_path(_asset(m)["media_ref"], must_exist=True)
    sf.write(path, stereo, rate, subtype="FLOAT")
    _asset(m).update(raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), channels=2)
    m["segments"] = [sample_segment("fixture-a", start=2.00003, end=6.00003),
                     sample_segment("fixture-b", segment_id="rejected", kind="rejected")]
    result = export_segments(m, cfg, "artifact:military_signature/trimmed")
    record = result["clips"][0]
    clip, clip_rate = sf.read(external_path(record["clip_ref"]), dtype="float64", always_2d=True)
    expected, _ = read_asset(_asset(m), cfg)
    assert record["start_frame"] == 32000 and record["end_frame"] == 96001
    assert clip_rate == rate and clip.shape[1] == 2
    np.testing.assert_array_equal(clip, expected[32000:96001])
    assert result["rejections"] == [m["segments"][1]]
    with pytest.raises(FileExistsError):
        export_segments(m, cfg, "artifact:military_signature/trimmed")


def test_raw_hash_mismatch_fails_before_audio_is_used(bundle):
    m, cfg = bundle
    _asset(m)["raw_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        read_asset(_asset(m), cfg)


def test_real_input_and_repository_outputs_are_blocked(bundle):
    m, cfg = bundle
    _asset(m)["media_kind"] = "real"
    with pytest.raises(ValueError, match="synthetic-only"):
        validate(m, cfg)
    with pytest.raises(ValueError, match="data: or artifact:"):
        external_path("repo:output/forbidden.wav")
    with pytest.raises(ValueError):
        external_path("data:../ABVID/forbidden.wav")

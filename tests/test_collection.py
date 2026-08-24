from __future__ import annotations

from pathlib import Path

import pytest

from vehicle_audio.collection import (
    _retry_delay_seconds,
    collect_sources,
    is_license_allowed,
    load_catalog,
    sync_catalog_metadata,
)


CATALOG = Path(__file__).parents[1] / "configs" / "audio_sources.yaml"


def test_catalog_has_unique_sources_and_required_layout() -> None:
    document, sources = load_catalog(CATALOG)
    assert document["version"] == 2
    assert len(sources) >= 6
    assert len({source.id for source in sources}) == len(sources)
    assert all(not Path(source.output_path).is_absolute() for source in sources)
    assert all(".." not in Path(source.output_path).parts for source in sources)


def test_catalog_has_tracked_and_wheeled_targets_and_backgrounds() -> None:
    _, sources = load_catalog(CATALOG)
    targets = {source.vehicle_class for source in sources if source.kind == "target"}
    categories = {source.category for source in sources if source.kind == "background"}
    assert {"tracked", "wheeled"} <= targets
    assert {"rain", "generic environmental ambience"} <= categories
    conditions = {
        source.operating_condition for source in sources if source.kind == "target"
    }
    assert {"accelerating", "decelerating", "steady_speed", "startup", "mixed"} <= conditions


def test_target_output_directory_matches_recording_session() -> None:
    _, sources = load_catalog(CATALOG)
    targets = [source for source in sources if source.kind == "target"]

    assert all(source.recording_session for source in targets)
    for source in targets:
        assert Path(source.output_path).parent.name == source.recording_session


def test_known_mpf_reposts_share_one_recording_session() -> None:
    _, sources = load_catalog(CATALOG)
    by_id = {source.id: source for source in sources}

    assert (
        by_id["target-tracked-mpf-firepower"].recording_session
        == by_id["target-tracked-bae-mpf-arrival"].recording_session
    )


def test_ambiguous_tiger131_provenance_remains_review_gated() -> None:
    _, sources = load_catalog(CATALOG)
    tiger = next(
        source
        for source in sources
        if source.id == "target-tracked-tiger131-engine-start"
    )

    assert tiger.status == "review_required"
    assert tiger.review_reason


def test_dvids_humvee_candidate_is_regated_after_access_denial() -> None:
    _, sources = load_catalog(CATALOG)
    humvee = next(
        source
        for source in sources
        if source.id == "candidate-target-wheeled-humvee-southern-strike-2022"
    )

    assert humvee.provider == "direct"
    assert humvee.vehicle_class == "wheeled"
    assert humvee.recording_session == "dvids_southern_strike_humvee_driving_2022"
    assert humvee.expected_license == "Public domain"
    assert humvee.status == "review_required"
    assert humvee.review_reason
    assert "approved one collection attempt on 2026-08-23" in humvee.review_reason
    assert "HTTP 403" in humvee.review_reason


def test_pdsounds_car_is_admitted_after_exact_ingest_and_human_review() -> None:
    _, sources = load_catalog(CATALOG)
    car = next(
        source
        for source in sources
        if source.id == "candidate-target-wheeled-car-start-drive-pdsounds-194"
    )

    assert car.provider == "wikimedia_commons"
    assert car.file_title == "File:Starting a car and driving.ogg"
    assert car.vehicle_class == "wheeled"
    assert car.recording_session == "pdsounds_194_stephan_car_start_drive_2007_04_26"
    assert car.expected_license == "Public domain"
    assert car.status == "approved"
    assert car.admitted_to_corpus is True
    assert car.review_reason is None
    assert [
        (segment.operating_condition, segment.start_seconds, segment.end_seconds)
        for segment in car.condition_segments
    ] == [
        ("startup", 14.5, 17.2),
        ("idle", 17.2, 20.5),
        ("mixed", 20.5, 41.5),
    ]
    assert "confirmed no speech" in car.notes


def test_sherman_locked_candidate_has_reviewed_approach_segment() -> None:
    _, sources = load_catalog(CATALOG)
    sherman = next(
        source
        for source in sources
        if source.id == "candidate-target-tracked-sherman-passby-gvn-43670951"
    )

    assert sherman.status == "approved"
    assert sherman.admitted_to_corpus is True
    assert sherman.provider == "wikimedia_commons"
    assert sherman.expected_license == "CC BY-SA 3.0"
    assert sherman.vehicle_class == "tracked"
    assert sherman.recording_session == "beeldengeluid_gvn_sherman_passby_43670951"
    assert [
        (segment.operating_condition, segment.start_seconds, segment.end_seconds)
        for segment in sherman.condition_segments
    ] == [("mixed", 25.0, 90.0)]
    assert "does not establish performance on modern tracked vehicles" in sherman.notes


def test_human_audio_review_updates_target_segments() -> None:
    _, sources = load_catalog(CATALOG)
    by_id = {source.id: source for source in sources}

    amx30 = by_id["candidate-target-tracked-retromobile-amx30-2015"]
    stug = by_id["target-tracked-stug-iiig-lappeenranta"]
    romanian = by_id["target-tracked-tr85m1-tank-range"]

    assert amx30.condition_segments[0].start_seconds == 9.0
    assert amx30.admitted_to_corpus is True
    assert stug.condition_segments[0].start_seconds == 15.0
    assert stug.admitted_to_corpus is True
    assert romanian.condition_segments == ()
    assert romanian.admitted_to_corpus is False
    assert "No interval is approved for corpus use" in romanian.notes


def test_catalog_license_url_fills_missing_provider_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "vehicle_audio.collection.resolve_commons_file",
        lambda _title: {
            "license": "Public domain",
            "license_url": None,
            "download_url": "https://upload.wikimedia.org/example.ogg",
            "sha1": "abc123",
            "size": 123,
        },
    )

    records = collect_sources(
        CATALOG,
        tmp_path,
        ["candidate-target-wheeled-car-start-drive-pdsounds-194"],
        dry_run=True,
    )

    assert records[0]["result"] == "ready"
    assert records[0]["license_url"] == (
        "https://commons.wikimedia.org/wiki/Template:PD-author"
    )


def test_license_allowlist_is_exact() -> None:
    assert is_license_allowed("Public domain", ["Public domain"])
    assert is_license_allowed("CC BY 4.0", ["CC BY 4.0"])
    assert not is_license_allowed("CC BY-NC 4.0", ["CC BY 4.0"])
    assert not is_license_allowed(None, ["Public domain"])


def test_retry_delay_honors_provider_cooldown() -> None:
    assert _retry_delay_seconds("600", attempt=0) == 600.0
    assert _retry_delay_seconds("1200", attempt=0) == 900.0
    assert _retry_delay_seconds("invalid", attempt=2) == 30.0
    assert _retry_delay_seconds(None, attempt=1) == 20.0


def test_catalog_rejects_path_traversal(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """version: 1\ncollection_policy:\n  allowed_licenses: [Public domain]\nsources:\n  - id: bad\n    kind: target\n    provider: wikimedia_commons\n    source_page: https://example.invalid\n    output_path: ../outside.wav\n    status: approved\n    expected_license: Public domain\n""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="output_path"):
        load_catalog(path)


def test_catalog_rejects_non_boolean_corpus_admission(tmp_path: Path) -> None:
    path = tmp_path / "bad-admission.yaml"
    path.write_text(
        """version: 1\ncollection_policy:\n  allowed_licenses: [Public domain]\nsources:\n  - id: bad\n    kind: target\n    provider: direct\n    source_page: https://example.invalid\n    output_path: targets/tracked/session/clip.wav\n    status: approved\n    expected_license: Public domain\n    admitted_to_corpus: yes-please\n""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="admitted_to_corpus"):
        load_catalog(path)


def test_sync_catalog_metadata_updates_collected_target_sidecar(tmp_path: Path) -> None:
    _, sources = load_catalog(CATALOG)
    source = next(item for item in sources if item.id == "target-wheeled-abarth-205-goodwood")
    audio_path = tmp_path / source.output_path
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"placeholder")
    audio_path.with_suffix(".json").write_text(
        '{"source_id": "target-wheeled-abarth-205-goodwood"}\n',
        encoding="utf-8",
    )

    records = sync_catalog_metadata(CATALOG, tmp_path)

    record = next(item for item in records if item["source_id"] == source.id)
    assert record["result"] == "updated"
    metadata = __import__("json").loads(audio_path.with_suffix(".json").read_text())
    assert metadata["operating_condition"] == "accelerating"
    assert metadata["vehicle_class"] == "wheeled"
    assert metadata["admitted_to_corpus"] is True

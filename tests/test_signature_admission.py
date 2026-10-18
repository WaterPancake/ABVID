"""Offline boundary checks for the separately scoped real-media collector."""
import json

import pytest
import numpy as np

from abvid.signature import admission
from abvid.signature.admission_annotations import validate_intervals
from abvid.signature import admission_review
from abvid.signature.admission_reuse_votes import votes as packed_votes
from abvid.signature.admission_reuse_correlation import correlation as accelerated_correlation
from abvid.data.overlap import correlation as original_correlation


def test_canonical_authorization_rejects_budget_drift(tmp_path, monkeypatch):
    original = admission.load_config()
    assert original["storage"]["maximum_total_bytes"] == 5_000_000_000
    assert original["collection"]["maximum_asset_seconds"] == 1200
    original["authorization"]["model_fitting"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(original))
    monkeypatch.setattr(admission, "resolve", lambda reference: changed)
    with pytest.raises(ValueError, match="Changed P1"):
        admission.load_config()


def test_network_guard_refuses_private_or_unrelated_urls():
    for url in [
        "http://www.dvidshub.net/video/1",
        "https://127.0.0.1/video/1",
        "https://www.dvidshub.net@evil.test/video/1",
        "https://d34w7g4gy10iej.cloudfront.net:443/video/example.mp4",
        "file:///etc/passwd",
    ]:
        with pytest.raises(ValueError):
            admission.check_url(url, media=True)
    admission.check_url("https://d34w7g4gy10iej.cloudfront.net/video/example.mp4", media=True)
    with pytest.raises(ValueError):
        admission.check_url("https://d34w7g4gy10iej.cloudfront.net/private/example.mp4", media=True)


def test_finite_playlist_does_not_trim_an_overlong_asset():
    playlist = "#EXTM3U\n#EXTINF:1199,\na.ts\n#EXTINF:2,\nb.ts\n#EXT-X-ENDLIST\n"
    with pytest.raises(ValueError, match="exceeds"):
        admission.validate_playlist(playlist, "https://d34w7g4gy10iej.cloudfront.net/video/a/index.m3u8", 1200)
    for unsafe in [
        playlist.replace("1199", "nan"),
        playlist.replace("1199", "2").replace("a.ts", "https://evil.test/a.ts"),
        playlist.replace("1199", "2").replace("#EXT-X-ENDLIST", "#EXT-X-KEY:METHOD=AES-128"),
        playlist.replace("1199", "2").replace("#EXT-X-ENDLIST", ""),
    ]:
        with pytest.raises(ValueError):
            admission.validate_playlist(unsafe, "https://d34w7g4gy10iej.cloudfront.net/video/a/index.m3u8", 1200)
    assert admission.validate_playlist(
        playlist.replace("1199", "1198"),
        "https://d34w7g4gy10iej.cloudfront.net/video/a/index.m3u8", 1200
    ) == 1200


def test_rendition_selection_avoids_lower_quality_audio_only_and_above_720p():
    base = "https://d34w7g4gy10iej.cloudfront.net/video/a/"
    master = (
        "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=64000,CODECS=mp4a\n" + base + "audio.m3u8\n"
        "#EXT-X-STREAM-INF:BANDWIDTH=2000000,RESOLUTION=1280x720\n" + base + "720.m3u8\n"
        "#EXT-X-STREAM-INF:BANDWIDTH=5000000,RESOLUTION=1920x1080\n" + base + "1080.m3u8\n"
    )
    assert admission.select_hls(master)["url"] == base + "720.m3u8"


def test_rights_banner_does_not_override_asset_restriction():
    row = {
        "caption": "Editorial use only. Leopard tank training.",
        "copyright_info": "DVIDS copyright notice",
        "unit": "Natochannel",
        "public_domain_banner": True,
    }
    assert admission.rights_decision(row)["decision"] == "quarantine"
    row.update(caption="M1151 driver training.", unit="128th Mobile Public Affairs Detachment")
    assert admission.rights_decision(row)["decision"] == "collect_for_local_noncommercial_research"
    row["public_domain_banner"] = False
    assert admission.rights_decision(row)["decision"] == "quarantine"


def test_caption_names_do_not_invent_tank_identity_or_sessions():
    assert admission.caption_classes("M1 heavy wrecker truck at a parade") == []
    assert admission.caption_classes("General Robert Abrams visits a clinic") == []
    assert admission.caption_classes("M1A2 Abrams tanks and M109A6 Paladin convoy") == ["abrams", "m109"]


def test_storage_budget_counts_data_and_artifacts_and_rejects_symlink(tmp_path, monkeypatch):
    data, artifacts = tmp_path / "data", tmp_path / "artifacts"
    data.mkdir()
    artifacts.mkdir()
    (data / "source.mp4").write_bytes(b"a" * 13)
    (artifacts / "viewer.html").write_bytes(b"b" * 17)
    cfg = {"storage": {"data_root": "data:test", "artifact_root": "artifact:test"}}
    monkeypatch.setattr(admission, "external_path", lambda ref: data if ref.startswith("data:") else artifacts)
    assert admission.storage_used(cfg) == 30
    (data / "alias").symlink_to(artifacts / "viewer.html")
    with pytest.raises(ValueError, match="Symlinks"):
        admission.storage_used(cfg)


def test_oversize_download_is_rejected_before_any_payload(tmp_path, monkeypatch):
    class Response:
        headers = {"Content-Length": "1001"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, *args):
            raise AssertionError("Oversize media payload must not be read")

    monkeypatch.setattr(admission, "response", lambda *args, **kwargs: Response())
    monkeypatch.setattr(admission, "available_bytes", lambda cfg: 1100)
    cfg = {"collection": {"maximum_encoded_asset_bytes": 1000},
           "storage": {"reserved_workspace_bytes": 100}}
    destination = tmp_path / "source.mp4"
    with pytest.raises(OSError, match="Advertised"):
        admission.download_file("https://d34w7g4gy10iej.cloudfront.net/video/a.mp4", destination, cfg)
    assert not destination.exists()


def test_octet_stream_playlist_requires_an_actual_hls_header(tmp_path, monkeypatch):
    class Headers:
        def get_content_type(self):
            return "application/octet-stream"

    class Response:
        headers = Headers()
        payload = b"#EXTM3U\n#EXTINF:2,\na.ts\n#EXT-X-ENDLIST\n"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            return self.payload

    mock = Response()
    monkeypatch.setattr(admission, "response", lambda *args, **kwargs: mock)
    monkeypatch.setattr(admission, "available_bytes", lambda cfg: 10000)
    cfg = {"collection": {"maximum_metadata_bytes": 1000}}
    url = "https://d34w7g4gy10iej.cloudfront.net/video/a.m3u8"
    assert admission.fetch_text(url, cfg, tmp_path / "playlist.m3u8", media=True).startswith("#EXTM3U")
    mock.payload = b"<html>not a playlist</html>"
    with pytest.raises(ValueError, match="HLS manifest"):
        admission.fetch_text(url, cfg, tmp_path / "wrong.m3u8", media=True)
    assert not (tmp_path / "wrong.m3u8").exists()


def test_review_controls_have_no_caption_identity_or_fixture_dependency():
    key = "review-" + "a" * 24
    html = admission_review.card(key, 1, 4.0, np.ones((32000, 1), dtype=np.float32))
    assert f'/audio/{key}.m4a' in html
    assert "Recording 1" in html and html.count("<svg") == 2
    for leaked in ["fixture", "Abrams", "Leopard", "dvidshub", "__BLIND_ID__", "__AUDIO_ROUTE__"]:
        assert leaked not in html
    with pytest.raises(ValueError, match="opaque"):
        admission_review.card("source-1754", 1, 4.0, np.ones((32000, 1)))


def test_human_interval_import_enforces_review_completeness_and_contamination():
    key = "review-" + "a" * 24
    recording = [{"blind_id": key, "duration_seconds": 8.0}]
    interval = {
        "blind_id": key, "start_seconds": 2.0, "end_seconds": 6.0,
        "kind": "foreground", "audible_vehicle_count": 1, "motion_state": "unknown",
        "contaminants": [], "rejection_reasons": [],
    }
    assert validate_intervals([interval], recording) == [interval]
    with pytest.raises(ValueError, match="Every recording"):
        validate_intervals([], recording)
    with pytest.raises(ValueError, match="Contaminated"):
        validate_intervals([{**interval, "contaminants": ["narration"]}], recording)
    with pytest.raises(ValueError, match="overlap"):
        validate_intervals([interval, {**interval, "start_seconds": 3.0}], recording)
    with pytest.raises(ValueError, match="finite"):
        validate_intervals([{**interval, "start_seconds": True}], recording)
    with pytest.raises(ValueError, match="Unexpected"):
        validate_intervals([{**interval, "experimental_role": "test"}], recording)


def test_packed_landmark_votes_preserve_reference_ids_negative_offsets_and_counts():
    rng = np.random.default_rng(17)
    refs = rng.integers(0, 100, size=2000, dtype=np.int64)
    times = rng.integers(0, 120, size=2000, dtype=np.int64)
    starts = rng.integers(0, 1800, size=400, dtype=np.int64)
    ends = starts + rng.integers(0, 200, size=400, dtype=np.int64)
    query_times = rng.integers(0, 2000, size=400, dtype=np.int64)
    pairs = np.concatenate([
        np.column_stack((refs[a:b], times[a:b] - t))
        for a, b, t in zip(starts, ends, query_times) if b > a
    ])
    expected_keys, expected_counts = np.unique(pairs, axis=0, return_counts=True)
    keys, counts = packed_votes(refs, times, starts, ends, query_times)
    assert np.array_equal(keys, expected_keys)
    assert np.array_equal(counts, expected_counts)
    keys, counts = packed_votes(refs, times, np.zeros(0, np.int64),
                               np.zeros(0, np.int64), np.zeros(0, np.int64))
    assert keys.shape == (0, 2) and len(counts) == 0


def test_accelerated_lag_search_returns_original_scores_for_edges_and_dynamic_range():
    rng = np.random.default_rng(29)
    for n in (64, 512, 32000):
        reference = rng.normal(size=n)
        query = np.r_[rng.normal(size=256), -.45 * reference, rng.normal(size=256)]
        for start in (-300, 0, 128, 256, len(query), len(query) - n):
            assert accelerated_correlation(reference, query, start, 256) == original_correlation(
                reference, query, start, 256)
        quiet = query * 1e-12
        quiet[0] = 1.0
        assert accelerated_correlation(reference, quiet, 256, 256) == original_correlation(
            reference, quiet, 256, 256)
    for reference, query in [
        (np.zeros(32000), np.zeros(33000)),
        (np.ones(32000), np.ones(33000)),
        (np.sin(np.arange(32000) / 7), np.sin(np.arange(33000) / 7)),
    ]:
        assert accelerated_correlation(reference, query, 400, 256) == original_correlation(
            reference, query, 400, 256)

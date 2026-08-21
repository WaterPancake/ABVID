from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import pytest

from vehicle_audio.audio import load_audio, save_audio
from vehicle_audio.real_corpus import RealCorpusConfig, prepare_real_corpus


def _write_real_source(
    root: Path,
    vehicle_class: str,
    session: str,
    frequency: float,
) -> None:
    sample_rate = 8_000
    time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    mono = 0.25 * torch.sin(2.0 * torch.pi * frequency * time)
    waveform = torch.stack((0.5 * mono, mono))
    path = root / vehicle_class / session / "source.wav"
    save_audio(path, waveform, sample_rate)
    sidecar = {
        "source_id": f"source-{vehicle_class}-{session}",
        "source_page": f"https://example.test/{vehicle_class}/{session}",
        "recording_session": session,
        "vehicle_class": vehicle_class,
        "vehicle_model": f"{vehicle_class}-{session}",
        "operating_condition": "steady_speed",
        "license": "CC0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "attribution": "Test fixture",
        "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "content_review_status": "reviewed_segment",
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")


def test_real_corpus_is_grouped_deterministic_and_does_not_infer_snr(tmp_path: Path) -> None:
    targets = tmp_path / "targets"
    for vehicle_class, base_frequency in (("tracked", 150.0), ("wheeled", 350.0)):
        for session_index in range(3):
            _write_real_source(
                targets,
                vehicle_class,
                f"{vehicle_class}_session_{session_index}",
                base_frequency + session_index,
            )
    config = RealCorpusConfig(
        sample_rate=8_000,
        window_seconds=0.25,
        hop_seconds=0.25,
        channel=1,
        max_windows_per_session=2,
    )
    first = prepare_real_corpus(config, targets, tmp_path / "first")
    second = prepare_real_corpus(config, targets, tmp_path / "second")

    assert first["observation_count"] == 12
    assert first["recording_session_counts"] == {"tracked": 3, "wheeled": 3}
    assert first["full_protocol_ready"] is True
    first_manifest = (tmp_path / "first" / "real_manifest.jsonl").read_bytes()
    second_manifest = (tmp_path / "second" / "real_manifest.jsonl").read_bytes()
    assert first_manifest == second_manifest

    rows = [json.loads(line) for line in first_manifest.splitlines()]
    assert all(row["snr_db"] is None for row in rows)
    assert all(row["source_domain"] == "real_recording" for row in rows)
    assert all(row["selected_channel"] == 1 for row in rows)
    assert all(row["provenance_complete"] for row in rows)
    for row in rows:
        first_audio = tmp_path / "first" / row["audio_path"]
        second_audio = tmp_path / "second" / row["audio_path"]
        assert first_audio.read_bytes() == second_audio.read_bytes()
        waveform, sample_rate = load_audio(first_audio)
        assert waveform.shape == (1, 2_000)
        assert sample_rate == 8_000
        assert torch.isfinite(waveform).all()


def test_real_corpus_rejects_reviewed_segment_beyond_source_duration(
    tmp_path: Path,
) -> None:
    targets = tmp_path / "targets"
    _write_real_source(targets, "tracked", "tracked_session", 150.0)
    _write_real_source(targets, "wheeled", "wheeled_session", 350.0)
    sidecar_path = targets / "tracked" / "tracked_session" / "source.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["condition_segments"] = [
        {
            "operating_condition": "steady_speed",
            "start_seconds": 0.25,
            "end_seconds": 1.25,
        }
    ]
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ValueError, match="reviewed segment ends outside"):
        prepare_real_corpus(
            RealCorpusConfig(sample_rate=8_000, window_seconds=0.25),
            targets,
            tmp_path / "output",
        )

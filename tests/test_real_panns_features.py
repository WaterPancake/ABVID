from __future__ import annotations

import json
from pathlib import Path

import pytest

from vehicle_audio.real_panns_features import extract_real_panns_features


def test_real_panns_extractor_rejects_protected_source_before_model_loading(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "real_manifest.jsonl"
    rows = []
    for vehicle_class in ("tracked", "wheeled"):
        for index in range(5):
            source_id = (
                "protected"
                if vehicle_class == "tracked" and index == 0
                else f"source-{vehicle_class}-{index}"
            )
            rows.append(
                {
                    "sample_id": f"{vehicle_class}-{index}",
                    "audio_path": "missing.wav",
                    "metadata_path": "missing.json",
                    "vehicle_class": vehicle_class,
                    "recording_session": f"{vehicle_class}-session-{index}",
                    "source_domain": "real_recording",
                    "source_id": source_id,
                    "sample_rate": 16_000,
                    "num_channels": 1,
                    "num_samples": 32_000,
                    "window_start_sample": 0,
                    "snr_db": None,
                    "provenance_complete": True,
                    "content_review_status": "reviewed_segment",
                }
            )
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="contains protected sources"):
        extract_real_panns_features(
            manifest,
            tmp_path / "missing-checkpoint.pth",
            tmp_path / "features.pt",
            forbidden_source_ids=["protected"],
        )

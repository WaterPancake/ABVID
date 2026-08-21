from __future__ import annotations

import json
from pathlib import Path

import torch

from vehicle_audio.audio import save_audio
from vehicle_audio.config import GenerationConfig
from vehicle_audio.factorial_dataset import FactorialDatasetGenerator


def _inputs(root: Path) -> tuple[Path, Path]:
    sample_rate = 8_000
    time = torch.arange(1_600, dtype=torch.float32) / sample_rate
    targets = root / "targets"
    backgrounds = root / "backgrounds"
    for vehicle_class, frequency, geometry in (
        ("tracked", 180.0, "near_broadside"),
        ("wheeled", 380.0, "far_offset"),
    ):
        path = targets / vehicle_class / f"{vehicle_class}.wav"
        save_audio(
            path,
            (0.2 * torch.sin(2 * torch.pi * frequency * time)).unsqueeze(0),
            sample_rate,
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "vehicle_class": vehicle_class,
                    "vehicle_id": f"toy_{vehicle_class}",
                    "recording_session": f"toy_{vehicle_class}_session",
                    "source_domain": "procedural_synthetic",
                    "operating_condition": "idle",
                    "source_listener_geometry": {"geometry_id": geometry},
                }
            ),
            encoding="utf-8",
        )
    for category, offset in (("rain", 0), ("road traffic", 1)):
        generator = torch.Generator().manual_seed(100 + offset)
        path = backgrounds / category.replace(" ", "_") / "background.wav"
        save_audio(
            path,
            (0.1 * torch.randn(1, 1_600, generator=generator)),
            sample_rate,
        )
        path.with_suffix(".json").write_text(
            json.dumps({"category": category}), encoding="utf-8"
        )
    return targets, backgrounds


def _config() -> GenerationConfig:
    return GenerationConfig.from_mapping(
        {
            "audio": {"sample_rate": 8_000, "duration_seconds": 0.05},
            "gain": {"enabled": False},
            "noise": {
                "enabled": True,
                "probability": 1.0,
                "snr_db_choices": [10.0, 0.0],
            },
            "lowpass": {"enabled": False},
            "highpass": {"enabled": False},
            "bandpass": {"enabled": False},
            "random_eq": {"enabled": False},
            "resampling": {"enabled": False},
            "clipping": {"enabled": False},
            "compression": {"enabled": False},
            "reverb": {"enabled": False},
            "microphone_response": {
                "enabled": True,
                "probability": 1.0,
                "anchor_frequencies_hz": [0.0, 1_000.0, 4_000.0],
                "min_gain_db": -4.0,
                "max_gain_db": 2.0,
            },
        }
    )


def _generate(inputs: Path, output: Path) -> dict[str, object]:
    targets, backgrounds = _inputs(inputs)
    return FactorialDatasetGenerator(
        _config(), targets, backgrounds, output
    ).generate(seed=42)


def test_factorial_generation_is_balanced_paired_and_deterministic(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    first = tmp_path / "first"
    second = tmp_path / "second"
    summary = _generate(inputs, first)
    _generate(inputs, second)

    assert summary["generated_samples"] == 16
    assert summary["base_event_count"] == 2
    assert summary["vehicle_class_counts"] == {"tracked": 8, "wheeled": 8}
    assert summary["background_category_counts"] == {
        "rain": 8,
        "road traffic": 8,
    }
    assert summary["snr_db_counts"] == {0.0: 8, 10.0: 8}
    assert summary["microphone_response_counts"] == {"False": 8, "True": 8}
    assert (first / "manifest.jsonl").read_bytes() == (
        second / "manifest.jsonl"
    ).read_bytes()

    records = [
        json.loads(line)
        for line in (first / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for record in records:
        assert record["factorial_dataset_version"] == 1
        assert abs(
            record["augmentations"]["noise"]["measured_snr_db_at_mix"]
            - record["snr_db"]
        ) < 0.05
        other = next(row for row in records if row["sample_id"] == record["sample_id"])
        assert (first / record["corrupted_path"]).read_bytes() == (
            second / other["corrupted_path"]
        ).read_bytes()

    tracked_rain = [
        record
        for record in records
        if record["vehicle_class"] == "tracked"
        and record["background_category"] == "rain"
    ]
    assert len({(row["snr_db"], row["augmentations"]["microphone_response"]["applied"]) for row in tracked_rain}) == 4
    assert len({(first / row["clean_path"]).read_bytes() for row in tracked_rain}) == 1
    without_microphone = {
        row["snr_db"]: (first / row["corrupted_path"]).read_bytes()
        for row in tracked_rain
        if not row["augmentations"]["microphone_response"]["applied"]
    }
    assert without_microphone[0.0] != without_microphone[10.0]
    for snr_db in (0.0, 10.0):
        pair = [row for row in tracked_rain if row["snr_db"] == snr_db]
        assert (first / pair[0]["corrupted_path"]).read_bytes() != (
            first / pair[1]["corrupted_path"]
        ).read_bytes()

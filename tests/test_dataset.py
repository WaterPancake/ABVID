from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from vehicle_audio.config import GenerationConfig
from vehicle_audio.dataset import DatasetGenerator, discover_target_recordings
from vehicle_audio.manifest import REQUIRED_MANIFEST_FIELDS


def _write_wave(path: Path, values: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, values.astype(np.float32), sample_rate, format="WAV", subtype="FLOAT")


def _build_inputs(root: Path) -> tuple[Path, Path, Path]:
    target_root = root / "targets"
    background_root = root / "backgrounds"
    ir_root = root / "impulse_responses"

    time = np.arange(2_400) / 12_000.0
    # A stereo source checks that generation never collapses target channels.
    target = np.stack(
        [0.15 * np.sin(2 * np.pi * 170 * time), 0.1 * np.sin(2 * np.pi * 260 * time)],
        axis=1,
    )
    target_path = target_root / "tracked" / "session_alpha" / "clip.wav"
    _write_wave(target_path, target, 12_000)
    target_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "vehicle_class": "tracked",
                "vehicle_model": None,
                "recording_session": "session_alpha",
                "source_domain": "synthetic_toy",
                "operating_condition": "mixed",
                "condition_segments": [
                    {
                        "operating_condition": "accelerating",
                        "start_seconds": 0.05,
                        "end_seconds": 0.18,
                        "notes": "Synthetic test segment.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    generator = np.random.default_rng(7)
    background_path = background_root / "wind" / "clip.wav"
    background = np.stack(
        [
            generator.normal(0.0, 0.1, 1_600),
            generator.normal(0.0, 0.1, 1_600),
        ],
        axis=1,
    )
    _write_wave(background_path, background, 8_000)
    background_path.with_suffix(".json").write_text(
        json.dumps({"category": "wind"}), encoding="utf-8"
    )

    impulse_response = np.zeros(160, dtype=np.float32)
    impulse_response[[0, 40, 90]] = [1.0, 0.25, 0.1]
    _write_wave(ir_root / "room.wav", impulse_response, 8_000)
    return target_root, background_root, ir_root


def _test_config() -> GenerationConfig:
    return GenerationConfig.from_mapping(
        {
            "audio": {"sample_rate": 8_000, "duration_seconds": 0.1},
            "gain": {"enabled": True, "probability": 1.0, "min_db": -6.0, "max_db": 0.0},
            "noise": {"enabled": True, "probability": 1.0, "snr_db_choices": [5.0]},
            "lowpass": {
                "enabled": True,
                "probability": 1.0,
                "min_cutoff_hz": 2_000.0,
                "max_cutoff_hz": 3_000.0,
                "order": 2,
            },
            "highpass": {"enabled": False},
            "bandpass": {"enabled": False},
            "random_eq": {"enabled": False},
            "resampling": {
                "enabled": True,
                "probability": 1.0,
                "intermediate_rates": [4_000],
            },
            "clipping": {
                "enabled": True,
                "probability": 1.0,
                "min_threshold": 0.8,
                "max_threshold": 0.8,
            },
            "compression": {"enabled": False},
            "reverb": {"enabled": True, "probability": 1.0},
            "microphone_response": {"enabled": False},
        }
    )


def test_discovery_skips_explicitly_unadmitted_target(tmp_path) -> None:
    target_root = tmp_path / "targets"
    admitted = target_root / "tracked" / "admitted" / "clip.wav"
    pending = target_root / "wheeled" / "pending" / "clip.wav"
    _write_wave(admitted, np.zeros(800, dtype=np.float32), 8_000)
    _write_wave(pending, np.zeros(800, dtype=np.float32), 8_000)
    pending.with_suffix(".json").write_text(
        json.dumps({"admitted_to_corpus": False}),
        encoding="utf-8",
    )

    sources = discover_target_recordings(target_root)

    assert [source.path for source in sources] == [admitted]


def _run_generation(inputs_root: Path, output_root: Path, seed: int) -> dict[str, object]:
    targets, backgrounds, impulse_responses = _build_inputs(inputs_root)
    return DatasetGenerator(
        _test_config(),
        targets,
        backgrounds,
        output_root,
        impulse_responses_root=impulse_responses,
    ).generate(num_samples=2, seed=seed)


def test_dataset_has_paired_files_valid_manifest_shape_and_sample_rate(tmp_path) -> None:
    output = tmp_path / "generated"
    summary = _run_generation(tmp_path / "inputs", output, seed=42)

    assert summary["generated_samples"] == 2
    records = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert len(records) == 2
    for record in records:
        assert REQUIRED_MANIFEST_FIELDS <= record.keys()
        assert record["recording_session"] == "session_alpha"
        assert record["vehicle_class"] == "tracked"
        assert record["source_domain"] == "synthetic_toy"
        assert record["operating_condition"] == "accelerating"
        assert record["target_segment_start_seconds"] == 0.05
        assert record["target_segment_end_seconds"] == 0.18
        assert 600 <= record["target_crop_start_frame"] <= 960
        assert record["target_crop_start_frame"] + 1_200 <= 2_160
        assert record["background_category"] == "wind"
        assert record["sample_rate"] == 8_000
        assert record["num_channels"] == 2
        assert record["num_samples"] == 800
        assert record["snr_db"] == 5.0
        assert record["target_original_sample_rate"] == 12_000
        assert record["background_original_num_channels"] == 2
        assert record["background_channel_selected"] == 0
        assert (output / record["clean_path"]).exists()
        assert (output / record["corrupted_path"]).exists()
        metadata = json.loads((output / record["metadata_path"]).read_text())
        assert metadata == record
        clean_info = sf.info(output / record["clean_path"])
        corrupted_info = sf.info(output / record["corrupted_path"])
        assert (clean_info.samplerate, clean_info.channels, clean_info.frames) == (8_000, 2, 800)
        assert (corrupted_info.samplerate, corrupted_info.channels, corrupted_info.frames) == (
            8_000,
            2,
            800,
        )


def test_fixed_seed_is_byte_reproducible_and_different_seed_changes_output(tmp_path) -> None:
    inputs = tmp_path / "inputs"
    first = tmp_path / "first"
    second = tmp_path / "second"
    third = tmp_path / "third"
    _run_generation(inputs, first, seed=123)
    _run_generation(inputs, second, seed=123)
    _run_generation(inputs, third, seed=124)

    first_manifest = (first / "manifest.jsonl").read_bytes()
    second_manifest = (second / "manifest.jsonl").read_bytes()
    third_manifest = (third / "manifest.jsonl").read_bytes()
    assert first_manifest == second_manifest
    assert first_manifest != third_manifest

    first_record = json.loads(first_manifest.splitlines()[0])
    second_record = json.loads(second_manifest.splitlines()[0])
    third_record = json.loads(third_manifest.splitlines()[0])
    assert (first / first_record["corrupted_path"]).read_bytes() == (
        second / second_record["corrupted_path"]
    ).read_bytes()
    assert (first / first_record["corrupted_path"]).read_bytes() != (
        third / third_record["corrupted_path"]
    ).read_bytes()

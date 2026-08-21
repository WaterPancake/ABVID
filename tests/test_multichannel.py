from __future__ import annotations

from pathlib import Path
import json

import pytest
import torch

from vehicle_audio.array_dataset import ArrayDatasetGenerator, REQUIRED_ARRAY_FIELDS
from vehicle_audio.audio import load_audio, save_audio
from vehicle_audio.mixing import measure_snr_db
from vehicle_audio.multichannel import (
    ArrayMicrophoneResponseConfig,
    ArrayReverberationConfig,
    ArraySimulationConfig,
    angular_error_degrees,
    beamformed_snr_db,
    fractional_delay,
    gcc_phat_azimuth,
    gcc_phat_delay_samples,
    load_array_config,
    simulate_microphone_array,
    srp_phat_azimuth,
)


def _anechoic_config() -> ArraySimulationConfig:
    return ArraySimulationConfig(
        sample_rate=8_000,
        duration_seconds=1.0,
        reverberation=ArrayReverberationConfig(enabled=False),
        microphone_response=ArrayMicrophoneResponseConfig(
            enabled=False,
            anchor_frequencies_hz=(0.0, 4_000.0),
        ),
        independent_sensor_noise_fraction=0.0,
    )


def test_checked_array_config_is_valid() -> None:
    config = load_array_config(Path("configs/multichannel.yaml"))

    assert config.sample_rate == 16_000
    assert config.num_samples == 32_000
    assert config.microphone_counts == (1, 2, 4)
    assert len(config.microphone_positions_m) == 4


def test_fractional_delay_and_gcc_phat_recover_known_delay() -> None:
    source = torch.randn((1, 4_000), generator=torch.Generator().manual_seed(4))
    delayed = fractional_delay(source, 3.25)

    estimate = gcc_phat_delay_samples(delayed[0], source[0], interpolation=16)

    assert delayed.shape == source.shape
    assert estimate == pytest.approx(3.25, abs=0.15)


def test_array_simulation_is_deterministic_finite_and_has_requested_snr() -> None:
    config = _anechoic_config()
    source_generator = torch.Generator().manual_seed(12)
    source = torch.randn((1, config.num_samples), generator=source_generator)
    background = torch.randn(
        (len(config.microphone_positions_m), config.num_samples),
        generator=source_generator,
    )

    first = simulate_microphone_array(
        source,
        background,
        config,
        azimuth_deg=30.0,
        source_distance_m=20.0,
        snr_db=5.0,
        generator=torch.Generator().manual_seed(91),
    )
    second = simulate_microphone_array(
        source,
        background,
        config,
        azimuth_deg=30.0,
        source_distance_m=20.0,
        snr_db=5.0,
        generator=torch.Generator().manual_seed(91),
    )

    assert first.observation.shape == (4, config.num_samples)
    assert torch.isfinite(first.observation).all()
    torch.testing.assert_close(first.observation, second.observation)
    assert first.metadata == second.metadata
    assert measure_snr_db(first.propagated_clean, first.scaled_noise) == pytest.approx(
        5.0, abs=1e-5
    )


def test_gcc_and_srp_phat_recover_known_broadside_azimuth() -> None:
    config = _anechoic_config()
    generator = torch.Generator().manual_seed(72)
    source = torch.randn((1, config.num_samples), generator=generator)
    background = torch.randn((4, config.num_samples), generator=generator)
    result = simulate_microphone_array(
        source,
        background,
        config,
        azimuth_deg=-30.0,
        source_distance_m=30.0,
        snr_db=30.0,
        generator=generator,
    )

    gcc_estimate = gcc_phat_azimuth(
        result.propagated_clean,
        config.microphone_positions_m,
        config.sample_rate,
    )
    srp_estimate, scores = srp_phat_azimuth(
        result.propagated_clean,
        config.microphone_positions_m,
        config.sample_rate,
    )

    assert scores.shape == (171,)
    assert angular_error_degrees(gcc_estimate, -30.0) < 3.0
    assert angular_error_degrees(srp_estimate, -30.0) < 3.0


def test_oracle_delay_and_sum_improves_independent_noise_snr() -> None:
    config = _anechoic_config()
    generator = torch.Generator().manual_seed(19)
    source = torch.randn((1, config.num_samples), generator=generator)
    background = torch.randn((4, config.num_samples), generator=generator)
    result = simulate_microphone_array(
        source,
        background,
        config,
        azimuth_deg=45.0,
        source_distance_m=20.0,
        snr_db=0.0,
        generator=generator,
    )

    single_snr = measure_snr_db(
        result.propagated_clean[:1],
        result.scaled_noise[:1],
    )
    array_snr = beamformed_snr_db(
        result.propagated_clean,
        result.scaled_noise,
        config.microphone_positions_m,
        45.0,
        config.sample_rate,
    )

    assert array_snr > single_snr + 4.0


def _array_dataset_inputs(root: Path) -> tuple[Path, Path]:
    sample_rate = 8_000
    source_root = root / "sources"
    source_root.mkdir(parents=True)
    records = []
    time = torch.arange(1_600) / sample_rate
    for index, vehicle_class in enumerate(("tracked", "wheeled")):
        relative = Path(f"source_{index}") / "clean.wav"
        waveform = (0.2 * torch.sin(2 * torch.pi * (180 + 100 * index) * time)).reshape(1, -1)
        save_audio(source_root / relative, waveform, sample_rate)
        records.append(
            {
                "sample_id": f"source_{index}",
                "recording_session": f"session_{vehicle_class}",
                "vehicle_class": vehicle_class,
                "vehicle_id": f"vehicle_{vehicle_class}",
                "source_domain": "synthetic_toy",
                "operating_condition": "steady_speed",
                "clean_path": relative.as_posix(),
                "sample_rate": sample_rate,
            }
        )
    manifest = source_root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    background_root = root / "backgrounds"
    noise = torch.randn((2, 2_000), generator=torch.Generator().manual_seed(81)) * 0.1
    save_audio(background_root / "wind" / "noise.wav", noise, sample_rate)
    return manifest, background_root


def test_array_dataset_is_paired_with_source_group_and_byte_reproducible(tmp_path) -> None:
    manifest, backgrounds = _array_dataset_inputs(tmp_path / "inputs")
    config = ArraySimulationConfig(
        sample_rate=8_000,
        duration_seconds=0.1,
        microphone_positions_m=((-0.04, 0.0, 1.0), (0.04, 0.0, 1.0)),
        microphone_counts=(1, 2),
        snr_db_choices=(5.0, -5.0),
        azimuth_deg_choices=(-30.0, 30.0),
        source_distance_m_choices=(10.0,),
        reverberation=ArrayReverberationConfig(enabled=False),
        microphone_response=ArrayMicrophoneResponseConfig(
            enabled=False,
            anchor_frequencies_hz=(0.0, 4_000.0),
        ),
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_summary = ArrayDatasetGenerator(config, manifest, backgrounds, first).generate(4, 42)
    ArrayDatasetGenerator(config, manifest, backgrounds, second).generate(4, 42)

    first_bytes = (first / "array_manifest.jsonl").read_bytes()
    assert first_bytes == (second / "array_manifest.jsonl").read_bytes()
    records = [json.loads(line) for line in first_bytes.splitlines()]
    assert first_summary["vehicle_class_counts"] == {"tracked": 2, "wheeled": 2}
    assert first_summary["base_event_count"] == 2
    assert records[0]["base_event_id"] == records[1]["base_event_id"]
    assert records[0]["source_sample_id"] == records[1]["source_sample_id"]
    assert records[0]["azimuth_deg"] == records[1]["azimuth_deg"]
    assert records[0]["background_channels"] == records[1]["background_channels"]
    assert (
        records[0]["array_simulation"]["microphone_response_gains_db"]
        == records[1]["array_simulation"]["microphone_response_gains_db"]
    )
    assert records[0]["snr_db"] == 5.0
    assert records[1]["snr_db"] == -5.0
    for record in records:
        assert REQUIRED_ARRAY_FIELDS <= record.keys()
        assert record["recording_session"].startswith("session_")
        assert record["num_channels"] == 2
        assert record["num_samples"] == 800
        assert record["measured_snr_db"] == pytest.approx(record["snr_db"], abs=1e-5)
        observation, rate = load_audio(first / record["observation_path"])
        assert rate == 8_000
        assert observation.shape == (2, 800)
        assert (first / record["observation_path"]).read_bytes() == (
            second / record["observation_path"]
        ).read_bytes()
        assert json.loads((first / record["metadata_path"]).read_text()) == record
    assert (first / records[0]["observation_path"]).read_bytes() != (
        first / records[1]["observation_path"]
    ).read_bytes()

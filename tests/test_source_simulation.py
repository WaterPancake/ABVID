from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf
import yaml

from vehicle_audio.dataset import discover_target_recordings
from vehicle_audio.source_simulation import (
    generate_controlled_corpus,
    load_synthetic_corpus_config,
)


def _config() -> dict[str, object]:
    profile = {
        "firing_events_per_revolution": 3.0,
        "harmonic_decay": 1.2,
        "engine_gain": 0.8,
        "mobility_rate_hz_per_mps": 1.5,
        "mobility_texture_gain": 0.5,
        "mechanical_noise_gain": 0.2,
    }
    return {
        "version": 1,
        "sample_rate": 2_000,
        "duration_seconds": 0.05,
        "runs_per_state": 1,
        "reference_distance_m": 10.0,
        "states": {
            "idle": {
                "rpm_start": 700,
                "rpm_end": 700,
                "throttle_start": 0.1,
                "throttle_end": 0.1,
                "speed_start_mps": 0,
                "speed_end_mps": 0,
                "acceleration_mps2": 0,
                "load": 0.2,
            },
            "accelerating": {
                "rpm_start": 900,
                "rpm_end": 1800,
                "throttle_start": 0.2,
                "throttle_end": 0.8,
                "speed_start_mps": 1,
                "speed_end_mps": 5,
                "acceleration_mps2": 80,
                "load": 0.8,
            },
        },
        "geometries": [
            {
                "geometry_id": "test",
                "source_position_m": [0, 0, 0],
                "listener_position_m": [0, 10, 1],
            }
        ],
        "vehicles": [
            {
                **profile,
                "vehicle_id": f"{vehicle_class}_{index}",
                "vehicle_class": vehicle_class,
                "vehicle_model": f"Test {vehicle_class} {index}",
            }
            for vehicle_class in ("tracked", "wheeled")
            for index in range(2)
        ],
    }


def _write_config(path: Path) -> None:
    path.write_text(yaml.safe_dump(_config(), sort_keys=False), encoding="utf-8")


def test_project_procedural_config_has_crossed_vehicle_and_state_coverage() -> None:
    config = load_synthetic_corpus_config("configs/procedural_vehicles.yaml")

    assert len(config.vehicles) == 8
    assert {vehicle.vehicle_class for vehicle in config.vehicles} == {"tracked", "wheeled"}
    assert {state.name for state in config.states} == {
        "idle",
        "accelerating",
        "steady_speed",
        "decelerating",
    }


def test_controlled_corpus_is_deterministic_finite_and_discoverable(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_summary = generate_controlled_corpus(config_path, first, seed=17)
    second_summary = generate_controlled_corpus(config_path, second, seed=17)

    assert first_summary["generated_sources"] == 8
    assert first_summary["manifest_sha256"] == second_summary["manifest_sha256"]
    first_records = [
        json.loads(line) for line in (first / "source_manifest.jsonl").read_text().splitlines()
    ]
    for record in first_records:
        relative = Path(record["source_audio_path"])
        assert (first / relative).read_bytes() == (second / relative).read_bytes()
        info = sf.info(first / relative)
        assert (info.samplerate, info.channels, info.frames) == (2_000, 1, 100)
        assert record["source_domain"] == "procedural_synthetic"
        assert record["engine_state"] in {"idle", "accelerating"}
        assert record["source_listener_geometry"]["distance_m"] > 0

    discovered = discover_target_recordings(first)
    assert len(discovered) == 8
    assert {source.vehicle_id for source in discovered} == {
        "tracked_0",
        "tracked_1",
        "wheeled_0",
        "wheeled_1",
    }
    assert {source.source_domain for source in discovered} == {"procedural_synthetic"}


def test_controlled_corpus_changes_with_seed(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_controlled_corpus(config_path, first, seed=17)
    generate_controlled_corpus(config_path, second, seed=18)

    relative = Path("tracked/sim_tracked_0_idle_run_00/source.wav")
    assert (first / relative).read_bytes() != (second / relative).read_bytes()

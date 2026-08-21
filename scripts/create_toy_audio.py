#!/usr/bin/env python3
"""Create tiny deterministic input recordings for a local smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def _write_audio(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform.astype(np.float32), sample_rate, format="WAV", subtype="FLOAT")


def _write_metadata(path: Path, metadata: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def create_toy_inputs(root: Path, sample_rate: int = 16_000) -> None:
    rng = np.random.default_rng(20260818)
    duration_seconds = 5.0
    time = np.arange(round(duration_seconds * sample_rate), dtype=np.float64) / sample_rate

    wheeled = 0.22 * np.sin(2 * np.pi * 92 * time)
    wheeled += 0.10 * np.sin(2 * np.pi * 184 * time)
    wheeled *= 0.75 + 0.25 * np.sin(2 * np.pi * 2.1 * time)
    wheeled_path = root / "targets" / "wheeled" / "toy_wheel_session" / "engine.wav"
    _write_audio(wheeled_path, wheeled, sample_rate)
    _write_metadata(
        wheeled_path.with_suffix(".json"),
        {
            "vehicle_class": "wheeled",
            "vehicle_model": None,
            "recording_session": "toy_wheel_session",
            "source_domain": "synthetic_toy",
            "operating_condition": "steady_speed",
        },
    )

    clatter_envelope = np.maximum(0.0, np.sin(2 * np.pi * 7.5 * time)) ** 8
    tracked = 0.18 * np.sin(2 * np.pi * 68 * time)
    tracked += clatter_envelope * 0.16 * np.sin(2 * np.pi * 520 * time)
    tracked_path = root / "targets" / "tracked" / "toy_track_session" / "engine.wav"
    _write_audio(tracked_path, tracked, sample_rate)
    _write_metadata(
        tracked_path.with_suffix(".json"),
        {
            "vehicle_class": "tracked",
            "vehicle_model": None,
            "recording_session": "toy_track_session",
            "source_domain": "synthetic_toy",
            "operating_condition": "steady_speed",
        },
    )

    white_noise = rng.normal(0.0, 1.0, time.size)
    wind = np.convolve(white_noise, np.ones(180) / 180.0, mode="same")
    wind /= max(np.max(np.abs(wind)), 1e-12)
    wind_path = root / "backgrounds" / "wind" / "wind.wav"
    _write_audio(wind_path, 0.25 * wind, sample_rate)
    _write_metadata(wind_path.with_suffix(".json"), {"category": "wind"})

    traffic = 0.08 * rng.normal(size=time.size)
    traffic += 0.12 * np.sin(2 * np.pi * 47 * time + 0.3 * np.sin(2 * np.pi * 0.2 * time))
    traffic_path = root / "backgrounds" / "road_traffic" / "traffic.wav"
    _write_audio(traffic_path, traffic, sample_rate)
    _write_metadata(traffic_path.with_suffix(".json"), {"category": "road traffic"})

    ir_time = np.arange(round(0.18 * sample_rate), dtype=np.float64) / sample_rate
    impulse_response = np.exp(-ir_time * 24.0) * rng.normal(0.0, 0.07, ir_time.size)
    impulse_response[0] = 1.0
    _write_audio(root / "impulse_responses" / "toy_shelter.wav", impulse_response, sample_rate)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(".artifacts/toy_inputs"))
    parser.add_argument("--sample-rate", type=int, default=16_000)
    args = parser.parse_args()
    create_toy_inputs(args.output, args.sample_rate)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Deterministic procedural clean-ish vehicle sources for Milestone 3.

This module is deliberately separate from environmental/microphone augmentation.
It produces a controlled source-domain corpus, not real-world recordings and not
a high-fidelity physical vehicle simulator.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as torch_functional
import yaml

from vehicle_audio.audio import save_audio
from vehicle_audio.manifest import write_json
from vehicle_audio.metadata import validate_operating_condition


@dataclass(frozen=True)
class OperatingState:
    name: str
    rpm_start: float
    rpm_end: float
    throttle_start: float
    throttle_end: float
    speed_start_mps: float
    speed_end_mps: float
    acceleration_mps2: float
    load: float

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any]) -> "OperatingState":
        validate_operating_condition(name)
        state = cls(name=name, **{key: float(item) for key, item in value.items()})
        if state.rpm_start <= 0 or state.rpm_end <= 0:
            raise ValueError(f"state {name!r} RPM values must be positive")
        if not 0 <= state.throttle_start <= 1 or not 0 <= state.throttle_end <= 1:
            raise ValueError(f"state {name!r} throttle values must be in [0, 1]")
        if state.speed_start_mps < 0 or state.speed_end_mps < 0:
            raise ValueError(f"state {name!r} speed values must be nonnegative")
        if not 0 <= state.load <= 1:
            raise ValueError(f"state {name!r} load must be in [0, 1]")
        return state


@dataclass(frozen=True)
class Geometry:
    geometry_id: str
    source_position_m: tuple[float, float, float]
    listener_position_m: tuple[float, float, float]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Geometry":
        source = tuple(float(item) for item in value["source_position_m"])
        listener = tuple(float(item) for item in value["listener_position_m"])
        if len(source) != 3 or len(listener) != 3:
            raise ValueError("source and listener positions must be three-dimensional")
        return cls(str(value["geometry_id"]), source, listener)

    @property
    def distance_m(self) -> float:
        return math.sqrt(
            sum((source - listener) ** 2 for source, listener in zip(
                self.source_position_m, self.listener_position_m, strict=True
            ))
        )


@dataclass(frozen=True)
class VehicleProfile:
    vehicle_id: str
    vehicle_class: str
    vehicle_model: str
    firing_events_per_revolution: float
    harmonic_decay: float
    engine_gain: float
    mobility_rate_hz_per_mps: float
    mobility_texture_gain: float
    mechanical_noise_gain: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VehicleProfile":
        profile = cls(
            vehicle_id=str(value["vehicle_id"]),
            vehicle_class=str(value["vehicle_class"]),
            vehicle_model=str(value["vehicle_model"]),
            firing_events_per_revolution=float(value["firing_events_per_revolution"]),
            harmonic_decay=float(value["harmonic_decay"]),
            engine_gain=float(value["engine_gain"]),
            mobility_rate_hz_per_mps=float(value["mobility_rate_hz_per_mps"]),
            mobility_texture_gain=float(value["mobility_texture_gain"]),
            mechanical_noise_gain=float(value["mechanical_noise_gain"]),
        )
        if profile.vehicle_class not in {"tracked", "wheeled"}:
            raise ValueError(f"invalid vehicle class: {profile.vehicle_class!r}")
        positive = (
            profile.firing_events_per_revolution,
            profile.harmonic_decay,
            profile.engine_gain,
            profile.mobility_rate_hz_per_mps,
            profile.mobility_texture_gain,
            profile.mechanical_noise_gain,
        )
        if any(item <= 0 for item in positive):
            raise ValueError(f"profile values must be positive for {profile.vehicle_id!r}")
        return profile


@dataclass(frozen=True)
class SyntheticCorpusConfig:
    version: int
    sample_rate: int
    duration_seconds: float
    runs_per_state: int
    reference_distance_m: float
    states: tuple[OperatingState, ...]
    geometries: tuple[Geometry, ...]
    vehicles: tuple[VehicleProfile, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SyntheticCorpusConfig":
        states_value = value.get("states")
        if not isinstance(states_value, Mapping):
            raise TypeError("states must be a mapping")
        geometries_value = value.get("geometries")
        vehicles_value = value.get("vehicles")
        if not isinstance(geometries_value, list) or not isinstance(vehicles_value, list):
            raise TypeError("geometries and vehicles must be lists")
        config = cls(
            version=int(value.get("version", 1)),
            sample_rate=int(value["sample_rate"]),
            duration_seconds=float(value["duration_seconds"]),
            runs_per_state=int(value["runs_per_state"]),
            reference_distance_m=float(value.get("reference_distance_m", 10.0)),
            states=tuple(
                OperatingState.from_mapping(str(name), state)
                for name, state in states_value.items()
            ),
            geometries=tuple(Geometry.from_mapping(item) for item in geometries_value),
            vehicles=tuple(VehicleProfile.from_mapping(item) for item in vehicles_value),
        )
        if config.sample_rate <= 0 or config.duration_seconds <= 0:
            raise ValueError("sample rate and duration must be positive")
        if config.runs_per_state <= 0 or config.reference_distance_m <= 0:
            raise ValueError("runs_per_state and reference_distance_m must be positive")
        if len(config.states) < 2 or not config.geometries:
            raise ValueError("at least two states and one geometry are required")
        vehicle_ids = [vehicle.vehicle_id for vehicle in config.vehicles]
        if len(vehicle_ids) != len(set(vehicle_ids)):
            raise ValueError("vehicle_id values must be unique")
        class_counts = Counter(vehicle.vehicle_class for vehicle in config.vehicles)
        if any(class_counts[name] < 2 for name in ("tracked", "wheeled")):
            raise ValueError("controlled corpus requires at least two vehicles per class")
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_synthetic_corpus_config(path: str | Path) -> SyntheticCorpusConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise TypeError("synthetic corpus config must contain a mapping")
    return SyntheticCorpusConfig.from_mapping(value)


def _derived_seed(master_seed: int, *parts: str) -> int:
    payload = "\x1f".join((str(master_seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _smooth_noise(noise: torch.Tensor, width: int) -> torch.Tensor:
    if width % 2 == 0:
        width += 1
    return torch_functional.avg_pool1d(
        noise.reshape(1, 1, -1),
        kernel_size=width,
        stride=1,
        padding=width // 2,
    ).reshape(-1)


def synthesize_vehicle_source(
    profile: VehicleProfile,
    state: OperatingState,
    geometry: Geometry,
    *,
    sample_rate: int,
    duration_seconds: float,
    reference_distance_m: float,
    seed: int,
) -> torch.Tensor:
    """Synthesize one mono clean-ish source with recorded control variables."""

    num_samples = round(sample_rate * duration_seconds)
    progress = torch.linspace(0.0, 1.0, num_samples, dtype=torch.float64)
    rpm = state.rpm_start + progress * (state.rpm_end - state.rpm_start)
    throttle = state.throttle_start + progress * (
        state.throttle_end - state.throttle_start
    )
    speed = state.speed_start_mps + progress * (
        state.speed_end_mps - state.speed_start_mps
    )
    firing_frequency = rpm / 60.0 * profile.firing_events_per_revolution
    engine_phase = 2.0 * math.pi * torch.cumsum(firing_frequency, dim=0) / sample_rate

    generator = torch.Generator().manual_seed(seed)
    engine = torch.zeros(num_samples, dtype=torch.float64)
    for harmonic in range(1, 9):
        phase_offset = 2.0 * math.pi * float(torch.rand((), generator=generator))
        engine += torch.sin(harmonic * engine_phase + phase_offset) / (
            harmonic ** profile.harmonic_decay
        )
    engine_envelope = profile.engine_gain * (0.25 + 0.75 * throttle) * (0.6 + 0.4 * state.load)
    engine *= engine_envelope

    white_noise = torch.randn(num_samples, generator=generator, dtype=torch.float64)
    low_noise = _smooth_noise(white_noise, max(3, sample_rate // 160))
    high_noise = white_noise - _smooth_noise(white_noise, max(3, sample_rate // 80))
    mechanical = profile.mechanical_noise_gain * (0.35 + 0.65 * throttle) * low_noise

    mobility_frequency = 1.5 + speed * profile.mobility_rate_hz_per_mps
    mobility_phase = 2.0 * math.pi * torch.cumsum(mobility_frequency, dim=0) / sample_rate
    if profile.vehicle_class == "tracked":
        link_impulses = torch.relu(torch.sin(mobility_phase)) ** 10
        carrier_frequency = 260.0 + 30.0 * profile.firing_events_per_revolution
        carrier_phase = 2.0 * math.pi * carrier_frequency * progress * duration_seconds
        mobility = profile.mobility_texture_gain * (
            link_impulses * torch.sin(carrier_phase) + 0.35 * low_noise
        )
    else:
        tire_level = (speed / max(state.speed_start_mps, state.speed_end_mps, 1.0)).sqrt()
        axle_tone = torch.sin(mobility_phase)
        mobility = profile.mobility_texture_gain * tire_level * (
            0.75 * high_noise + 0.25 * axle_tone
        )

    waveform = engine + mechanical + mobility
    waveform /= waveform.abs().amax().clamp_min(1e-12)
    attenuation = min(1.0, reference_distance_m / max(geometry.distance_m, 1.0))
    waveform *= 0.82 * attenuation
    fade_samples = min(num_samples // 10, round(0.05 * sample_rate))
    if fade_samples > 1:
        fade = torch.linspace(0.0, 1.0, fade_samples, dtype=waveform.dtype)
        waveform[:fade_samples] *= fade
        waveform[-fade_samples:] *= fade.flip(0)
    waveform = waveform.to(torch.float32).reshape(1, -1)
    if not torch.isfinite(waveform).all():
        raise RuntimeError("procedural source produced NaN or Inf")
    return waveform


def generate_controlled_corpus(
    config_path: str | Path,
    output_root: str | Path,
    *,
    seed: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate a crossed vehicle/state/run corpus and provenance manifest."""

    if seed < 0:
        raise ValueError("seed must be nonnegative")
    config = load_synthetic_corpus_config(config_path)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for vehicle in config.vehicles:
        for state in config.states:
            for run_index in range(config.runs_per_state):
                geometry = config.geometries[run_index % len(config.geometries)]
                simulation_run = (
                    f"sim_{vehicle.vehicle_id}_{state.name}_run_{run_index:02d}"
                )
                source_seed = _derived_seed(seed, vehicle.vehicle_id, state.name, str(run_index))
                relative_audio = (
                    Path(vehicle.vehicle_class) / simulation_run / "source.wav"
                )
                audio_path = root / relative_audio
                sidecar_path = audio_path.with_suffix(".json")
                if not overwrite and (audio_path.exists() or sidecar_path.exists()):
                    raise FileExistsError(
                        f"controlled source exists (use overwrite): {audio_path}"
                    )
                waveform = synthesize_vehicle_source(
                    vehicle,
                    state,
                    geometry,
                    sample_rate=config.sample_rate,
                    duration_seconds=config.duration_seconds,
                    reference_distance_m=config.reference_distance_m,
                    seed=source_seed,
                )
                save_audio(audio_path, waveform, config.sample_rate)
                geometry_metadata = {
                    "geometry_id": geometry.geometry_id,
                    "source_position_m": list(geometry.source_position_m),
                    "listener_position_m": list(geometry.listener_position_m),
                    "distance_m": geometry.distance_m,
                }
                metadata: dict[str, Any] = {
                    "source_domain": "procedural_synthetic",
                    "synthesis_method": "harmonic_engine_and_mobility_texture_v1",
                    "vehicle_id": vehicle.vehicle_id,
                    "vehicle_class": vehicle.vehicle_class,
                    "vehicle_model": vehicle.vehicle_model,
                    "recording_session": simulation_run,
                    "simulation_run": simulation_run,
                    "operating_condition": state.name,
                    "engine_state": state.name,
                    "rpm": {"start": state.rpm_start, "end": state.rpm_end},
                    "throttle": {
                        "start": state.throttle_start,
                        "end": state.throttle_end,
                    },
                    "speed_mps": {
                        "start": state.speed_start_mps,
                        "end": state.speed_end_mps,
                    },
                    "acceleration_mps2": state.acceleration_mps2,
                    "load": state.load,
                    "source_listener_geometry": geometry_metadata,
                    "source_seed": source_seed,
                    "sample_rate": config.sample_rate,
                    "duration_seconds": config.duration_seconds,
                    "num_channels": 1,
                    "num_samples": waveform.shape[-1],
                    "profile": asdict(vehicle),
                    "source_audio_path": relative_audio.as_posix(),
                }
                write_json(sidecar_path, metadata)
                record = {
                    **metadata,
                    "sidecar_path": sidecar_path.relative_to(root).as_posix(),
                    "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                }
                records.append(record)

    manifest_path = root / "source_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    config_snapshot = {
        "corpus_type": "procedural_synthetic",
        "config_path": str(config_path),
        "config": config.to_dict(),
        "seed": seed,
        "source_manifest": manifest_path.name,
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    write_json(root / "corpus.json", config_snapshot)
    return {
        "generated_sources": len(records),
        "vehicle_class_counts": dict(Counter(record["vehicle_class"] for record in records)),
        "vehicle_count": len(config.vehicles),
        "vehicle_ids": [vehicle.vehicle_id for vehicle in config.vehicles],
        "operating_state_counts": dict(
            Counter(record["operating_condition"] for record in records)
        ),
        "recording_session_count": len({record["recording_session"] for record in records}),
        "manifest": str(manifest_path),
        "manifest_sha256": config_snapshot["source_manifest_sha256"],
        "source_domain": "procedural_synthetic",
    }

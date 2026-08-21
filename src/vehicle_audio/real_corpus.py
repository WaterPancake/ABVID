"""Deterministic fixed-window corpus preparation for real vehicle recordings.

Real recordings retain their native acoustic conditions.  The preparer only
selects one explicit channel, resamples, and windows; it never invents an SNR or
applies synthetic corruption.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import soundfile as sf
import torch
import yaml

from vehicle_audio.audio import force_num_samples, resample_audio, save_audio
from vehicle_audio.dataset import TargetSource, discover_target_recordings
from vehicle_audio.manifest import write_json


REAL_CORPUS_VERSION = 1
REQUIRED_PROVENANCE_FIELDS = (
    "source_id",
    "source_page",
    "recording_session",
    "vehicle_class",
    "license",
    "raw_sha256",
)


@dataclass(frozen=True)
class RealCorpusConfig:
    sample_rate: int = 16_000
    window_seconds: float = 2.0
    hop_seconds: float = 1.0
    channel: int = 0
    max_windows_per_session: int | None = 64
    minimum_rms: float = 0.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.window_seconds <= 0 or self.hop_seconds <= 0:
            raise ValueError("window_seconds and hop_seconds must be positive")
        if self.channel < 0:
            raise ValueError("channel must be nonnegative")
        if self.max_windows_per_session is not None and self.max_windows_per_session <= 0:
            raise ValueError("max_windows_per_session must be positive or null")
        if self.minimum_rms < 0 or not math.isfinite(self.minimum_rms):
            raise ValueError("minimum_rms must be finite and nonnegative")

    @property
    def window_samples(self) -> int:
        return round(self.sample_rate * self.window_seconds)

    @property
    def hop_samples(self) -> int:
        return round(self.sample_rate * self.hop_seconds)


def load_real_corpus_config(path: str | Path) -> RealCorpusConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise TypeError("real corpus config must contain a mapping")
    return RealCorpusConfig(**dict(value))


@dataclass(frozen=True)
class _WindowCandidate:
    source: TargetSource
    start_sample: int
    sidecar: Mapping[str, Any]


def _source_sidecar(source: TargetSource) -> Mapping[str, Any]:
    path = source.path.with_suffix(".json")
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"target sidecar must contain an object: {path}")
    return value


def _window_candidates(
    sources: Sequence[TargetSource], config: RealCorpusConfig
) -> list[_WindowCandidate]:
    by_session: dict[str, list[_WindowCandidate]] = defaultdict(list)
    for source in sources:
        sidecar = _source_sidecar(source)
        with sf.SoundFile(source.path) as audio_file:
            duration_seconds = audio_file.frames / audio_file.samplerate
            duration_tolerance = 1.0 / audio_file.samplerate
        if (
            source.segment_start_seconds is not None
            and source.segment_start_seconds >= duration_seconds
        ):
            raise ValueError(
                f"reviewed segment starts outside {source.path}: "
                f"{source.segment_start_seconds} >= {duration_seconds}"
            )
        if (
            source.segment_end_seconds is not None
            and source.segment_end_seconds > duration_seconds + duration_tolerance
        ):
            raise ValueError(
                f"reviewed segment ends outside {source.path}: "
                f"{source.segment_end_seconds} > {duration_seconds}"
            )
        region_start = round((source.segment_start_seconds or 0.0) * config.sample_rate)
        region_end = round(
            (
                source.segment_end_seconds
                if source.segment_end_seconds is not None
                else duration_seconds
            )
            * config.sample_rate
        )
        final_start = region_end - config.window_samples
        if final_start < region_start:
            continue
        for start_sample in range(region_start, final_start + 1, config.hop_samples):
            by_session[source.recording_session].append(
                _WindowCandidate(source, start_sample, sidecar)
            )

    selected: list[_WindowCandidate] = []
    for session in sorted(by_session):
        candidates = sorted(
            by_session[session],
            key=lambda value: (value.source.path.as_posix(), value.start_sample),
        )
        maximum = config.max_windows_per_session
        if maximum is not None and len(candidates) > maximum:
            # Even spacing avoids selecting only the beginning of a long recording.
            indices = [round(index * (len(candidates) - 1) / (maximum - 1)) for index in range(maximum)] if maximum > 1 else [len(candidates) // 2]
            candidates = [candidates[index] for index in indices]
        selected.extend(candidates)
    return sorted(
        selected,
        key=lambda value: (
            value.source.vehicle_class,
            value.source.recording_session,
            value.source.path.as_posix(),
            value.start_sample,
        ),
    )


def _read_window(candidate: _WindowCandidate, config: RealCorpusConfig) -> tuple[torch.Tensor, int, int]:
    source = candidate.source
    start_seconds = candidate.start_sample / config.sample_rate
    with sf.SoundFile(source.path) as audio_file:
        original_rate = int(audio_file.samplerate)
        original_channels = int(audio_file.channels)
        if config.channel >= original_channels:
            raise ValueError(
                f"configured channel {config.channel} but {source.path} has "
                f"{original_channels} channels"
            )
        start_frame = round(start_seconds * original_rate)
        logical_frames = math.ceil(config.window_samples * original_rate / config.sample_rate)
        margin = 128 if original_rate != config.sample_rate else 0
        read_start = max(0, start_frame - margin)
        read_end = min(audio_file.frames, start_frame + logical_frames + margin)
        audio_file.seek(read_start)
        samples = audio_file.read(read_end - read_start, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples[:, config.channel].copy()).unsqueeze(0)
    if original_rate != config.sample_rate:
        waveform = resample_audio(waveform, original_rate, config.sample_rate)
    crop_offset = round((start_frame - read_start) * config.sample_rate / original_rate)
    waveform = force_num_samples(
        waveform[..., crop_offset : crop_offset + config.window_samples],
        config.window_samples,
    )
    return waveform, original_rate, original_channels


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_real_manifest_record(record: Mapping[str, Any]) -> None:
    required = {
        "sample_id",
        "audio_path",
        "metadata_path",
        "vehicle_class",
        "recording_session",
        "source_domain",
        "source_id",
        "sample_rate",
        "num_channels",
        "num_samples",
        "window_start_sample",
        "snr_db",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"real manifest record is missing fields: {sorted(missing)}")
    if record["vehicle_class"] not in {"tracked", "wheeled"}:
        raise ValueError("real vehicle_class must be tracked or wheeled")
    if record["source_domain"] != "real_recording":
        raise ValueError("real corpus records must use source_domain='real_recording'")
    if record["snr_db"] is not None:
        raise ValueError("native real recordings must not be assigned an inferred SNR")
    for field in ("sample_rate", "num_channels", "num_samples"):
        if not isinstance(record[field], int) or record[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")


def audit_real_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    minimum_sessions_per_class: int = 3,
) -> dict[str, Any]:
    if not records:
        raise ValueError("real manifest is empty")
    sessions_by_class = {
        vehicle_class: sorted(
            {
                str(record["recording_session"])
                for record in records
                if record["vehicle_class"] == vehicle_class
            }
        )
        for vehicle_class in ("tracked", "wheeled")
    }
    missing_provenance = sorted(
        {
            str(record["source_id"])
            for record in records
            if not bool(record.get("provenance_complete"))
        }
    )
    unreviewed_sources = sorted(
        {
            str(record["source_id"])
            for record in records
            if str(record.get("content_review_status")) != "reviewed_segment"
        }
    )
    session_ready = all(
        len(sessions) >= minimum_sessions_per_class
        for sessions in sessions_by_class.values()
    )
    return {
        "real_corpus_version": REAL_CORPUS_VERSION,
        "observation_count": len(records),
        "class_counts": dict(
            sorted(Counter(str(record["vehicle_class"]) for record in records).items())
        ),
        "recording_sessions_by_class": sessions_by_class,
        "recording_session_counts": {
            name: len(values) for name, values in sessions_by_class.items()
        },
        "minimum_sessions_per_class": minimum_sessions_per_class,
        "session_split_ready": session_ready,
        "missing_provenance_source_ids": missing_provenance,
        "provenance_complete": not missing_provenance,
        "unreviewed_source_ids": unreviewed_sources,
        "content_review_complete": not unreviewed_sources,
        "full_protocol_ready": session_ready and not missing_provenance and not unreviewed_sources,
        "synthetic_to_real_smoke_ready": all(sessions_by_class.values()),
        "blocking_reasons": [
            reason
            for condition, reason in (
                (
                    not session_ready,
                    "at least three independent recording sessions per class are required",
                ),
                (bool(missing_provenance), "one or more sources lack required provenance"),
                (
                    bool(unreviewed_sources),
                    "one or more sources lack reviewed vehicle-audio segment boundaries",
                ),
            )
            if condition
        ],
    }


def prepare_real_corpus(
    config: RealCorpusConfig,
    targets_root: str | Path,
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    targets_path = Path(targets_root)
    output = Path(output_root)
    manifest_path = output / "real_manifest.jsonl"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"real manifest exists (use overwrite): {manifest_path}")
    sources = discover_target_recordings(targets_path)
    if not sources:
        raise ValueError(f"no target recordings found under {targets_path}")
    if {source.vehicle_class for source in sources} != {"tracked", "wheeled"}:
        raise ValueError("real corpus requires both tracked and wheeled recordings")
    if any(source.source_domain != "real_recording" for source in sources):
        raise ValueError("real target root contains a non-real source_domain")

    candidates = _window_candidates(sources, config)
    if not candidates:
        raise ValueError("no full windows can be extracted with the configured duration")
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {source.path: _sha256(source.path) for source in sources}
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        waveform, original_rate, original_channels = _read_window(candidate, config)
        rms = float(waveform.square().mean().sqrt())
        if rms < config.minimum_rms:
            continue
        source = candidate.source
        sidecar = candidate.sidecar
        source_id = str(sidecar.get("source_id", source.source_id))
        stable_key = (
            f"{source_id}|{source.source_id}|{source.recording_session}|"
            f"{source.operating_condition}|{source.segment_start_seconds}|"
            f"{source.segment_end_seconds}|{candidate.start_sample}|"
            f"{config.sample_rate}|{config.window_samples}|{config.channel}"
        )
        digest = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:16]
        sample_id = f"real_{source.vehicle_class}_{digest}"
        relative_dir = Path(sample_id)
        audio_relative = relative_dir / "observation.wav"
        metadata_relative = relative_dir / "metadata.json"
        save_audio(output / audio_relative, waveform, config.sample_rate)
        provenance_complete = all(sidecar.get(field) not in (None, "") for field in REQUIRED_PROVENANCE_FIELDS)
        content_review_status = (
            "reviewed_segment"
            if source.segment_start_seconds is not None
            else str(sidecar.get("content_review_status", "whole_recording_unsegmented"))
        )
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "audio_path": audio_relative.as_posix(),
            # Compatibility alias for feature loaders; no corruption is implied.
            "corrupted_path": audio_relative.as_posix(),
            "metadata_path": metadata_relative.as_posix(),
            "source_domain": "real_recording",
            "observation_domain": "native_real_recording",
            "snr_db": None,
            "vehicle_class": source.vehicle_class,
            "vehicle_model": source.vehicle_model,
            "vehicle_id": source.vehicle_id,
            "recording_session": source.recording_session,
            "operating_condition": source.operating_condition,
            "engine_state": source.engine_state,
            "source_id": source_id,
            "target_source": source.path.relative_to(targets_path.resolve()).as_posix(),
            "source_page": sidecar.get("source_page"),
            "original_media_source": sidecar.get("source_page"),
            "license": sidecar.get("license"),
            "license_url": sidecar.get("license_url"),
            "attribution": sidecar.get("attribution"),
            "raw_sha256": sidecar.get("raw_sha256"),
            "normalized_source_sha256": source_hashes[source.path],
            "provenance_complete": provenance_complete,
            "content_review_status": content_review_status,
            "sample_rate": config.sample_rate,
            "num_channels": 1,
            "num_samples": config.window_samples,
            "duration_seconds": config.window_seconds,
            "selected_channel": config.channel,
            "original_sample_rate": original_rate,
            "original_num_channels": original_channels,
            "window_start_sample": candidate.start_sample,
            "window_start_seconds": candidate.start_sample / config.sample_rate,
            "window_end_seconds": (
                candidate.start_sample + config.window_samples
            )
            / config.sample_rate,
            "rms": rms,
            "real_corpus_version": REAL_CORPUS_VERSION,
            "real_corpus_config": asdict(config),
        }
        validate_real_manifest_record(record)
        write_json(output / metadata_relative, record)
        records.append(record)

    if not records:
        raise ValueError("all real windows were rejected by minimum_rms")
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    manifest_sha256 = _sha256(manifest_path)
    audit = audit_real_manifest(records)
    audit.update(
        {
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "dataset_version": manifest_sha256,
            "configuration": asdict(config),
        }
    )
    (output / "corpus_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return audit

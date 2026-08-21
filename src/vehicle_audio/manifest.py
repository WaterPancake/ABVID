"""Manifest records and JSON/JSONL serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "sample_id",
        "target_source",
        "background_source",
        "recording_session",
        "vehicle_class",
        "vehicle_model",
        "source_domain",
        "operating_condition",
        "snr_db",
        "gain_db",
        "augmentation_seed",
        "sample_rate",
    }
)


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    target_source: str
    background_source: str
    background_category: str
    recording_session: str
    vehicle_class: str
    vehicle_model: str | None
    source_domain: str
    vehicle_id: str | None
    simulation_run: str | None
    operating_condition: str
    engine_state: str | None
    rpm: dict[str, float] | None
    throttle: dict[str, float] | None
    speed_mps: dict[str, float] | None
    acceleration_mps2: float | None
    load: float | None
    source_listener_geometry: dict[str, Any] | None
    target_segment_start_seconds: float | None
    target_segment_end_seconds: float | None
    snr_db: float | None
    gain_db: float
    augmentation_seed: int
    sample_rate: int
    num_channels: int
    num_samples: int
    duration_seconds: float
    target_original_sample_rate: int
    background_original_sample_rate: int
    background_original_num_channels: int
    background_channel_selected: int | None
    target_crop_start_frame: int
    target_crop_start_sample: int
    target_repeated: bool
    background_crop_start_frame: int
    background_crop_start_sample: int
    background_repeated: bool
    clean_path: str
    corrupted_path: str
    metadata_path: str
    augmentations: dict[str, Any]
    generation_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        validate_manifest_record(record)
        return record


def validate_manifest_record(record: Mapping[str, Any]) -> None:
    missing = REQUIRED_MANIFEST_FIELDS - set(record)
    if missing:
        raise ValueError(f"manifest record is missing required fields: {sorted(missing)}")
    nonempty_strings = (
        "sample_id",
        "target_source",
        "recording_session",
        "vehicle_class",
        "source_domain",
        "operating_condition",
    )
    for field_name in nonempty_strings:
        if not isinstance(record[field_name], str) or not record[field_name]:
            raise ValueError(f"manifest field {field_name!r} must be a nonempty string")
    if not isinstance(record["augmentation_seed"], int):
        raise TypeError("augmentation_seed must be an integer")
    if not isinstance(record["sample_rate"], int) or record["sample_rate"] <= 0:
        raise ValueError("sample_rate must be a positive integer")


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_manifest_record(record)
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False))
            handle.write("\n")

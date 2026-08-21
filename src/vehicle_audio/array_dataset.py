"""Deterministic multichannel dataset generation from clean paired-source clips."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import torch

from vehicle_audio.audio import load_audio, load_audio_crop, save_audio, temporal_crop
from vehicle_audio.dataset import BackgroundSource, discover_background_recordings
from vehicle_audio.manifest import write_json
from vehicle_audio.multichannel import ArraySimulationConfig, simulate_microphone_array


REQUIRED_ARRAY_FIELDS = frozenset(
    {
        "sample_id",
        "source_sample_id",
        "recording_session",
        "vehicle_class",
        "source_domain",
        "snr_db",
        "azimuth_deg",
        "source_distance_m",
        "array_seed",
        "sample_rate",
        "num_channels",
        "num_samples",
        "observation_path",
        "metadata_path",
    }
)


def validate_array_manifest_record(record: Mapping[str, Any]) -> None:
    missing = REQUIRED_ARRAY_FIELDS - set(record)
    if missing:
        raise ValueError(f"array manifest record is missing fields: {sorted(missing)}")
    for field_name in (
        "sample_id",
        "source_sample_id",
        "recording_session",
        "vehicle_class",
        "source_domain",
        "observation_path",
        "metadata_path",
    ):
        if not isinstance(record[field_name], str) or not record[field_name]:
            raise ValueError(f"array field {field_name!r} must be a nonempty string")
    if record["vehicle_class"] not in {"tracked", "wheeled"}:
        raise ValueError("array vehicle_class must be tracked or wheeled")
    if not isinstance(record["array_seed"], int):
        raise TypeError("array_seed must be an integer")
    for field_name in ("sample_rate", "num_channels", "num_samples"):
        if not isinstance(record[field_name], int) or record[field_name] <= 0:
            raise ValueError(f"array field {field_name!r} must be a positive integer")


def _load_source_manifest(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = path.read_bytes()
    records = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if not records:
        raise ValueError(f"source manifest is empty: {path}")
    classes = {str(record.get("vehicle_class")) for record in records}
    if classes != {"tracked", "wheeled"}:
        raise ValueError(f"source manifest must contain tracked and wheeled, found {classes}")
    required = {"sample_id", "recording_session", "clean_path", "sample_rate"}
    for index, record in enumerate(records):
        missing = required - set(record)
        if missing:
            raise ValueError(f"source record {index} is missing {sorted(missing)}")
    return records, hashlib.sha256(payload).hexdigest()


def _select_source_record(
    records: Sequence[Mapping[str, Any]],
    base_event_index: int,
    selector: random.Random,
) -> Mapping[str, Any]:
    """Cycle through complete source sessions before reusing one.

    This prevents a finite random draw from omitting source identities or
    operating states and gives the grouped evaluation comparable coverage per
    class. A random clean crop from the selected session is still chosen with
    the base-event seed.
    """

    classes = ("tracked", "wheeled")
    vehicle_class = classes[base_event_index % len(classes)]
    class_records = [
        record for record in records if str(record["vehicle_class"]) == vehicle_class
    ]
    by_session: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in class_records:
        by_session[str(record["recording_session"])].append(record)
    sessions = sorted(by_session)
    session_cycle_index = (base_event_index // len(classes)) % len(sessions)
    candidates = by_session[sessions[session_cycle_index]]
    return candidates[selector.randrange(len(candidates))]


def _load_background_channels(
    background: BackgroundSource,
    config: ArraySimulationConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    channels: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    for microphone_index in range(len(config.microphone_positions_m)):
        crop = load_audio_crop(
            background.path,
            config.sample_rate,
            config.num_samples,
            generator,
            repeat_if_short=True,
            randomize_start=True,
        )
        original_channels = crop.waveform.shape[0]
        selected_channel = config.background_channel
        if selected_channel is None:
            if original_channels != 1:
                raise ValueError(
                    f"background_channel is null but {background.path} has "
                    f"{original_channels} channels"
                )
            channel = crop.waveform
        else:
            if selected_channel >= original_channels:
                raise ValueError(
                    f"configured background channel {selected_channel} but "
                    f"{background.path} has {original_channels} channels"
                )
            channel = crop.waveform[selected_channel : selected_channel + 1]
        channels.append(channel)
        metadata.append(
            {
                "microphone_index": microphone_index,
                "original_sample_rate": crop.original_sample_rate,
                "original_num_channels": original_channels,
                "selected_channel": selected_channel,
                "crop_start_frame": crop.start_frame,
                "crop_start_sample": crop.start_sample,
                "repeated": crop.repeated,
            }
        )
    return torch.cat(channels, dim=0), metadata


class ArrayDatasetGenerator:
    """Generate multichannel observations while preserving source-session groups."""

    def __init__(
        self,
        config: ArraySimulationConfig,
        source_manifest: str | Path,
        backgrounds_root: str | Path,
        output_root: str | Path,
    ) -> None:
        self.config = config
        self.source_manifest = Path(source_manifest)
        self.backgrounds_root = Path(backgrounds_root)
        self.output_root = Path(output_root)

    def generate(
        self,
        num_samples: int,
        seed: int,
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        if num_samples <= 0 or seed < 0:
            raise ValueError("num_samples must be positive and seed must be nonnegative")
        if self.config.paired_snr_sweep and num_samples % len(self.config.snr_db_choices):
            raise ValueError(
                "paired_snr_sweep requires num_samples to be divisible by the "
                "number of configured SNR values"
            )
        records, source_manifest_sha256 = _load_source_manifest(self.source_manifest)
        backgrounds = discover_background_recordings(self.backgrounds_root)
        if not backgrounds:
            raise ValueError(f"no backgrounds found under {self.backgrounds_root}")
        manifest_path = self.output_root / "array_manifest.jsonl"
        if manifest_path.exists() and not overwrite:
            raise FileExistsError(f"array manifest exists (use overwrite): {manifest_path}")
        self.output_root.mkdir(parents=True, exist_ok=True)

        root = self.source_manifest.parent
        master_selector = random.Random(seed)
        snr_count = len(self.config.snr_db_choices)
        base_event_count = (
            num_samples // snr_count if self.config.paired_snr_sweep else num_samples
        )
        base_event_seeds = [master_selector.randrange(0, 2**63) for _ in range(base_event_count)]
        output_records: list[dict[str, Any]] = []
        for sample_index in range(num_samples):
            if self.config.paired_snr_sweep:
                base_event_index = sample_index // snr_count
                snr_index = sample_index % snr_count
                array_seed = base_event_seeds[base_event_index]
            else:
                base_event_index = sample_index
                snr_index = None
                array_seed = base_event_seeds[base_event_index]
            selector = random.Random(array_seed)
            generator = torch.Generator(device="cpu").manual_seed(array_seed)
            source_record = _select_source_record(records, base_event_index, selector)
            background = backgrounds[selector.randrange(len(backgrounds))]
            source, source_rate = load_audio(root / str(source_record["clean_path"]))
            if source_rate != self.config.sample_rate:
                raise ValueError(
                    f"source rate {source_rate} does not match configured "
                    f"{self.config.sample_rate}; resampling must be explicit upstream"
                )
            if self.config.source_channel >= source.shape[0]:
                raise ValueError(
                    f"configured source channel {self.config.source_channel} but "
                    f"{source_record['clean_path']} has {source.shape[0]} channels"
                )
            selected_source = source[
                self.config.source_channel : self.config.source_channel + 1
            ]
            source_crop = temporal_crop(
                selected_source,
                self.config.num_samples,
                generator,
                repeat_if_short=True,
                randomize_start=True,
            )
            background_audio, background_channels = _load_background_channels(
                background,
                self.config,
                generator,
            )
            snr_db = float(
                self.config.snr_db_choices[snr_index]
                if snr_index is not None
                else self.config.snr_db_choices[
                    selector.randrange(len(self.config.snr_db_choices))
                ]
            )
            azimuth_deg = float(
                self.config.azimuth_deg_choices[
                    selector.randrange(len(self.config.azimuth_deg_choices))
                ]
            )
            distance_m = float(
                self.config.source_distance_m_choices[
                    selector.randrange(len(self.config.source_distance_m_choices))
                ]
            )
            result = simulate_microphone_array(
                source_crop.waveform,
                background_audio,
                self.config,
                azimuth_deg=azimuth_deg,
                source_distance_m=distance_m,
                snr_db=snr_db,
                generator=generator,
            )

            sample_id = f"array_{sample_index:06d}_{array_seed:016x}"
            base_event_id = f"base_event_{base_event_index:06d}_{array_seed:016x}"
            relative_dir = Path(sample_id)
            observation_relative = relative_dir / "observation.wav"
            metadata_relative = relative_dir / "metadata.json"
            save_audio(
                self.output_root / observation_relative,
                result.observation,
                self.config.sample_rate,
            )
            record: dict[str, Any] = {
                "sample_id": sample_id,
                "source_sample_id": str(source_record["sample_id"]),
                "source_manifest": str(self.source_manifest),
                "source_manifest_sha256": source_manifest_sha256,
                "source_clean_path": str(source_record["clean_path"]),
                "source_channel_selected": self.config.source_channel,
                "source_crop_start_sample": source_crop.start_sample,
                "source_repeated": source_crop.repeated,
                "target_source": source_record.get("target_source"),
                "recording_session": str(source_record["recording_session"]),
                "simulation_run": source_record.get("simulation_run"),
                "vehicle_class": str(source_record["vehicle_class"]),
                "vehicle_id": source_record.get("vehicle_id"),
                "vehicle_model": source_record.get("vehicle_model"),
                "operating_condition": source_record.get("operating_condition"),
                "engine_state": source_record.get("engine_state"),
                "source_domain": str(source_record.get("source_domain", "unspecified")),
                "background_source": background.source_id,
                "background_category": background.category,
                "background_channels": background_channels,
                "snr_db": snr_db,
                "measured_snr_db": result.metadata["measured_snr_db"],
                "azimuth_deg": azimuth_deg,
                "source_distance_m": distance_m,
                "array_seed": array_seed,
                "base_event_id": base_event_id,
                "base_event_index": base_event_index,
                "snr_variant_index": snr_index,
                "paired_snr_sweep": self.config.paired_snr_sweep,
                "sample_rate": self.config.sample_rate,
                "num_channels": result.observation.shape[0],
                "num_samples": result.observation.shape[-1],
                "duration_seconds": self.config.duration_seconds,
                "observation_path": observation_relative.as_posix(),
                "metadata_path": metadata_relative.as_posix(),
                "array_simulation": result.metadata,
                "array_config": self.config.to_dict(),
            }
            validate_array_manifest_record(record)
            write_json(self.output_root / metadata_relative, record)
            output_records.append(record)

        with manifest_path.open("w", encoding="utf-8") as handle:
            for record in output_records:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        return {
            "generated_samples": len(output_records),
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "num_microphones": len(self.config.microphone_positions_m),
            "paired_snr_sweep": self.config.paired_snr_sweep,
            "base_event_count": base_event_count,
            "vehicle_class_counts": dict(
                sorted(Counter(record["vehicle_class"] for record in output_records).items())
            ),
            "operating_condition_counts": dict(
                sorted(
                    Counter(
                        str(record["operating_condition"]) for record in output_records
                    ).items()
                )
            ),
            "background_category_counts": dict(
                sorted(
                    Counter(record["background_category"] for record in output_records).items()
                )
            ),
            "snr_db_counts": dict(
                sorted(Counter(str(record["snr_db"]) for record in output_records).items())
            ),
            "recording_session_count": len(
                {record["recording_session"] for record in output_records}
            ),
        }

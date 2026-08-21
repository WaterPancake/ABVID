"""Balanced paired nuisance/SNR corpus generation for Milestone 6."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from vehicle_audio.audio import load_audio_crop, save_audio
from vehicle_audio.augment import AugmentationPipeline, apply_frequency_response
from vehicle_audio.config import GenerationConfig
from vehicle_audio.dataset import (
    BackgroundSource,
    discover_background_recordings,
    discover_impulse_responses,
    discover_target_recordings,
)
from vehicle_audio.manifest import ManifestRecord, write_json, write_jsonl


FACTORIAL_DATASET_VERSION = 1


def _derived_seed(seed: int, *parts: object) -> int:
    payload = "|".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def _microphone_gains(config: GenerationConfig, seed: int) -> list[float]:
    generator = torch.Generator().manual_seed(seed)
    minimum = config.microphone_response.min_gain_db
    span = config.microphone_response.max_gain_db - minimum
    return [
        float(minimum + span * torch.rand((), generator=generator).item())
        for _ in config.microphone_response.anchor_frequencies_hz
    ]


class FactorialDatasetGenerator:
    """Cross every procedural source with controlled nuisance factors.

    All SNR and microphone variants of a base event share the clean crop,
    background crop, and non-microphone augmentation realization. This makes the
    factor comparisons paired rather than relying on post-hoc random subsets.
    """

    def __init__(
        self,
        config: GenerationConfig,
        targets_root: str | Path,
        backgrounds_root: str | Path,
        output_root: str | Path,
        *,
        impulse_responses_root: str | Path | None = None,
    ) -> None:
        self.config = config
        self.targets_root = Path(targets_root)
        self.backgrounds_root = Path(backgrounds_root)
        self.output_root = Path(output_root)
        self.impulse_responses_root = (
            None if impulse_responses_root is None else Path(impulse_responses_root)
        )

    def generate(self, *, seed: int, corruption_views: int = 1) -> dict[str, Any]:
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        if corruption_views <= 0:
            raise ValueError("corruption_views must be positive")
        if not self.config.noise.enabled or self.config.noise.probability != 1.0:
            raise ValueError("factorial generation requires always-applied noise")
        if not self.config.microphone_response.enabled:
            raise ValueError("factorial generation requires microphone_response.enabled")
        manifest_path = self.output_root / "manifest.jsonl"
        if manifest_path.exists():
            raise FileExistsError(f"factorial manifest already exists: {manifest_path}")

        targets = sorted(discover_target_recordings(self.targets_root), key=lambda x: x.source_id)
        backgrounds = discover_background_recordings(self.backgrounds_root)
        impulse_responses = discover_impulse_responses(self.impulse_responses_root)
        if not targets or not backgrounds:
            raise ValueError("factorial generation requires target and background recordings")
        backgrounds_by_category: dict[str, list[BackgroundSource]] = {}
        for background in backgrounds:
            backgrounds_by_category.setdefault(background.category, []).append(background)
        for values in backgrounds_by_category.values():
            values.sort(key=lambda value: value.source_id)

        # Microphone response is applied after the shared corruption realization so
        # on/off variants differ only by the controlled response.
        pipeline_config = replace(
            self.config,
            microphone_response=replace(
                self.config.microphone_response,
                enabled=False,
                probability=0.0,
            ),
        )
        pipeline = AugmentationPipeline(pipeline_config)
        ir_pairs: Sequence[tuple[str, Path]] = [
            (source.source_id, source.path) for source in impulse_responses
        ]
        sample_rate = self.config.audio.sample_rate
        target_length = self.config.audio.num_samples
        records: list[dict[str, Any]] = []

        for target_index, target_source in enumerate(targets):
            for view_index in range(corruption_views):
                base_event_id = f"event_{target_index:03d}_view_{view_index:02d}"
                target_seed = _derived_seed(seed, base_event_id, "target")
                target_generator = torch.Generator().manual_seed(target_seed)
                target_crop = load_audio_crop(
                    target_source.path,
                    sample_rate,
                    target_length,
                    target_generator,
                    repeat_if_short=self.config.temporal_crop.repeat_if_short,
                    randomize_start=self.config.temporal_crop.enabled,
                    source_start_seconds=target_source.segment_start_seconds,
                    source_end_seconds=target_source.segment_end_seconds,
                )
                clean = target_crop.waveform.to(torch.float32)

                for category_index, category in enumerate(sorted(backgrounds_by_category)):
                    category_sources = backgrounds_by_category[category]
                    source_index = _derived_seed(
                        seed, base_event_id, category, "source"
                    ) % len(category_sources)
                    background_source = category_sources[source_index]
                    background_seed = _derived_seed(
                        seed, base_event_id, category, "background_crop"
                    )
                    background_generator = torch.Generator().manual_seed(background_seed)
                    background_crop = load_audio_crop(
                        background_source.path,
                        sample_rate,
                        target_length,
                        background_generator,
                        repeat_if_short=self.config.temporal_crop.repeat_if_short,
                        randomize_start=self.config.temporal_crop.enabled,
                    )
                    background_original_channels = background_crop.waveform.shape[0]
                    background = background_crop.waveform
                    selected_channel = self.config.audio.background_channel
                    if selected_channel is not None:
                        if selected_channel >= background_original_channels:
                            raise ValueError(
                                f"configured background channel {selected_channel} but "
                                f"{background_source.path} has {background_original_channels}"
                            )
                        background = background[
                            selected_channel : selected_channel + 1
                        ]

                    corruption_seed = _derived_seed(
                        seed, base_event_id, category, "corruption"
                    )
                    microphone_seed = _derived_seed(
                        seed, base_event_id, category, "microphone"
                    )
                    microphone_gains = _microphone_gains(
                        self.config, microphone_seed
                    )
                    for snr_index, snr_db in enumerate(self.config.noise.snr_db_choices):
                        pipeline_generator = torch.Generator().manual_seed(
                            corruption_seed
                        )
                        base_result = pipeline(
                            clean,
                            background,
                            pipeline_generator,
                            impulse_responses=ir_pairs,
                            snr_db_override=snr_db,
                        )
                        for microphone_applied in (False, True):
                            waveform = base_result.waveform
                            augmentations = deepcopy(base_result.parameters)
                            if microphone_applied:
                                waveform = apply_frequency_response(
                                    waveform,
                                    sample_rate,
                                    self.config.microphone_response.anchor_frequencies_hz,
                                    microphone_gains,
                                )
                            augmentations["microphone_response"] = {
                                "applied": microphone_applied,
                                "anchor_frequencies_hz": list(
                                    self.config.microphone_response.anchor_frequencies_hz
                                ),
                                "anchor_gains_db": (
                                    microphone_gains if microphone_applied else []
                                ),
                                "controlled_factor": True,
                                "application_stage": "final_observation",
                            }
                            augmentations["temporal_crop"] = {
                                "applied": self.config.temporal_crop.enabled,
                                "output_num_samples": target_length,
                                "repeat_if_short": self.config.temporal_crop.repeat_if_short,
                            }
                            sample_index = len(records)
                            sample_id = (
                                f"factorial_{sample_index:06d}_"
                                f"{_derived_seed(seed, sample_index):016x}"
                            )
                            relative_dir = Path(sample_id)
                            clean_relative = relative_dir / "clean.wav"
                            corrupted_relative = relative_dir / "corrupted.wav"
                            metadata_relative = relative_dir / "metadata.json"
                            save_audio(
                                self.output_root / clean_relative,
                                clean,
                                sample_rate,
                            )
                            save_audio(
                                self.output_root / corrupted_relative,
                                waveform,
                                sample_rate,
                            )
                            record = ManifestRecord(
                                sample_id=sample_id,
                                target_source=target_source.source_id,
                                background_source=background_source.source_id,
                                background_category=category,
                                recording_session=target_source.recording_session,
                                vehicle_class=target_source.vehicle_class,
                                vehicle_model=target_source.vehicle_model,
                                source_domain=target_source.source_domain,
                                vehicle_id=target_source.vehicle_id,
                                simulation_run=target_source.simulation_run,
                                operating_condition=target_source.operating_condition,
                                engine_state=target_source.engine_state,
                                rpm=target_source.rpm,
                                throttle=target_source.throttle,
                                speed_mps=target_source.speed_mps,
                                acceleration_mps2=target_source.acceleration_mps2,
                                load=target_source.load,
                                source_listener_geometry=target_source.source_listener_geometry,
                                target_segment_start_seconds=target_source.segment_start_seconds,
                                target_segment_end_seconds=target_source.segment_end_seconds,
                                snr_db=float(snr_db),
                                gain_db=float(augmentations["gain"]["gain_db"]),
                                augmentation_seed=corruption_seed,
                                sample_rate=sample_rate,
                                num_channels=clean.shape[0],
                                num_samples=clean.shape[-1],
                                duration_seconds=clean.shape[-1] / sample_rate,
                                target_original_sample_rate=target_crop.original_sample_rate,
                                background_original_sample_rate=background_crop.original_sample_rate,
                                background_original_num_channels=background_original_channels,
                                background_channel_selected=selected_channel,
                                target_crop_start_frame=target_crop.start_frame,
                                target_crop_start_sample=target_crop.start_sample,
                                target_repeated=target_crop.repeated,
                                background_crop_start_frame=background_crop.start_frame,
                                background_crop_start_sample=background_crop.start_sample,
                                background_repeated=background_crop.repeated,
                                clean_path=clean_relative.as_posix(),
                                corrupted_path=corrupted_relative.as_posix(),
                                metadata_path=metadata_relative.as_posix(),
                                augmentations=augmentations,
                                generation_config=self.config.to_dict(),
                            ).to_dict()
                            geometry = target_source.source_listener_geometry or {}
                            record.update(
                                {
                                    "factorial_dataset_version": FACTORIAL_DATASET_VERSION,
                                    "factorial_seed": seed,
                                    "base_event_id": base_event_id,
                                    "corruption_view": view_index,
                                    "controlled_factors": {
                                        "background_category": category,
                                        "background_category_index": category_index,
                                        "snr_db": float(snr_db),
                                        "snr_index": snr_index,
                                        "microphone_response_applied": microphone_applied,
                                        "geometry_id": geometry.get("geometry_id"),
                                    },
                                }
                            )
                            write_json(self.output_root / metadata_relative, record)
                            records.append(record)

        write_jsonl(manifest_path, records)
        return self._summary(records, manifest_path, corruption_views)

    @staticmethod
    def _summary(
        records: Sequence[Mapping[str, Any]],
        manifest_path: Path,
        corruption_views: int,
    ) -> dict[str, Any]:
        return {
            "factorial_dataset_version": FACTORIAL_DATASET_VERSION,
            "generated_samples": len(records),
            "base_event_count": len(
                {str(record["base_event_id"]) for record in records}
            ),
            "corruption_views": corruption_views,
            "manifest": str(manifest_path),
            "vehicle_class_counts": dict(
                sorted(Counter(str(record["vehicle_class"]) for record in records).items())
            ),
            "background_category_counts": dict(
                sorted(
                    Counter(str(record["background_category"]) for record in records).items()
                )
            ),
            "snr_db_counts": dict(
                sorted(Counter(float(record["snr_db"]) for record in records).items())
            ),
            "microphone_response_counts": dict(
                sorted(
                    Counter(
                        str(bool(record["augmentations"]["microphone_response"]["applied"]))
                        for record in records
                    ).items()
                )
            ),
        }

"""Source discovery and deterministic paired-dataset generation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import torch

from vehicle_audio.audio import load_audio_crop, save_audio
from vehicle_audio.augment import AugmentationPipeline
from vehicle_audio.config import GenerationConfig
from vehicle_audio.manifest import ManifestRecord, write_json, write_jsonl
from vehicle_audio.metadata import ConditionSegment, validate_operating_condition


AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".ogg", ".aif", ".aiff"})


@dataclass(frozen=True)
class TargetSource:
    path: Path
    source_id: str
    vehicle_class: str
    vehicle_model: str | None
    recording_session: str
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
    segment_start_seconds: float | None = None
    segment_end_seconds: float | None = None


@dataclass(frozen=True)
class BackgroundSource:
    path: Path
    source_id: str
    category: str


@dataclass(frozen=True)
class ImpulseResponseSource:
    path: Path
    source_id: str


def _audio_paths(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"audio root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"audio root is not a directory: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )


def _load_sidecar(audio_path: Path) -> Mapping[str, Any]:
    sidecar_path = audio_path.with_suffix(".json")
    if not sidecar_path.exists():
        return {}
    with sidecar_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"metadata sidecar must contain a JSON object: {sidecar_path}")
    return value


def discover_target_recordings(root: str | Path) -> list[TargetSource]:
    """Discover targets and read/infer class, model, and session identifiers.

    A sibling ``clip.json`` can specify ``vehicle_class``, ``vehicle_model``,
    and ``recording_session``.  Otherwise the first directory below the root is
    the class, and the immediate parent directory is the session when present.
    """

    root_path = Path(root).resolve()
    sources: list[TargetSource] = []
    for audio_path in _audio_paths(root_path):
        relative = audio_path.relative_to(root_path)
        sidecar = _load_sidecar(audio_path)
        # A normalized artifact may be staged beside admitted recordings while its
        # rights or content review is still pending.  Explicitly gated files must
        # never enter augmentation or evaluation through directory discovery.
        if sidecar.get("admitted_to_corpus") is False:
            continue
        inferred_class = relative.parts[0] if len(relative.parts) > 1 else None
        inferred_session = relative.parent.name if len(relative.parts) > 2 else relative.stem
        vehicle_class_value = sidecar.get("vehicle_class", inferred_class)
        if vehicle_class_value is None:
            raise ValueError(
                f"cannot infer vehicle class for {audio_path}; place it in a class "
                "subdirectory or add a JSON sidecar"
            )
        vehicle_class = str(vehicle_class_value)
        recording_session = str(sidecar.get("recording_session", inferred_session))
        vehicle_model_value = sidecar.get("vehicle_model")
        vehicle_model = None if vehicle_model_value is None else str(vehicle_model_value)
        source_domain = str(sidecar.get("source_domain", "real_recording"))
        vehicle_id_value = sidecar.get("vehicle_id")
        vehicle_id = None if vehicle_id_value is None else str(vehicle_id_value)
        simulation_run_value = sidecar.get("simulation_run")
        simulation_run = None if simulation_run_value is None else str(simulation_run_value)
        engine_state_value = sidecar.get("engine_state")
        engine_state = None if engine_state_value is None else str(engine_state_value)
        rpm = sidecar.get("rpm")
        throttle = sidecar.get("throttle")
        speed_mps = sidecar.get("speed_mps")
        acceleration_value = sidecar.get("acceleration_mps2")
        acceleration_mps2 = (
            None if acceleration_value is None else float(acceleration_value)
        )
        load_value = sidecar.get("load")
        load = None if load_value is None else float(load_value)
        source_listener_geometry = sidecar.get("source_listener_geometry")
        if not vehicle_class or not recording_session:
            raise ValueError(f"empty target metadata identifier for {audio_path}")
        raw_segments = sidecar.get("condition_segments", [])
        if not isinstance(raw_segments, list):
            raise TypeError(f"condition_segments must be a list for {audio_path}")
        segments = [ConditionSegment.from_mapping(value) for value in raw_segments]
        if segments:
            for segment in segments:
                sources.append(
                    TargetSource(
                        path=audio_path,
                        source_id=relative.as_posix(),
                        vehicle_class=vehicle_class,
                        vehicle_model=vehicle_model,
                        recording_session=recording_session,
                        source_domain=source_domain,
                        vehicle_id=vehicle_id,
                        simulation_run=simulation_run,
                        operating_condition=segment.operating_condition,
                        engine_state=engine_state,
                        rpm=rpm,
                        throttle=throttle,
                        speed_mps=speed_mps,
                        acceleration_mps2=acceleration_mps2,
                        load=load,
                        source_listener_geometry=source_listener_geometry,
                        segment_start_seconds=segment.start_seconds,
                        segment_end_seconds=segment.end_seconds,
                    )
                )
        else:
            operating_condition = validate_operating_condition(
                str(sidecar.get("operating_condition", "unknown"))
            )
            sources.append(
                TargetSource(
                    path=audio_path,
                    source_id=relative.as_posix(),
                    vehicle_class=vehicle_class,
                    vehicle_model=vehicle_model,
                    recording_session=recording_session,
                    source_domain=source_domain,
                    vehicle_id=vehicle_id,
                    simulation_run=simulation_run,
                    operating_condition=operating_condition,
                    engine_state=engine_state,
                    rpm=rpm,
                    throttle=throttle,
                    speed_mps=speed_mps,
                    acceleration_mps2=acceleration_mps2,
                    load=load,
                    source_listener_geometry=source_listener_geometry,
                )
            )
    return sources


def discover_background_recordings(root: str | Path) -> list[BackgroundSource]:
    """Discover backgrounds, taking ``category`` from a sidecar when present."""

    root_path = Path(root).resolve()
    sources: list[BackgroundSource] = []
    for audio_path in _audio_paths(root_path):
        relative = audio_path.relative_to(root_path)
        sidecar = _load_sidecar(audio_path)
        inferred_category = relative.parts[0] if len(relative.parts) > 1 else "generic environmental ambience"
        category = str(sidecar.get("category", inferred_category))
        if not category:
            raise ValueError(f"empty background category for {audio_path}")
        sources.append(
            BackgroundSource(
                path=audio_path,
                source_id=relative.as_posix(),
                category=category,
            )
        )
    return sources


def discover_impulse_responses(root: str | Path | None) -> list[ImpulseResponseSource]:
    if root is None:
        return []
    root_path = Path(root).resolve()
    if not root_path.exists():
        return []
    return [
        ImpulseResponseSource(path=path, source_id=path.relative_to(root_path).as_posix())
        for path in _audio_paths(root_path)
    ]


def select_target_source(
    targets: Sequence[TargetSource],
    selector: random.Random,
    strategy: str,
) -> TargetSource:
    """Select a source without letting source-count imbalance dictate labels.

    ``class_condition_balanced`` first samples a vehicle class, then one of the
    conditions available for that class, and finally a source in that cell.
    """

    if not targets:
        raise ValueError("cannot select from an empty target list")
    if strategy == "uniform_source":
        return targets[selector.randrange(len(targets))]

    classes = sorted({target.vehicle_class for target in targets})
    vehicle_class = classes[selector.randrange(len(classes))]
    class_targets = [target for target in targets if target.vehicle_class == vehicle_class]
    if strategy == "class_balanced":
        return class_targets[selector.randrange(len(class_targets))]
    if strategy == "class_condition_balanced":
        conditions = sorted({target.operating_condition for target in class_targets})
        condition = conditions[selector.randrange(len(conditions))]
        cell = [
            target for target in class_targets if target.operating_condition == condition
        ]
        return cell[selector.randrange(len(cell))]
    raise ValueError(f"unknown target sampling strategy: {strategy!r}")


class DatasetGenerator:
    """Generate paired clean/corrupted samples and a JSONL manifest."""

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

    def generate(self, num_samples: int, seed: int) -> dict[str, Any]:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if seed < 0:
            raise ValueError("seed must be nonnegative")

        targets = discover_target_recordings(self.targets_root)
        backgrounds = discover_background_recordings(self.backgrounds_root)
        impulse_responses = discover_impulse_responses(self.impulse_responses_root)
        if not targets:
            raise ValueError(f"no supported target audio found under {self.targets_root}")
        if not backgrounds:
            raise ValueError(f"no supported background audio found under {self.backgrounds_root}")

        self.output_root.mkdir(parents=True, exist_ok=True)
        selector = random.Random(seed)
        pipeline = AugmentationPipeline(self.config)
        records: list[dict[str, Any]] = []
        target_length = self.config.audio.num_samples
        output_sample_rate = self.config.audio.sample_rate
        ir_pairs: Sequence[tuple[str, Path]] = [
            (source.source_id, source.path) for source in impulse_responses
        ]

        for sample_index in range(num_samples):
            augmentation_seed = selector.randrange(0, 2**63)
            sample_selector = random.Random(augmentation_seed)
            target_source = select_target_source(
                targets,
                sample_selector,
                self.config.sampling.target_strategy,
            )
            background_source = backgrounds[sample_selector.randrange(len(backgrounds))]
            torch_generator = torch.Generator(device="cpu")
            torch_generator.manual_seed(augmentation_seed)

            crop_randomized = bool(
                self.config.temporal_crop.enabled
                and float(torch.rand((), generator=torch_generator).item())
                < self.config.temporal_crop.probability
            )

            target_crop = load_audio_crop(
                target_source.path,
                output_sample_rate,
                target_length,
                torch_generator,
                repeat_if_short=self.config.temporal_crop.repeat_if_short,
                randomize_start=crop_randomized,
                source_start_seconds=target_source.segment_start_seconds,
                source_end_seconds=target_source.segment_end_seconds,
            )
            clean = target_crop.waveform.to(torch.float32)

            background_crop = load_audio_crop(
                background_source.path,
                output_sample_rate,
                target_length,
                torch_generator,
                repeat_if_short=self.config.temporal_crop.repeat_if_short,
                randomize_start=crop_randomized,
            )
            background_original_num_channels = background_crop.waveform.shape[0]
            background_channel_selected = self.config.audio.background_channel
            background_waveform = background_crop.waveform
            if background_channel_selected is not None:
                if background_channel_selected >= background_original_num_channels:
                    raise ValueError(
                        f"configured background channel {background_channel_selected} but "
                        f"{background_source.path} has {background_original_num_channels} channels"
                    )
                background_waveform = background_waveform[
                    background_channel_selected : background_channel_selected + 1
                ]

            result = pipeline(
                clean,
                background_waveform,
                torch_generator,
                impulse_responses=ir_pairs,
            )
            result.parameters["temporal_crop"] = {
                "applied": crop_randomized,
                "output_num_samples": target_length,
                "repeat_if_short": self.config.temporal_crop.repeat_if_short,
            }
            sample_id = f"sample_{sample_index:06d}_{augmentation_seed:016x}"
            relative_sample_dir = Path(sample_id)
            sample_dir = self.output_root / relative_sample_dir
            clean_relative = relative_sample_dir / "clean.wav"
            corrupted_relative = relative_sample_dir / "corrupted.wav"
            metadata_relative = relative_sample_dir / "metadata.json"
            save_audio(self.output_root / clean_relative, clean, output_sample_rate)
            save_audio(self.output_root / corrupted_relative, result.waveform, output_sample_rate)

            noise_parameters = result.parameters["noise"]
            record = ManifestRecord(
                sample_id=sample_id,
                target_source=target_source.source_id,
                background_source=background_source.source_id,
                background_category=background_source.category,
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
                snr_db=noise_parameters["snr_db"],
                gain_db=float(result.parameters["gain"]["gain_db"]),
                augmentation_seed=augmentation_seed,
                sample_rate=output_sample_rate,
                num_channels=clean.shape[0],
                num_samples=clean.shape[-1],
                duration_seconds=clean.shape[-1] / output_sample_rate,
                target_original_sample_rate=target_crop.original_sample_rate,
                background_original_sample_rate=background_crop.original_sample_rate,
                background_original_num_channels=background_original_num_channels,
                background_channel_selected=background_channel_selected,
                target_crop_start_frame=target_crop.start_frame,
                target_crop_start_sample=target_crop.start_sample,
                target_repeated=target_crop.repeated,
                background_crop_start_frame=background_crop.start_frame,
                background_crop_start_sample=background_crop.start_sample,
                background_repeated=background_crop.repeated,
                clean_path=clean_relative.as_posix(),
                corrupted_path=corrupted_relative.as_posix(),
                metadata_path=metadata_relative.as_posix(),
                augmentations=result.parameters,
                generation_config=self.config.to_dict(),
            ).to_dict()
            write_json(sample_dir / "metadata.json", record)
            records.append(record)

        manifest_path = self.output_root / "manifest.jsonl"
        write_jsonl(manifest_path, records)
        return _summarize(records, manifest_path, len(targets), len(backgrounds), len(impulse_responses))


def _summarize(
    records: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    num_target_sources: int,
    num_background_sources: int,
    num_impulse_responses: int,
) -> dict[str, Any]:
    classes = Counter(str(record["vehicle_class"]) for record in records)
    conditions = Counter(str(record["operating_condition"]) for record in records)
    class_conditions = Counter(
        f"{record['vehicle_class']}:{record['operating_condition']}" for record in records
    )
    categories = Counter(str(record["background_category"]) for record in records)
    snr_values = Counter(
        "none" if record["snr_db"] is None else f"{float(record['snr_db']):g}"
        for record in records
    )
    return {
        "generated_samples": len(records),
        "manifest": str(manifest_path),
        "discovered_target_sources": num_target_sources,
        "discovered_background_sources": num_background_sources,
        "discovered_impulse_responses": num_impulse_responses,
        "vehicle_class_counts": dict(sorted(classes.items())),
        "operating_condition_counts": dict(sorted(conditions.items())),
        "vehicle_class_condition_counts": dict(sorted(class_conditions.items())),
        "background_category_counts": dict(sorted(categories.items())),
        "snr_db_counts": dict(sorted(snr_values.items())),
    }

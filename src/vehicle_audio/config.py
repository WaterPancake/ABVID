"""Typed configuration for dataset generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def _validate_range(low: float, high: float, name: str) -> None:
    if low > high:
        raise ValueError(f"{name} minimum must not exceed maximum")


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    duration_seconds: float = 4.0
    background_channel: int | None = 0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("audio.sample_rate must be positive")
        if self.duration_seconds <= 0:
            raise ValueError("audio.duration_seconds must be positive")
        if self.background_channel is not None and self.background_channel < 0:
            raise ValueError("audio.background_channel must be nonnegative or null")

    @property
    def num_samples(self) -> int:
        return round(self.sample_rate * self.duration_seconds)


@dataclass(frozen=True)
class TemporalCropConfig:
    enabled: bool = True
    probability: float = 1.0
    repeat_if_short: bool = True

    def __post_init__(self) -> None:
        _validate_probability(self.probability, "temporal_crop.probability")


@dataclass(frozen=True)
class SamplingConfig:
    target_strategy: str = "class_condition_balanced"

    def __post_init__(self) -> None:
        allowed = {
            "uniform_source",
            "class_balanced",
            "class_condition_balanced",
            "class_session_condition_balanced",
        }
        if self.target_strategy not in allowed:
            raise ValueError(
                "sampling.target_strategy must be one of "
                f"{sorted(allowed)}, got {self.target_strategy!r}"
            )


@dataclass(frozen=True)
class GainConfig:
    enabled: bool = True
    probability: float = 1.0
    min_db: float = -12.0
    max_db: float = 3.0

    def __post_init__(self) -> None:
        _validate_probability(self.probability, "gain.probability")
        _validate_range(self.min_db, self.max_db, "gain")


@dataclass(frozen=True)
class NoiseConfig:
    enabled: bool = True
    probability: float = 1.0
    snr_db_choices: tuple[float, ...] = (
        30.0,
        20.0,
        10.0,
        5.0,
        0.0,
        -5.0,
        -10.0,
    )

    def __post_init__(self) -> None:
        _validate_probability(self.probability, "noise.probability")
        object.__setattr__(self, "snr_db_choices", tuple(float(x) for x in self.snr_db_choices))
        if not self.snr_db_choices:
            raise ValueError("noise.snr_db_choices must not be empty")


@dataclass(frozen=True)
class LowpassConfig:
    enabled: bool = True
    probability: float = 0.35
    min_cutoff_hz: float = 2_500.0
    max_cutoff_hz: float = 7_500.0
    order: int = 4

    def __post_init__(self) -> None:
        _validate_probability(self.probability, "lowpass.probability")
        _validate_range(self.min_cutoff_hz, self.max_cutoff_hz, "lowpass cutoff")
        if self.min_cutoff_hz <= 0 or self.order <= 0:
            raise ValueError("lowpass cutoff and order must be positive")


@dataclass(frozen=True)
class HighpassConfig:
    enabled: bool = True
    probability: float = 0.25
    min_cutoff_hz: float = 30.0
    max_cutoff_hz: float = 300.0
    order: int = 4

    def __post_init__(self) -> None:
        _validate_probability(self.probability, "highpass.probability")
        _validate_range(self.min_cutoff_hz, self.max_cutoff_hz, "highpass cutoff")
        if self.min_cutoff_hz <= 0 or self.order <= 0:
            raise ValueError("highpass cutoff and order must be positive")


@dataclass(frozen=True)
class BandpassConfig:
    enabled: bool = True
    probability: float = 0.15
    min_low_hz: float = 40.0
    max_low_hz: float = 400.0
    min_high_hz: float = 2_500.0
    max_high_hz: float = 7_500.0
    order: int = 3

    def __post_init__(self) -> None:
        _validate_probability(self.probability, "bandpass.probability")
        _validate_range(self.min_low_hz, self.max_low_hz, "bandpass low cutoff")
        _validate_range(self.min_high_hz, self.max_high_hz, "bandpass high cutoff")
        if self.min_low_hz <= 0 or self.order <= 0:
            raise ValueError("bandpass cutoffs and order must be positive")
        if self.max_low_hz >= self.min_high_hz:
            raise ValueError("bandpass low-cutoff range must be below high-cutoff range")


@dataclass(frozen=True)
class RandomEqConfig:
    enabled: bool = True
    probability: float = 0.35
    anchor_frequencies_hz: tuple[float, ...] = (
        60.0,
        150.0,
        400.0,
        1_000.0,
        2_500.0,
        6_000.0,
        8_000.0,
    )
    min_gain_db: float = -5.0
    max_gain_db: float = 5.0

    def __post_init__(self) -> None:
        _validate_probability(self.probability, "random_eq.probability")
        _validate_range(self.min_gain_db, self.max_gain_db, "random_eq gain")
        object.__setattr__(
            self,
            "anchor_frequencies_hz",
            tuple(float(x) for x in self.anchor_frequencies_hz),
        )
        if len(self.anchor_frequencies_hz) < 2:
            raise ValueError("random_eq requires at least two frequency anchors")


@dataclass(frozen=True)
class ResamplingConfig:
    enabled: bool = True
    probability: float = 0.25
    intermediate_rates: tuple[int, ...] = (4_000, 8_000, 12_000)

    def __post_init__(self) -> None:
        _validate_probability(self.probability, "resampling.probability")
        object.__setattr__(self, "intermediate_rates", tuple(int(x) for x in self.intermediate_rates))
        if not self.intermediate_rates or any(rate <= 0 for rate in self.intermediate_rates):
            raise ValueError("resampling.intermediate_rates must contain positive rates")


@dataclass(frozen=True)
class ClippingConfig:
    enabled: bool = True
    probability: float = 0.15
    min_threshold: float = 0.65
    max_threshold: float = 0.95

    def __post_init__(self) -> None:
        _validate_probability(self.probability, "clipping.probability")
        _validate_range(self.min_threshold, self.max_threshold, "clipping threshold")
        if self.min_threshold <= 0 or self.max_threshold > 1:
            raise ValueError("clipping thresholds must be in (0, 1]")


@dataclass(frozen=True)
class CompressionConfig:
    enabled: bool = True
    probability: float = 0.25
    min_threshold: float = 0.15
    max_threshold: float = 0.5
    min_ratio: float = 2.0
    max_ratio: float = 6.0

    def __post_init__(self) -> None:
        _validate_probability(self.probability, "compression.probability")
        _validate_range(self.min_threshold, self.max_threshold, "compression threshold")
        _validate_range(self.min_ratio, self.max_ratio, "compression ratio")
        if self.min_threshold <= 0 or self.min_ratio < 1:
            raise ValueError("compression threshold must be positive and ratio at least 1")


@dataclass(frozen=True)
class ReverbConfig:
    enabled: bool = True
    probability: float = 0.4

    def __post_init__(self) -> None:
        _validate_probability(self.probability, "reverb.probability")


@dataclass(frozen=True)
class MicrophoneResponseConfig:
    enabled: bool = True
    probability: float = 0.5
    anchor_frequencies_hz: tuple[float, ...] = (0.0, 80.0, 250.0, 1_000.0, 4_000.0, 8_000.0)
    min_gain_db: float = -6.0
    max_gain_db: float = 3.0

    def __post_init__(self) -> None:
        _validate_probability(self.probability, "microphone_response.probability")
        _validate_range(self.min_gain_db, self.max_gain_db, "microphone_response gain")
        object.__setattr__(
            self,
            "anchor_frequencies_hz",
            tuple(float(x) for x in self.anchor_frequencies_hz),
        )
        if len(self.anchor_frequencies_hz) < 2:
            raise ValueError("microphone_response requires at least two frequency anchors")


@dataclass(frozen=True)
class GenerationConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    temporal_crop: TemporalCropConfig = field(default_factory=TemporalCropConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    gain: GainConfig = field(default_factory=GainConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    lowpass: LowpassConfig = field(default_factory=LowpassConfig)
    highpass: HighpassConfig = field(default_factory=HighpassConfig)
    bandpass: BandpassConfig = field(default_factory=BandpassConfig)
    random_eq: RandomEqConfig = field(default_factory=RandomEqConfig)
    resampling: ResamplingConfig = field(default_factory=ResamplingConfig)
    clipping: ClippingConfig = field(default_factory=ClippingConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    reverb: ReverbConfig = field(default_factory=ReverbConfig)
    microphone_response: MicrophoneResponseConfig = field(default_factory=MicrophoneResponseConfig)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "GenerationConfig":
        builders = {
            "audio": AudioConfig,
            "temporal_crop": TemporalCropConfig,
            "sampling": SamplingConfig,
            "gain": GainConfig,
            "noise": NoiseConfig,
            "lowpass": LowpassConfig,
            "highpass": HighpassConfig,
            "bandpass": BandpassConfig,
            "random_eq": RandomEqConfig,
            "resampling": ResamplingConfig,
            "clipping": ClippingConfig,
            "compression": CompressionConfig,
            "reverb": ReverbConfig,
            "microphone_response": MicrophoneResponseConfig,
        }
        unknown = set(values) - set(builders)
        if unknown:
            raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
        kwargs: dict[str, Any] = {}
        for name, builder in builders.items():
            section = values.get(name)
            if section is not None:
                if not isinstance(section, Mapping):
                    raise TypeError(f"configuration section {name!r} must be a mapping")
                kwargs[name] = builder(**section)
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> GenerationConfig:
    """Load and validate a YAML generation configuration."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("the top-level YAML value must be a mapping")
    return GenerationConfig.from_mapping(raw)

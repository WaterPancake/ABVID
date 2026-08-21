"""Deterministic microphone-array simulation, localization, and beamforming.

Azimuth is measured in degrees from array broadside: zero degrees points along
positive ``y`` and positive angles rotate toward positive ``x``.  Public audio
tensors use ``[channels, samples]`` throughout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from vehicle_audio.audio import validate_waveform
from vehicle_audio.augment import apply_frequency_response, convolve_impulse_response
from vehicle_audio.mixing import measure_snr_db, mix_at_snr


def _as_positions(values: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], ...]:
    positions: list[tuple[float, float, float]] = []
    for value in values:
        if len(value) != 3:
            raise ValueError("every microphone position must contain x, y, and z")
        position = tuple(float(coordinate) for coordinate in value)
        if not all(math.isfinite(coordinate) for coordinate in position):
            raise ValueError("microphone positions must be finite")
        positions.append(position)
    return tuple(positions)


@dataclass(frozen=True)
class ArrayReverberationConfig:
    enabled: bool = True
    num_reflections: int = 3
    min_delay_ms: float = 2.0
    max_delay_ms: float = 14.0
    min_gain: float = 0.03
    max_gain: float = 0.12

    def __post_init__(self) -> None:
        if self.num_reflections < 0:
            raise ValueError("reverberation.num_reflections must be nonnegative")
        if not 0 <= self.min_delay_ms <= self.max_delay_ms:
            raise ValueError("reverberation delay range is invalid")
        if not 0 <= self.min_gain <= self.max_gain < 1:
            raise ValueError("reverberation gains must satisfy 0 <= min <= max < 1")


@dataclass(frozen=True)
class ArrayMicrophoneResponseConfig:
    enabled: bool = True
    anchor_frequencies_hz: tuple[float, ...] = (
        0.0,
        80.0,
        250.0,
        1_000.0,
        4_000.0,
        8_000.0,
    )
    min_gain_db: float = -2.0
    max_gain_db: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "anchor_frequencies_hz",
            tuple(float(value) for value in self.anchor_frequencies_hz),
        )
        if len(self.anchor_frequencies_hz) < 2:
            raise ValueError("microphone response requires at least two anchors")
        if any(
            second <= first
            for first, second in zip(
                self.anchor_frequencies_hz,
                self.anchor_frequencies_hz[1:],
            )
        ):
            raise ValueError("microphone-response anchors must be strictly increasing")
        if self.min_gain_db > self.max_gain_db:
            raise ValueError("microphone-response gain range is invalid")


@dataclass(frozen=True)
class ArraySimulationConfig:
    sample_rate: int = 16_000
    duration_seconds: float = 2.0
    speed_of_sound_mps: float = 343.0
    source_channel: int = 0
    background_channel: int | None = 0
    microphone_positions_m: tuple[tuple[float, float, float], ...] = (
        (-0.12, 0.0, 1.2),
        (-0.04, 0.0, 1.2),
        (0.04, 0.0, 1.2),
        (0.12, 0.0, 1.2),
    )
    microphone_counts: tuple[int, ...] = (1, 2, 4)
    paired_snr_sweep: bool = True
    snr_db_choices: tuple[float, ...] = (30.0, 20.0, 10.0, 5.0, 0.0, -5.0, -10.0)
    azimuth_deg_choices: tuple[float, ...] = (
        -75.0,
        -60.0,
        -45.0,
        -30.0,
        -15.0,
        0.0,
        15.0,
        30.0,
        45.0,
        60.0,
        75.0,
    )
    source_distance_m_choices: tuple[float, ...] = (12.0, 20.0, 35.0)
    independent_sensor_noise_fraction: float = 0.03
    reverberation: ArrayReverberationConfig = field(default_factory=ArrayReverberationConfig)
    microphone_response: ArrayMicrophoneResponseConfig = field(
        default_factory=ArrayMicrophoneResponseConfig
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "microphone_positions_m",
            _as_positions(self.microphone_positions_m),
        )
        object.__setattr__(
            self, "microphone_counts", tuple(int(value) for value in self.microphone_counts)
        )
        object.__setattr__(
            self, "snr_db_choices", tuple(float(value) for value in self.snr_db_choices)
        )
        object.__setattr__(
            self,
            "azimuth_deg_choices",
            tuple(float(value) for value in self.azimuth_deg_choices),
        )
        object.__setattr__(
            self,
            "source_distance_m_choices",
            tuple(float(value) for value in self.source_distance_m_choices),
        )
        if self.sample_rate <= 0 or self.duration_seconds <= 0:
            raise ValueError("sample_rate and duration_seconds must be positive")
        if self.speed_of_sound_mps <= 0:
            raise ValueError("speed_of_sound_mps must be positive")
        if self.source_channel < 0:
            raise ValueError("source_channel must be nonnegative")
        if self.background_channel is not None and self.background_channel < 0:
            raise ValueError("background_channel must be nonnegative or null")
        if len(self.microphone_positions_m) < 2:
            raise ValueError("at least two microphone positions are required")
        if len(set(self.microphone_positions_m)) != len(self.microphone_positions_m):
            raise ValueError("microphone positions must be unique")
        if not self.microphone_counts or self.microphone_counts[0] != 1:
            raise ValueError("microphone_counts must start with the one-microphone baseline")
        if tuple(sorted(set(self.microphone_counts))) != self.microphone_counts:
            raise ValueError("microphone_counts must be unique and increasing")
        if self.microphone_counts[-1] > len(self.microphone_positions_m):
            raise ValueError("microphone_counts exceeds the configured array size")
        if not self.snr_db_choices or not self.azimuth_deg_choices:
            raise ValueError("SNR and azimuth choices must not be empty")
        if any(abs(value) >= 90 for value in self.azimuth_deg_choices):
            raise ValueError("linear-array azimuths must remain within (-90, 90) degrees")
        if not self.source_distance_m_choices or any(
            value <= 0 for value in self.source_distance_m_choices
        ):
            raise ValueError("source distances must be positive")
        if self.independent_sensor_noise_fraction < 0:
            raise ValueError("independent_sensor_noise_fraction must be nonnegative")
        nyquist = self.sample_rate / 2.0
        if self.microphone_response.anchor_frequencies_hz[-1] > nyquist:
            raise ValueError("microphone-response anchors must not exceed Nyquist")

    @property
    def num_samples(self) -> int:
        return round(self.sample_rate * self.duration_seconds)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArraySimulationConfig":
        raw = dict(value)
        raw["reverberation"] = ArrayReverberationConfig(**raw.get("reverberation", {}))
        raw["microphone_response"] = ArrayMicrophoneResponseConfig(
            **raw.get("microphone_response", {})
        )
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_array_config(path: str | Path) -> ArraySimulationConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise TypeError("array config must contain a mapping")
    return ArraySimulationConfig.from_mapping(value)


@dataclass(frozen=True)
class ArraySimulationResult:
    observation: torch.Tensor
    propagated_clean: torch.Tensor
    scaled_noise: torch.Tensor
    metadata: dict[str, Any]


def azimuth_unit_vector(azimuth_deg: float) -> torch.Tensor:
    """Return the source direction for broadside-referenced azimuth."""

    angle = math.radians(float(azimuth_deg))
    return torch.tensor((math.sin(angle), math.cos(angle), 0.0), dtype=torch.float64)


def fractional_delay(waveform: torch.Tensor, delay_samples: float) -> torch.Tensor:
    """Shift audio by a fractional number of samples using linear interpolation.

    Positive values delay the signal and negative values advance it. Samples that
    move outside the observation window are replaced with zeros; no circular wrap
    is introduced.
    """

    validate_waveform(waveform)
    if not math.isfinite(delay_samples):
        raise ValueError("delay_samples must be finite")
    num_samples = waveform.shape[-1]
    source_position = (
        torch.arange(num_samples, device=waveform.device, dtype=torch.float64)
        - float(delay_samples)
    )
    lower = torch.floor(source_position).to(torch.long)
    upper = lower + 1
    upper_weight = (source_position - lower.to(source_position.dtype)).to(waveform.dtype)
    lower_weight = 1.0 - upper_weight
    output = torch.zeros_like(waveform)
    lower_valid = (lower >= 0) & (lower < num_samples)
    upper_valid = (upper >= 0) & (upper < num_samples)
    if lower_valid.any():
        output[:, lower_valid] += (
            waveform[:, lower[lower_valid]] * lower_weight[lower_valid].unsqueeze(0)
        )
    if upper_valid.any():
        output[:, upper_valid] += (
            waveform[:, upper[upper_valid]] * upper_weight[upper_valid].unsqueeze(0)
        )
    return output


def far_field_arrival_delays_seconds(
    microphone_positions_m: Sequence[Sequence[float]],
    azimuth_deg: float,
    speed_of_sound_mps: float = 343.0,
) -> torch.Tensor:
    """Return relative far-field arrival delays with microphone zero as reference."""

    if speed_of_sound_mps <= 0:
        raise ValueError("speed_of_sound_mps must be positive")
    positions = torch.tensor(_as_positions(microphone_positions_m), dtype=torch.float64)
    direction = azimuth_unit_vector(azimuth_deg)
    delays = -(positions @ direction) / float(speed_of_sound_mps)
    return delays - delays[0]


def _uniform(
    generator: torch.Generator, minimum: float, maximum: float
) -> float:
    if minimum == maximum:
        return float(minimum)
    return float(minimum + float(torch.rand((), generator=generator)) * (maximum - minimum))


def _random_impulse_response(
    sample_rate: int,
    config: ArrayReverberationConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    if not config.enabled or config.num_reflections == 0:
        return torch.ones((1, 1), dtype=torch.float32)
    max_delay = max(1, round(config.max_delay_ms * sample_rate / 1_000.0))
    min_delay = max(1, round(config.min_delay_ms * sample_rate / 1_000.0))
    impulse_response = torch.zeros((1, max_delay + 1), dtype=torch.float32)
    impulse_response[0, 0] = 1.0
    for _ in range(config.num_reflections):
        delay = int(
            torch.randint(min_delay, max_delay + 1, (1,), generator=generator).item()
        )
        gain = _uniform(generator, config.min_gain, config.max_gain)
        sign = -1.0 if float(torch.rand((), generator=generator)) < 0.5 else 1.0
        impulse_response[0, delay] += sign * gain
    return impulse_response


def simulate_microphone_array(
    source: torch.Tensor,
    background: torch.Tensor,
    config: ArraySimulationConfig,
    *,
    azimuth_deg: float,
    source_distance_m: float,
    snr_db: float,
    generator: torch.Generator,
) -> ArraySimulationResult:
    """Create one synchronized array observation with exact controlled SNR."""

    validate_waveform(source)
    validate_waveform(background)
    if source.shape[0] != 1:
        raise ValueError("array simulation requires one explicitly selected source channel")
    num_microphones = len(config.microphone_positions_m)
    if background.shape[0] == 1:
        background = background.expand(num_microphones, -1).clone()
    elif background.shape[0] != num_microphones:
        raise ValueError("background must be mono or contain one channel per microphone")
    if source.shape[-1] != config.num_samples or background.shape[-1] != config.num_samples:
        raise ValueError("source and background lengths must match the array configuration")
    if source_distance_m <= 0:
        raise ValueError("source_distance_m must be positive")
    if abs(azimuth_deg) >= 90:
        raise ValueError("linear-array azimuth must be within (-90, 90) degrees")

    positions = torch.tensor(config.microphone_positions_m, dtype=torch.float64)
    array_center = positions.mean(dim=0)
    source_position = array_center + source_distance_m * azimuth_unit_vector(azimuth_deg)
    distances = torch.linalg.vector_norm(source_position.unsqueeze(0) - positions, dim=1)
    absolute_arrival_seconds = distances / config.speed_of_sound_mps
    relative_arrival_seconds = absolute_arrival_seconds - absolute_arrival_seconds.min()

    clean_channels: list[torch.Tensor] = []
    noise_channels: list[torch.Tensor] = []
    impulse_responses: list[list[float]] = []
    microphone_gains: list[list[float]] = []
    for microphone_index in range(num_microphones):
        delayed = fractional_delay(
            source,
            float(relative_arrival_seconds[microphone_index]) * config.sample_rate,
        )
        attenuation = 1.0 / max(float(distances[microphone_index]), 1.0)
        clean_channel = delayed * attenuation
        impulse_response = _random_impulse_response(
            config.sample_rate,
            config.reverberation,
            generator,
        )
        clean_channel = convolve_impulse_response(clean_channel, impulse_response)

        raw_noise = background[microphone_index : microphone_index + 1].clone()
        if config.independent_sensor_noise_fraction:
            sensor_noise = torch.randn(
                raw_noise.shape,
                generator=generator,
                dtype=raw_noise.dtype,
                device=raw_noise.device,
            )
            raw_rms = raw_noise.square().mean().sqrt().clamp_min(1e-8)
            sensor_noise = sensor_noise / sensor_noise.square().mean().sqrt().clamp_min(1e-8)
            raw_noise += (
                config.independent_sensor_noise_fraction * raw_rms * sensor_noise
            )

        gains = [
            _uniform(
                generator,
                config.microphone_response.min_gain_db,
                config.microphone_response.max_gain_db,
            )
            for _ in config.microphone_response.anchor_frequencies_hz
        ]
        if config.microphone_response.enabled:
            clean_channel = apply_frequency_response(
                clean_channel,
                config.sample_rate,
                config.microphone_response.anchor_frequencies_hz,
                gains,
            )
            raw_noise = apply_frequency_response(
                raw_noise,
                config.sample_rate,
                config.microphone_response.anchor_frequencies_hz,
                gains,
            )
        else:
            gains = [0.0 for _ in config.microphone_response.anchor_frequencies_hz]
        clean_channels.append(clean_channel)
        noise_channels.append(raw_noise)
        impulse_responses.append(impulse_response.flatten().tolist())
        microphone_gains.append(gains)

    propagated_clean = torch.cat(clean_channels, dim=0)
    unscaled_noise = torch.cat(noise_channels, dim=0)
    mix = mix_at_snr(propagated_clean, unscaled_noise, snr_db)
    if not torch.isfinite(mix.mixture).all():
        raise RuntimeError("array simulation produced NaN or Inf")
    metadata = {
        "azimuth_deg": float(azimuth_deg),
        "source_distance_m": float(source_distance_m),
        "source_position_m": source_position.tolist(),
        "microphone_positions_m": [list(position) for position in config.microphone_positions_m],
        "path_distances_m": distances.tolist(),
        "absolute_arrival_seconds": absolute_arrival_seconds.tolist(),
        "relative_arrival_seconds": relative_arrival_seconds.tolist(),
        "attenuation_linear": [1.0 / max(float(value), 1.0) for value in distances],
        "requested_snr_db": float(snr_db),
        "measured_snr_db": mix.measured_snr_db,
        "microphone_response_gains_db": microphone_gains,
        "impulse_responses": impulse_responses,
    }
    return ArraySimulationResult(
        observation=mix.mixture,
        propagated_clean=propagated_clean,
        scaled_noise=mix.scaled_noise,
        metadata=metadata,
    )


def _gcc_phat_correlation(
    signal: torch.Tensor,
    reference: torch.Tensor,
    *,
    interpolation: int = 8,
    max_delay_samples: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if signal.ndim != 1 or reference.ndim != 1 or signal.shape != reference.shape:
        raise ValueError("GCC-PHAT inputs must be equal-length one-dimensional tensors")
    if interpolation <= 0:
        raise ValueError("interpolation must be positive")
    fft_size = 1
    while fft_size < signal.numel() + reference.numel():
        fft_size *= 2
    signal_spectrum = torch.fft.rfft(signal.to(torch.float64), n=fft_size)
    reference_spectrum = torch.fft.rfft(reference.to(torch.float64), n=fft_size)
    cross_spectrum = signal_spectrum * reference_spectrum.conj()
    cross_spectrum /= cross_spectrum.abs().clamp_min(1e-12)
    correlation_size = fft_size * interpolation
    correlation = torch.fft.irfft(cross_spectrum, n=correlation_size)
    maximum = correlation_size // 2
    if max_delay_samples is not None:
        maximum = min(maximum, math.ceil(abs(max_delay_samples) * interpolation))
    centered = torch.cat((correlation[-maximum:], correlation[: maximum + 1]))
    lags = torch.arange(-maximum, maximum + 1, dtype=torch.float64) / interpolation
    return centered, lags


def gcc_phat_delay_samples(
    signal: torch.Tensor,
    reference: torch.Tensor,
    *,
    interpolation: int = 8,
    max_delay_samples: float | None = None,
) -> float:
    """Estimate how many samples ``signal`` lags ``reference``."""

    correlation, lags = _gcc_phat_correlation(
        signal,
        reference,
        interpolation=interpolation,
        max_delay_samples=max_delay_samples,
    )
    return float(lags[int(correlation.argmax())])


def gcc_phat_azimuth(
    observation: torch.Tensor,
    microphone_positions_m: Sequence[Sequence[float]],
    sample_rate: int,
    *,
    speed_of_sound_mps: float = 343.0,
    interpolation: int = 8,
) -> float:
    """Estimate broadside azimuth by GCC-PHAT TDOAs and least squares."""

    validate_waveform(observation)
    positions = torch.tensor(_as_positions(microphone_positions_m), dtype=torch.float64)
    if observation.shape[0] != len(positions):
        raise ValueError("observation channel count does not match microphone positions")
    if observation.shape[0] < 2:
        raise ValueError("GCC-PHAT azimuth requires at least two microphones")
    reference = observation[0] - observation[0].mean()
    maximum_delay = torch.linalg.vector_norm(positions - positions[0], dim=1).max()
    maximum_delay_samples = float(maximum_delay) / speed_of_sound_mps * sample_rate
    delays = []
    for index in range(1, observation.shape[0]):
        delays.append(
            gcc_phat_delay_samples(
                observation[index] - observation[index].mean(),
                reference,
                interpolation=interpolation,
                max_delay_samples=maximum_delay_samples,
            )
            / sample_rate
        )
    delta_positions = positions[1:, :2] - positions[0, :2]
    design = -delta_positions / speed_of_sound_mps
    estimate = torch.linalg.lstsq(design, torch.tensor(delays, dtype=torch.float64)).solution
    if int(torch.linalg.matrix_rank(design)) < 2:
        # The configured ULA lies on x, so TDOA determines sin(azimuth). The
        # positive-y solution is selected by the documented (-90, 90) field.
        if float(delta_positions[:, 0].abs().max()) < float(
            delta_positions[:, 1].abs().max()
        ):
            raise ValueError("GCC-PHAT broadside azimuth expects an x-axis linear array")
        sine = max(-1.0, min(1.0, float(estimate[0])))
        estimate = torch.tensor(
            (sine, math.sqrt(max(0.0, 1.0 - sine * sine))),
            dtype=torch.float64,
        )
    else:
        if float(torch.linalg.vector_norm(estimate)) < 1e-12:
            return 0.0
        estimate /= torch.linalg.vector_norm(estimate)
    azimuth = math.degrees(math.atan2(float(estimate[0]), float(estimate[1])))
    # A linear array only determines sin(azimuth); constrain to its unambiguous range.
    return float(max(-89.999, min(89.999, azimuth)))


def srp_phat_azimuth(
    observation: torch.Tensor,
    microphone_positions_m: Sequence[Sequence[float]],
    sample_rate: int,
    *,
    speed_of_sound_mps: float = 343.0,
    azimuth_grid_deg: Sequence[float] | None = None,
    interpolation: int = 8,
) -> tuple[float, torch.Tensor]:
    """Estimate azimuth with pairwise steered-response-power PHAT."""

    validate_waveform(observation)
    positions = _as_positions(microphone_positions_m)
    if observation.shape[0] != len(positions) or observation.shape[0] < 2:
        raise ValueError("SRP-PHAT requires matching positions for at least two channels")
    grid = tuple(
        float(value)
        for value in (
            azimuth_grid_deg
            if azimuth_grid_deg is not None
            else range(-85, 86)
        )
    )
    if not grid:
        raise ValueError("azimuth grid must not be empty")
    scores = torch.zeros(len(grid), dtype=torch.float64)
    position_tensor = torch.tensor(positions, dtype=torch.float64)
    maximum_aperture = max(
        float(torch.linalg.vector_norm(position_tensor[first] - position_tensor[second]))
        for first in range(len(positions))
        for second in range(first)
    )
    maximum_delay_samples = maximum_aperture / speed_of_sound_mps * sample_rate
    centered = observation - observation.mean(dim=1, keepdim=True)
    pair_data: list[tuple[int, int, torch.Tensor, torch.Tensor]] = []
    for first in range(observation.shape[0]):
        for second in range(first):
            correlation, lags = _gcc_phat_correlation(
                centered[first],
                centered[second],
                interpolation=interpolation,
                max_delay_samples=maximum_delay_samples,
            )
            pair_data.append((first, second, correlation, lags))

    for grid_index, azimuth in enumerate(grid):
        delays = far_field_arrival_delays_seconds(
            positions,
            azimuth,
            speed_of_sound_mps,
        ) * sample_rate
        score = 0.0
        for first, second, correlation, lags in pair_data:
            predicted = float(delays[first] - delays[second])
            fractional_index = (predicted - float(lags[0])) * interpolation
            lower = math.floor(fractional_index)
            upper = lower + 1
            if 0 <= lower < len(correlation):
                weight = fractional_index - lower
                lower_value = float(correlation[lower])
                upper_value = (
                    float(correlation[upper]) if upper < len(correlation) else lower_value
                )
                score += (1.0 - weight) * lower_value + weight * upper_value
        scores[grid_index] = score
    best_index = int(scores.argmax())
    return grid[best_index], scores


def delay_and_sum_beamform(
    observation: torch.Tensor,
    microphone_positions_m: Sequence[Sequence[float]],
    azimuth_deg: float,
    sample_rate: int,
    *,
    speed_of_sound_mps: float = 343.0,
) -> torch.Tensor:
    """Steer and average array channels into one waveform."""

    validate_waveform(observation)
    positions = _as_positions(microphone_positions_m)
    if observation.shape[0] != len(positions):
        raise ValueError("observation channel count does not match microphone positions")
    delays = far_field_arrival_delays_seconds(
        positions,
        azimuth_deg,
        speed_of_sound_mps,
    )
    aligned = [
        fractional_delay(
            observation[index : index + 1],
            -float(delays[index]) * sample_rate,
        )
        for index in range(observation.shape[0])
    ]
    return torch.stack(aligned, dim=0).mean(dim=0)


def angular_error_degrees(estimate: float, target: float) -> float:
    difference = (float(estimate) - float(target) + 180.0) % 360.0 - 180.0
    return abs(difference)


def beamformed_snr_db(
    propagated_clean: torch.Tensor,
    scaled_noise: torch.Tensor,
    microphone_positions_m: Sequence[Sequence[float]],
    azimuth_deg: float,
    sample_rate: int,
    *,
    speed_of_sound_mps: float = 343.0,
) -> float:
    clean = delay_and_sum_beamform(
        propagated_clean,
        microphone_positions_m,
        azimuth_deg,
        sample_rate,
        speed_of_sound_mps=speed_of_sound_mps,
    )
    noise = delay_and_sum_beamform(
        scaled_noise,
        microphone_positions_m,
        azimuth_deg,
        sample_rate,
        speed_of_sound_mps=speed_of_sound_mps,
    )
    return measure_snr_db(clean, noise)

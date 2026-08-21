"""Composable waveform augmentations and the seeded corruption pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import signal as scipy_signal
import torch
import torch.nn.functional as torch_functional

from vehicle_audio.audio import (
    force_num_samples,
    load_audio,
    resample_audio,
    validate_waveform,
)
from vehicle_audio.config import GenerationConfig
from vehicle_audio.mixing import mix_at_snr


@dataclass(frozen=True)
class AugmentationResult:
    waveform: torch.Tensor
    parameters: dict[str, Any]


def apply_gain(waveform: torch.Tensor, gain_db: float) -> torch.Tensor:
    validate_waveform(waveform)
    return waveform * (10.0 ** (float(gain_db) / 20.0))


def _apply_sos_filter(waveform: torch.Tensor, sos: np.ndarray) -> torch.Tensor:
    validate_waveform(waveform)
    original_device = waveform.device
    filtered = scipy_signal.sosfilt(
        sos,
        waveform.detach().cpu().numpy().astype(np.float64, copy=False),
        axis=-1,
    ).astype(np.float32, copy=False)
    return torch.from_numpy(filtered.copy()).to(device=original_device, dtype=waveform.dtype)


def lowpass_filter(
    waveform: torch.Tensor,
    sample_rate: int,
    cutoff_hz: float,
    *,
    order: int = 4,
) -> torch.Tensor:
    """Apply a Butterworth low-pass filter without changing shape."""

    _validate_cutoff(cutoff_hz, sample_rate)
    sos = scipy_signal.butter(order, cutoff_hz, btype="lowpass", fs=sample_rate, output="sos")
    return _apply_sos_filter(waveform, sos)


def highpass_filter(
    waveform: torch.Tensor,
    sample_rate: int,
    cutoff_hz: float,
    *,
    order: int = 4,
) -> torch.Tensor:
    """Apply a Butterworth high-pass filter without changing shape."""

    _validate_cutoff(cutoff_hz, sample_rate)
    sos = scipy_signal.butter(order, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return _apply_sos_filter(waveform, sos)


def bandpass_filter(
    waveform: torch.Tensor,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    *,
    order: int = 3,
) -> torch.Tensor:
    """Apply a Butterworth band-pass filter without changing shape."""

    _validate_cutoff(low_hz, sample_rate)
    _validate_cutoff(high_hz, sample_rate)
    if low_hz >= high_hz:
        raise ValueError("band-pass low cutoff must be below high cutoff")
    sos = scipy_signal.butter(
        order,
        (low_hz, high_hz),
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    return _apply_sos_filter(waveform, sos)


def _validate_cutoff(cutoff_hz: float, sample_rate: int) -> None:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not 0.0 < cutoff_hz < sample_rate / 2.0:
        raise ValueError(
            f"filter cutoff {cutoff_hz} must be between 0 and Nyquist ({sample_rate / 2})"
        )


def apply_frequency_response(
    waveform: torch.Tensor,
    sample_rate: int,
    anchor_frequencies_hz: Sequence[float],
    anchor_gains_db: Sequence[float],
) -> torch.Tensor:
    """Approximate an EQ or microphone response in the frequency domain.

    Gain is linearly interpolated in dB between frequency anchors.  The same
    response is applied independently to every channel; channels are not mixed.
    """

    validate_waveform(waveform)
    if len(anchor_frequencies_hz) != len(anchor_gains_db) or len(anchor_frequencies_hz) < 2:
        raise ValueError("frequency response requires equally sized anchor arrays")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    frequencies = np.asarray(anchor_frequencies_hz, dtype=np.float64)
    gains_db = np.asarray(anchor_gains_db, dtype=np.float64)
    if not np.all(np.isfinite(frequencies)) or not np.all(np.isfinite(gains_db)):
        raise ValueError("frequency-response anchors must be finite")
    if np.any(frequencies < 0) or np.any(np.diff(frequencies) <= 0):
        raise ValueError("frequency anchors must be nonnegative and strictly increasing")

    fft_frequencies = np.fft.rfftfreq(waveform.shape[-1], d=1.0 / sample_rate)
    interpolated_db = np.interp(fft_frequencies, frequencies, gains_db)
    linear_gain = torch.from_numpy(np.power(10.0, interpolated_db / 20.0)).to(
        device=waveform.device,
        dtype=waveform.dtype,
    )
    spectrum = torch.fft.rfft(waveform, dim=-1)
    return torch.fft.irfft(spectrum * linear_gain, n=waveform.shape[-1], dim=-1)


def resampling_degradation(
    waveform: torch.Tensor,
    sample_rate: int,
    intermediate_sample_rate: int,
) -> torch.Tensor:
    """Round-trip through a lower rate and return at the original rate/length."""

    validate_waveform(waveform)
    if intermediate_sample_rate >= sample_rate:
        raise ValueError("intermediate_sample_rate must be below sample_rate")
    degraded = resample_audio(waveform, sample_rate, intermediate_sample_rate)
    restored = resample_audio(degraded, intermediate_sample_rate, sample_rate)
    return force_num_samples(restored, waveform.shape[-1])


def hard_clip(waveform: torch.Tensor, threshold: float) -> torch.Tensor:
    """Apply symmetric finite hard clipping."""

    validate_waveform(waveform)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("clipping threshold must be in (0, 1]")
    return torch.clamp(waveform, min=-float(threshold), max=float(threshold))


def dynamic_range_compress(
    waveform: torch.Tensor,
    threshold: float,
    ratio: float,
) -> torch.Tensor:
    """Apply a memoryless symmetric compressor above ``threshold``.

    For magnitude ``x`` above the threshold, the output magnitude is
    ``threshold + (x - threshold) / ratio``.
    """

    validate_waveform(waveform)
    if threshold <= 0.0:
        raise ValueError("compression threshold must be positive")
    if ratio < 1.0:
        raise ValueError("compression ratio must be at least 1")
    magnitude = waveform.abs()
    compressed_magnitude = torch.where(
        magnitude <= threshold,
        magnitude,
        threshold + (magnitude - threshold) / ratio,
    )
    return waveform.sign() * compressed_magnitude


def convolve_impulse_response(waveform: torch.Tensor, impulse_response: torch.Tensor) -> torch.Tensor:
    """Apply causal per-channel FIR convolution and preserve input length."""

    validate_waveform(waveform)
    validate_waveform(impulse_response)
    channels = waveform.shape[0]
    if impulse_response.shape[0] == 1 and channels > 1:
        impulse_response = impulse_response.expand(channels, -1).clone()
    elif impulse_response.shape[0] != channels:
        raise ValueError(
            f"cannot apply a {impulse_response.shape[0]}-channel impulse response "
            f"to a {channels}-channel waveform"
        )

    # L1 normalization prevents arbitrary IR recording gain from dominating.
    normalization = impulse_response.abs().sum(dim=-1, keepdim=True).clamp_min(1e-12)
    kernels = (impulse_response / normalization).flip(-1).unsqueeze(1)
    padded = torch_functional.pad(waveform.unsqueeze(0), (impulse_response.shape[-1] - 1, 0))
    convolved = torch_functional.conv1d(padded, kernels, groups=channels)
    return convolved.squeeze(0)[..., : waveform.shape[-1]]


def _uniform(generator: torch.Generator, minimum: float, maximum: float) -> float:
    if minimum == maximum:
        return float(minimum)
    draw = float(torch.rand((), generator=generator).item())
    return float(minimum + draw * (maximum - minimum))


def _choose(generator: torch.Generator, values: Sequence[Any]) -> Any:
    index = int(torch.randint(len(values), (1,), generator=generator).item())
    return values[index]


def _enabled(enabled: bool, probability: float, generator: torch.Generator) -> bool:
    return bool(enabled and float(torch.rand((), generator=generator).item()) < probability)


class AugmentationPipeline:
    """Compose independently configured augmentations using one seeded RNG."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    def __call__(
        self,
        target: torch.Tensor,
        background: torch.Tensor,
        generator: torch.Generator,
        *,
        impulse_responses: Sequence[tuple[str, Path]] = (),
    ) -> AugmentationResult:
        validate_waveform(target)
        validate_waveform(background)
        sample_rate = self.config.audio.sample_rate
        output = target.clone()
        parameters: dict[str, Any] = {}

        apply_augmentation = _enabled(
            self.config.gain.enabled,
            self.config.gain.probability,
            generator,
        )
        gain_db = (
            _uniform(generator, self.config.gain.min_db, self.config.gain.max_db)
            if apply_augmentation
            else 0.0
        )
        output = apply_gain(output, gain_db)
        parameters["gain"] = {"applied": apply_augmentation, "gain_db": gain_db}

        apply_augmentation = _enabled(
            self.config.reverb.enabled,
            self.config.reverb.probability,
            generator,
        )
        if apply_augmentation and impulse_responses:
            ir_source, ir_path = _choose(generator, impulse_responses)
            impulse_response, ir_sample_rate = load_audio(ir_path)
            if ir_sample_rate != sample_rate:
                impulse_response = resample_audio(impulse_response, ir_sample_rate, sample_rate)
            output = convolve_impulse_response(output, impulse_response)
            parameters["reverb"] = {
                "applied": True,
                "impulse_response_source": ir_source,
                "original_sample_rate": ir_sample_rate,
            }
        else:
            parameters["reverb"] = {
                "applied": False,
                "reason": "no_impulse_responses" if apply_augmentation else "not_selected",
            }

        apply_augmentation = _enabled(
            self.config.lowpass.enabled,
            self.config.lowpass.probability,
            generator,
        )
        if apply_augmentation:
            nyquist_margin = sample_rate / 2.0 * 0.999
            maximum = min(self.config.lowpass.max_cutoff_hz, nyquist_margin)
            minimum = min(self.config.lowpass.min_cutoff_hz, maximum)
            cutoff = _uniform(generator, minimum, maximum)
            output = lowpass_filter(output, sample_rate, cutoff, order=self.config.lowpass.order)
            parameters["lowpass"] = {"applied": True, "cutoff_hz": cutoff}
        else:
            parameters["lowpass"] = {"applied": False}

        apply_augmentation = _enabled(
            self.config.highpass.enabled,
            self.config.highpass.probability,
            generator,
        )
        if apply_augmentation:
            nyquist_margin = sample_rate / 2.0 * 0.999
            maximum = min(self.config.highpass.max_cutoff_hz, nyquist_margin)
            minimum = min(self.config.highpass.min_cutoff_hz, maximum)
            cutoff = _uniform(generator, minimum, maximum)
            output = highpass_filter(output, sample_rate, cutoff, order=self.config.highpass.order)
            parameters["highpass"] = {"applied": True, "cutoff_hz": cutoff}
        else:
            parameters["highpass"] = {"applied": False}

        apply_augmentation = _enabled(
            self.config.bandpass.enabled,
            self.config.bandpass.probability,
            generator,
        )
        if apply_augmentation:
            nyquist_margin = sample_rate / 2.0 * 0.999
            low_maximum = min(self.config.bandpass.max_low_hz, nyquist_margin)
            low_minimum = min(self.config.bandpass.min_low_hz, low_maximum)
            high_maximum = min(self.config.bandpass.max_high_hz, nyquist_margin)
            high_minimum = min(self.config.bandpass.min_high_hz, high_maximum)
            low_cutoff = _uniform(generator, low_minimum, low_maximum)
            if high_minimum <= low_cutoff:
                raise ValueError("configured band-pass ranges are invalid for the output sample rate")
            high_cutoff = _uniform(generator, high_minimum, high_maximum)
            output = bandpass_filter(
                output,
                sample_rate,
                low_cutoff,
                high_cutoff,
                order=self.config.bandpass.order,
            )
            parameters["bandpass"] = {
                "applied": True,
                "low_hz": low_cutoff,
                "high_hz": high_cutoff,
            }
        else:
            parameters["bandpass"] = {"applied": False}

        apply_augmentation = _enabled(
            self.config.random_eq.enabled,
            self.config.random_eq.probability,
            generator,
        )
        if apply_augmentation:
            gains = [
                _uniform(
                    generator,
                    self.config.random_eq.min_gain_db,
                    self.config.random_eq.max_gain_db,
                )
                for _ in self.config.random_eq.anchor_frequencies_hz
            ]
            output = apply_frequency_response(
                output,
                sample_rate,
                self.config.random_eq.anchor_frequencies_hz,
                gains,
            )
            parameters["random_eq"] = {
                "applied": True,
                "anchor_frequencies_hz": list(self.config.random_eq.anchor_frequencies_hz),
                "anchor_gains_db": gains,
            }
        else:
            parameters["random_eq"] = {"applied": False}

        apply_augmentation = _enabled(
            self.config.noise.enabled,
            self.config.noise.probability,
            generator,
        )
        if apply_augmentation:
            snr_db = float(_choose(generator, self.config.noise.snr_db_choices))
            mix = mix_at_snr(output, background, snr_db)
            output = mix.mixture
            parameters["noise"] = {
                "applied": True,
                "snr_db": snr_db,
                "measured_snr_db_at_mix": mix.measured_snr_db,
            }
        else:
            parameters["noise"] = {"applied": False, "snr_db": None}

        apply_augmentation = _enabled(
            self.config.microphone_response.enabled,
            self.config.microphone_response.probability,
            generator,
        )
        if apply_augmentation:
            gains = [
                _uniform(
                    generator,
                    self.config.microphone_response.min_gain_db,
                    self.config.microphone_response.max_gain_db,
                )
                for _ in self.config.microphone_response.anchor_frequencies_hz
            ]
            output = apply_frequency_response(
                output,
                sample_rate,
                self.config.microphone_response.anchor_frequencies_hz,
                gains,
            )
            parameters["microphone_response"] = {
                "applied": True,
                "anchor_frequencies_hz": list(
                    self.config.microphone_response.anchor_frequencies_hz
                ),
                "anchor_gains_db": gains,
            }
        else:
            parameters["microphone_response"] = {"applied": False}

        valid_intermediate_rates = [
            rate for rate in self.config.resampling.intermediate_rates if rate < sample_rate
        ]
        apply_augmentation = _enabled(
            self.config.resampling.enabled,
            self.config.resampling.probability,
            generator,
        )
        if apply_augmentation and valid_intermediate_rates:
            intermediate_rate = int(_choose(generator, valid_intermediate_rates))
            output = resampling_degradation(output, sample_rate, intermediate_rate)
            parameters["resampling"] = {
                "applied": True,
                "intermediate_sample_rate": intermediate_rate,
                "output_sample_rate": sample_rate,
            }
        else:
            parameters["resampling"] = {
                "applied": False,
                "reason": "no_rate_below_output" if apply_augmentation else "not_selected",
            }

        apply_augmentation = _enabled(
            self.config.compression.enabled,
            self.config.compression.probability,
            generator,
        )
        if apply_augmentation:
            threshold = _uniform(
                generator,
                self.config.compression.min_threshold,
                self.config.compression.max_threshold,
            )
            ratio = _uniform(
                generator,
                self.config.compression.min_ratio,
                self.config.compression.max_ratio,
            )
            output = dynamic_range_compress(output, threshold, ratio)
            parameters["compression"] = {
                "applied": True,
                "threshold": threshold,
                "ratio": ratio,
            }
        else:
            parameters["compression"] = {"applied": False}

        apply_augmentation = _enabled(
            self.config.clipping.enabled,
            self.config.clipping.probability,
            generator,
        )
        if apply_augmentation:
            threshold = _uniform(
                generator,
                self.config.clipping.min_threshold,
                self.config.clipping.max_threshold,
            )
            output = hard_clip(output, threshold)
            parameters["clipping"] = {"applied": True, "threshold": threshold}
        else:
            parameters["clipping"] = {"applied": False}

        if output.shape != target.shape:
            raise RuntimeError(
                f"augmentation pipeline changed shape from {tuple(target.shape)} "
                f"to {tuple(output.shape)}"
            )
        if not torch.isfinite(output).all():
            raise RuntimeError("augmentation pipeline produced NaN or Inf")
        return AugmentationResult(output.to(torch.float32), parameters)


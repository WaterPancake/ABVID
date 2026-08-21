"""Controlled signal/noise mixing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from vehicle_audio.audio import validate_waveform


@dataclass(frozen=True)
class MixResult:
    mixture: torch.Tensor
    scaled_noise: torch.Tensor
    requested_snr_db: float
    measured_snr_db: float


def signal_power(waveform: torch.Tensor) -> torch.Tensor:
    """Return mean-square power across all channels and samples."""

    validate_waveform(waveform)
    return waveform.to(torch.float64).square().mean()


def measure_snr_db(signal: torch.Tensor, noise: torch.Tensor) -> float:
    """Measure SNR as ``10 log10(P_signal / P_noise)``."""

    if signal.shape != noise.shape:
        raise ValueError("signal and noise must have identical shapes")
    signal_value = float(signal_power(signal))
    noise_value = float(signal_power(noise))
    if signal_value <= 0.0 or noise_value <= 0.0:
        raise ValueError("SNR is undefined for a silent signal or silent noise")
    return 10.0 * math.log10(signal_value / noise_value)


def match_noise_channels(noise: torch.Tensor, num_signal_channels: int) -> torch.Tensor:
    """Broadcast mono noise or require an exact multichannel match.

    Mono background represents one observation shared by all target channels.
    Multichannel background is never downmixed or channel-selected.
    """

    validate_waveform(noise)
    if num_signal_channels <= 0:
        raise ValueError("num_signal_channels must be positive")
    if noise.shape[0] == num_signal_channels:
        return noise
    if noise.shape[0] == 1:
        return noise.expand(num_signal_channels, -1).clone()
    raise ValueError(
        f"cannot mix {noise.shape[0]}-channel noise with a "
        f"{num_signal_channels}-channel signal without an explicit channel mapping"
    )


def mix_at_snr(signal: torch.Tensor, noise: torch.Tensor, snr_db: float) -> MixResult:
    """Scale noise to a requested SNR and add it to the signal.

    If ``P_s`` and ``P_n`` are the current mean-square powers, the required
    noise amplitude scale is ``sqrt(P_s / (P_n * 10 ** (SNR_dB / 10)))``.
    """

    validate_waveform(signal)
    noise = match_noise_channels(noise, signal.shape[0])
    if signal.shape[-1] != noise.shape[-1]:
        raise ValueError("signal and noise must have the same number of samples")

    signal_value = signal_power(signal)
    noise_value = signal_power(noise)
    if float(signal_value) <= 0.0:
        raise ValueError("cannot mix at a controlled SNR when the signal is silent")
    if float(noise_value) <= 0.0:
        raise ValueError("cannot mix at a controlled SNR when the noise is silent")

    target_ratio = 10.0 ** (float(snr_db) / 10.0)
    scale = torch.sqrt(signal_value / (noise_value * target_ratio)).to(signal.dtype)
    scaled_noise = noise * scale
    mixture = signal + scaled_noise
    measured = measure_snr_db(signal, scaled_noise)
    return MixResult(mixture, scaled_noise, float(snr_db), measured)


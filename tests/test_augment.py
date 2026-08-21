from __future__ import annotations

import pytest
import torch

from vehicle_audio.augment import (
    apply_frequency_response,
    bandpass_filter,
    convolve_impulse_response,
    dynamic_range_compress,
    hard_clip,
    highpass_filter,
    lowpass_filter,
    resampling_degradation,
)


@pytest.fixture
def stereo_waveform() -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    return torch.randn((2, 4_000), generator=generator) * 0.2


def test_shape_preserving_augmentations(stereo_waveform: torch.Tensor) -> None:
    sample_rate = 8_000
    outputs = [
        lowpass_filter(stereo_waveform, sample_rate, 2_500.0),
        highpass_filter(stereo_waveform, sample_rate, 100.0),
        bandpass_filter(stereo_waveform, sample_rate, 100.0, 2_500.0),
        apply_frequency_response(
            stereo_waveform,
            sample_rate,
            [0.0, 1_000.0, 4_000.0],
            [-3.0, 2.0, -5.0],
        ),
        resampling_degradation(stereo_waveform, sample_rate, 4_000),
        dynamic_range_compress(stereo_waveform, 0.2, 4.0),
        hard_clip(stereo_waveform, 0.4),
    ]

    for output in outputs:
        assert output.shape == stereo_waveform.shape
        assert torch.isfinite(output).all()


def test_mono_impulse_response_broadcasts_without_mixing_channels() -> None:
    waveform = torch.zeros((2, 20))
    waveform[0, 0] = 1.0
    waveform[1, 0] = 2.0
    impulse_response = torch.tensor([[1.0, 0.5, 0.25]])

    convolved = convolve_impulse_response(waveform, impulse_response)

    assert convolved.shape == waveform.shape
    torch.testing.assert_close(convolved[1], 2.0 * convolved[0])


def test_clipping_remains_finite_and_bounded() -> None:
    waveform = torch.tensor([[-1.0e30, -2.0, 0.0, 2.0, 1.0e30]])
    clipped = hard_clip(waveform, 0.75)

    assert torch.isfinite(clipped).all()
    assert float(clipped.abs().max()) <= 0.75


def test_resampling_degradation_rejects_implicit_output_rate_change(
    stereo_waveform: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="below sample_rate"):
        resampling_degradation(stereo_waveform, 8_000, 12_000)


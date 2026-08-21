from __future__ import annotations

import pytest
import torch

from vehicle_audio.mixing import measure_snr_db, mix_at_snr


@pytest.mark.parametrize("requested_snr", [30.0, 10.0, 0.0, -10.0])
def test_controlled_snr_is_accurate(requested_snr: float) -> None:
    generator = torch.Generator().manual_seed(123)
    signal = torch.randn((2, 16_000), generator=generator) * 0.1
    mono_noise = torch.randn((1, 16_000), generator=generator)

    result = mix_at_snr(signal, mono_noise, requested_snr)

    assert result.mixture.shape == signal.shape
    assert result.measured_snr_db == pytest.approx(requested_snr, abs=1e-5)
    assert measure_snr_db(signal, result.scaled_noise) == pytest.approx(requested_snr, abs=1e-5)


def test_multichannel_noise_mismatch_is_not_silently_downmixed() -> None:
    signal = torch.ones((1, 100))
    noise = torch.ones((2, 100))
    with pytest.raises(ValueError, match="explicit channel mapping"):
        mix_at_snr(signal, noise, 5.0)


def test_silent_noise_is_rejected() -> None:
    with pytest.raises(ValueError, match="noise is silent"):
        mix_at_snr(torch.ones((1, 100)), torch.zeros((1, 100)), 5.0)


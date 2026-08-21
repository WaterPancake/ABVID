from __future__ import annotations

import torch

from vehicle_audio.audio import load_audio, load_audio_crop, save_audio, temporal_crop


def test_audio_round_trip_preserves_channels_samples_and_rate(tmp_path) -> None:
    waveform = torch.stack(
        (
            torch.linspace(-0.5, 0.5, 401),
            torch.linspace(0.25, -0.25, 401),
        )
    )
    path = tmp_path / "stereo.wav"
    save_audio(path, waveform, 12_000)

    restored, sample_rate = load_audio(path)

    assert sample_rate == 12_000
    assert restored.shape == (2, 401)
    torch.testing.assert_close(restored, waveform)


def test_float_wav_serialization_is_byte_reproducible(tmp_path) -> None:
    waveform = torch.randn((2, 401), generator=torch.Generator().manual_seed(5))
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"

    save_audio(first, waveform, 12_000)
    save_audio(second, waveform, 12_000)

    assert first.read_bytes() == second.read_bytes()


def test_temporal_crop_is_seeded_and_has_requested_length() -> None:
    waveform = torch.arange(20, dtype=torch.float32).reshape(1, -1)
    first_generator = torch.Generator().manual_seed(17)
    second_generator = torch.Generator().manual_seed(17)

    first = temporal_crop(waveform, 8, first_generator)
    second = temporal_crop(waveform, 8, second_generator)

    assert first.waveform.shape == (1, 8)
    assert first.start_sample == second.start_sample
    torch.testing.assert_close(first.waveform, second.waveform)


def test_temporal_crop_repeats_short_multichannel_audio() -> None:
    waveform = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = temporal_crop(waveform, 11, torch.Generator().manual_seed(3))

    assert result.waveform.shape == (2, 11)
    assert result.repeated is True


def test_load_audio_crop_stays_inside_annotated_source_region(tmp_path) -> None:
    sample_rate = 1_000
    waveform = torch.arange(sample_rate, dtype=torch.float32).reshape(1, -1)
    path = tmp_path / "ramp.wav"
    save_audio(path, waveform, sample_rate)

    crop = load_audio_crop(
        path,
        output_sample_rate=sample_rate,
        output_num_samples=100,
        generator=torch.Generator().manual_seed(29),
        source_start_seconds=0.2,
        source_end_seconds=0.5,
    )

    assert 200 <= crop.start_frame <= 400
    assert crop.start_frame + crop.waveform.shape[-1] <= 500
    assert crop.waveform[0, 0].item() == crop.start_frame

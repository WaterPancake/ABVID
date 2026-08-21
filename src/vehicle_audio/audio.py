"""Audio I/O and length/sample-rate utilities.

All public functions use tensors shaped ``[num_channels, num_samples]``.  No
function implicitly collapses or selects channels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import struct

import soundfile as sf
import torch
import torch.nn.functional as torch_functional
import torchaudio.functional as audio_functional


@dataclass(frozen=True)
class CropResult:
    waveform: torch.Tensor
    start_sample: int
    repeated: bool


@dataclass(frozen=True)
class LoadedCropResult:
    waveform: torch.Tensor
    original_sample_rate: int
    start_frame: int
    start_sample: int
    repeated: bool


def validate_waveform(waveform: torch.Tensor) -> None:
    if waveform.ndim != 2:
        raise ValueError(
            "waveform must have shape [num_channels, num_samples], "
            f"got {tuple(waveform.shape)}"
        )
    if waveform.shape[0] < 1 or waveform.shape[1] < 1:
        raise ValueError("waveform must contain at least one channel and one sample")
    if not waveform.is_floating_point():
        raise TypeError("waveform must use a floating-point dtype")


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    """Load audio without changing its sample rate or channel count."""

    samples, sample_rate = sf.read(
        Path(path),
        dtype="float32",
        always_2d=True,
    )
    # soundfile returns [samples, channels]; copy gives Torch writable storage.
    waveform = torch.from_numpy(samples.T.copy())
    validate_waveform(waveform)
    return waveform, int(sample_rate)


def save_audio(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Save a floating-point WAV while preserving every channel."""

    validate_waveform(waveform)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not torch.isfinite(waveform).all():
        raise ValueError("refusing to save a waveform containing NaN or Inf")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = waveform.detach().cpu().to(torch.float32).T.contiguous().numpy()
    sf.write(output_path, samples, sample_rate, format="WAV", subtype="FLOAT")
    _zero_wav_peak_timestamp(output_path)


def _zero_wav_peak_timestamp(path: Path) -> None:
    """Remove libsndfile's wall-clock value from a float-WAV ``PEAK`` chunk.

    libsndfile stores the current Unix time in that chunk even when waveform
    samples are identical. Zeroing only this informational field makes paired
    outputs byte-reproducible across separate CLI processes.
    """

    payload = bytearray(path.read_bytes())
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise ValueError(f"expected a RIFF/WAVE file after writing: {path}")
    offset = 12
    while offset + 8 <= len(payload):
        chunk_id = payload[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", payload, offset + 4)[0]
        data_offset = offset + 8
        chunk_end = data_offset + chunk_size
        if chunk_end > len(payload):
            raise ValueError(f"invalid WAV chunk size in {path}")
        if chunk_id == b"PEAK" and chunk_size >= 8:
            payload[data_offset + 4 : data_offset + 8] = b"\x00\x00\x00\x00"
            path.write_bytes(payload)
            return
        offset = chunk_end + (chunk_size % 2)


def resample_audio(
    waveform: torch.Tensor,
    original_sample_rate: int,
    new_sample_rate: int,
) -> torch.Tensor:
    """Explicitly resample audio, preserving its channel dimension."""

    validate_waveform(waveform)
    if original_sample_rate <= 0 or new_sample_rate <= 0:
        raise ValueError("sample rates must be positive")
    if original_sample_rate == new_sample_rate:
        return waveform.clone()
    return audio_functional.resample(waveform, original_sample_rate, new_sample_rate)


def load_audio_crop(
    path: str | Path,
    output_sample_rate: int,
    output_num_samples: int,
    generator: torch.Generator,
    *,
    repeat_if_short: bool = True,
    randomize_start: bool = True,
    source_start_seconds: float | None = None,
    source_end_seconds: float | None = None,
) -> LoadedCropResult:
    """Seek, decode, and resample only the source window needed for one crop.

    Long recordings are not loaded in full. A small margin is decoded around the
    logical crop so the resampling filter has context at both boundaries.
    ``start_frame`` is measured in the original file rate; ``start_sample`` is the
    equivalent offset at the configured output rate.
    """

    if output_sample_rate <= 0 or output_num_samples <= 0:
        raise ValueError("output sample rate and length must be positive")

    input_path = Path(path)
    with sf.SoundFile(input_path) as audio_file:
        original_sample_rate = int(audio_file.samplerate)
        source_frames = int(audio_file.frames)
        if source_frames <= 0:
            raise ValueError(f"audio file contains no frames: {input_path}")

        region_start = (
            0
            if source_start_seconds is None
            else round(float(source_start_seconds) * original_sample_rate)
        )
        region_end = (
            source_frames
            if source_end_seconds is None
            else round(float(source_end_seconds) * original_sample_rate)
        )
        region_end = min(region_end, source_frames)
        if region_start < 0 or region_start >= region_end:
            raise ValueError(
                f"invalid source region [{source_start_seconds}, {source_end_seconds}] "
                f"for {input_path} ({source_frames / original_sample_rate:.3f} seconds)"
            )
        region_frames = region_end - region_start

        logical_source_frames = math.ceil(
            output_num_samples * original_sample_rate / output_sample_rate
        )
        if region_frames >= logical_source_frames:
            max_start = region_frames - logical_source_frames
            relative_start = (
                int(torch.randint(max_start + 1, (1,), generator=generator).item())
                if max_start and randomize_start
                else 0
            )
            start_frame = region_start + relative_start
            # The default torchaudio sinc kernel is compact; 128 input frames is
            # ample context for current supported sample-rate ratios.
            margin_frames = 128 if original_sample_rate != output_sample_rate else 0
            read_start = max(region_start, start_frame - margin_frames)
            read_end = min(
                region_end,
                start_frame + logical_source_frames + margin_frames,
            )
            audio_file.seek(read_start)
            samples = audio_file.read(
                read_end - read_start,
                dtype="float32",
                always_2d=True,
            )
            waveform = torch.from_numpy(samples.T.copy())
            validate_waveform(waveform)
            if original_sample_rate != output_sample_rate:
                waveform = resample_audio(
                    waveform,
                    original_sample_rate,
                    output_sample_rate,
                )
            crop_offset = round(
                (start_frame - read_start) * output_sample_rate / original_sample_rate
            )
            waveform = force_num_samples(
                waveform[..., crop_offset : crop_offset + output_num_samples],
                output_num_samples,
            )
            return LoadedCropResult(
                waveform=waveform,
                original_sample_rate=original_sample_rate,
                start_frame=start_frame,
                start_sample=round(start_frame * output_sample_rate / original_sample_rate),
                repeated=False,
            )

        audio_file.seek(region_start)
        samples = audio_file.read(region_frames, dtype="float32", always_2d=True)

    waveform = torch.from_numpy(samples.T.copy())
    validate_waveform(waveform)
    if original_sample_rate != output_sample_rate:
        waveform = resample_audio(waveform, original_sample_rate, output_sample_rate)
    crop = temporal_crop(
        waveform,
        output_num_samples,
        generator,
        repeat_if_short=repeat_if_short,
        randomize_start=randomize_start,
    )
    start_frame = region_start + round(
        crop.start_sample * original_sample_rate / output_sample_rate
    )
    return LoadedCropResult(
        waveform=crop.waveform,
        original_sample_rate=original_sample_rate,
        start_frame=start_frame,
        start_sample=round(start_frame * output_sample_rate / original_sample_rate),
        repeated=crop.repeated,
    )


def force_num_samples(waveform: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Trim or right-pad by at most a small resampling-rounding difference."""

    validate_waveform(waveform)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    difference = num_samples - waveform.shape[-1]
    if difference > 0:
        return torch_functional.pad(waveform, (0, difference))
    return waveform[..., :num_samples]


def temporal_crop(
    waveform: torch.Tensor,
    num_samples: int,
    generator: torch.Generator,
    *,
    repeat_if_short: bool = True,
    randomize_start: bool = True,
) -> CropResult:
    """Crop to length, optionally randomizing the start and looping short audio."""

    validate_waveform(waveform)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    available = waveform.shape[-1]
    if available >= num_samples:
        max_start = available - num_samples
        start = (
            int(torch.randint(max_start + 1, (1,), generator=generator).item())
            if max_start and randomize_start
            else 0
        )
        return CropResult(waveform[..., start : start + num_samples].clone(), start, False)

    if not repeat_if_short:
        return CropResult(force_num_samples(waveform, num_samples), 0, False)

    # Pick a phase in the short recording before looping it to the requested size.
    start = (
        int(torch.randint(available, (1,), generator=generator).item())
        if available > 1 and randomize_start
        else 0
    )
    repetitions = math.ceil((num_samples + start) / available)
    looped = waveform.repeat(1, repetitions)
    return CropResult(looped[..., start : start + num_samples].clone(), start, True)

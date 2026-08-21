#!/usr/bin/env python3
"""Render clean/corrupted waveforms and log-Mel spectrograms to PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchaudio.transforms as transforms

from vehicle_audio.audio import load_audio


def _log_mel(waveform: torch.Tensor, sample_rate: int, channel: int) -> torch.Tensor:
    mel = transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
    )(waveform[channel : channel + 1])
    return transforms.AmplitudeToDB(stype="power")(mel).squeeze(0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("sample_inspection.png"))
    parser.add_argument("--channel", type=int, default=0)
    args = parser.parse_args()

    clean, clean_rate = load_audio(args.sample_dir / "clean.wav")
    corrupted, corrupted_rate = load_audio(args.sample_dir / "corrupted.wav")
    if clean_rate != corrupted_rate:
        raise ValueError("paired files have different sample rates")
    if clean.shape[0] != corrupted.shape[0]:
        raise ValueError("paired files have different channel counts")
    if not 0 <= args.channel < clean.shape[0]:
        raise ValueError(
            f"channel must be in [0, {clean.shape[0] - 1}], got {args.channel}"
        )

    seconds = torch.arange(clean.shape[-1]) / clean_rate
    figure, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    axes[0, 0].plot(seconds.numpy(), clean[args.channel].numpy(), linewidth=0.6)
    axes[0, 0].set(title=f"Clean waveform (channel {args.channel})", xlabel="Time (s)")
    axes[0, 1].plot(seconds.numpy(), corrupted[args.channel].numpy(), linewidth=0.6)
    axes[0, 1].set(title=f"Corrupted waveform (channel {args.channel})", xlabel="Time (s)")
    for axis, waveform, title in (
        (axes[1, 0], clean, f"Clean log-Mel (channel {args.channel})"),
        (axes[1, 1], corrupted, f"Corrupted log-Mel (channel {args.channel})"),
    ):
        image = axis.imshow(
            _log_mel(waveform, clean_rate, args.channel).numpy(),
            origin="lower",
            aspect="auto",
        )
        axis.set(title=title, xlabel="Frame", ylabel="Mel bin")
        figure.colorbar(image, ax=axis, label="dB")
    figure.savefig(args.output, dpi=150)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

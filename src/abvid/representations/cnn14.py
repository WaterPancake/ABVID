"""Frozen PANNs CNN14 (Kong et al., 2020) embeddings from the official checkpoint.

The module tree mirrors the official parameter names so the checkpoint loads with
``strict=True``. The STFT kernels and mel matrix come from the checkpoint itself,
so the log-mel front end is the released one. Input: 32 kHz mono float32.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

RATE, N_FFT, HOP, MELS = 32000, 1024, 320, 64


class _Stft(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_real = nn.Conv1d(1, N_FFT // 2 + 1, N_FFT, stride=HOP, bias=False)
        self.conv_imag = nn.Conv1d(1, N_FFT // 2 + 1, N_FFT, stride=HOP, bias=False)


class _Spectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft = _Stft()

    def forward(self, x):
        x = F.pad(x[:, None, :], (N_FFT // 2, N_FFT // 2), mode="reflect")
        real, imag = self.stft.conv_real(x), self.stft.conv_imag(x)
        return (real ** 2 + imag ** 2).transpose(1, 2)[:, None]  # batch x 1 x time x freq


class _Logmel(nn.Module):
    def __init__(self):
        super().__init__()
        self.melW = nn.Parameter(torch.zeros(N_FFT // 2 + 1, MELS), requires_grad=False)

    def forward(self, power):
        mel = torch.matmul(power, self.melW)
        return 10.0 * torch.log10(torch.clamp(mel, min=1e-10))  # ref = 1.0, top_db = None


class _ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn1, self.bn2 = nn.BatchNorm2d(cout), nn.BatchNorm2d(cout)

    def forward(self, x, pool):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool)


class Cnn14(nn.Module):
    def __init__(self, classes=527):
        super().__init__()
        self.spectrogram_extractor, self.logmel_extractor = _Spectrogram(), _Logmel()
        self.bn0 = nn.BatchNorm2d(MELS)
        widths = [1, 64, 128, 256, 512, 1024, 2048]
        for i in range(6):
            setattr(self, f"conv_block{i + 1}", _ConvBlock(widths[i], widths[i + 1]))
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, classes)

    def forward(self, wave):
        x = self.logmel_extractor(self.spectrogram_extractor(wave))
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for i in range(6):
            x = getattr(self, f"conv_block{i + 1}")(x, (2, 2) if i < 5 else (1, 1))
        x = torch.mean(x, dim=3)
        x = torch.max(x, dim=2)[0] + torch.mean(x, dim=2)
        embedding = F.relu_(self.fc1(x))
        return embedding, torch.sigmoid(self.fc_audioset(embedding))


def load(checkpoint, expected_sha256):
    data = Path(checkpoint).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("CNN14 checkpoint hash mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    model = Cnn14()
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def embed(model, waves):
    """Batch x samples float32 at 32 kHz -> (embeddings, AudioSet clipwise probabilities)."""
    if waves.ndim != 2 or not torch.isfinite(waves).all():
        raise ValueError("Expected finite batch x samples waveforms")
    return model(waves)

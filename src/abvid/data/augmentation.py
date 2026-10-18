"""Conventional waveform augmentation for CAST-5A arm RA (shape-only profile).

Three cascaded RBJ peaking filters, then one recorded background excerpt at a
drawn SNR. Every random draw is returned so the augmentation is replayable.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

RATE = 16000
SETTINGS = {"filters": 3, "f0_hz": (100.0, 3000.0), "q": (0.7, 2.0), "gain_db": (-6.0, 6.0),
            "snr_db": (10.0, 30.0), "floor_rms": 1e-7}


def shape(x, floor=SETTINGS["floor_rms"]):
    """Remove the mean and scale to unit RMS; reject near-silent input."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Expected a finite mono waveform")
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x * x)))
    if rms < floor:
        raise ValueError("Waveform RMS below the declared floor")
    return x / rms


def peaking(f0, q, gain_db, rate=RATE):
    """RBJ audio-EQ-cookbook peaking biquad, normalized so a[0] == 1."""
    if not 0 < f0 < rate / 2 or q <= 0:
        raise ValueError("Invalid peaking filter parameters")
    a_lin = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * f0 / rate
    alpha = math.sin(w0) / (2 * q)
    b = np.array([1 + alpha * a_lin, -2 * math.cos(w0), 1 - alpha * a_lin])
    a = np.array([1 + alpha / a_lin, -2 * math.cos(w0), 1 - alpha / a_lin])
    return b / a[0], a / a[0]


def draw(rng, settings=SETTINGS):
    lo, hi = np.log(settings["f0_hz"])
    filters = [{"f0_hz": float(np.exp(rng.uniform(lo, hi))), "q": float(rng.uniform(*settings["q"])),
                "gain_db": float(rng.uniform(*settings["gain_db"]))} for _ in range(settings["filters"])]
    return {"filters": filters, "snr_db": float(rng.uniform(*settings["snr_db"]))}


def excerpt(background, rng, samples=32000):
    """Uniform-start window of a background observation, unit RMS."""
    background = np.asarray(background, dtype=np.float64)
    if len(background) < samples:
        raise ValueError("Background shorter than the observation")
    start = int(rng.integers(0, len(background) - samples + 1))
    return shape(background[start:start + samples]), start


def augment(x, background_excerpt, params):
    """EQ the shape-normalized observation, add background at the drawn SNR, renormalize."""
    y = shape(x)
    for f in params["filters"]:
        b, a = peaking(f["f0_hz"], f["q"], f["gain_db"])
        y = lfilter(b, a, y)
    y = shape(y)
    noise = shape(background_excerpt)
    return shape(y + 10 ** (-params["snr_db"] / 20) * noise).astype(np.float32)

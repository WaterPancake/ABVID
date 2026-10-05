"""Numeric components extracted from CAST/src/cast/audio.py."""
from math import gcd


import numpy as np


from scipy.signal import resample_poly


import torch


def observation(x, sr, cfg):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all() or not isinstance(sr, int) or sr <= 0:
        raise ValueError("Expected finite two-stored-channel waveform and positive sample rate")
    x = x.mean(axis=1)
    a = cfg["audio"]
    for rate in (a["intermediate_rate_hz"], a["sample_rate_hz"]):
        if sr != rate:
            g = gcd(sr, rate)
            x = resample_poly(x, rate//g, sr//g, window=("kaiser", a["kaiser_beta"]), padtype="constant")
        sr = rate
    n = a["samples"]
    if len(x) < n:
        raise ValueError("Short waveform; padding forbidden")
    start = (len(x)-n)//2
    x = np.asarray(x[start:start+n], dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite observation")
    return x


def normalize(x, cfg):
    dc = float(np.mean(x, dtype=np.float64))
    ac = x.astype(np.float64)-dc
    rms = float(np.sqrt(np.mean(ac**2)))
    if not np.isfinite(rms) or rms < cfg["normalization"]["silence_rms"]:
        raise ValueError("silent_or_near_silent")
    return torch.tensor(ac/rms, dtype=torch.float32), {"dc": dc, "ac_rms": rms,
           "raw_rms": float(np.sqrt(np.mean(x.astype(np.float64)**2))), "peak": float(np.max(np.abs(x)))}


def shape(x, floor=1e-12):
    x = x-x.mean(dim=-1, keepdim=True)
    return x / x.square().mean(dim=-1, keepdim=True).clamp_min(floor**2).sqrt()

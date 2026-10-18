"""Hash-verified, synthetic-only media IO for P0 preparation tools."""
from __future__ import annotations

import numpy as np
import soundfile as sf

from abvid.provenance.hashing import sha256

from .config import external_path, validate


def read_asset(asset, config):
    validate(config)
    if asset["media_kind"] != "synthetic_fixture":
        raise ValueError("Real-media processing is not authorized in P0")
    if not asset["media_ref"].startswith(config["storage"]["fixture_root"] + "/"):
        raise ValueError("Input lies outside the configured synthetic fixture subtree")
    path = external_path(asset["media_ref"], must_exist=True)
    if sha256(path) != asset["raw_sha256"]:
        raise ValueError("Raw SHA256 mismatch")
    info = sf.info(path)
    if (info.samplerate != asset["native_sample_rate_hz"] or info.channels != asset["channels"] or
            abs(info.duration - asset["duration_seconds"]) > 0.5 / info.samplerate):
        raise ValueError("Declared audio metadata does not match the file")
    if info.duration > config["trimming"]["maximum_asset_seconds"]:
        raise ValueError("Audio exceeds bounded P0 duration")
    x, rate = sf.read(path, dtype="float64", always_2d=True)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Empty or nonfinite waveform")
    return x, rate

"""Deterministic offline preprocessing and inner-only grid selection."""

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
from scipy.signal import butter, sosfiltfilt


@dataclass(frozen=True)
class Preprocessing:
    remove_dc: bool = False
    highpass_hz: int = 0
    rms_dbfs: int | None = None

    @property
    def name(self):
        return f"dc{int(self.remove_dc)}_hp{self.highpass_hz}_rms{self.rms_dbfs if self.rms_dbfs is not None else 'off'}"

    def to_dict(self):
        return asdict(self)


def preprocessing_grid(dc_values, highpass_values, rms_values):
    return [Preprocessing(*v) for v in product(dc_values, highpass_values, rms_values)]


def preprocess(waveforms, config, sample_rate=16000):
    """[batch, samples], DC -> offline zero-phase HPF -> bounded scalar RMS gain.

    The second-order Butterworth is applied forward/backward (fourth-order
    effective magnitude response). Reflection padding is within each window;
    no outside audio/noise estimate is consulted. This is not causal streaming.
    """
    x = np.asarray(waveforms, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < 16 or not np.isfinite(x).all():
        raise ValueError("expected finite [batch, samples>=16] waveforms")
    if not 0 <= config.highpass_hz < sample_rate / 2:
        raise ValueError("high-pass frequency must be below Nyquist")
    if config.rms_dbfs is not None and not -100 < config.rms_dbfs < 0:
        raise ValueError("RMS target must be between -100 and 0 dBFS")
    y = x.astype(np.float64)
    if config.remove_dc:
        y -= y.mean(axis=1, keepdims=True)
    if config.highpass_hz:
        sos = butter(
            2, config.highpass_hz, btype="highpass", fs=sample_rate, output="sos"
        )
        y = sosfiltfilt(sos, y, axis=-1)
    gain = np.ones(len(y), dtype=np.float64)
    if config.rms_dbfs is not None:
        rms = np.sqrt(np.mean(y * y, axis=1))
        active = rms > 1e-8
        gain[active] = np.clip(
            10 ** (config.rms_dbfs / 20) / rms[active],
            10 ** (-12 / 20),
            10 ** (12 / 20),
        )
        peak = np.max(np.abs(y), axis=1)
        # Bound the gain to avoid creating clipping; do not clip the waveform.
        gain[active] = np.minimum(gain[active], 0.99 / np.maximum(peak[active], 1e-12))
        y *= gain[:, None]
    y = np.asarray(y, dtype=np.float32)
    if not np.isfinite(y).all():
        raise ValueError("preprocessing produced nonfinite samples")
    return y, gain


def selection_key(config, fold, grid_index):
    """Uses inner validation only; outer metrics cannot influence selection."""
    candidate = next(
        c
        for c in fold["candidate_selection"]
        if c["regularization_c"] == fold["selected_regularization_c"]
    )
    complexity = (
        int(config.remove_dc)
        + int(config.highpass_hz > 0)
        + int(config.rms_dbfs is not None)
    )
    return (
        candidate["mean_balanced_accuracy"],
        candidate["median_balanced_accuracy"],
        -complexity,
        -config.highpass_hz,
        -grid_index,
        -candidate["regularization_c"],
    )

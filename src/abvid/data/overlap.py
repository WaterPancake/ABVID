"""Label-blind cross-corpus waveform overlap audit.

Finds reference clips whose audio reappears anywhere inside query recordings,
tolerating time shifts, gain, polarity and resampling. Peak-pair (landmark)
hashes on an 8 kHz log spectrogram propose candidates; each candidate is
confirmed by normalized waveform correlation. Inputs are identifiers and audio
loaders only; outputs are match identities and scores, never labels or model
outputs. Codec or time-stretch variants beyond these tolerances may be missed.
"""
from __future__ import annotations

from math import gcd

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import resample_poly

RATE = 8000
FFT, HOP = 512, 128
SETTINGS = {"rate": RATE, "fft": FFT, "hop": HOP, "neighbourhood": (15, 15), "frame_threshold_db": 10, "peaks_per_frame": 2,
            "fan_out": 5, "max_dt": 63, "max_df": 63, "min_votes": 12, "min_vote_fraction": 0.1,
            "min_correlation": 0.9, "lag_search_samples": 2 * HOP}


def mono8k(audio, rate):
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Expected finite mono or multichannel audio")
    if rate != RATE:
        g = gcd(int(rate), RATE)
        x = resample_poly(x, RATE // g, int(rate) // g)
    return x - x.mean()


def spectrogram(x):
    if len(x) < FFT:
        return np.zeros((0, FFT // 2 + 1))
    frames = np.lib.stride_tricks.sliding_window_view(x, FFT)[::HOP] * np.hanning(FFT)
    return 20 * np.log10(np.abs(np.fft.rfft(frames, axis=1)) + 1e-9)


def landmarks(x, settings=SETTINGS):
    """Return (hash, anchor_frame) arrays; invariant to gain and polarity."""
    s = spectrogram(x)
    if not len(s):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    # Thresholds and budgets are per frame, so loud passages cannot starve quiet ones
    # and a clip yields the same peaks wherever it sits inside a longer recording.
    floor = np.median(s, axis=1, keepdims=True) + settings["frame_threshold_db"]
    is_peak = (s == maximum_filter(s, size=settings["neighbourhood"])) & (s > floor)
    t, f = np.nonzero(is_peak)
    order = np.lexsort((-s[t, f], t))
    t, f = t[order], f[order]
    rank = np.arange(len(t)) - np.searchsorted(t, t, "left")
    t, f = t[rank < settings["peaks_per_frame"]], f[rank < settings["peaks_per_frame"]]
    order = np.lexsort((f, t))
    t, f = t[order], f[order]
    hashes, anchors = [], []
    for i in range(len(t)):
        paired = 0
        for j in range(i + 1, len(t)):
            dt = t[j] - t[i]
            if dt > settings["max_dt"]:
                break
            df = int(f[j]) - int(f[i])
            if dt == 0 or abs(df) > settings["max_df"]:
                continue
            hashes.append((int(f[i]) << 14) | ((df + 64) << 7) | int(dt))
            anchors.append(t[i])
            paired += 1
            if paired == settings["fan_out"]:
                break
    return np.asarray(hashes, np.int64), np.asarray(anchors, np.int64)


class Index:
    """Sorted hash index over reference clips."""

    def __init__(self, settings=SETTINGS):
        self.settings, self.ids, self.audio, self.parts = settings, [], [], []

    def add(self, clip_id, audio, rate):
        x = mono8k(audio, rate)
        h, t = landmarks(x, self.settings)
        index = len(self.ids)
        self.ids.append(clip_id)
        self.audio.append(x)
        self.parts.append((h, np.full(len(h), index, np.int64), t))

    def freeze(self):
        h, ref, t = (np.concatenate(p) if self.parts else np.zeros(0, np.int64) for p in zip(*self.parts))
        order = np.argsort(h, kind="stable")
        self.hash, self.ref, self.time = h[order], ref[order], t[order]
        self.counts = np.bincount(self.ref, minlength=len(self.ids))
        return self


def correlation(reference, query, start, lag):
    """Best absolute normalized correlation of reference against query near start."""
    best = 0.0
    n = len(reference)
    for shift in range(-lag, lag + 1):
        a = start + shift
        if a < 0 or a + n > len(query):
            continue
        q = query[a:a + n]
        denominator = np.sqrt(np.dot(reference, reference) * np.dot(q, q))
        if denominator > 0:
            best = max(best, abs(float(np.dot(reference, q) / denominator)))
    return best


def match(index, query_id, audio, rate):
    """Return candidate rows for one query; no label or score fields exist."""
    st = index.settings
    x = mono8k(audio, rate)
    h, t = landmarks(x, st)
    lo, hi = np.searchsorted(index.hash, h, "left"), np.searchsorted(index.hash, h, "right")
    hits = [(index.ref[a:b], index.time[a:b] - tq) for a, b, tq in zip(lo, hi, t) if b > a]
    if not hits:
        return []
    ref = np.concatenate([r for r, _ in hits])
    offset = np.concatenate([o for _, o in hits])
    keys, votes = np.unique(np.stack([ref, offset], 1), axis=0, return_counts=True)
    rows = []
    for (r, o), v in zip(keys, votes):
        fraction = v / max(index.counts[r], 1)
        if v < st["min_votes"] or fraction < st["min_vote_fraction"]:
            continue
        start = int(-o) * HOP
        score = correlation(index.audio[r], x, start, st["lag_search_samples"])
        rows.append({"query_id": query_id, "reference_id": index.ids[r], "offset_s": start / RATE,
                     "votes": int(v), "vote_fraction": float(fraction), "correlation": score,
                     "accepted": score >= st["min_correlation"]})
    best = {}
    for row in rows:
        key = row["reference_id"]
        if key not in best or (row["correlation"], row["votes"]) > (best[key]["correlation"], best[key]["votes"]):
            best[key] = row
    return sorted(best.values(), key=lambda r: (-r["correlation"], r["reference_id"]))

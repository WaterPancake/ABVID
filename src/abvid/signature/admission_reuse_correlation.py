"""Accelerate local lag search while returning original direct-dot scores.

FFT scores nominate possible winning lags. Ambiguous/ill-conditioned cases use
the original full search; nominated lags are scored with its exact arithmetic.
"""
import numpy as np
from scipy.signal import correlate

from abvid.data.overlap import correlation as original


def correlation(reference, query, start, lag):
    n = len(reference)
    first = max(0, start - lag)
    last = min(len(query) - n, start + lag)
    if not n or first > last:
        return 0.0
    ref = np.asarray(reference, dtype=np.float64)
    window = np.asarray(query[first:last + n], dtype=np.float64)
    reference_energy = np.dot(ref, ref)
    if not reference_energy > 0:
        return original(reference, query, start, lag)
    cumulative = np.concatenate(([0.0], np.cumsum(window * window, dtype=np.float64)))
    energy = cumulative[n:] - cumulative[:-n]
    # A conservative cancellation guard sends quiet windows next to strong
    # transients back through the original arithmetic.
    error_bound = np.finfo(np.float64).eps * len(window) * cumulative[-1]
    if (not np.isfinite(cumulative[-1]) or
            np.any(energy <= max(error_bound * 1e8, np.finfo(np.float64).tiny))):
        return original(reference, query, start, lag)
    numerator = correlate(window, ref, mode="valid", method="fft")
    score_estimates = np.abs(numerator) / np.sqrt(reference_energy * energy)
    if not np.isfinite(score_estimates).all():
        return original(reference, query, start, lag)
    candidates = np.flatnonzero(score_estimates >= score_estimates.max() - 1e-7)
    best = 0.0
    for offset in candidates:
        a = first + int(offset)
        q = query[a:a + n]
        denominator = np.sqrt(np.dot(reference, reference) * np.dot(q, q))
        if denominator > 0:
            best = max(best, abs(float(np.dot(reference, q) / denominator)))
    return best

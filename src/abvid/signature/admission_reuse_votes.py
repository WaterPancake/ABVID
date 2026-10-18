"""Exact packed-integer voting for the unchanged landmark reuse criterion."""
import numpy as np

from abvid.data.overlap import HOP, RATE, landmarks, mono8k
from .admission_reuse_correlation import correlation


def votes(ref, reference_time, lo, hi, query_time):
    if not len(reference_time) or not len(query_time):
        return np.zeros((0, 2), np.int64), np.zeros(0, np.int64)
    minimum = int(reference_time.min()) - int(query_time.max())
    maximum = int(reference_time.max()) - int(query_time.min())
    span = maximum - minimum + 1
    largest = int(ref.max(initial=0)) * span + span - 1
    dtype = np.uint32 if largest <= np.iinfo(np.uint32).max else np.uint64
    if largest > np.iinfo(np.int64).max:
        raise ValueError("Packed vote key would exceed exact integer range")
    parts = [
        (ref[a:b] * span + reference_time[a:b] - int(t) - minimum).astype(dtype)
        for a, b, t in zip(lo, hi, query_time) if b > a
    ]
    if not parts:
        return np.zeros((0, 2), np.int64), np.zeros(0, np.int64)
    packed, counts = np.unique(np.concatenate(parts), return_counts=True)
    packed = packed.astype(np.int64)
    keys = np.column_stack((packed // span, packed % span + minimum))
    return keys, counts


def match(index, query_id, audio, rate):
    settings = index.settings
    x = mono8k(audio, rate)
    hashes, times = landmarks(x, settings)
    lo = np.searchsorted(index.hash, hashes, "left")
    hi = np.searchsorted(index.hash, hashes, "right")
    keys, counts = votes(index.ref, index.time, lo, hi, times)
    if not len(keys):
        return []
    # This is the same integer/float criterion as the original per-key loop.
    fractions = counts / np.maximum(index.counts[keys[:, 0]], 1)
    selected = (counts >= settings["min_votes"]) & (fractions >= settings["min_vote_fraction"])
    rows = []
    for (r, offset), count, fraction in zip(keys[selected], counts[selected], fractions[selected]):
        start = int(-offset) * HOP
        score = correlation(index.audio[r], x, start, settings["lag_search_samples"])
        rows.append({
            "query_id": query_id, "reference_id": index.ids[r], "offset_s": start / RATE,
            "votes": int(count), "vote_fraction": float(fraction), "correlation": score,
            "accepted": score >= settings["min_correlation"],
        })
    best = {}
    for row in rows:
        key = row["reference_id"]
        if key not in best or (row["correlation"], row["votes"]) > (best[key]["correlation"], best[key]["votes"]):
            best[key] = row
    return sorted(best.values(), key=lambda r: (-r["correlation"], r["reference_id"]))

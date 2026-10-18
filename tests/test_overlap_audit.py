import numpy as np

from abvid.data.overlap import RATE, Index, match


def clip(seed, seconds=2.0, rate=RATE):
    """Vehicle-like test signal: harmonics with drifting pitch plus coloured noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    f0 = rng.uniform(40, 90) * (1 + 0.2 * np.sin(2 * np.pi * rng.uniform(.2, .6) * t))
    phase = 2 * np.pi * np.cumsum(f0) / rate
    tonal = sum(rng.uniform(.2, 1) * np.sin(k * phase + rng.uniform(0, 6.3)) for k in range(1, 25))
    noise = np.convolve(rng.standard_normal(len(t)), np.hanning(9), "same")
    return tonal + 0.3 * noise


def index_of(*clips):
    index = Index()
    for name, x in clips:
        index.add(name, x, RATE)
    return index.freeze()


def test_embedded_reference_is_found_with_offset_gain_and_polarity():
    reference = clip(1)
    index = index_of(("ref_a", reference), ("ref_b", clip(2)))
    rng = np.random.default_rng(9)
    query = 0.05 * rng.standard_normal(20 * RATE)
    start = int(7.3 * RATE)
    query[start:start + len(reference)] += -0.4 * reference
    rows = match(index, "query", np.stack([query, query], 1), RATE)
    accepted = [r for r in rows if r["accepted"]]
    assert [r["reference_id"] for r in accepted] == ["ref_a"]
    assert abs(accepted[0]["offset_s"] - 7.3) < 0.05 and accepted[0]["correlation"] > 0.95


def test_resampled_query_is_matched():
    reference = clip(3)
    index = index_of(("ref", reference))
    up = np.interp(np.arange(len(reference) * 6) / 6, np.arange(len(reference)), reference)
    query = np.concatenate([np.zeros(48000 * 3), up, np.zeros(48000 * 2)])
    rows = [r for r in match(index, "q48k", query, 48000) if r["accepted"]]
    assert rows and rows[0]["reference_id"] == "ref" and abs(rows[0]["offset_s"] - 3.0) < 0.05


def test_unrelated_audio_is_not_accepted_and_rows_carry_no_labels():
    index = index_of(("ref_a", clip(4)), ("ref_b", clip(5)))
    query = np.concatenate([clip(6, 10.0), clip(7, 10.0)])
    rows = match(index, "query", query, RATE)
    assert not any(r["accepted"] for r in rows)
    allowed = {"query_id", "reference_id", "offset_s", "votes", "vote_fraction", "correlation", "accepted"}
    assert all(set(r) == allowed for r in rows)


def test_silent_or_short_query_returns_no_rows():
    index = index_of(("ref", clip(8)))
    assert match(index, "silence", np.zeros(RATE), RATE) == []
    assert match(index, "short", np.ones(100), RATE) == []


def busy(seed, seconds):
    """Dense broadband traffic-like background with many strong spectral peaks."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    clicks = np.zeros(n)
    clicks[rng.integers(n, size=400)] = rng.standard_normal(400) * 20
    return (sum(clip(seed * 10 + k, seconds) for k in range(4)) + 0.5 * rng.standard_normal(n)
            + np.convolve(clicks, np.hanning(5), "same"))


def test_quiet_insert_inside_busy_recording_is_found():
    """Regression: a global peak budget let loud passages starve a quiet reused clip.

    The earlier global top-N selection finds nothing here; per-frame selection does.
    """
    reference = clip(10)
    index = index_of(("ref", reference), ("other", clip(11)))
    query = busy(3, 20.0)
    start = int(8.0 * RATE)
    query[start:start + len(reference)] = 0.25 * reference
    rows = [r for r in match(index, "query", query, RATE) if r["accepted"]]
    assert [r["reference_id"] for r in rows] == ["ref"] and abs(rows[0]["offset_s"] - 8.0) < 0.05

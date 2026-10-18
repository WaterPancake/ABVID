"""CAST-5A learning stage: observations, RA banks, features, heads and scores.

Baseline environment (frozen BEATs, MFCC, scikit-learn). Every head is fitted on
its own rows only and saved before any held prediction is computed. Scores use
the H1 fixed argmax rule plus threshold-free metrics.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from abvid.cast.grouped_design import hash_seed
from abvid.data.audio import observation
from abvid.data.augmentation import augment, draw, excerpt, shape

ARMS = ("R", "RR", "RA", "RU", "RC", "U", "C")
RA_ID = "cast5a_ra_v1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def real_observation(path, expected_sha256, preprocessing):
    """Hash-checked H1 route, then the shape-only profile (float32, 32,000)."""
    data = Path(path).read_bytes()
    if sha(data) != expected_sha256:
        raise ValueError(f"Source bytes changed: {path}")
    x, sr = sf.read(BytesIO(data), dtype="float64", always_2d=True)
    return shape(observation(x, sr, preprocessing)).astype(np.float32)


def background_pool(audio_dir, sources):
    """(site, date) -> group from admitted rows; map SE CH34 background files to groups."""
    groups = {}
    for row in sources:
        key = (row["site_id"], row["calendar_date"])
        if groups.setdefault(key, row["provenance_group_id"]) != row["provenance_group_id"]:
            raise ValueError("Site/date maps to more than one group")
    pool = defaultdict(list)
    for path in sorted(Path(audio_dir).glob("*_SE_CH34-BG.wav")):
        stamp, site = path.name.split("_")[:2]
        group = groups.get((site, stamp[:10]))
        if group is not None:
            pool[group].append(path)
    return dict(pool)


class Backgrounds:
    """Lazy background observations with recorded file hashes."""

    def __init__(self, preprocessing):
        self.preprocessing, self.cache = preprocessing, {}

    def get(self, path):
        if path not in self.cache:
            data = Path(path).read_bytes()
            x, sr = sf.read(BytesIO(data), dtype="float64", always_2d=True)
            self.cache[path] = (observation(x, sr, self.preprocessing).astype(np.float64), sha(data))
        return self.cache[path]


def ra_copy(partition_id, file_id, x, training_groups, pool, backgrounds):
    """One replayable RA copy: draw EQ/SNR, then the background, then its window."""
    rng = np.random.Generator(np.random.PCG64(hash_seed([RA_ID, partition_id, file_id])))
    params = draw(rng)
    candidates = sorted((p for g in sorted(training_groups) for p in pool.get(g, [])), key=lambda p: p.name)
    if not candidates:
        raise ValueError("No background from the role's training groups")
    path = candidates[int(rng.integers(len(candidates)))]
    wave, digest = backgrounds.get(path)
    window, start = excerpt(wave, rng)
    record = {"partition": partition_id, "parent": file_id, "background": path.name,
              "background_sha256": digest, "background_start": start, **params}
    return augment(x, window, params), record


def training_rows(arm, real, ra, ru, rc):
    """Stack (features, labels) for one arm; each input is (X, y) for this role."""
    if arm == "R":
        parts = [real]
    elif arm == "RR":
        parts = [real, real]
    elif arm == "RA":
        parts = [real, ra]
    elif arm == "RU":
        parts = [real, ru]
    elif arm == "RC":
        parts = [real, rc]
    elif arm == "U":
        parts = [ru]
    elif arm == "C":
        parts = [rc]
    else:
        raise ValueError(f"Unknown arm {arm}")
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    expected = {"R": 190, "RR": 380, "RA": 380, "RU": 380, "RC": 380, "U": 190, "C": 190}[arm]
    if np.bincount(y, minlength=2).tolist() != [expected, expected]:
        raise ValueError(f"{arm}: class budget differs from the protocol")
    return x, y


def head_metrics(y, p_truck):
    """Fixed-rule and threshold-free metrics for one head on its held fold."""
    from abvid.evaluation.ranking import level_counts, weighted_metrics
    values = weighted_metrics(level_counts(p_truck, y, np.zeros(len(y), int), 1), np.ones(1))
    pred = (p_truck > 0.5).astype(int)
    cm = np.bincount(y * 2 + pred, minlength=4).reshape(2, 2)
    out = {k: float(v[0]) for k, v in values.items()}
    out["confusion"] = cm.tolist()
    out["truck_precision"] = float(cm[1, 1] / cm[:, 1].sum()) if cm[:, 1].sum() else 0.0
    return out


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)).encode()

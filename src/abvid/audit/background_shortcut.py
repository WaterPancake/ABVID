"""Background-shortcut audit, data side: feature pin, clip roles, site folds, separation.

Roles reuse the admission rules unchanged. Folds hold out whole provider sites, which is
coarser than any candidate session, so no session is inferred.
"""
import json
from collections import Counter

import numpy as np

from abvid.data.admission import parse_idmt
from abvid.provenance.hashing import sha256

CLASSES = ("car", "truck")
SEPARATION_KEYS = ("file_id", "paired_event_id", "candidate_group_id", "site_id")
CONFIG_SCHEMA = "background_shortcut_v0"
_CONFIG_KEYS = {"schema", "status", "features", "selection", "folds", "budget_per_class",
                "seeds", "classifier"}


def load_config(path):
    with open(path) as f:
        config = json.load(f)
    unknown, missing = set(config) - _CONFIG_KEYS, _CONFIG_KEYS - set(config)
    if unknown or missing:
        raise ValueError(f"config keys: unknown {sorted(unknown)}, missing {sorted(missing)}")
    if config["schema"] != CONFIG_SCHEMA:
        raise ValueError(f"unsupported config schema {config['schema']!r}")
    if config["folds"] != "leave_one_site_out":
        raise ValueError("only leave_one_site_out folds are implemented")
    return config


def load_features(path, expected_sha256):
    """-> (ids, X). The file must hash to the pin recorded when the features were made."""
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"features sha256 {actual} does not match pin {expected_sha256}")
    with np.load(path, allow_pickle=False) as z:
        ids, X = [str(i) for i in z["ids"]], np.asarray(z["X"])
    if X.ndim != 2 or len(ids) != X.shape[0]:
        raise ValueError("feature rows and ids disagree")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate clip ids")
    if not np.isfinite(X).all():
        raise ValueError("non-finite features")
    return ids, X


def role(row):
    """'vehicle', 'background' or 'excluded' for one parsed admission row."""
    reasons = row["metadata_exclusion_reasons"]
    if row["provider_class"] in CLASSES and not reasons:
        return "vehicle"
    # Admission marks all background out of scope for car/truck; keep the clips recorded on
    # the vehicle clips' sensor and channel pair so the microphone is held fixed.
    if (row["provider_class"] == "background" and reasons == ["class_out_of_scope_v1.1"]
            and row["device_id"] == "SE" and row["provider_channel_pair"] == "CH34"):
        return "background"
    return "excluded"


def clip_table(ids):
    """One row per feature row, in feature order (row["index"] == position)."""
    rows = []
    for index, name in enumerate(ids):
        parsed = parse_idmt(name)
        if not parsed["metadata_parse_ok"]:
            raise ValueError(f"unparsed IDMT clip name: {name}")
        rows.append(dict(
            index=index, name=name, role=role(parsed), site_id=parsed["site_id"],
            provider_class=parsed["provider_class"], class_id=parsed["class_id"],
            **{key: parsed[key] for key in SEPARATION_KEYS if key != "site_id"}))
    return rows


def site_summary(rows):
    out = {}
    for site in sorted({r["site_id"] for r in rows if r["role"] != "excluded"}):
        n = Counter(r["provider_class"] for r in rows
                    if r["site_id"] == site and r["role"] != "excluded")
        vehicles = n["car"] + n["truck"]
        out[site] = dict(car=n["car"], truck=n["truck"], background=n["background"],
                         truck_share=n["truck"] / vehicles if vehicles else None)
    return out


def assert_separated(rows, train, test, keys=SEPARATION_KEYS):
    """Raise when a provenance key is shared across roles. An unknown value raises too:
    separation that cannot be shown is not assumed."""
    for key in keys:
        a, b = ({rows[i][key] for i in side} for side in (train, test))
        if None in a or None in b:
            raise ValueError(f"{key} is unknown on some rows; separation cannot be shown")
        if a & b:
            raise ValueError(f"{key} leakage: {sorted(a & b)[:3]}")


def leave_one_site_out(rows, *, budget, seed):
    """One fold per site with car, truck and background clips. Training draws `budget` clips
    per class from the other sites, so every head starts from the same balanced prior and a
    site difference in background P(truck) cannot come from a different training prior."""
    rng = np.random.default_rng(seed)
    vehicle = [r for r in rows if r["role"] == "vehicle"]
    eligible = sorted(site for site, n in site_summary(rows).items()
                      if n["car"] and n["truck"] and n["background"])
    folds = []
    for held in eligible:
        train = []
        for cls in CLASSES:
            pool = sorted(r["index"] for r in vehicle
                          if r["site_id"] != held and r["provider_class"] == cls)
            if len(pool) < budget:
                raise ValueError(f"holding out {held}: {len(pool)} {cls} clips for budget {budget}")
            train += rng.choice(pool, size=budget, replace=False).tolist()
        test_vehicle = [r["index"] for r in vehicle if r["site_id"] == held]
        test_background = [r["index"] for r in rows
                           if r["role"] == "background" and r["site_id"] == held]
        assert_separated(rows, train, test_vehicle + test_background)
        folds.append(dict(held_out_site=held, seed=seed, train=sorted(train),
                          test_vehicle=test_vehicle, test_background=test_background))
    return folds

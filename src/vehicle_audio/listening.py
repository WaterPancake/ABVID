"""Prediction-independent listening selection and strict response scoring."""

from collections import defaultdict
import random

import numpy as np


def select_anchors(records, sources, seed=42):
    rng = random.Random(seed)
    candidates = defaultdict(list)
    for r in sorted(records, key=lambda r: r["sample_id"]):
        center = (r["window_start_seconds"] + r["window_end_seconds"]) / 2
        if any(
            s["start_seconds"] <= center - 3
            and center + 3 <= s["end_seconds"]
            and s["operating_condition"] == r["operating_condition"]
            for s in sources[r["source_id"]]["condition_segments"]
        ):
            candidates[r["recording_session"]].append(r)
    sessions = sorted({r["recording_session"] for r in records})
    if any(not candidates[s] for s in sessions):
        raise ValueError("a session has no eligible six-second context")
    selected = [rng.choice(candidates[s]) for s in sessions]
    eligible = []
    for first in selected:
        if first["vehicle_class"] != "wheeled":
            continue
        rest = [
            r
            for r in candidates[first["recording_session"]]
            if abs(r["window_start_seconds"] - first["window_start_seconds"]) >= 6
        ]
        if rest:
            eligible.append(rest)
    if len(eligible) < 2:
        raise ValueError("cannot balance classes without overlapping context")
    selected += [rng.choice(group) for group in rng.sample(eligible, 2)]
    counts = {
        c: sum(r["vehicle_class"] == c for r in selected)
        for c in ("tracked", "wheeled")
    }
    if counts != {"tracked": 7, "wheeled": 7}:
        raise ValueError(f"unexpected class counts: {counts}")
    return selected


def score_responses(key, response):
    if response.get("test_id") != key["test_id"]:
        raise ValueError("wrong test ID")
    rows = response.get("responses", [])
    expected = {t["id"]: t for t in key["trials"]}
    if len(rows) != len(expected) or {r["id"] for r in rows} != set(expected):
        raise ValueError("missing, duplicate, or unknown trials")
    if [r["id"] for r in rows] != [t["id"] for t in key["trials"]]:
        raise ValueError("unexpected trial order")
    results = {}
    for duration in (2, 6):
        by_session = defaultdict(list)
        classes = {}
        confusion = {
            c: {p: 0 for p in ("tracked", "wheeled", "unsure")}
            for c in ("tracked", "wheeled")
        }
        committed = correct = total = 0
        for row in rows:
            trial = expected[row["id"]]
            if row["choice"] not in {"tracked", "wheeled", "unsure"}:
                raise ValueError("invalid response choice")
            if (
                type(row.get("confidence")) is not int
                or not 1 <= row["confidence"] <= 5
            ):
                raise ValueError("confidence must be 1..5")
            seconds = row.get("response_seconds")
            if (
                not isinstance(seconds, (int, float))
                or not np.isfinite(seconds)
                or seconds < 0
            ):
                raise ValueError("invalid response time")
            if type(row.get("plays")) is not int or row["plays"] < 1:
                raise ValueError("trial must have been played")
            if trial["seconds"] != duration:
                continue
            actual, prediction = trial["vehicle_class"], row["choice"]
            hit = int(actual == prediction)
            session = trial["recording_session"]
            by_session[session].append(hit)
            classes[session] = actual
            confusion[actual][prediction] += 1
            committed += prediction != "unsure"
            correct += hit
            total += 1
        recalls = {
            c: float(
                np.mean([np.mean(v) for s, v in by_session.items() if classes[s] == c])
            )
            for c in ("tracked", "wheeled")
        }
        rng = np.random.default_rng(42)
        draws = []
        for c in ("tracked", "wheeled"):
            values = [np.mean(v) for s, v in by_session.items() if classes[s] == c]
            draws.append(
                rng.choice(values, (10000, len(values)), replace=True).mean(axis=1)
            )
        results[f"{duration}s"] = {
            "trials": total,
            "accuracy": correct / total,
            "session_macro_balanced_accuracy": float(np.mean(list(recalls.values()))),
            "recall": recalls,
            "coverage": committed / total,
            "conditional_accuracy": correct / committed if committed else None,
            "confusion": confusion,
            "session_recall": {s: float(np.mean(v)) for s, v in by_session.items()},
            "descriptive_session_bootstrap_95_percent": np.quantile(
                (draws[0] + draws[1]) / 2, [0.025, 0.975]
            ).tolist(),
        }
    return results

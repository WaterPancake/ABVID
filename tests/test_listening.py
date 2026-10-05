import copy
import json
from pathlib import Path

import pytest

from vehicle_audio.development_corpus import (
    MANIFEST,
    load_development,
    validate_records,
)
from vehicle_audio.listening import score_responses, select_anchors


def test_selection_is_deterministic_balanced_and_reviewed():
    if not MANIFEST.exists():
        pytest.skip("local reviewed corpus not present")
    records, sources, _, _ = load_development()
    selected = select_anchors(records, sources)
    assert selected == select_anchors(list(reversed(records)), sources)
    assert len(selected) == 14
    assert len({r["recording_session"] for r in selected}) == 12
    assert sum(r["vehicle_class"] == "wheeled" for r in selected) == 7
    for a in selected:
        center = a["window_start_seconds"] + 1
        assert any(
            s["start_seconds"] <= center - 3 < center + 3 <= s["end_seconds"]
            for s in sources[a["source_id"]]["condition_segments"]
        )
        for b in selected:
            if a != b and a["recording_session"] == b["recording_session"]:
                assert abs(a["window_start_seconds"] - b["window_start_seconds"]) >= 6


def test_protected_startup_and_review_guards():
    if not MANIFEST.exists():
        pytest.skip("local reviewed corpus not present")
    import yaml

    records = [json.loads(x) for x in MANIFEST.read_text().splitlines()]
    catalog, policy, archive = [
        yaml.safe_load(Path(p).read_text())
        for p in (
            "configs/audio_sources.yaml",
            "configs/benchmark_v0_1.yaml",
            "configs/archived_startup_intervals.yaml",
        )
    ]
    for mutation in (
        {"source_id": policy["future_confirmation_source_ids"][0]},
        {"operating_condition": "startup"},
        {"window_end_seconds": 9999},
    ):
        modified = copy.deepcopy(records)
        modified[0].update(mutation)
        with pytest.raises(ValueError):
            validate_records(modified, catalog, policy, archive)


def example_responses():
    key = {"test_id": "unit", "trials": []}
    rows = []
    for seconds in (2, 6):
        for label in ("tracked", "wheeled"):
            ident = f"{seconds}-{label}"
            key["trials"].append(
                {
                    "id": ident,
                    "seconds": seconds,
                    "vehicle_class": label,
                    "recording_session": label,
                }
            )
            rows.append(
                {
                    "id": ident,
                    "choice": label,
                    "confidence": 3,
                    "plays": 1,
                    "response_seconds": 5,
                }
            )
    return key, {"test_id": "unit", "responses": rows}


def test_scoring_unsure_and_strict_validation():
    key, response = example_responses()
    assert score_responses(key, response)["2s"]["session_macro_balanced_accuracy"] == 1
    response["responses"][0]["choice"] = "unsure"
    score = score_responses(key, response)["2s"]
    assert score["session_macro_balanced_accuracy"] == 0.5
    assert score["coverage"] == 0.5 and score["conditional_accuracy"] == 1
    for field, value in (
        ("choice", "tank"),
        ("confidence", 0),
        ("response_seconds", float("nan")),
        ("plays", 0),
    ):
        bad = copy.deepcopy(response)
        bad["responses"][0][field] = value
        with pytest.raises(ValueError):
            score_responses(key, bad)
    response["responses"].pop()
    with pytest.raises(ValueError):
        score_responses(key, response)


def test_participant_payload_contains_no_answers():
    root = Path(".artifacts/blind_listening_v1")
    if not root.exists():
        pytest.skip("generated listening artifact not present")
    html = (root / "participant/index.html").read_text()
    key = json.loads((root / "private/answer_key.json").read_text())
    assert len(key["trials"]) == 28
    for t in key["trials"]:
        assert t["recording_session"] not in html
        assert t["source_id"] not in html
        assert t["anchor_sample_id"] not in html
    assert not (root / "participant/answer_key.json").exists()

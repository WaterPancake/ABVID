import copy

import numpy as np
import pytest

from vehicle_audio.state_pair_diagnostic import (
    fit_head,
    make_tasks,
    metrics,
    replay_head,
    validate_split,
)


CONFIG = {"C": 1, "solver": "lbfgs", "max_iter": 5000, "seed": 42}


def rows():
    return [
        {
            "sample_id": f"{label}-{start}",
            "source_id": str(label),
            "recording_session": str(label),
            "window_start_seconds": start,
            "window_end_seconds": start + 2,
            "pair_label": label,
            "operating_condition": "idle" if start == 0 else "steady_speed",
        }
        for start in (0, 3)
        for label in (0, 1)
    ]


def test_shared_sessions_default_reject_and_explicit_exception():
    r = rows()
    with pytest.raises(ValueError, match="opt-in"):
        validate_split(r[:2], r[2:])
    assert validate_split(r[:2], r[2:], allow_within_session=True) == ["0", "1"]


def test_exception_never_permits_audio_overlap_or_missing_class():
    r = rows()
    r[2]["window_start_seconds"] = 1
    with pytest.raises(ValueError, match="audio overlaps"):
        validate_split(r[:2], r[2:], allow_within_session=True)
    with pytest.raises(ValueError, match="both pair members"):
        validate_split(r[:1], r[2:], allow_within_session=True)


def test_split_uses_interval_identity_not_source_model():
    r = rows()
    extra = {
        **r[2],
        "sample_id": "other-model",
        "window_start_seconds": 8,
        "window_end_seconds": 10,
    }
    r.append(extra)
    inventory = {
        "intervals": [
            {
                "sample_ids": [x["sample_id"]],
                "reviewed_interval_model": "other"
                if x is extra
                else f"model-{x['pair_label']}",
            }
            for x in r
        ]
    }
    config = {
        "pairs": [
            {
                "id": "unit",
                "members": [
                    {"source_id": str(k), "model": f"model-{k}"} for k in (0, 1)
                ],
            }
        ],
        "directions": [["idle", "steady_speed"], ["steady_speed", "idle"]],
        "domain": "diagnostic",
    }
    with pytest.raises(ValueError):
        make_tasks(r, inventory, config)
    tasks = make_tasks(r, inventory, config, allow_within_session=True)
    assert len(tasks) == 2
    assert all(len(t["train"]) == len(t["test"]) == 2 for t in tasks)
    assert all(not t["benchmark_eligible"] for t in tasks)
    assert tasks[0]["train"] == tasks[1]["test"]


def test_weighted_scaler_train_only_and_replay_exact():
    x = np.array([[0, 1], [2, 1], [10, 1]], dtype=float)
    y = np.array([0, 0, 1])
    state, scaler, model = fit_head(x, y, CONFIG)
    # Equal class mass: mean of class means (1 and 10), not raw mean 4.
    assert state["mean"] == [5.5, 1.0]
    saved = copy.deepcopy(state)
    test = np.array([[-1e6, 1], [1e6, 1]])
    probability = replay_head(test, state)
    assert state == saved
    np.testing.assert_allclose(
        probability, model.predict_proba(scaler.transform(test))[:, 1], atol=1e-12
    )
    again, _, _ = fit_head(x, y, CONFIG)
    assert again == saved
    with pytest.raises(ValueError):
        fit_head(x * np.nan, y, CONFIG)


def test_balanced_accuracy_not_majority_accuracy():
    score = metrics([0, 0, 0, 1], [0, 0, 0, 0])
    assert score["accuracy"] == 0.75
    assert score["balanced_accuracy"] == 0.5
    assert score["recall_A"] == 1 and score["recall_B"] == 0
    assert score["confusion_true_rows_predicted_columns_A_B"] == [[3, 0], [1, 0]]
    assert metrics([0, 1], [0.49, 0.5])["balanced_accuracy"] == 1
    with pytest.raises(ValueError):
        metrics([0, 1], [0, float("nan")])

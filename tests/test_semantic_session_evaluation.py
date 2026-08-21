from __future__ import annotations

import numpy as np
import pytest
import torch

from vehicle_audio.semantic_session_evaluation import (
    SEMANTIC_AUDIOSET_FEATURES,
    nested_leave_session_pair_out,
    session_balanced_sample_weights,
)


pytest.importorskip("sklearn")


def test_semantic_feature_indices_are_unique_and_auditable() -> None:
    indices = [index for index, _ in SEMANTIC_AUDIOSET_FEATURES]
    assert len(indices) == 35
    assert len(indices) == len(set(indices))
    assert all(name for _, name in SEMANTIC_AUDIOSET_FEATURES)


def test_session_balanced_weights_equalize_classes_and_sessions() -> None:
    records = [
        {"vehicle_class": "tracked", "recording_session": "tracked_a"},
        {"vehicle_class": "tracked", "recording_session": "tracked_a"},
        {"vehicle_class": "tracked", "recording_session": "tracked_b"},
        {"vehicle_class": "wheeled", "recording_session": "wheeled_a"},
        {"vehicle_class": "wheeled", "recording_session": "wheeled_a"},
        {"vehicle_class": "wheeled", "recording_session": "wheeled_a"},
        {"vehicle_class": "wheeled", "recording_session": "wheeled_b"},
    ]

    weights = session_balanced_sample_weights(records, tuple(range(len(records))))

    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(len(records))
    assert weights[:3].sum() == pytest.approx(weights[3:].sum())
    assert weights[:2].sum() == pytest.approx(weights[2])
    assert weights[3:6].sum() == pytest.approx(weights[6])


def test_nested_session_evaluation_has_disjoint_grouped_outer_folds() -> None:
    records: list[dict[str, object]] = []
    feature_rows: list[list[float]] = []
    labels: list[int] = []
    generator = np.random.default_rng(123)
    for class_index, vehicle_class in enumerate(("tracked", "wheeled")):
        sign = -1.0 if class_index == 0 else 1.0
        for session_index in range(3):
            session = f"{vehicle_class}_{session_index}"
            for sample_index in range(3):
                records.append(
                    {
                        "sample_id": f"{session}_{sample_index}",
                        "vehicle_class": vehicle_class,
                        "recording_session": session,
                    }
                )
                feature_rows.append(
                    [
                        sign + 0.02 * generator.standard_normal(),
                        0.02 * generator.standard_normal(),
                    ]
                )
                labels.append(class_index)

    result, states, splits = nested_leave_session_pair_out(
        torch.tensor(feature_rows, dtype=torch.float32),
        torch.tensor(labels),
        records,
        c_values=(0.1, 1.0),
        seed=42,
    )

    assert result["aggregate"]["outer_fold_count"] == 9
    assert result["aggregate"]["balanced_accuracy"]["mean"] >= 0.9
    assert result["aggregate"]["both_heldout_sessions_correct_rate"] >= 0.8
    assert result["regularization_ensemble_aggregate"]["outer_fold_count"] == 9
    assert (
        result["regularization_ensemble_aggregate"]["balanced_accuracy"]["mean"]
        >= 0.9
    )
    assert len(states) == 9
    assert all(
        set(state) == {"nested_selected", "regularization_ensemble"}
        for state in states.values()
    )
    assert all(
        set(state["regularization_ensemble"]) == {"c_0.1", "c_1"}
        for state in states.values()
    )
    assert all(
        fold["regularization_ensemble_metrics"]["snr_available"] is False
        and set(fold["regularization_ensemble_session_predictions"])
        == {fold["tracked_test_session"], fold["wheeled_test_session"]}
        for fold in result["folds"]
    )
    assert len(splits["folds"]) == 9
    for fold in splits["folds"]:
        train_ids = set(fold["train_sample_ids"])
        test_ids = set(fold["test_sample_ids"])
        assert train_ids.isdisjoint(test_ids)
        assert len(fold["test_sessions"]["tracked"]) == 1
        assert len(fold["test_sessions"]["wheeled"]) == 1

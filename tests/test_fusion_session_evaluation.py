from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
import torch

from vehicle_audio.baseline import FeatureConfig
from vehicle_audio.fusion_session_evaluation import (
    _validate_candidates,
    load_or_extract_classical_features,
    nested_leave_session_pair_out_fusion,
)


pytest.importorskip("sklearn")


def _toy_records_and_features() -> tuple[
    list[dict[str, str]], torch.Tensor, torch.Tensor, torch.Tensor
]:
    records: list[dict[str, str]] = []
    semantic_rows: list[list[float]] = []
    classical_rows: list[list[float]] = []
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
                semantic_rows.append(
                    [sign + 0.02 * generator.standard_normal(), float(session_index)]
                )
                classical_rows.append(
                    [sign + 0.02 * generator.standard_normal(), float(sample_index)]
                )
                labels.append(class_index)
    return (
        records,
        torch.tensor(semantic_rows, dtype=torch.float32),
        torch.tensor(classical_rows, dtype=torch.float32),
        torch.tensor(labels),
    )


def test_candidate_validation_rejects_unordered_or_boundary_values() -> None:
    assert _validate_candidates((0.1, 1.0), (0.25, 0.5)) == (
        (0.1, 1.0),
        (0.25, 0.5),
    )
    with pytest.raises(ValueError, match="unique and increasing"):
        _validate_candidates((1.0, 0.1), (0.25, 0.5))
    with pytest.raises(ValueError, match="strictly between"):
        _validate_candidates((0.1, 1.0), (0.0, 0.5))


def test_nested_fusion_uses_disjoint_outer_pairs_and_inner_thresholds() -> None:
    records, semantic, classical, labels = _toy_records_and_features()

    result, states, splits = nested_leave_session_pair_out_fusion(
        semantic,
        classical,
        labels,
        records,
        c_values=(0.1, 1.0),
        threshold_values=(0.25, 0.5, 0.75),
        seed=42,
    )

    assert result["aggregate"]["outer_fold_count"] == 9
    assert result["aggregate"]["balanced_accuracy"]["mean"] >= 0.9
    assert result["aggregate"]["both_heldout_sessions_correct_rate"] >= 0.8
    assert len(states) == 9
    assert all(
        set(state)
        == {
            "semantic_regularization_ensemble",
            "classical_regularization_ensemble",
            "wheeled_threshold",
        }
        for state in states.values()
    )
    assert all(
        fold["selected_wheeled_threshold"] in {0.25, 0.5, 0.75}
        and len(fold["threshold_selection"]) == 3
        for fold in result["folds"]
    )
    for fold in splits["folds"]:
        assert set(fold["train_sample_ids"]).isdisjoint(fold["test_sample_ids"])
        assert len(fold["test_sessions"]["tracked"]) == 1
        assert len(fold["test_sessions"]["wheeled"]) == 1
        assert len(fold["inner_folds"]) == 4
        for inner_fold in fold["inner_folds"]:
            assert set(inner_fold["train_sample_ids"]).isdisjoint(
                inner_fold["validation_sample_ids"]
            )
            assert set(inner_fold["validation_sample_ids"]).isdisjoint(
                fold["test_sample_ids"]
            )


def test_real_classical_feature_cache_is_deterministic(tmp_path) -> None:
    waveform = 0.1 * np.sin(
        2.0 * np.pi * 180.0 * np.arange(16_000, dtype=np.float32) / 16_000
    )
    sf.write(tmp_path / "clip.wav", waveform, 16_000, subtype="FLOAT")
    records = [
        {
            "corrupted_path": "clip.wav",
            "sample_rate": 16_000,
            "vehicle_class": "tracked",
        }
    ]
    cache = tmp_path / "classical.pt"

    first_features, first_labels = load_or_extract_classical_features(
        records,
        tmp_path / "manifest.jsonl",
        "manifest-hash",
        cache,
        feature_config=FeatureConfig(),
    )
    second_features, second_labels = load_or_extract_classical_features(
        records,
        tmp_path / "manifest.jsonl",
        "manifest-hash",
        cache,
        feature_config=FeatureConfig(),
    )

    assert first_features.shape == (1, 53)
    assert torch.isfinite(first_features).all()
    assert torch.equal(first_features, second_features)
    assert torch.equal(first_labels, second_labels)

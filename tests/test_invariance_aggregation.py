from __future__ import annotations

import json
from pathlib import Path

import pytest

from vehicle_audio.invariance_aggregation import aggregate_invariance_runs


def _write_metrics(
    path: Path,
    *,
    seed: int,
    tracked_recall: float,
    wheeled_recall: float,
    dataset_version: str = "dataset-v1",
    with_real_adaptation: bool = False,
) -> None:
    balanced_accuracy = (tracked_recall + wheeled_recall) / 2.0
    evaluation = {
        "support": 20,
        "accuracy": balanced_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": balanced_accuracy - 0.05,
        "per_class_recall": {
            "tracked": tracked_recall,
            "wheeled": wheeled_recall,
        },
    }
    all_session_evaluation = {
        **evaluation,
        "session_macro_accuracy": balanced_accuracy,
        "session_balanced_accuracy": balanced_accuracy,
        "class_session_mean_recall": {
            "tracked": tracked_recall,
            "wheeled": wheeled_recall,
        },
    }
    payload = {
        "protocol_status": "complete",
        "git_commit": "abc123",
        "dataset_version": dataset_version,
        "training_config": {
            "invariance_evaluation_version": 6,
            "methods": ["standard_supervised"],
            "model": "toy",
            "split_seed": 42,
            "seed": seed,
            **(
                {"adaptation_source_method": "standard_supervised"}
                if with_real_adaptation
                else {}
            ),
        },
        "methods": {
            "standard_supervised": {
                "evaluations": {
                    "all_corruptions": evaluation,
                    "native_real": evaluation,
                    "native_real_all_sessions": all_session_evaluation,
                }
            }
        },
    }
    if with_real_adaptation:
        adapted_tracked = min(1.0, tracked_recall + 0.1)
        adapted_wheeled = min(1.0, wheeled_recall + 0.2)
        adapted_balanced = (adapted_tracked + adapted_wheeled) / 2.0
        payload["real_adaptation"] = [
            {
                "key": "25_percent_real",
                "requested_real_fraction": 0.25,
                "actual_real_fraction": 0.25,
                "real_training_support": 4,
                "real_training_class_support": {"tracked": 2, "wheeled": 2},
                "real_training_sample_ids": ["a", "b", "c", "d"],
                "native_real": {
                    "support": 20,
                    "accuracy": adapted_balanced,
                    "balanced_accuracy": adapted_balanced,
                    "macro_f1": adapted_balanced - 0.05,
                    "per_class_recall": {
                        "tracked": adapted_tracked,
                        "wheeled": adapted_wheeled,
                    },
                },
            }
        ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_invariance_runs_writes_validated_statistics(tmp_path: Path) -> None:
    first = tmp_path / "seed_1.json"
    second = tmp_path / "seed_2.json"
    _write_metrics(first, seed=1, tracked_recall=0.8, wheeled_recall=0.2)
    _write_metrics(second, seed=2, tracked_recall=0.6, wheeled_recall=0.4)

    result = aggregate_invariance_runs([first, second], tmp_path / "aggregate")

    assert result["protocol_status"] == "complete"
    assert result["run_count"] == 2
    assert [run["training_seed"] for run in result["source_runs"]] == [1, 2]
    summary = result["methods"]["standard_supervised"]["evaluations"]
    assert summary["native_real"]["metrics"]["balanced_accuracy"]["mean"] == 0.5
    assert summary["native_real"]["per_class_recall"]["tracked"]["mean"] == 0.7
    assert summary["native_real"]["per_class_recall"]["wheeled"]["mean"] == pytest.approx(0.3)
    all_sessions = summary["native_real_all_sessions"]
    assert all_sessions["metrics"]["session_balanced_accuracy"]["mean"] == 0.5
    assert all_sessions["class_session_mean_recall"]["tracked"]["mean"] == 0.7
    for name in (
        "aggregate_metrics.json",
        "aggregate_summary.csv",
        "aggregate_condition_comparison.png",
    ):
        assert (tmp_path / "aggregate" / name).is_file()


def test_aggregate_invariance_runs_rejects_dataset_mismatch(tmp_path: Path) -> None:
    first = tmp_path / "seed_1.json"
    second = tmp_path / "seed_2.json"
    _write_metrics(first, seed=1, tracked_recall=0.8, wheeled_recall=0.2)
    _write_metrics(
        second,
        seed=2,
        tracked_recall=0.6,
        wheeled_recall=0.4,
        dataset_version="different",
    )

    with pytest.raises(ValueError, match="dataset version mismatch"):
        aggregate_invariance_runs([first, second], tmp_path / "aggregate")


def test_aggregate_invariance_runs_summarizes_real_adaptation(tmp_path: Path) -> None:
    first = tmp_path / "seed_1.json"
    second = tmp_path / "seed_2.json"
    _write_metrics(
        first,
        seed=1,
        tracked_recall=0.8,
        wheeled_recall=0.2,
        with_real_adaptation=True,
    )
    _write_metrics(
        second,
        seed=2,
        tracked_recall=0.6,
        wheeled_recall=0.4,
        with_real_adaptation=True,
    )

    result = aggregate_invariance_runs([first, second], tmp_path / "aggregate")

    adaptation = result["real_adaptation"]
    assert adaptation["source_method"] == "standard_supervised"
    assert adaptation["zero_real_baseline"]["metrics"]["balanced_accuracy"][
        "mean"
    ] == 0.5
    fraction = adaptation["fractions"]["25_percent_real"]
    assert fraction["real_training_support"] == 4
    assert fraction["metrics"]["balanced_accuracy"]["mean"] == pytest.approx(0.65)
    assert fraction["per_class_recall"]["tracked"]["mean"] == pytest.approx(0.8)
    assert (tmp_path / "aggregate" / "aggregate_real_adaptation_curve.png").is_file()

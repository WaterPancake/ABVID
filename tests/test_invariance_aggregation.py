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
        },
        "methods": {
            "standard_supervised": {
                "evaluations": {
                    "all_corruptions": evaluation,
                    "native_real": evaluation,
                }
            }
        },
    }
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

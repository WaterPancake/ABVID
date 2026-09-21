"""Nested unseen-session evaluation on classical vehicle-audio features."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from vehicle_audio.baseline import FEATURE_IMPLEMENTATION_VERSION, FeatureConfig
from vehicle_audio.fusion_session_evaluation import load_or_extract_classical_features
from vehicle_audio.invariance_evaluation import _git_commit, _load_jsonl
from vehicle_audio.semantic_session_evaluation import (
    _require_sklearn,
    nested_leave_session_pair_out,
)


CLASSICAL_SESSION_EVALUATION_VERSION = 1
DEFAULT_REGULARIZATION_C = (0.001, 0.01, 0.1, 1.0, 10.0)


def evaluate_classical_sessions(
    real_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    feature_cache_path: str | Path | None = None,
    c_values: Sequence[float] = DEFAULT_REGULARIZATION_C,
    channel: int = 0,
    extraction_batch_size: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate the fixed classical feature baseline with nested session splits."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = Path(real_manifest_path)
    records, manifest_sha256 = _load_jsonl(manifest)
    cache = (
        Path(feature_cache_path)
        if feature_cache_path is not None
        else output / "real_classical_features.pt"
    )
    features, labels = load_or_extract_classical_features(
        records,
        manifest,
        manifest_sha256,
        cache,
        channel=channel,
        batch_size=extraction_batch_size,
    )
    evaluation, model_states, splits = nested_leave_session_pair_out(
        features,
        labels,
        records,
        c_values=c_values,
        seed=seed,
    )
    _, _, sklearn_version = _require_sklearn()
    training_config = {
        "classical_session_evaluation_version": CLASSICAL_SESSION_EVALUATION_VERSION,
        "models": [
            "nested_selected_logistic_regression",
            "regularization_probability_ensemble",
        ],
        "classical_feature_config": asdict(FeatureConfig()),
        "classical_feature_implementation_version": FEATURE_IMPLEMENTATION_VERSION,
        "regularization_c_candidates": [float(value) for value in c_values],
        "selection_metric": "inner_pair_mean_balanced_accuracy",
        "ensemble_rule": "equal_mean_of_probabilities_across_all_regularization_candidates",
        "sample_weighting": "equal_class_equal_session_equal_window_within_session",
        "channel": channel,
        "seed": seed,
        "scikit_learn_version": sklearn_version,
    }
    results = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset_version": manifest_sha256,
        "real_manifest": str(manifest),
        "real_manifest_sha256": manifest_sha256,
        "feature_cache": str(cache),
        "test_domain": "real -> real nested unseen recording-session pairs",
        "protocol_status": "complete",
        "training_config": training_config,
        **evaluation,
        "artifacts": {
            "checkpoint": "classical_probe_models.pt",
            "splits": "splits.json",
            "fold_csv": "outer_fold_metrics.csv",
            "fold_plot": "outer_fold_balanced_accuracy.png",
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "splits.json").write_text(
        json.dumps(splits, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "models": model_states,
            "training_config": training_config,
            "class_names": ["tracked", "wheeled"],
            "real_manifest_sha256": manifest_sha256,
        },
        output / "classical_probe_models.pt",
    )
    experiment = {
        "git_commit": results["git_commit"],
        "configuration": training_config,
        "random_seed": seed,
        "dataset_version": manifest_sha256,
        "train_test_split": "splits.json",
        "model_checkpoint": "classical_probe_models.pt",
        "metrics": "metrics.json",
        "test_domain": results["test_domain"],
        "protocol_status": results["protocol_status"],
    }
    (output / "experiment.json").write_text(
        json.dumps(experiment, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    rows = []
    for fold in results["folds"]:
        rows.append(
            {
                "fold_id": fold["fold_id"],
                "tracked_test_session": fold["tracked_test_session"],
                "wheeled_test_session": fold["wheeled_test_session"],
                "selected_regularization_c": fold["selected_regularization_c"],
                "selected_balanced_accuracy": fold["metrics"]["balanced_accuracy"],
                "selected_tracked_recall": fold["metrics"]["per_class_recall"][
                    "tracked"
                ],
                "selected_wheeled_recall": fold["metrics"]["per_class_recall"][
                    "wheeled"
                ],
                "ensemble_balanced_accuracy": fold["regularization_ensemble_metrics"][
                    "balanced_accuracy"
                ],
                "ensemble_tracked_recall": fold["regularization_ensemble_metrics"][
                    "per_class_recall"
                ]["tracked"],
                "ensemble_wheeled_recall": fold["regularization_ensemble_metrics"][
                    "per_class_recall"
                ]["wheeled"],
            }
        )
    with (output / "outer_fold_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    positions = np.arange(len(rows))
    selected = [float(row["selected_balanced_accuracy"]) for row in rows]
    ensemble = [float(row["ensemble_balanced_accuracy"]) for row in rows]
    figure, axis = plt.subplots(figsize=(10.5, 5.2))
    axis.bar(positions - 0.2, selected, width=0.4, label="nested-selected C")
    axis.bar(positions + 0.2, ensemble, width=0.4, label="regularization ensemble")
    axis.axhline(
        float(
            results["regularization_ensemble_aggregate"]["balanced_accuracy"]["mean"]
        ),
        color="tab:orange",
        linestyle="--",
        label="ensemble mean",
    )
    axis.set_xlabel("Held-out tracked/wheeled session pair")
    axis.set_ylabel("Balanced accuracy")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Nested unseen-session classical-feature evaluation")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "outer_fold_balanced_accuracy.png", dpi=160)
    plt.close(figure)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument(
        "--regularization-c", type=float, action="append", dest="c_values"
    )
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--extraction-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = evaluate_classical_sessions(
        args.real_manifest,
        args.output,
        feature_cache_path=args.feature_cache,
        c_values=DEFAULT_REGULARIZATION_C if args.c_values is None else args.c_values,
        channel=args.channel,
        extraction_batch_size=args.extraction_batch_size,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output),
                "protocol_status": results["protocol_status"],
                "outer_fold_count": results["aggregate"]["outer_fold_count"],
                "nested_selected": results["aggregate"],
                "regularization_ensemble": results["regularization_ensemble_aggregate"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

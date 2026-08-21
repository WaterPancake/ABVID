"""Nested unseen-session evaluation with semantic/classical late fusion."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from vehicle_audio.baseline import (
    CLASS_NAMES,
    FEATURE_IMPLEMENTATION_VERSION,
    ClassicalFeatureExtractor,
    FeatureConfig,
    _read_one_channel,
    classification_metrics,
)
from vehicle_audio.invariance_evaluation import _git_commit, _load_jsonl
from vehicle_audio.pretrained_evaluation import validate_panns_checkpoint
from vehicle_audio.semantic_session_evaluation import (
    SEMANTIC_AUDIOSET_FEATURES,
    _aggregate_folds,
    _evaluate_probe_ensemble,
    _fit_probe,
    _load_semantic_features,
    _require_sklearn,
    _sessions_by_class,
)


FUSION_SESSION_EVALUATION_VERSION = 1
DEFAULT_REGULARIZATION_C = (0.001, 0.01, 0.1, 1.0, 10.0)
DEFAULT_WHEELED_THRESHOLDS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)


def _validate_candidates(
    c_values: Sequence[float], threshold_values: Sequence[float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    regularization = tuple(float(value) for value in c_values)
    thresholds = tuple(float(value) for value in threshold_values)
    if not regularization or any(value <= 0 for value in regularization):
        raise ValueError("C candidates must be positive")
    if tuple(sorted(set(regularization))) != regularization:
        raise ValueError("C candidates must be unique and increasing")
    if not thresholds or any(not 0 < value < 1 for value in thresholds):
        raise ValueError("wheeled thresholds must be strictly between zero and one")
    if tuple(sorted(set(thresholds))) != thresholds:
        raise ValueError("wheeled thresholds must be unique and increasing")
    return regularization, thresholds


def _fit_regularization_ensemble(
    features: np.ndarray,
    labels: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
    c_values: Sequence[float],
    seed: int,
) -> tuple[list[tuple[Any, Any]], dict[str, dict[str, torch.Tensor]]]:
    probes: list[tuple[Any, Any]] = []
    states: dict[str, dict[str, torch.Tensor]] = {}
    for regularization_c in c_values:
        scaler, model, state = _fit_probe(
            features,
            labels,
            records,
            train_indices,
            regularization_c,
            seed,
        )
        probes.append((scaler, model))
        states[f"c_{regularization_c:g}"] = state
    return probes, states


def _fused_probabilities(
    semantic_features: np.ndarray,
    classical_features: np.ndarray,
    labels: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
    evaluation_indices: Sequence[int],
    c_values: Sequence[float],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    semantic_probes, semantic_states = _fit_regularization_ensemble(
        semantic_features,
        labels,
        records,
        train_indices,
        c_values,
        seed,
    )
    classical_probes, classical_states = _fit_regularization_ensemble(
        classical_features,
        labels,
        records,
        train_indices,
        c_values,
        seed,
    )
    _, semantic_probabilities = _evaluate_probe_ensemble(
        semantic_features,
        labels,
        evaluation_indices,
        semantic_probes,
    )
    _, classical_probabilities = _evaluate_probe_ensemble(
        classical_features,
        labels,
        evaluation_indices,
        classical_probes,
    )
    probabilities = 0.5 * (semantic_probabilities + classical_probabilities)
    return probabilities, {
        "semantic_regularization_ensemble": semantic_states,
        "classical_regularization_ensemble": classical_states,
    }


def _threshold_metrics(
    labels: np.ndarray,
    indices: Sequence[int],
    probabilities: np.ndarray,
    wheeled_threshold: float,
) -> dict[str, Any]:
    selected = np.asarray(sorted(int(index) for index in indices), dtype=np.int64)
    predictions = (probabilities[:, 1] >= wheeled_threshold).astype(np.int64)
    metrics = classification_metrics(
        torch.from_numpy(predictions),
        torch.from_numpy(labels[selected]),
        torch.zeros(len(selected)),
    )
    metrics.pop("per_snr")
    metrics["snr_available"] = False
    metrics["calibration_available"] = False
    return metrics


def _threshold_session_predictions(
    probabilities: np.ndarray,
    sorted_test_indices: Sequence[int],
    records: Sequence[Mapping[str, Any]],
    tracked_session: str,
    wheeled_session: str,
    wheeled_threshold: float,
) -> dict[str, Any]:
    predictions: dict[str, Any] = {}
    for vehicle_class, session in (
        ("tracked", tracked_session),
        ("wheeled", wheeled_session),
    ):
        positions = [
            position
            for position, record_index in enumerate(sorted_test_indices)
            if str(records[record_index]["recording_session"]) == session
        ]
        mean_probability = probabilities[positions].mean(axis=0)
        predicted_class = (
            "wheeled" if mean_probability[1] >= wheeled_threshold else "tracked"
        )
        predictions[session] = {
            "vehicle_class": vehicle_class,
            "predicted_class": predicted_class,
            "correct": predicted_class == vehicle_class,
            "mean_probability": {
                name: float(mean_probability[class_index])
                for class_index, name in enumerate(CLASS_NAMES)
            },
            "wheeled_threshold": wheeled_threshold,
            "support": len(positions),
        }
    return predictions


def nested_leave_session_pair_out_fusion(
    semantic_features: torch.Tensor,
    classical_features: torch.Tensor,
    labels: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    *,
    c_values: Sequence[float] = DEFAULT_REGULARIZATION_C,
    threshold_values: Sequence[float] = DEFAULT_WHEELED_THRESHOLDS,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Evaluate fixed equal fusion with inner-only decision-threshold selection."""

    if semantic_features.ndim != 2 or semantic_features.shape[0] != len(records):
        raise ValueError("semantic features must have shape [records, features]")
    if classical_features.ndim != 2 or classical_features.shape[0] != len(records):
        raise ValueError("classical features must have shape [records, features]")
    if labels.shape != (len(records),):
        raise ValueError("labels must have one value per record")
    if not torch.isfinite(semantic_features).all() or not torch.isfinite(
        classical_features
    ).all():
        raise ValueError("features contain NaN or Inf")
    regularization, thresholds = _validate_candidates(c_values, threshold_values)
    sessions = _sessions_by_class(records)
    all_indices = set(range(len(records)))
    semantic_array = semantic_features.detach().cpu().numpy().astype(np.float64)
    classical_array = classical_features.detach().cpu().numpy().astype(np.float64)
    label_array = labels.detach().cpu().numpy().astype(np.int64)
    expected_labels = np.asarray(
        [CLASS_NAMES.index(str(record["vehicle_class"])) for record in records]
    )
    if not np.array_equal(label_array, expected_labels):
        raise ValueError("feature-cache labels do not align with the real manifest")

    folds: list[dict[str, Any]] = []
    model_states: dict[str, Any] = {}
    split_folds: list[dict[str, Any]] = []
    for tracked_session, tracked_indices in sessions["tracked"].items():
        for wheeled_session, wheeled_indices in sessions["wheeled"].items():
            outer_test = set(tracked_indices) | set(wheeled_indices)
            outer_train = all_indices - outer_test
            inner_predictions: list[dict[str, Any]] = []
            inner_split_folds: list[dict[str, Any]] = []
            for inner_tracked in sessions["tracked"]:
                if inner_tracked == tracked_session:
                    continue
                for inner_wheeled in sessions["wheeled"]:
                    if inner_wheeled == wheeled_session:
                        continue
                    inner_validation = set(sessions["tracked"][inner_tracked]) | set(
                        sessions["wheeled"][inner_wheeled]
                    )
                    probabilities, _ = _fused_probabilities(
                        semantic_array,
                        classical_array,
                        label_array,
                        records,
                        outer_train - inner_validation,
                        inner_validation,
                        regularization,
                        seed,
                    )
                    inner_train = outer_train - inner_validation
                    inner_predictions.append(
                        {
                            "tracked_validation_session": inner_tracked,
                            "wheeled_validation_session": inner_wheeled,
                            "validation_indices": inner_validation,
                            "probabilities": probabilities,
                        }
                    )
                    inner_split_folds.append(
                        {
                            "tracked_validation_session": inner_tracked,
                            "wheeled_validation_session": inner_wheeled,
                            "train_sample_ids": [
                                str(records[index]["sample_id"])
                                for index in sorted(inner_train)
                            ],
                            "validation_sample_ids": [
                                str(records[index]["sample_id"])
                                for index in sorted(inner_validation)
                            ],
                        }
                    )

            threshold_results: list[dict[str, Any]] = []
            for threshold in thresholds:
                inner_folds = []
                for inner_fold in inner_predictions:
                    score = float(
                        _threshold_metrics(
                            label_array,
                            inner_fold["validation_indices"],
                            inner_fold["probabilities"],
                            threshold,
                        )["balanced_accuracy"]
                    )
                    inner_folds.append(
                        {
                            "tracked_validation_session": inner_fold[
                                "tracked_validation_session"
                            ],
                            "wheeled_validation_session": inner_fold[
                                "wheeled_validation_session"
                            ],
                            "balanced_accuracy": score,
                        }
                    )
                inner_scores = [
                    float(inner_fold["balanced_accuracy"])
                    for inner_fold in inner_folds
                ]
                threshold_results.append(
                    {
                        "wheeled_threshold": threshold,
                        "mean_balanced_accuracy": mean(inner_scores),
                        "median_balanced_accuracy": median(inner_scores),
                        "minimum_balanced_accuracy": min(inner_scores),
                        "inner_folds": inner_folds,
                    }
                )
            selected = max(
                threshold_results,
                key=lambda row: (
                    float(row["mean_balanced_accuracy"]),
                    float(row["median_balanced_accuracy"]),
                    float(row["minimum_balanced_accuracy"]),
                    -abs(float(row["wheeled_threshold"]) - 0.5),
                ),
            )
            threshold = float(selected["wheeled_threshold"])
            probabilities, state = _fused_probabilities(
                semantic_array,
                classical_array,
                label_array,
                records,
                outer_train,
                outer_test,
                regularization,
                seed,
            )
            metrics = _threshold_metrics(
                label_array,
                outer_test,
                probabilities,
                threshold,
            )
            sorted_test = sorted(outer_test)
            session_predictions = _threshold_session_predictions(
                probabilities,
                sorted_test,
                records,
                tracked_session,
                wheeled_session,
                threshold,
            )
            fold_id = hashlib.sha256(
                f"fusion:{tracked_session}:{wheeled_session}".encode("utf-8")
            ).hexdigest()[:16]
            state["wheeled_threshold"] = torch.tensor(threshold, dtype=torch.float32)
            model_states[fold_id] = state
            folds.append(
                {
                    "fold_id": fold_id,
                    "tracked_test_session": tracked_session,
                    "wheeled_test_session": wheeled_session,
                    "selected_wheeled_threshold": threshold,
                    "inner_selection_mean_balanced_accuracy": selected[
                        "mean_balanced_accuracy"
                    ],
                    "threshold_selection": threshold_results,
                    "metrics": metrics,
                    "session_predictions": session_predictions,
                }
            )
            split_folds.append(
                {
                    "fold_id": fold_id,
                    "train_sessions": {
                        vehicle_class: sorted(
                            session
                            for session in sessions[vehicle_class]
                            if session
                            != (
                                tracked_session
                                if vehicle_class == "tracked"
                                else wheeled_session
                            )
                        )
                        for vehicle_class in CLASS_NAMES
                    },
                    "test_sessions": {
                        "tracked": [tracked_session],
                        "wheeled": [wheeled_session],
                    },
                    "train_sample_ids": [
                        str(records[index]["sample_id"])
                        for index in sorted(outer_train)
                    ],
                    "test_sample_ids": [
                        str(records[index]["sample_id"])
                        for index in sorted(outer_test)
                    ],
                    "inner_folds": inner_split_folds,
                }
            )

    return {
        "folds": folds,
        "aggregate": _aggregate_folds(
            folds,
            metrics_field="metrics",
            session_predictions_field="session_predictions",
        ),
    }, model_states, {
        "strategy": "nested_leave_one_tracked_and_one_wheeled_session_out",
        "group_field": "recording_session",
        "folds": split_folds,
    }


def load_or_extract_classical_features(
    records: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    manifest_sha256: str,
    cache_path: Path,
    *,
    channel: int = 0,
    batch_size: int = 32,
    feature_config: FeatureConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract deterministic classical features from native-real windows."""

    config = FeatureConfig() if feature_config is None else feature_config
    expected = {
        "manifest_sha256": manifest_sha256,
        "channel": channel,
        "feature_config": asdict(config),
        "feature_implementation_version": FEATURE_IMPLEMENTATION_VERSION,
    }
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if all(cached.get(key) == value for key, value in expected.items()):
            features = cached.get("features")
            labels = cached.get("labels")
            if isinstance(features, torch.Tensor) and isinstance(labels, torch.Tensor):
                if features.ndim == 2 and features.shape[0] == len(records):
                    if labels.shape == (len(records),):
                        return features.to(torch.float32), labels.to(torch.long)
        raise ValueError("classical feature cache metadata or tensor shape mismatch")

    extractor = ClassicalFeatureExtractor(config).eval()
    root = manifest_path.parent
    feature_batches: list[torch.Tensor] = []
    waveforms: list[torch.Tensor] = []
    labels: list[int] = []

    def flush() -> None:
        if not waveforms:
            return
        lengths = {waveform.shape[-1] for waveform in waveforms}
        if len(lengths) != 1:
            raise ValueError("all real evaluation windows must have equal duration")
        with torch.inference_mode():
            feature_batches.append(extractor(torch.stack(waveforms)).cpu())
        waveforms.clear()

    for record in records:
        sample_rate = int(record["sample_rate"])
        if sample_rate != config.sample_rate:
            raise ValueError(
                f"real manifest sample rate {sample_rate} != {config.sample_rate}"
            )
        waveforms.append(
            _read_one_channel(
                root / str(record["corrupted_path"]),
                channel,
                config.sample_rate,
            )
        )
        labels.append(CLASS_NAMES.index(str(record["vehicle_class"])))
        if len(waveforms) == batch_size:
            flush()
    flush()
    features = torch.cat(feature_batches).to(torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    if not torch.isfinite(features).all():
        raise RuntimeError("classical feature extraction produced NaN or Inf")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"features": features, "labels": label_tensor, **expected},
        cache_path,
    )
    return features, label_tensor


def evaluate_fusion_sessions(
    real_manifest_path: str | Path,
    panns_feature_cache_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    c_values: Sequence[float] = DEFAULT_REGULARIZATION_C,
    threshold_values: Sequence[float] = DEFAULT_WHEELED_THRESHOLDS,
    channel: int = 0,
    extraction_batch_size: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = Path(real_manifest_path)
    records, manifest_sha256 = _load_jsonl(manifest)
    checkpoint = validate_panns_checkpoint(checkpoint_path)
    semantic_features, semantic_labels = _load_semantic_features(
        Path(panns_feature_cache_path),
        records,
        manifest_sha256,
        str(checkpoint["sha256"]),
    )
    classical_cache = output / "real_classical_features.pt"
    classical_features, classical_labels = load_or_extract_classical_features(
        records,
        manifest,
        manifest_sha256,
        classical_cache,
        channel=channel,
        batch_size=extraction_batch_size,
    )
    if not torch.equal(semantic_labels, classical_labels):
        raise ValueError("semantic and classical feature labels do not align")
    evaluation, model_states, splits = nested_leave_session_pair_out_fusion(
        semantic_features,
        classical_features,
        semantic_labels,
        records,
        c_values=c_values,
        threshold_values=threshold_values,
        seed=seed,
    )
    _, _, sklearn_version = _require_sklearn()
    training_config = {
        "fusion_session_evaluation_version": FUSION_SESSION_EVALUATION_VERSION,
        "model": "equal_probability_semantic_classical_regularization_ensembles",
        "fusion_weights": {"semantic": 0.5, "classical": 0.5},
        "semantic_audioset_features": [
            {"index": index, "name": name}
            for index, name in SEMANTIC_AUDIOSET_FEATURES
        ],
        "classical_feature_config": asdict(FeatureConfig()),
        "classical_feature_implementation_version": FEATURE_IMPLEMENTATION_VERSION,
        "regularization_c_candidates": [float(value) for value in c_values],
        "wheeled_threshold_candidates": [float(value) for value in threshold_values],
        "threshold_selection_metric": "inner_pair_mean_balanced_accuracy",
        "sample_weighting": "equal_class_equal_session_equal_window_within_session",
        "encoder_trainable": False,
        "scikit_learn_version": sklearn_version,
        "channel": channel,
        "seed": seed,
    }
    results = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset_version": manifest_sha256,
        "real_manifest": str(manifest),
        "real_manifest_sha256": manifest_sha256,
        "panns_feature_cache": str(panns_feature_cache_path),
        "classical_feature_cache": str(classical_cache),
        "test_domain": "real -> real nested unseen recording-session pairs",
        "protocol_status": "complete",
        "pretrained_encoder": checkpoint,
        "training_config": training_config,
        **evaluation,
        "artifacts": {
            "checkpoint": "fusion_probe_models.pt",
            "classical_feature_cache": "real_classical_features.pt",
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
            "pretrained_encoder": checkpoint,
            "class_names": list(CLASS_NAMES),
            "real_manifest_sha256": manifest_sha256,
        },
        output / "fusion_probe_models.pt",
    )
    experiment = {
        "git_commit": results["git_commit"],
        "configuration": training_config,
        "random_seed": seed,
        "dataset_version": manifest_sha256,
        "train_test_split": "splits.json",
        "model_checkpoint": "fusion_probe_models.pt",
        "metrics": "metrics.json",
        "test_domain": results["test_domain"],
        "protocol_status": results["protocol_status"],
    }
    (output / "experiment.json").write_text(
        json.dumps(experiment, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    csv_rows = [
        {
            "fold_id": fold["fold_id"],
            "tracked_test_session": fold["tracked_test_session"],
            "wheeled_test_session": fold["wheeled_test_session"],
            "selected_wheeled_threshold": fold["selected_wheeled_threshold"],
            "balanced_accuracy": fold["metrics"]["balanced_accuracy"],
            "macro_f1": fold["metrics"]["macro_f1"],
            "tracked_recall": fold["metrics"]["per_class_recall"]["tracked"],
            "wheeled_recall": fold["metrics"]["per_class_recall"]["wheeled"],
            "both_sessions_correct": all(
                prediction["correct"]
                for prediction in fold["session_predictions"].values()
            ),
        }
        for fold in results["folds"]
    ]
    with (output / "outer_fold_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    figure, axis = plt.subplots(figsize=(10.5, 5.2))
    values = [float(row["balanced_accuracy"]) for row in csv_rows]
    axis.bar(range(len(values)), values)
    axis.axhline(
        float(results["aggregate"]["balanced_accuracy"]["mean"]),
        color="black",
        linestyle="--",
        label="fold mean",
    )
    axis.set_xticks(range(len(values)), [str(index + 1) for index in range(len(values))])
    axis.set_xlabel("Held-out tracked/wheeled session pair")
    axis.set_ylabel("Balanced accuracy")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Nested unseen-session equal late fusion")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "outer_fold_balanced_accuracy.png", dpi=160)
    plt.close(figure)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--panns-feature-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regularization-c", type=float, action="append", dest="c_values")
    parser.add_argument(
        "--wheeled-threshold", type=float, action="append", dest="threshold_values"
    )
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--extraction-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = evaluate_fusion_sessions(
        args.real_manifest,
        args.panns_feature_cache,
        args.checkpoint,
        args.output,
        c_values=DEFAULT_REGULARIZATION_C if args.c_values is None else args.c_values,
        threshold_values=(
            DEFAULT_WHEELED_THRESHOLDS
            if args.threshold_values is None
            else args.threshold_values
        ),
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
                "balanced_accuracy": results["aggregate"]["balanced_accuracy"],
                "per_class_recall": results["aggregate"]["per_class_recall"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

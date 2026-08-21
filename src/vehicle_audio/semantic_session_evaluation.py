"""Nested unseen-session evaluation on semantic PANNs vehicle features."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from vehicle_audio.baseline import CLASS_NAMES, classification_metrics
from vehicle_audio.invariance_evaluation import _git_commit, _load_jsonl
from vehicle_audio.pretrained_evaluation import (
    PANN_AUDIOSET_DIMENSION,
    PRETRAINED_EVALUATION_VERSION,
    validate_panns_checkpoint,
)


SEMANTIC_SESSION_EVALUATION_VERSION = 1

# Fixed before evaluation from the AudioSet ontology. These features describe
# vehicles, road/rail motion, engines, and mechanical track-like sound. They exclude
# speech/music labels whose near-zero variance produced unstable shortcut weights.
SEMANTIC_AUDIOSET_FEATURES: tuple[tuple[int, str], ...] = (
    (135, "Rattle"),
    (300, "Vehicle"),
    (301, "Boat, Water vehicle"),
    (306, "Motor vehicle (road)"),
    (307, "Car"),
    (308, "Vehicle horn, car horn, honking"),
    (309, "Toot"),
    (310, "Car alarm"),
    (311, "Power windows, electric windows"),
    (312, "Skidding"),
    (313, "Tire squeal"),
    (314, "Car passing by"),
    (315, "Race car, auto racing"),
    (316, "Truck"),
    (317, "Air brake"),
    (318, "Air horn, truck horn"),
    (320, "Ice cream truck, ice cream van"),
    (321, "Bus"),
    (322, "Emergency vehicle"),
    (326, "Motorcycle"),
    (328, "Rail transport"),
    (329, "Train"),
    (330, "Train whistle"),
    (331, "Train horn"),
    (332, "Railroad car, train wagon"),
    (333, "Train wheels squealing"),
    (343, "Engine"),
    (344, "Light engine (high frequency)"),
    (348, "Medium engine (mid frequency)"),
    (349, "Heavy engine (low frequency)"),
    (350, "Engine knocking"),
    (351, "Engine starting"),
    (352, "Idling"),
    (353, "Accelerating, revving, vroom"),
    (489, "Clatter"),
)


def _require_sklearn() -> tuple[Any, Any, str]:
    try:
        import sklearn
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "semantic session evaluation requires `uv sync --extra pretrained`"
        ) from error
    return LogisticRegression, StandardScaler, str(sklearn.__version__)


def _sessions_by_class(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, tuple[int, ...]]]:
    grouped: dict[str, dict[str, list[int]]] = {
        vehicle_class: {} for vehicle_class in CLASS_NAMES
    }
    for index, record in enumerate(records):
        vehicle_class = str(record.get("vehicle_class"))
        session = str(record.get("recording_session", ""))
        if vehicle_class not in grouped:
            raise ValueError(f"unexpected vehicle class at record {index}: {vehicle_class}")
        if not session:
            raise ValueError(f"record {index} has no recording_session")
        grouped[vehicle_class].setdefault(session, []).append(index)
    result = {
        vehicle_class: {
            session: tuple(indices) for session, indices in sorted(sessions.items())
        }
        for vehicle_class, sessions in grouped.items()
    }
    for vehicle_class, sessions in result.items():
        if len(sessions) < 3:
            raise ValueError(
                "nested session evaluation requires at least three sessions per "
                f"class; found {len(sessions)} for {vehicle_class}"
            )
    return result


def session_balanced_sample_weights(
    records: Sequence[Mapping[str, Any]], indices: Sequence[int]
) -> np.ndarray:
    """Give each class and each session within a class equal total training mass."""

    selected = tuple(int(index) for index in indices)
    if not selected:
        raise ValueError("cannot weight an empty training partition")
    weights = np.zeros(len(selected), dtype=np.float64)
    for vehicle_class in CLASS_NAMES:
        positions_by_session: dict[str, list[int]] = {}
        for position, index in enumerate(selected):
            record = records[index]
            if str(record["vehicle_class"]) == vehicle_class:
                positions_by_session.setdefault(
                    str(record["recording_session"]), []
                ).append(position)
        if not positions_by_session:
            raise ValueError("training partition must contain both vehicle classes")
        for positions in positions_by_session.values():
            value = 1.0 / (
                len(CLASS_NAMES) * len(positions_by_session) * len(positions)
            )
            weights[positions] = value
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise RuntimeError("session-balanced weights are nonfinite or nonpositive")
    return weights * (len(weights) / weights.sum())


def _fit_probe(
    features: np.ndarray,
    labels: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
    regularization_c: float,
    seed: int,
) -> tuple[Any, Any, dict[str, torch.Tensor]]:
    LogisticRegression, StandardScaler, _ = _require_sklearn()
    indices = np.asarray(sorted(int(index) for index in train_indices), dtype=np.int64)
    weights = session_balanced_sample_weights(records, indices)
    scaler = StandardScaler().fit(features[indices], sample_weight=weights)
    model = LogisticRegression(
        C=regularization_c,
        max_iter=5_000,
        solver="lbfgs",
        random_state=seed,
    ).fit(scaler.transform(features[indices]), labels[indices], sample_weight=weights)
    if int(model.n_iter_[0]) >= int(model.max_iter):
        raise RuntimeError("semantic logistic probe failed to converge")
    state = {
        "feature_mean": torch.from_numpy(np.asarray(scaler.mean_)).to(torch.float32),
        "feature_scale": torch.from_numpy(np.asarray(scaler.scale_)).to(torch.float32),
        "coefficient": torch.from_numpy(np.asarray(model.coef_)).to(torch.float32),
        "intercept": torch.from_numpy(np.asarray(model.intercept_)).to(torch.float32),
    }
    return scaler, model, state


def _evaluate_probe(
    features: np.ndarray,
    labels: np.ndarray,
    indices: Sequence[int],
    scaler: Any,
    model: Any,
) -> tuple[dict[str, Any], np.ndarray]:
    selected = np.asarray(sorted(int(index) for index in indices), dtype=np.int64)
    probabilities = model.predict_proba(scaler.transform(features[selected]))
    predictions = probabilities.argmax(axis=1)
    metrics = classification_metrics(
        torch.from_numpy(predictions),
        torch.from_numpy(labels[selected]),
        torch.zeros(len(selected)),
        torch.from_numpy(probabilities).to(torch.float32),
    )
    metrics.pop("per_snr")
    metrics["snr_available"] = False
    return metrics, probabilities


def _metric_summary(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "sample_standard_deviation": stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def nested_leave_session_pair_out(
    features: torch.Tensor,
    labels: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    *,
    c_values: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0),
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    """Run nested leave-one-tracked/one-wheeled-session-out evaluation."""

    if features.ndim != 2 or features.shape[0] != len(records):
        raise ValueError("features must have shape [records, feature_dimension]")
    if labels.shape != (len(records),):
        raise ValueError("labels must have one value per record")
    if not torch.isfinite(features).all():
        raise ValueError("features contain NaN or Inf")
    candidates = tuple(float(value) for value in c_values)
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("C candidates must be positive")
    if tuple(sorted(set(candidates))) != candidates:
        raise ValueError("C candidates must be unique and increasing")
    sessions = _sessions_by_class(records)
    all_indices = set(range(len(records)))
    feature_array = features.detach().cpu().numpy().astype(np.float64, copy=False)
    label_array = labels.detach().cpu().numpy().astype(np.int64, copy=False)
    expected_labels = np.asarray(
        [CLASS_NAMES.index(str(record["vehicle_class"])) for record in records]
    )
    if not np.array_equal(label_array, expected_labels):
        raise ValueError("feature-cache labels do not align with the real manifest")

    folds: list[dict[str, Any]] = []
    model_states: dict[str, dict[str, torch.Tensor]] = {}
    split_folds: list[dict[str, Any]] = []
    for tracked_session, tracked_indices in sessions["tracked"].items():
        for wheeled_session, wheeled_indices in sessions["wheeled"].items():
            outer_test = set(tracked_indices) | set(wheeled_indices)
            remaining = {
                "tracked": [
                    session
                    for session in sessions["tracked"]
                    if session != tracked_session
                ],
                "wheeled": [
                    session
                    for session in sessions["wheeled"]
                    if session != wheeled_session
                ],
            }
            candidate_results: list[dict[str, Any]] = []
            for regularization_c in candidates:
                inner_scores: list[dict[str, Any]] = []
                for inner_tracked in remaining["tracked"]:
                    for inner_wheeled in remaining["wheeled"]:
                        inner_validation = set(sessions["tracked"][inner_tracked]) | set(
                            sessions["wheeled"][inner_wheeled]
                        )
                        inner_train = all_indices - outer_test - inner_validation
                        scaler, model, _ = _fit_probe(
                            feature_array,
                            label_array,
                            records,
                            inner_train,
                            regularization_c,
                            seed,
                        )
                        metrics, _ = _evaluate_probe(
                            feature_array,
                            label_array,
                            inner_validation,
                            scaler,
                            model,
                        )
                        inner_scores.append(
                            {
                                "tracked_validation_session": inner_tracked,
                                "wheeled_validation_session": inner_wheeled,
                                "balanced_accuracy": metrics["balanced_accuracy"],
                            }
                        )
                scores = [float(row["balanced_accuracy"]) for row in inner_scores]
                candidate_results.append(
                    {
                        "regularization_c": regularization_c,
                        "mean_balanced_accuracy": mean(scores),
                        "median_balanced_accuracy": median(scores),
                        "inner_folds": inner_scores,
                    }
                )
            selected = max(
                candidate_results,
                key=lambda row: (
                    float(row["mean_balanced_accuracy"]),
                    float(row["median_balanced_accuracy"]),
                    -float(row["regularization_c"]),
                ),
            )
            outer_train = all_indices - outer_test
            scaler, model, state = _fit_probe(
                feature_array,
                label_array,
                records,
                outer_train,
                float(selected["regularization_c"]),
                seed,
            )
            metrics, probabilities = _evaluate_probe(
                feature_array,
                label_array,
                outer_test,
                scaler,
                model,
            )
            sorted_test = sorted(outer_test)
            session_predictions: dict[str, Any] = {}
            for vehicle_class, session in (
                ("tracked", tracked_session),
                ("wheeled", wheeled_session),
            ):
                positions = [
                    position
                    for position, record_index in enumerate(sorted_test)
                    if str(records[record_index]["recording_session"]) == session
                ]
                mean_probability = probabilities[positions].mean(axis=0)
                predicted_class = CLASS_NAMES[int(mean_probability.argmax())]
                session_predictions[session] = {
                    "vehicle_class": vehicle_class,
                    "predicted_class": predicted_class,
                    "correct": predicted_class == vehicle_class,
                    "mean_probability": {
                        name: float(mean_probability[class_index])
                        for class_index, name in enumerate(CLASS_NAMES)
                    },
                    "support": len(positions),
                }
            fold_id = hashlib.sha256(
                f"{tracked_session}:{wheeled_session}".encode("utf-8")
            ).hexdigest()[:16]
            model_states[fold_id] = state
            folds.append(
                {
                    "fold_id": fold_id,
                    "tracked_test_session": tracked_session,
                    "wheeled_test_session": wheeled_session,
                    "selected_regularization_c": selected["regularization_c"],
                    "inner_selection_mean_balanced_accuracy": selected[
                        "mean_balanced_accuracy"
                    ],
                    "candidate_selection": candidate_results,
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
                            not in (
                                tracked_session
                                if vehicle_class == "tracked"
                                else wheeled_session,
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
                }
            )

    aggregate = {
        "outer_fold_count": len(folds),
        "balanced_accuracy": _metric_summary(
            [float(fold["metrics"]["balanced_accuracy"]) for fold in folds]
        ),
        "macro_f1": _metric_summary(
            [float(fold["metrics"]["macro_f1"]) for fold in folds]
        ),
        "per_class_recall": {
            vehicle_class: _metric_summary(
                [
                    float(fold["metrics"]["per_class_recall"][vehicle_class])
                    for fold in folds
                ]
            )
            for vehicle_class in CLASS_NAMES
        },
        "both_heldout_sessions_correct_rate": mean(
            [
                float(
                    all(
                        prediction["correct"]
                        for prediction in fold["session_predictions"].values()
                    )
                )
                for fold in folds
            ]
        ),
    }
    split_payload = {
        "strategy": "nested_leave_one_tracked_and_one_wheeled_session_out",
        "group_field": "recording_session",
        "folds": split_folds,
    }
    return {"folds": folds, "aggregate": aggregate}, model_states, split_payload


def _load_semantic_features(
    cache_path: Path,
    records: Sequence[Mapping[str, Any]],
    real_manifest_sha256: str,
    panns_checkpoint_sha256: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    cached = torch.load(cache_path, map_location="cpu", weights_only=True)
    expected = {
        "pretrained_evaluation_version": PRETRAINED_EVALUATION_VERSION,
        "real_manifest_sha256": real_manifest_sha256,
        "panns_checkpoint_sha256": panns_checkpoint_sha256,
    }
    for key, value in expected.items():
        if cached.get(key) != value:
            raise ValueError(f"PANNs feature cache {key} mismatch")
    outputs = cached.get("real_clipwise_outputs")
    labels = cached.get("real_labels")
    if not isinstance(outputs, torch.Tensor) or outputs.shape != (
        len(records),
        PANN_AUDIOSET_DIMENSION,
    ):
        raise ValueError("PANNs cache has the wrong native-real AudioSet shape")
    if not isinstance(labels, torch.Tensor) or labels.shape != (len(records),):
        raise ValueError("PANNs cache has the wrong native-real label shape")
    indices = [index for index, _ in SEMANTIC_AUDIOSET_FEATURES]
    return outputs[:, indices].to(torch.float32), labels.to(torch.long)


def evaluate_semantic_sessions(
    real_manifest_path: str | Path,
    feature_cache_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    c_values: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0),
    seed: int = 42,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = Path(real_manifest_path)
    records, manifest_sha256 = _load_jsonl(manifest)
    checkpoint = validate_panns_checkpoint(checkpoint_path)
    features, labels = _load_semantic_features(
        Path(feature_cache_path),
        records,
        manifest_sha256,
        str(checkpoint["sha256"]),
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
        "semantic_session_evaluation_version": SEMANTIC_SESSION_EVALUATION_VERSION,
        "model": "frozen_panns_semantic_outputs_session_balanced_logistic_regression",
        "encoder_trainable": False,
        "semantic_audioset_features": [
            {"index": index, "name": name}
            for index, name in SEMANTIC_AUDIOSET_FEATURES
        ],
        "regularization_c_candidates": [float(value) for value in c_values],
        "selection_metric": "inner_pair_mean_balanced_accuracy",
        "sample_weighting": "equal_class_equal_session_equal_window_within_session",
        "seed": seed,
        "scikit_learn_version": sklearn_version,
    }
    results = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset_version": manifest_sha256,
        "real_manifest": str(manifest),
        "real_manifest_sha256": manifest_sha256,
        "feature_cache": str(feature_cache_path),
        "test_domain": "real -> real nested unseen recording-session pairs",
        "protocol_status": "complete",
        "pretrained_encoder": checkpoint,
        "training_config": training_config,
        **evaluation,
        "artifacts": {
            "checkpoint": "semantic_probe_models.pt",
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
        output / "semantic_probe_models.pt",
    )
    experiment = {
        "git_commit": results["git_commit"],
        "configuration": training_config,
        "random_seed": seed,
        "dataset_version": manifest_sha256,
        "train_test_split": "splits.json",
        "model_checkpoint": "semantic_probe_models.pt",
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
            "selected_regularization_c": fold["selected_regularization_c"],
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
    axis.set_title("Nested unseen-session semantic PANNs evaluation")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "outer_fold_balanced_accuracy.png", dpi=160)
    plt.close(figure)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regularization-c", type=float, action="append", dest="c_values")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = evaluate_semantic_sessions(
        args.real_manifest,
        args.feature_cache,
        args.checkpoint,
        args.output,
        c_values=(0.001, 0.01, 0.1, 1.0, 10.0) if args.c_values is None else args.c_values,
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

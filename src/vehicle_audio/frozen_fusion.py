"""Freeze a development-selected fusion model for a future locked real pair."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from vehicle_audio.baseline import CLASS_NAMES, FEATURE_IMPLEMENTATION_VERSION, FeatureConfig
from vehicle_audio.fusion_session_evaluation import (
    DEFAULT_REGULARIZATION_C,
    DEFAULT_WHEELED_THRESHOLDS,
    FUSION_SESSION_EVALUATION_VERSION,
    _fit_regularization_ensemble,
    _fused_probabilities,
    _threshold_metrics,
    _threshold_session_predictions,
    _validate_candidates,
    load_or_extract_classical_features,
)
from vehicle_audio.invariance_evaluation import _git_commit, _load_jsonl
from vehicle_audio.pretrained_evaluation import validate_panns_checkpoint
from vehicle_audio.semantic_session_evaluation import (
    SEMANTIC_AUDIOSET_FEATURES,
    _aggregate_folds,
    _load_semantic_features,
    _require_sklearn,
    _sessions_by_class,
)


FROZEN_FUSION_VERSION = 1


def _validate_feature_inputs(
    semantic_features: torch.Tensor,
    classical_features: torch.Tensor,
    labels: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    semantic_array = semantic_features.detach().cpu().numpy().astype(np.float64)
    classical_array = classical_features.detach().cpu().numpy().astype(np.float64)
    label_array = labels.detach().cpu().numpy().astype(np.int64)
    expected_labels = np.asarray(
        [CLASS_NAMES.index(str(record["vehicle_class"])) for record in records]
    )
    if not np.array_equal(label_array, expected_labels):
        raise ValueError("feature labels do not align with the manifest")
    return semantic_array, classical_array, label_array


def select_and_fit_frozen_fusion(
    semantic_features: torch.Tensor,
    classical_features: torch.Tensor,
    labels: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    *,
    c_values: Sequence[float] = DEFAULT_REGULARIZATION_C,
    threshold_values: Sequence[float] = DEFAULT_WHEELED_THRESHOLDS,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Select one global threshold by grouped CV, then fit on all development data."""

    regularization, thresholds = _validate_candidates(c_values, threshold_values)
    semantic_array, classical_array, label_array = _validate_feature_inputs(
        semantic_features,
        classical_features,
        labels,
        records,
    )
    sessions = _sessions_by_class(records)
    all_indices = set(range(len(records)))
    heldout_predictions: list[dict[str, Any]] = []
    split_folds: list[dict[str, Any]] = []
    for tracked_session, tracked_indices in sessions["tracked"].items():
        for wheeled_session, wheeled_indices in sessions["wheeled"].items():
            heldout = set(tracked_indices) | set(wheeled_indices)
            training = all_indices - heldout
            probabilities, _ = _fused_probabilities(
                semantic_array,
                classical_array,
                label_array,
                records,
                training,
                heldout,
                regularization,
                seed,
            )
            fold_id = f"{tracked_session}::{wheeled_session}"
            heldout_predictions.append(
                {
                    "fold_id": fold_id,
                    "tracked_test_session": tracked_session,
                    "wheeled_test_session": wheeled_session,
                    "test_indices": heldout,
                    "probabilities": probabilities,
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
                        str(records[index]["sample_id"]) for index in sorted(training)
                    ],
                    "test_sample_ids": [
                        str(records[index]["sample_id"]) for index in sorted(heldout)
                    ],
                }
            )

    candidates: list[dict[str, Any]] = []
    for threshold in thresholds:
        fold_rows: list[dict[str, Any]] = []
        for fold in heldout_predictions:
            metrics = _threshold_metrics(
                label_array,
                fold["test_indices"],
                fold["probabilities"],
                threshold,
            )
            fold_rows.append(
                {
                    "fold_id": fold["fold_id"],
                    "tracked_test_session": fold["tracked_test_session"],
                    "wheeled_test_session": fold["wheeled_test_session"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "tracked_recall": metrics["per_class_recall"]["tracked"],
                    "wheeled_recall": metrics["per_class_recall"]["wheeled"],
                }
            )
        scores = [float(row["balanced_accuracy"]) for row in fold_rows]
        candidates.append(
            {
                "wheeled_threshold": threshold,
                "mean_balanced_accuracy": mean(scores),
                "median_balanced_accuracy": median(scores),
                "minimum_balanced_accuracy": min(scores),
                "folds": fold_rows,
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            float(row["mean_balanced_accuracy"]),
            float(row["median_balanced_accuracy"]),
            float(row["minimum_balanced_accuracy"]),
            -abs(float(row["wheeled_threshold"]) - 0.5),
        ),
    )
    threshold = float(selected["wheeled_threshold"])
    selected_folds: list[dict[str, Any]] = []
    for fold in heldout_predictions:
        metrics = _threshold_metrics(
            label_array,
            fold["test_indices"],
            fold["probabilities"],
            threshold,
        )
        sorted_test = sorted(fold["test_indices"])
        selected_folds.append(
            {
                "fold_id": fold["fold_id"],
                "tracked_test_session": fold["tracked_test_session"],
                "wheeled_test_session": fold["wheeled_test_session"],
                "metrics": metrics,
                "session_predictions": _threshold_session_predictions(
                    fold["probabilities"],
                    sorted_test,
                    records,
                    fold["tracked_test_session"],
                    fold["wheeled_test_session"],
                    threshold,
                ),
            }
        )

    semantic_probes, semantic_states = _fit_regularization_ensemble(
        semantic_array,
        label_array,
        records,
        all_indices,
        regularization,
        seed,
    )
    classical_probes, classical_states = _fit_regularization_ensemble(
        classical_array,
        label_array,
        records,
        all_indices,
        regularization,
        seed,
    )
    del semantic_probes, classical_probes
    frozen_state = {
        "semantic_regularization_ensemble": semantic_states,
        "classical_regularization_ensemble": classical_states,
        "wheeled_threshold": torch.tensor(threshold, dtype=torch.float32),
    }
    development_selection = {
        "selection_protocol_status": "development_selection_only",
        "performance_claim": "not a locked or independent test",
        "selection_strategy": (
            "leave_one_tracked_and_one_wheeled_session_out_global_threshold"
        ),
        "selected_wheeled_threshold": threshold,
        "threshold_candidates": candidates,
        "selected_threshold_folds": selected_folds,
        "selected_threshold_aggregate": _aggregate_folds(
            selected_folds,
            metrics_field="metrics",
            session_predictions_field="session_predictions",
        ),
    }
    splits = {
        "strategy": "leave_one_tracked_and_one_wheeled_session_out_development_selection",
        "group_field": "recording_session",
        "folds": split_folds,
    }
    return development_selection, frozen_state, splits


def _fingerprints(records: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    fields = (
        "recording_session",
        "source_id",
        "original_media_source",
        "normalized_source_sha256",
        "raw_sha256",
    )
    return {
        field: sorted(
            {
                str(record[field])
                for record in records
                if record.get(field) not in (None, "")
            }
        )
        for field in fields
    }


def validate_locked_records(
    locked_records: Sequence[Mapping[str, Any]],
    development_fingerprints: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """Reject a purported locked corpus that overlaps development provenance."""

    sessions: dict[str, set[str]] = {name: set() for name in CLASS_NAMES}
    required_provenance = (
        "source_id",
        "original_media_source",
        "normalized_source_sha256",
    )
    for index, record in enumerate(locked_records):
        vehicle_class = str(record.get("vehicle_class"))
        recording_session = str(record.get("recording_session", ""))
        if vehicle_class not in sessions:
            raise ValueError(
                f"unexpected locked vehicle class at record {index}: {vehicle_class}"
            )
        if not recording_session:
            raise ValueError(f"locked record {index} has no recording_session")
        for field in required_provenance:
            if record.get(field) in (None, ""):
                raise ValueError(f"locked record {index} has no {field}")
        if record.get("provenance_complete") is not True:
            raise ValueError(f"locked record {index} has incomplete provenance")
        sessions[vehicle_class].add(recording_session)
    for vehicle_class, class_sessions in sessions.items():
        if len(class_sessions) != 1:
            raise ValueError(
                "locked confirmation requires exactly one session per class; "
                f"found {len(class_sessions)} for {vehicle_class}"
            )
    if sessions["tracked"] & sessions["wheeled"]:
        raise ValueError("locked recording_session cannot contain both classes")
    locked = _fingerprints(locked_records)
    for field, values in locked.items():
        overlap = set(values) & set(development_fingerprints.get(field, ()))
        if overlap:
            raise ValueError(
                f"locked corpus overlaps development {field}: {sorted(overlap)}"
            )
    return locked


def _probe_ensemble_probabilities(
    features: torch.Tensor,
    states: Mapping[str, Mapping[str, torch.Tensor]],
) -> torch.Tensor:
    if not states:
        raise ValueError("frozen probe ensemble is empty")
    feature_tensor = features.detach().cpu().to(torch.float32)
    probabilities: list[torch.Tensor] = []
    for state in states.values():
        feature_mean = state["feature_mean"].to(torch.float32)
        feature_scale = state["feature_scale"].to(torch.float32)
        coefficient = state["coefficient"].to(torch.float32)
        intercept = state["intercept"].to(torch.float32)
        if feature_mean.shape != (feature_tensor.shape[1],):
            raise ValueError("frozen probe feature dimension mismatch")
        logits = (
            ((feature_tensor - feature_mean) / feature_scale)
            @ coefficient.squeeze(0)
            + intercept.squeeze(0)
        )
        wheeled = torch.sigmoid(logits)
        probabilities.append(torch.stack((1.0 - wheeled, wheeled), dim=1))
    return torch.stack(probabilities).mean(dim=0)


def predict_frozen_fusion(
    semantic_features: torch.Tensor,
    classical_features: torch.Tensor,
    frozen_state: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the serialized equal-fusion probe ensembles to new features."""

    if semantic_features.shape[0] != classical_features.shape[0]:
        raise ValueError("semantic and classical feature rows do not align")
    semantic_probabilities = _probe_ensemble_probabilities(
        semantic_features,
        frozen_state["semantic_regularization_ensemble"],
    )
    classical_probabilities = _probe_ensemble_probabilities(
        classical_features,
        frozen_state["classical_regularization_ensemble"],
    )
    probabilities = 0.5 * (semantic_probabilities + classical_probabilities)
    threshold = float(frozen_state["wheeled_threshold"])
    predictions = (probabilities[:, 1] >= threshold).to(torch.long)
    return probabilities, predictions


def freeze_fusion_model(
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
    frozen_path = output / "frozen_fusion_model.pt"
    if frozen_path.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen model; choose a new output: {frozen_path}"
        )
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
    classical_cache = output / "development_classical_features.pt"
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
    selection, frozen_state, splits = select_and_fit_frozen_fusion(
        semantic_features,
        classical_features,
        semantic_labels,
        records,
        c_values=c_values,
        threshold_values=threshold_values,
        seed=seed,
    )
    _, _, sklearn_version = _require_sklearn()
    config = {
        "frozen_fusion_version": FROZEN_FUSION_VERSION,
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
        "selected_wheeled_threshold": selection["selected_wheeled_threshold"],
        "sample_weighting": "equal_class_equal_session_equal_window_within_session",
        "channel": channel,
        "seed": seed,
        "scikit_learn_version": sklearn_version,
    }
    git_commit = _git_commit()
    development_fingerprints = _fingerprints(records)
    torch.save(
        {
            "frozen_fusion_version": FROZEN_FUSION_VERSION,
            "state": frozen_state,
            "configuration": config,
            "class_names": list(CLASS_NAMES),
            "development_manifest_sha256": manifest_sha256,
            "development_fingerprints": development_fingerprints,
            "pretrained_encoder": checkpoint,
            "git_commit": git_commit,
        },
        frozen_path,
    )
    selection_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "dataset_version": manifest_sha256,
        "development_manifest": str(manifest),
        "development_manifest_sha256": manifest_sha256,
        "test_domain": "real -> real development selection; no locked test",
        "protocol_status": "frozen_awaiting_locked_evaluation",
        "configuration": config,
        "pretrained_encoder": checkpoint,
        "development_fingerprints": development_fingerprints,
        "locked_evaluation_requirements": {
            "sessions_per_class": 1,
            "required_provenance_fields": [
                "recording_session",
                "source_id",
                "original_media_source",
                "normalized_source_sha256",
                "provenance_complete",
            ],
            "must_not_overlap_development_fingerprints": True,
            "must_not_be_used_for_model_or_threshold_selection": True,
            "evaluation_policy": "single frozen-checkpoint evaluation",
        },
        **selection,
        "artifacts": {
            "frozen_model": "frozen_fusion_model.pt",
            "development_splits": "development_splits.json",
            "classical_feature_cache": "development_classical_features.pt",
        },
    }
    (output / "development_selection.json").write_text(
        json.dumps(selection_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "development_splits.json").write_text(
        json.dumps(splits, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    experiment = {
        "git_commit": git_commit,
        "configuration": config,
        "random_seed": seed,
        "dataset_version": manifest_sha256,
        "train_test_split": "development_splits.json",
        "model_checkpoint": "frozen_fusion_model.pt",
        "metrics": "development_selection.json",
        "test_domain": selection_payload["test_domain"],
        "protocol_status": selection_payload["protocol_status"],
    }
    (output / "experiment.json").write_text(
        json.dumps(experiment, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return selection_payload


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
    result = freeze_fusion_model(
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
                "protocol_status": result["protocol_status"],
                "performance_claim": result["performance_claim"],
                "selected_wheeled_threshold": result[
                    "selected_wheeled_threshold"
                ],
                "development_balanced_accuracy": result[
                    "selected_threshold_aggregate"
                ]["balanced_accuracy"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

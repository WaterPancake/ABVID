"""Evaluate a frozen fusion model once on a fresh locked real-session pair."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from vehicle_audio.baseline import (
    CLASS_NAMES,
    FEATURE_IMPLEMENTATION_VERSION,
    FeatureConfig,
)
from vehicle_audio.frozen_fusion import (
    FROZEN_FUSION_VERSION,
    predict_frozen_fusion,
    validate_locked_records,
)
from vehicle_audio.fusion_session_evaluation import (
    _threshold_metrics,
    _threshold_session_predictions,
    load_or_extract_classical_features,
)
from vehicle_audio.invariance_evaluation import _git_commit, _load_jsonl
from vehicle_audio.pretrained_evaluation import (
    PANN_AUDIOSET_DIMENSION,
    PANN_SAMPLE_RATE,
    PRETRAINED_EVALUATION_VERSION,
    _extract_panns_features,
    validate_panns_checkpoint,
)
from vehicle_audio.semantic_session_evaluation import SEMANTIC_AUDIOSET_FEATURES


LOCKED_FUSION_EVALUATION_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_or_extract_locked_semantic_features(
    records: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    manifest_sha256: str,
    checkpoint: Mapping[str, Any],
    cache_path: Path,
    *,
    channel: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the same fixed semantic PANN outputs used during development."""

    sample_rates = {int(record["sample_rate"]) for record in records}
    if len(sample_rates) != 1:
        raise ValueError(
            f"locked records must use one sample rate, found {sorted(sample_rates)}"
        )
    sample_rate = next(iter(sample_rates))
    expected = {
        "locked_fusion_evaluation_version": LOCKED_FUSION_EVALUATION_VERSION,
        "pretrained_evaluation_version": PRETRAINED_EVALUATION_VERSION,
        "manifest_sha256": manifest_sha256,
        "panns_checkpoint_sha256": checkpoint["sha256"],
        "source_sample_rate": sample_rate,
        "panns_sample_rate": PANN_SAMPLE_RATE,
        "channel": channel,
    }
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if not all(cached.get(key) == value for key, value in expected.items()):
            raise ValueError("locked semantic feature cache metadata mismatch")
        outputs = cached.get("clipwise_outputs")
        labels = cached.get("labels")
        if not isinstance(outputs, torch.Tensor) or outputs.shape != (
            len(records),
            PANN_AUDIOSET_DIMENSION,
        ):
            raise ValueError("locked semantic cache has the wrong output shape")
        if not isinstance(labels, torch.Tensor) or labels.shape != (len(records),):
            raise ValueError("locked semantic cache has the wrong label shape")
    else:
        try:
            from panns_inference import AudioTagging
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "locked PANNs extraction requires `uv sync --extra pretrained`"
            ) from error
        tagging = AudioTagging(checkpoint_path=checkpoint["path"], device="cpu")
        _, outputs = _extract_panns_features(
            records,
            manifest_path,
            "corrupted_path",
            tagging,
            channel=channel,
            sample_rate=sample_rate,
            batch_size=batch_size,
            label="locked real",
        )
        labels = torch.tensor(
            [CLASS_NAMES.index(str(record["vehicle_class"])) for record in records],
            dtype=torch.long,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"clipwise_outputs": outputs, "labels": labels, **expected},
            cache_path,
        )
    indices = [index for index, _ in SEMANTIC_AUDIOSET_FEATURES]
    return outputs[:, indices].to(torch.float32), labels.to(torch.long)


def summarize_locked_predictions(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    wheeled_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor]:
    """Compute window and recording-level results for exactly one session pair."""

    if probabilities.shape != (len(records), len(CLASS_NAMES)):
        raise ValueError("locked probabilities have the wrong shape")
    if labels.shape != (len(records),):
        raise ValueError("locked labels have the wrong shape")
    sessions: dict[str, set[str]] = {name: set() for name in CLASS_NAMES}
    for record in records:
        sessions[str(record["vehicle_class"])].add(str(record["recording_session"]))
    if any(len(values) != 1 for values in sessions.values()):
        raise ValueError("locked summary requires exactly one session per class")
    tracked_session = next(iter(sessions["tracked"]))
    wheeled_session = next(iter(sessions["wheeled"]))
    indices = tuple(range(len(records)))
    probability_array = probabilities.detach().cpu().numpy()
    metrics = _threshold_metrics(
        labels.detach().cpu().numpy(),
        indices,
        probability_array,
        wheeled_threshold,
    )
    session_predictions = _threshold_session_predictions(
        probability_array,
        indices,
        records,
        tracked_session,
        wheeled_session,
        wheeled_threshold,
    )
    predictions = (probabilities[:, 1] >= wheeled_threshold).to(torch.long)
    return metrics, session_predictions, predictions


def evaluate_locked_fusion(
    locked_manifest_path: str | Path,
    frozen_model_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    channel: int = 0,
    extraction_batch_size: int = 32,
) -> dict[str, Any]:
    output = Path(output_dir)
    metrics_path = output / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(
            "refusing to overwrite a locked evaluation; choose a new output only "
            "for a genuinely different frozen model/locked manifest"
        )
    output.mkdir(parents=True, exist_ok=True)
    manifest = Path(locked_manifest_path)
    records, manifest_sha256 = _load_jsonl(manifest)
    frozen_path = Path(frozen_model_path)
    frozen = torch.load(frozen_path, map_location="cpu", weights_only=True)
    if frozen.get("frozen_fusion_version") != FROZEN_FUSION_VERSION:
        raise ValueError("unsupported frozen fusion checkpoint version")
    if frozen.get("class_names") != list(CLASS_NAMES):
        raise ValueError("frozen fusion class names do not match")
    locked_fingerprints = validate_locked_records(
        records,
        frozen["development_fingerprints"],
    )
    if manifest_sha256 == frozen["development_manifest_sha256"]:
        raise ValueError("locked manifest is identical to the development manifest")
    checkpoint = validate_panns_checkpoint(checkpoint_path)
    if checkpoint["sha256"] != frozen["pretrained_encoder"]["sha256"]:
        raise ValueError("locked PANNs checkpoint differs from the frozen encoder")
    configuration = frozen["configuration"]
    expected_semantic_features = [
        {"index": index, "name": name} for index, name in SEMANTIC_AUDIOSET_FEATURES
    ]
    if configuration["semantic_audioset_features"] != expected_semantic_features:
        raise ValueError("frozen semantic feature definition differs from this evaluator")
    if configuration["classical_feature_config"] != asdict(FeatureConfig()):
        raise ValueError("frozen classical feature definition differs from this evaluator")
    if (
        int(configuration["classical_feature_implementation_version"])
        != FEATURE_IMPLEMENTATION_VERSION
    ):
        raise ValueError("frozen classical feature implementation version differs")
    if int(configuration["channel"]) != channel:
        raise ValueError("locked channel differs from the frozen model channel")
    semantic_features, semantic_labels = load_or_extract_locked_semantic_features(
        records,
        manifest,
        manifest_sha256,
        checkpoint,
        output / "locked_panns_features.pt",
        channel=channel,
        batch_size=extraction_batch_size,
    )
    classical_features, classical_labels = load_or_extract_classical_features(
        records,
        manifest,
        manifest_sha256,
        output / "locked_classical_features.pt",
        channel=channel,
        batch_size=extraction_batch_size,
    )
    if not torch.equal(semantic_labels, classical_labels):
        raise ValueError("locked semantic and classical labels do not align")
    probabilities, predictions = predict_frozen_fusion(
        semantic_features,
        classical_features,
        frozen["state"],
    )
    threshold = float(frozen["state"]["wheeled_threshold"])
    metrics, session_predictions, summarized_predictions = summarize_locked_predictions(
        probabilities,
        semantic_labels,
        records,
        threshold,
    )
    if not torch.equal(predictions, summarized_predictions):
        raise RuntimeError("locked prediction implementations disagree")
    results = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "locked_fusion_evaluation_version": LOCKED_FUSION_EVALUATION_VERSION,
        "dataset_version": manifest_sha256,
        "locked_manifest": str(manifest),
        "locked_manifest_sha256": manifest_sha256,
        "locked_fingerprints": locked_fingerprints,
        "development_manifest_sha256": frozen["development_manifest_sha256"],
        "frozen_model": str(frozen_path),
        "frozen_model_sha256": _sha256(frozen_path),
        "frozen_model_git_commit": frozen["git_commit"],
        "frozen_configuration": configuration,
        "test_domain": "real development -> fresh locked real session pair",
        "protocol_status": "complete",
        "pretrained_encoder": checkpoint,
        "metrics": metrics,
        "session_predictions": session_predictions,
        "artifacts": {
            "panns_feature_cache": "locked_panns_features.pt",
            "classical_feature_cache": "locked_classical_features.pt",
            "split": "locked_split.json",
            "window_predictions": "window_predictions.csv",
        },
    }
    locked_split = {
        "strategy": "fresh_locked_pair_disjoint_from_frozen_development_corpus",
        "development_manifest_sha256": frozen["development_manifest_sha256"],
        "locked_manifest_sha256": manifest_sha256,
        "locked_fingerprints": locked_fingerprints,
        "test_sessions": {
            vehicle_class: sorted(
                {
                    str(record["recording_session"])
                    for record in records
                    if str(record["vehicle_class"]) == vehicle_class
                }
            )
            for vehicle_class in CLASS_NAMES
        },
        "test_sample_ids": [str(record["sample_id"]) for record in records],
        "overlap_audit": {
            "recording_session": 0,
            "source_id": 0,
            "original_media_source": 0,
            "normalized_source_sha256": 0,
            "raw_sha256": 0,
        },
    }
    (output / "locked_split.json").write_text(
        json.dumps(locked_split, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    experiment = {
        "git_commit": results["git_commit"],
        "configuration": configuration,
        "random_seed": configuration["seed"],
        "dataset_version": manifest_sha256,
        "train_test_split": "locked_split.json",
        "model_checkpoint": str(frozen_path),
        "model_checkpoint_sha256": results["frozen_model_sha256"],
        "metrics": "metrics.json",
        "test_domain": results["test_domain"],
        "protocol_status": results["protocol_status"],
    }
    (output / "experiment.json").write_text(
        json.dumps(experiment, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output / "window_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_id",
                "recording_session",
                "target_class",
                "predicted_class",
                "tracked_probability",
                "wheeled_probability",
                "wheeled_threshold",
            ),
        )
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "recording_session": record["recording_session"],
                    "target_class": record["vehicle_class"],
                    "predicted_class": CLASS_NAMES[int(predictions[index])],
                    "tracked_probability": float(probabilities[index, 0]),
                    "wheeled_probability": float(probabilities[index, 1]),
                    "wheeled_threshold": threshold,
                }
            )
    metrics_path.write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-manifest", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--extraction-batch-size", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = evaluate_locked_fusion(
        args.locked_manifest,
        args.frozen_model,
        args.checkpoint,
        args.output,
        channel=args.channel,
        extraction_batch_size=args.extraction_batch_size,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output),
                "protocol_status": results["protocol_status"],
                "test_domain": results["test_domain"],
                "balanced_accuracy": results["metrics"]["balanced_accuracy"],
                "per_class_recall": results["metrics"]["per_class_recall"],
                "session_predictions": results["session_predictions"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

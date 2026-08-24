"""Post-hoc evaluation of frozen historical checkpoints on one real corpus.

This module is deliberately evaluation-only: it does not train models, tune
thresholds, or choose a checkpoint based on the audit corpus.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import soundfile as sf
import torch
import yaml

from vehicle_audio.baseline import (
    CLASS_NAMES,
    FEATURE_IMPLEMENTATION_VERSION,
    ClassicalFeatureExtractor,
    FeatureConfig,
    LogMelExtractor,
    LogisticRegressionBaseline,
    SmallAudioCNN,
    classification_metrics,
)
from vehicle_audio.invariance_evaluation import _git_commit, _model_for_method
from vehicle_audio.pretrained_evaluation import (
    StandardizedLinearProbe,
    _extract_panns_features,
    validate_panns_checkpoint,
)


HISTORICAL_CHECKPOINT_AUDIT_VERSION = 1
SUPPORTED_KINDS = {
    "standalone",
    "classical_bundle",
    "cnn_bundle",
    "invariance_bundle",
    "panns_bundle",
    "panns_adaptation_bundle",
    "preserved_result",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_audit_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the predeclared checkpoint roster."""

    config_path = Path(path)
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("audit config must contain a mapping")
    if value.get("audit_version") != HISTORICAL_CHECKPOINT_AUDIT_VERSION:
        raise ValueError("unsupported historical checkpoint audit version")
    if not isinstance(value.get("selection_policy"), str):
        raise ValueError("audit config must document its selection_policy")
    models = value.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("audit config must declare at least one model")
    identifiers: set[str] = set()
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            raise ValueError(f"model entry {index} must be a mapping")
        required = {"id", "milestone", "architecture", "kind", "training_domain"}
        missing = sorted(required - set(model))
        if missing:
            raise ValueError(f"model entry {index} is missing {missing}")
        identifier = str(model["id"])
        if identifier in identifiers:
            raise ValueError(f"duplicate model id: {identifier}")
        identifiers.add(identifier)
        if model["kind"] not in SUPPORTED_KINDS:
            raise ValueError(f"unsupported model kind: {model['kind']}")
        for field in ("checkpoint",):
            if field not in model:
                raise ValueError(f"model {identifier} is missing {field}")
        if model["kind"] in {
            "classical_bundle",
            "cnn_bundle",
            "invariance_bundle",
            "panns_bundle",
            "panns_adaptation_bundle",
        } and "state_key" not in model:
            raise ValueError(f"model {identifier} is missing state_key")
        if model["kind"] == "preserved_result":
            if "metrics" not in model or "predictions" not in model:
                raise ValueError(
                    f"preserved result {identifier} needs metrics and predictions"
                )
    return value


def _load_manifest(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = path.read_bytes()
    records = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records, hashlib.sha256(payload).hexdigest()


def validate_audit_manifest(
    records: Sequence[Mapping[str, Any]], manifest: Path, channel: int
) -> dict[str, Any]:
    """Require reviewed, native-real, group-consistent fixed-length audio."""

    session_classes: dict[str, str] = {}
    sample_ids: set[str] = set()
    sample_rates: set[int] = set()
    sample_counts: set[int] = set()
    class_counts = {name: 0 for name in CLASS_NAMES}
    root = manifest.parent
    for index, record in enumerate(records):
        for field in (
            "sample_id",
            "audio_path",
            "vehicle_class",
            "recording_session",
            "sample_rate",
            "num_samples",
        ):
            if field not in record:
                raise ValueError(f"manifest record {index} has no {field}")
        sample_id = str(record["sample_id"])
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        vehicle_class = str(record["vehicle_class"])
        if vehicle_class not in CLASS_NAMES:
            raise ValueError(f"unsupported vehicle_class: {vehicle_class}")
        class_counts[vehicle_class] += 1
        session = str(record["recording_session"])
        previous_class = session_classes.setdefault(session, vehicle_class)
        if previous_class != vehicle_class:
            raise ValueError(f"recording session spans classes: {session}")
        if record.get("source_domain") != "real_recording":
            raise ValueError("audit manifest must contain native real recordings")
        if record.get("content_review_status") != "reviewed_segment":
            raise ValueError("every audit segment must have completed content review")
        if not bool(record.get("provenance_complete")):
            raise ValueError("every audit segment must have complete provenance")
        sample_rate = int(record["sample_rate"])
        num_samples = int(record["num_samples"])
        sample_rates.add(sample_rate)
        sample_counts.add(num_samples)
        audio_path = root / str(record["audio_path"])
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        info = sf.info(audio_path)
        if info.samplerate != sample_rate or info.frames != num_samples:
            raise ValueError(f"manifest/audio shape mismatch: {audio_path}")
        if not 0 <= channel < info.channels:
            raise ValueError(
                f"requested channel {channel} but {audio_path} has {info.channels}"
            )
    if any(count == 0 for count in class_counts.values()):
        raise ValueError("audit manifest must contain both tracked and wheeled samples")
    if len(sample_rates) != 1 or len(sample_counts) != 1:
        raise ValueError("audit clips must have one sample rate and fixed duration")
    return {
        "num_windows": len(records),
        "class_counts": class_counts,
        "sessions": [
            {"recording_session": session, "vehicle_class": vehicle_class}
            for session, vehicle_class in sorted(session_classes.items())
        ],
        "sample_rate": next(iter(sample_rates)),
        "num_samples": next(iter(sample_counts)),
        "channel": channel,
    }


def summarize_probabilities(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor]:
    """Apply the native argmax decision and summarize windows and sessions."""

    if probabilities.shape != (len(records), len(CLASS_NAMES)):
        raise ValueError("probabilities have the wrong shape")
    if labels.shape != (len(records),):
        raise ValueError("labels have the wrong shape")
    if not torch.isfinite(probabilities).all():
        raise ValueError("probabilities contain NaN or Inf")
    predictions = probabilities.argmax(dim=1)
    metrics = classification_metrics(
        predictions,
        labels,
        torch.zeros(len(records)),
        probabilities,
    )
    metrics.pop("per_snr")
    metrics["snr_available"] = False
    metrics["decision_rule"] = "argmax (equivalent to wheeled probability >= 0.5)"
    session_predictions: dict[str, Any] = {}
    sessions = sorted({str(record["recording_session"]) for record in records})
    for session in sessions:
        indices = [
            index
            for index, record in enumerate(records)
            if str(record["recording_session"]) == session
        ]
        target_classes = {str(records[index]["vehicle_class"]) for index in indices}
        if len(target_classes) != 1:
            raise ValueError(f"recording session spans classes: {session}")
        target_class = next(iter(target_classes))
        mean_probability = probabilities[indices].mean(dim=0)
        predicted_class = CLASS_NAMES[int(mean_probability.argmax())]
        session_predictions[session] = {
            "vehicle_class": target_class,
            "predicted_class": predicted_class,
            "correct": predicted_class == target_class,
            "mean_probability": {
                name: float(mean_probability[index])
                for index, name in enumerate(CLASS_NAMES)
            },
            "support": len(indices),
            "decision_rule": "argmax",
        }
    return metrics, session_predictions, predictions


def _feature_config(checkpoint: Mapping[str, Any]) -> FeatureConfig:
    mapping = checkpoint.get("feature_config")
    if mapping is None:
        mapping = checkpoint.get("training_config", {}).get("feature_config")
    if not isinstance(mapping, Mapping):
        raise ValueError("checkpoint has no feature configuration")
    return FeatureConfig(**{field: mapping[field] for field in asdict(FeatureConfig())})


def _read_waveforms(
    records: Sequence[Mapping[str, Any]], manifest: Path, channel: int, sample_rate: int
) -> tuple[torch.Tensor, torch.Tensor]:
    waveforms: list[torch.Tensor] = []
    labels: list[int] = []
    root = manifest.parent
    for record in records:
        samples, observed_rate = sf.read(
            root / str(record["audio_path"]), dtype="float32", always_2d=True
        )
        if observed_rate != sample_rate:
            raise ValueError("feature sample rate does not match audit audio")
        waveforms.append(torch.from_numpy(samples[:, channel].copy()))
        labels.append(CLASS_NAMES.index(str(record["vehicle_class"])))
    lengths = {len(waveform) for waveform in waveforms}
    if len(lengths) != 1:
        raise ValueError("audit waveforms do not have equal length")
    return torch.stack(waveforms), torch.tensor(labels, dtype=torch.long)


def _extract_native_features(
    records: Sequence[Mapping[str, Any]],
    manifest: Path,
    output: Path,
    config: FeatureConfig,
    manifest_sha256: str,
    *,
    channel: int,
    batch_size: int,
) -> dict[str, Any]:
    cache_path = output / "native_features.pt"
    expected = {
        "historical_checkpoint_audit_version": HISTORICAL_CHECKPOINT_AUDIT_VERSION,
        "feature_implementation_version": FEATURE_IMPLEMENTATION_VERSION,
        "feature_config": asdict(config),
        "manifest_sha256": manifest_sha256,
        "channel": channel,
    }
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if all(cached.get(key) == value for key, value in expected.items()):
            return cached
    waveforms, labels = _read_waveforms(
        records, manifest, channel, config.sample_rate
    )
    classical_extractor = ClassicalFeatureExtractor(config).eval()
    log_mel_extractor = LogMelExtractor(config).eval()
    classical_batches: list[torch.Tensor] = []
    log_mel_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(waveforms), batch_size):
            batch = waveforms[start : start + batch_size]
            classical_batches.append(classical_extractor(batch).cpu())
            log_mel_batches.append(log_mel_extractor(batch).cpu())
    result = {
        **expected,
        "classical_features": torch.cat(classical_batches),
        "log_mel_features": torch.cat(log_mel_batches),
        "labels": labels,
    }
    if not torch.isfinite(result["classical_features"]).all() or not torch.isfinite(
        result["log_mel_features"]
    ).all():
        raise RuntimeError("native feature extraction produced NaN or Inf")
    torch.save(result, cache_path)
    return result


def _extract_panns_cache(
    records: Sequence[Mapping[str, Any]],
    manifest: Path,
    output: Path,
    manifest_sha256: str,
    panns_checkpoint: Path,
    *,
    channel: int,
    sample_rate: int,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_metadata = validate_panns_checkpoint(panns_checkpoint)
    cache_path = output / "panns_features.pt"
    expected = {
        "historical_checkpoint_audit_version": HISTORICAL_CHECKPOINT_AUDIT_VERSION,
        "manifest_sha256": manifest_sha256,
        "panns_checkpoint_sha256": checkpoint_metadata["sha256"],
        "source_sample_rate": sample_rate,
        "channel": channel,
    }
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if all(cached.get(key) == value for key, value in expected.items()):
            return cached, checkpoint_metadata
    try:
        from panns_inference import AudioTagging
    except ImportError as error:
        raise RuntimeError(
            "PANNs audit requires `uv sync --extra pretrained`"
        ) from error
    tagging = AudioTagging(checkpoint_path=str(panns_checkpoint), device="cpu")
    embeddings, clipwise_outputs = _extract_panns_features(
        records,
        manifest,
        "audio_path",
        tagging,
        channel=channel,
        sample_rate=sample_rate,
        batch_size=batch_size,
        label="historical audit",
    )
    result = {
        **expected,
        "embeddings": embeddings,
        "clipwise_outputs": clipwise_outputs,
    }
    torch.save(result, cache_path)
    return result, checkpoint_metadata


def _checkpoint_provenance(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "class_names",
        "manifest_sha256",
        "synthetic_manifest_sha256",
        "real_manifest_sha256",
        "training_config",
        "pretrained_encoder",
        "best_epoch",
    )
    return {field: checkpoint[field] for field in fields if field in checkpoint}


def _torch_probabilities(
    specification: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    native_features: Mapping[str, Any],
) -> torch.Tensor:
    kind = str(specification["kind"])
    if kind == "standalone":
        model_kind = str(checkpoint["model_kind"])
        state = checkpoint["model_state_dict"]
        if model_kind == "classical":
            features = native_features["classical_features"]
            normalization = checkpoint["feature_normalization"]
            features = (features - normalization["mean"]) / normalization[
                "std"
            ].clamp_min(1e-6)
            model = LogisticRegressionBaseline(features.shape[1], len(CLASS_NAMES))
        elif model_kind == "cnn":
            features = native_features["log_mel_features"]
            model = SmallAudioCNN(len(CLASS_NAMES))
        else:
            raise ValueError(f"unsupported standalone model kind: {model_kind}")
    elif kind == "classical_bundle":
        bundle = checkpoint["models"][str(specification["state_key"])]
        features = native_features["classical_features"]
        features = (features - bundle["feature_mean"]) / bundle[
            "feature_std"
        ].clamp_min(1e-6)
        model = LogisticRegressionBaseline(features.shape[1], len(CLASS_NAMES))
        state = bundle["model_state_dict"]
    elif kind == "cnn_bundle":
        features = native_features["log_mel_features"]
        model = SmallAudioCNN(len(CLASS_NAMES))
        state = checkpoint["models"][str(specification["state_key"])]
    elif kind == "invariance_bundle":
        features = native_features["log_mel_features"]
        method = str(specification["state_key"])
        model = _model_for_method(method)
        state = checkpoint["models"][method]
    else:
        raise ValueError(f"not a native-feature checkpoint: {kind}")
    model.load_state_dict(state)
    model.eval()
    with torch.inference_mode():
        return torch.softmax(model(features), dim=1).cpu()


def _panns_probabilities(
    specification: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    panns_features: Mapping[str, Any],
) -> torch.Tensor:
    feature_space = str(specification["feature_space"])
    features = (
        panns_features["embeddings"]
        if feature_space == "embedding"
        else panns_features["clipwise_outputs"]
    )
    collection = (
        checkpoint["real_adaptation_models"]
        if specification["kind"] == "panns_adaptation_bundle"
        else checkpoint["models"]
    )
    state = collection[str(specification["state_key"])]
    model = StandardizedLinearProbe(
        state["feature_mean"], state["feature_standard_deviation"], len(CLASS_NAMES)
    )
    model.load_state_dict(state)
    model.eval()
    with torch.inference_mode():
        return torch.softmax(model(features), dim=1).cpu()


def _preserved_result(
    specification: Mapping[str, Any],
    repo_root: Path,
    records: Sequence[Mapping[str, Any]],
    manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor, torch.Tensor, dict[str, Any]]:
    metrics_path = repo_root / str(specification["metrics"])
    predictions_path = repo_root / str(specification["predictions"])
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if payload.get("locked_manifest_sha256") != manifest_sha256:
        raise ValueError("preserved result does not match the audit manifest")
    rows_by_id: dict[str, dict[str, str]] = {}
    with predictions_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows_by_id[str(row["sample_id"])] = row
    if set(rows_by_id) != {str(record["sample_id"]) for record in records}:
        raise ValueError("preserved predictions do not align with the audit manifest")
    probabilities = torch.tensor(
        [
            [
                float(rows_by_id[str(record["sample_id"])]["tracked_probability"]),
                float(rows_by_id[str(record["sample_id"])]["wheeled_probability"]),
            ]
            for record in records
        ],
        dtype=torch.float32,
    )
    predictions = torch.tensor(
        [
            CLASS_NAMES.index(
                rows_by_id[str(record["sample_id"])]["predicted_class"]
            )
            for record in records
        ],
        dtype=torch.long,
    )
    metadata = {
        key: payload[key]
        for key in (
            "created_at_utc",
            "frozen_model_git_commit",
            "frozen_model_sha256",
            "frozen_configuration",
            "protocol_status",
            "test_domain",
        )
        if key in payload
    }
    return (
        payload["metrics"],
        payload["session_predictions"],
        predictions,
        probabilities,
        metadata,
    )


def _comparison_row(result: Mapping[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    sessions = result["session_predictions"]
    return {
        "model_id": result["model_id"],
        "milestone": result["milestone"],
        "architecture": result["architecture"],
        "training_domain": result["training_domain"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "tracked_recall": metrics["per_class_recall"]["tracked"],
        "wheeled_recall": metrics["per_class_recall"]["wheeled"],
        "sessions_correct": sum(bool(value["correct"]) for value in sessions.values()),
        "session_count": len(sessions),
    }


def evaluate_historical_checkpoints(
    config_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    repo_root: str | Path = ".",
    panns_checkpoint: str | Path = ".artifacts/models/panns/Cnn14_mAP=0.431.pth",
    batch_size: int = 16,
) -> dict[str, Any]:
    """Evaluate the complete predeclared roster without changing any model."""

    config_path = Path(config_path)
    manifest_path = Path(manifest_path)
    output = Path(output_path)
    root = Path(repo_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite historical audit: {output}")
    output.mkdir(parents=True)
    config = load_audit_config(config_path)
    records, manifest_sha256 = _load_manifest(manifest_path)
    manifest_summary = validate_audit_manifest(
        records, manifest_path, int(config["channel"])
    )

    loaded_checkpoints: dict[Path, dict[str, Any]] = {}
    native_specs = [
        specification
        for specification in config["models"]
        if specification["kind"]
        in {"standalone", "classical_bundle", "cnn_bundle", "invariance_bundle"}
    ]
    feature_configs: set[str] = set()
    parsed_feature_config: FeatureConfig | None = None
    for specification in native_specs:
        path = root / str(specification["checkpoint"])
        checkpoint = loaded_checkpoints.setdefault(
            path, torch.load(path, map_location="cpu", weights_only=True)
        )
        current = _feature_config(checkpoint)
        feature_configs.add(json.dumps(asdict(current), sort_keys=True))
        parsed_feature_config = current
    if len(feature_configs) != 1 or parsed_feature_config is None:
        raise ValueError("native checkpoints must use one shared feature configuration")
    if parsed_feature_config.sample_rate != manifest_summary["sample_rate"]:
        raise ValueError("checkpoint and audit sample rates differ")
    native_features = _extract_native_features(
        records,
        manifest_path,
        output,
        parsed_feature_config,
        manifest_sha256,
        channel=int(config["channel"]),
        batch_size=batch_size,
    )
    labels = native_features["labels"]

    panns_specs = [
        specification
        for specification in config["models"]
        if specification["kind"] in {"panns_bundle", "panns_adaptation_bundle"}
    ]
    panns_features: dict[str, Any] | None = None
    panns_metadata: dict[str, Any] | None = None
    if panns_specs:
        panns_features, panns_metadata = _extract_panns_cache(
            records,
            manifest_path,
            output,
            manifest_sha256,
            root / Path(panns_checkpoint),
            channel=int(config["channel"]),
            sample_rate=manifest_summary["sample_rate"],
            batch_size=batch_size,
        )

    results: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for specification in config["models"]:
        checkpoint_path = root / str(specification["checkpoint"])
        kind = str(specification["kind"])
        if kind == "preserved_result":
            metrics, sessions, predictions, probabilities, checkpoint_metadata = (
                _preserved_result(
                    specification, root, records, manifest_sha256
                )
            )
        else:
            checkpoint = loaded_checkpoints.setdefault(
                checkpoint_path,
                torch.load(checkpoint_path, map_location="cpu", weights_only=True),
            )
            checkpoint_metadata = _checkpoint_provenance(checkpoint)
            if kind in {"panns_bundle", "panns_adaptation_bundle"}:
                if panns_features is None:
                    raise RuntimeError("PANNs features were not extracted")
                probabilities = _panns_probabilities(
                    specification, checkpoint, panns_features
                )
            else:
                probabilities = _torch_probabilities(
                    specification, checkpoint, native_features
                )
            metrics, sessions, predictions = summarize_probabilities(
                probabilities, labels, records
            )
        result = {
            "model_id": specification["id"],
            "milestone": int(specification["milestone"]),
            "architecture": specification["architecture"],
            "kind": kind,
            "state_key": specification.get("state_key"),
            "training_domain": specification["training_domain"],
            "checkpoint": str(specification["checkpoint"]),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_metadata": checkpoint_metadata,
            "metrics": metrics,
            "session_predictions": sessions,
        }
        results.append(result)
        for index, record in enumerate(records):
            prediction_rows.append(
                {
                    "model_id": specification["id"],
                    "milestone": specification["milestone"],
                    "sample_id": record["sample_id"],
                    "recording_session": record["recording_session"],
                    "target_class": record["vehicle_class"],
                    "predicted_class": CLASS_NAMES[int(predictions[index])],
                    "tracked_probability": float(probabilities[index, 0]),
                    "wheeled_probability": float(probabilities[index, 1]),
                }
            )

    comparison = [_comparison_row(result) for result in results]
    payload = {
        "historical_checkpoint_audit_version": HISTORICAL_CHECKPOINT_AUDIT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "seed": int(config["seed"]),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "selection_policy": config["selection_policy"],
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "dataset_version": manifest_sha256,
        "manifest_summary": manifest_summary,
        "test_domain": "historical frozen checkpoint -> reviewed native real pair",
        "protocol_status": "post_hoc_diagnostic_not_independent_locked_test",
        "limitations": [
            "The corpus was already opened for the frozen Milestone 6 evaluation.",
            "The two classes contain one recording session each, so session-level support is two.",
            "The tracked class is one historical Sherman and the wheeled class is one passenger car.",
            "No model, threshold, or preprocessing choice may be tuned on these audit results.",
            "Milestone 4 multichannel localization and beamforming are not evaluable from mono audio.",
        ],
        "feature_config": asdict(parsed_feature_config),
        "feature_implementation_version": FEATURE_IMPLEMENTATION_VERSION,
        "pretrained_encoder": panns_metadata,
        "results": results,
        "comparison": comparison,
        "artifacts": {
            "comparison": "comparison.csv",
            "window_predictions": "window_predictions.csv",
            "native_feature_cache": "native_features.pt",
            "panns_feature_cache": "panns_features.pt" if panns_specs else None,
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    with (output / "window_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit frozen Milestone 2-6 checkpoints on one reviewed real corpus"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path("."), type=Path)
    parser.add_argument(
        "--panns-checkpoint",
        default=Path(".artifacts/models/panns/Cnn14_mAP=0.431.pth"),
        type=Path,
    )
    parser.add_argument("--batch-size", default=16, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    payload = evaluate_historical_checkpoints(
        arguments.config,
        arguments.manifest,
        arguments.output,
        repo_root=arguments.repo_root,
        panns_checkpoint=arguments.panns_checkpoint,
        batch_size=arguments.batch_size,
    )
    print(json.dumps(payload["comparison"], indent=2))
    print(f"Saved audit to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

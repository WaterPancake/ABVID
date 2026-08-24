"""Source-held-out PANNs probe adaptation using low-SNR real-source augmentations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import itertools
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

from vehicle_audio.baseline import CLASS_NAMES, classification_metrics
from vehicle_audio.invariance_evaluation import _git_commit, _load_jsonl
from vehicle_audio.pretrained_evaluation import (
    PANN_AUDIOSET_DIMENSION,
    PRETRAINED_EVALUATION_VERSION,
    StandardizedLinearProbe,
    _extract_panns_features,
    validate_panns_checkpoint,
)
from vehicle_audio.semantic_session_evaluation import session_balanced_sample_weights


LOW_SNR_ADAPTATION_VERSION = 1


@dataclass(frozen=True)
class LowSnrAdaptationConfig:
    evaluation_version: int
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    channel: int
    base_state_key: str
    training_input: str
    split_strategy: str
    sample_weighting: str
    model_selection: str

    def __post_init__(self) -> None:
        if self.evaluation_version != LOW_SNR_ADAPTATION_VERSION:
            raise ValueError("unsupported low-SNR adaptation evaluation version")
        if self.seed < 0 or self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("seed must be nonnegative and training counts positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer values are invalid")
        if self.channel < 0:
            raise ValueError("channel must be nonnegative")
        if not self.base_state_key:
            raise ValueError("base_state_key must not be empty")


def load_low_snr_adaptation_config(path: str | Path) -> LowSnrAdaptationConfig:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("low-SNR adaptation config must contain a mapping")
    return LowSnrAdaptationConfig(**dict(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_low_snr_records(
    records: Sequence[Mapping[str, Any]], manifest: Path
) -> dict[str, Any]:
    """Validate paired controlled-SNR data derived only from admitted real sources."""

    root = manifest.parent
    sessions: dict[str, set[str]] = {name: set() for name in CLASS_NAMES}
    class_counts = {name: 0 for name in CLASS_NAMES}
    snr_counts: dict[str, int] = {}
    sample_ids: set[str] = set()
    sample_rates: set[int] = set()
    for index, record in enumerate(records):
        required = {
            "sample_id",
            "clean_path",
            "corrupted_path",
            "vehicle_class",
            "recording_session",
            "source_domain",
            "snr_db",
            "sample_rate",
            "augmentations",
        }
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"record {index} is missing fields: {missing}")
        sample_id = str(record["sample_id"])
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample id: {sample_id}")
        sample_ids.add(sample_id)
        vehicle_class = str(record["vehicle_class"])
        if vehicle_class not in CLASS_NAMES:
            raise ValueError(f"unexpected vehicle class: {vehicle_class}")
        if record["source_domain"] != "real_recording":
            raise ValueError("low-SNR adaptation sources must be real recordings")
        session = str(record["recording_session"])
        if not session:
            raise ValueError("every adaptation record needs a recording session")
        sessions[vehicle_class].add(session)
        class_counts[vehicle_class] += 1
        snr = float(record["snr_db"])
        snr_counts[f"{snr:g}"] = snr_counts.get(f"{snr:g}", 0) + 1
        sample_rates.add(int(record["sample_rate"]))
        for field in ("clean_path", "corrupted_path"):
            if not (root / str(record[field])).is_file():
                raise FileNotFoundError(root / str(record[field]))
        measured = float(
            record["augmentations"]["noise"]["measured_snr_db_at_mix"]
        )
        if abs(measured - snr) > 0.05:
            raise ValueError("controlled SNR validation failed")
    if len(sample_rates) != 1:
        raise ValueError("adaptation records must share one sample rate")
    for vehicle_class, class_sessions in sessions.items():
        if len(class_sessions) < 3:
            raise ValueError(
                "leave-session-pair-out adaptation requires at least three "
                f"{vehicle_class} sessions"
            )
    return {
        "sample_count": len(records),
        "class_counts": class_counts,
        "snr_counts": dict(sorted(snr_counts.items(), key=lambda item: -float(item[0]))),
        "sample_rate": next(iter(sample_rates)),
        "sessions_by_class": {
            name: sorted(values) for name, values in sessions.items()
        },
    }


def leave_session_pair_out_folds(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Hold out every Cartesian pair of one tracked and one wheeled session."""

    sessions = {
        vehicle_class: sorted(
            {
                str(record["recording_session"])
                for record in records
                if record["vehicle_class"] == vehicle_class
            }
        )
        for vehicle_class in CLASS_NAMES
    }
    if any(len(values) < 3 for values in sessions.values()):
        raise ValueError("outer folds require at least three sessions per class")
    folds: list[dict[str, Any]] = []
    all_indices = set(range(len(records)))
    for fold_index, (tracked_session, wheeled_session) in enumerate(
        itertools.product(sessions["tracked"], sessions["wheeled"])
    ):
        test = tuple(
            index
            for index, record in enumerate(records)
            if str(record["recording_session"])
            in {tracked_session, wheeled_session}
        )
        train = tuple(sorted(all_indices - set(test)))
        folds.append(
            {
                "fold_index": fold_index,
                "tracked_test_session": tracked_session,
                "wheeled_test_session": wheeled_session,
                "train_indices": train,
                "test_indices": test,
            }
        )
    return folds


def _probe_from_state(state: Mapping[str, torch.Tensor]) -> StandardizedLinearProbe:
    model = StandardizedLinearProbe(
        state["feature_mean"], state["feature_standard_deviation"], len(CLASS_NAMES)
    )
    model.load_state_dict(state)
    return model


def adapt_probe_fixed_epochs(
    initial_state: Mapping[str, torch.Tensor],
    features: torch.Tensor,
    labels: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
    """Fine-tune only the frozen probe head with equal class/session mass."""

    selected = tuple(sorted(int(index) for index in train_indices))
    if not selected:
        raise ValueError("training indices must not be empty")
    weights = torch.from_numpy(
        session_balanced_sample_weights(records, selected)
    ).to(torch.float32)
    model = _probe_from_state(initial_state).to(device)
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(features[list(selected)], labels[list(selected)], weights),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    torch.manual_seed(seed)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        weight_sum = 0.0
        for feature_batch, label_batch, weight_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            losses = F.cross_entropy(
                model(feature_batch.to(device)),
                label_batch.to(device),
                reduction="none",
            )
            batch_weights = weight_batch.to(device)
            loss = (losses * batch_weights).sum() / batch_weights.sum()
            loss.backward()
            optimizer.step()
            loss_sum += float((losses.detach() * batch_weights).sum().cpu())
            weight_sum += float(batch_weights.sum().cpu())
        history.append({"epoch": float(epoch), "train_loss": loss_sum / weight_sum})
    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.state_dict().items()
    }
    return state, history


def _probabilities(
    state: Mapping[str, torch.Tensor],
    features: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model = _probe_from_state(state).to(device).eval()
    batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            logits = model(features[start : start + batch_size].to(device))
            batches.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(batches)


def _metrics(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    snrs: torch.Tensor,
) -> dict[str, Any]:
    return classification_metrics(
        probabilities.argmax(dim=1), labels, snrs, probabilities
    )


def _session_predictions(
    probabilities: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for session in sorted({str(record["recording_session"]) for record in records}):
        indices = [
            index
            for index, record in enumerate(records)
            if str(record["recording_session"]) == session
        ]
        classes = {str(records[index]["vehicle_class"]) for index in indices}
        if len(classes) != 1:
            raise ValueError(f"session spans classes: {session}")
        target = next(iter(classes))
        mean_probability = probabilities[indices].mean(dim=0)
        predicted = CLASS_NAMES[int(mean_probability.argmax())]
        result[session] = {
            "vehicle_class": target,
            "predicted_class": predicted,
            "correct": predicted == target,
            "support": len(indices),
            "mean_probability": {
                name: float(mean_probability[index])
                for index, name in enumerate(CLASS_NAMES)
            },
        }
    return result


def _extract_feature_cache(
    records: Sequence[Mapping[str, Any]],
    manifest: Path,
    output: Path,
    manifest_sha256: str,
    panns_checkpoint: Path,
    *,
    channel: int,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    encoder = validate_panns_checkpoint(panns_checkpoint)
    cache_path = output / "panns_features.pt"
    expected = {
        "low_snr_adaptation_version": LOW_SNR_ADAPTATION_VERSION,
        "pretrained_evaluation_version": PRETRAINED_EVALUATION_VERSION,
        "manifest_sha256": manifest_sha256,
        "panns_checkpoint_sha256": encoder["sha256"],
        "channel": channel,
    }
    if cache_path.is_file():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if all(cache.get(key) == value for key, value in expected.items()):
            return cache, encoder
    try:
        from panns_inference import AudioTagging
    except ImportError as error:
        raise RuntimeError(
            "low-SNR adaptation requires `uv sync --extra pretrained`"
        ) from error
    tagging = AudioTagging(checkpoint_path=str(panns_checkpoint), device="cpu")
    _, clean_outputs = _extract_panns_features(
        records,
        manifest,
        "clean_path",
        tagging,
        channel=channel,
        sample_rate=int(records[0]["sample_rate"]),
        batch_size=batch_size,
        label="low-SNR real clean",
    )
    _, corrupted_outputs = _extract_panns_features(
        records,
        manifest,
        "corrupted_path",
        tagging,
        channel=channel,
        sample_rate=int(records[0]["sample_rate"]),
        batch_size=batch_size,
        label="low-SNR real corrupted",
    )
    labels = torch.tensor(
        [CLASS_NAMES.index(str(record["vehicle_class"])) for record in records],
        dtype=torch.long,
    )
    snrs = torch.tensor([float(record["snr_db"]) for record in records])
    cache = {
        **expected,
        "clean_clipwise_outputs": clean_outputs,
        "corrupted_clipwise_outputs": corrupted_outputs,
        "labels": labels,
        "snrs": snrs,
    }
    torch.save(cache, cache_path)
    return cache, encoder


def run_low_snr_adaptation(
    config_path: str | Path,
    manifest_path: str | Path,
    base_probe_bundle: str | Path,
    panns_checkpoint: str | Path,
    output_path: str | Path,
    *,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """Adapt and evaluate with outer source-session pairs held out."""

    config_path = Path(config_path)
    manifest_path = Path(manifest_path)
    bundle_path = Path(base_probe_bundle)
    panns_path = Path(panns_checkpoint)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite low-SNR adaptation: {output}")
    output.mkdir(parents=True)
    config = load_low_snr_adaptation_config(config_path)
    records, manifest_sha256 = _load_jsonl(manifest_path)
    corpus_summary = validate_low_snr_records(records, manifest_path)
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    if tuple(bundle.get("class_names", ())) != tuple(CLASS_NAMES):
        raise ValueError("base probe class names are incompatible")
    if config.base_state_key not in bundle.get("models", {}):
        raise ValueError("base probe state is missing")
    initial_state = bundle["models"][config.base_state_key]
    if initial_state["feature_mean"].shape != (PANN_AUDIOSET_DIMENSION,):
        raise ValueError("base state is not an AudioSet-output probe")
    cache, encoder = _extract_feature_cache(
        records,
        manifest_path,
        output,
        manifest_sha256,
        panns_path,
        channel=config.channel,
        batch_size=config.batch_size,
    )
    clean_features = cache["clean_clipwise_outputs"]
    corrupted_features = cache["corrupted_clipwise_outputs"]
    labels = cache["labels"]
    snrs = cache["snrs"]
    device = torch.device(device_name)
    base_corrupted_probabilities = _probabilities(
        initial_state,
        corrupted_features,
        batch_size=config.batch_size,
        device=device,
    )
    base_clean_probabilities = _probabilities(
        initial_state,
        clean_features,
        batch_size=config.batch_size,
        device=device,
    )

    folds = leave_session_pair_out_folds(records)
    probability_sum = torch.zeros_like(base_corrupted_probabilities)
    prediction_count = torch.zeros(len(records), dtype=torch.long)
    fold_results: list[dict[str, Any]] = []
    fold_states: dict[str, dict[str, torch.Tensor]] = {}
    for fold in folds:
        fold_index = int(fold["fold_index"])
        state, history = adapt_probe_fixed_epochs(
            initial_state,
            corrupted_features,
            labels,
            records,
            fold["train_indices"],
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            seed=config.seed + fold_index,
            device=device,
        )
        key = f"fold_{fold_index:02d}"
        fold_states[key] = state
        probabilities = _probabilities(
            state,
            corrupted_features,
            batch_size=config.batch_size,
            device=device,
        )
        indices = list(fold["test_indices"])
        probability_sum[indices] += probabilities[indices]
        prediction_count[indices] += 1
        fold_results.append(
            {
                "fold_id": key,
                "tracked_test_session": fold["tracked_test_session"],
                "wheeled_test_session": fold["wheeled_test_session"],
                "train_support": len(fold["train_indices"]),
                "test_support": len(indices),
                "final_train_loss": history[-1]["train_loss"],
                "metrics": _metrics(
                    probabilities[indices], labels[indices], snrs[indices]
                ),
            }
        )
    if (prediction_count == 0).any():
        raise RuntimeError("outer folds failed to predict every record")
    crossval_probabilities = probability_sum / prediction_count.unsqueeze(1)

    all_indices = tuple(range(len(records)))
    final_state, final_history = adapt_probe_fixed_epochs(
        initial_state,
        corrupted_features,
        labels,
        records,
        all_indices,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        seed=config.seed + 10_000,
        device=device,
    )
    results = {
        "base_frozen_probe": {
            "corrupted": _metrics(base_corrupted_probabilities, labels, snrs),
            "clean": _metrics(base_clean_probabilities, labels, snrs),
            "session_predictions": _session_predictions(
                base_corrupted_probabilities, records
            ),
        },
        "low_snr_real_adapted_crossval": {
            "corrupted": _metrics(crossval_probabilities, labels, snrs),
            "session_predictions": _session_predictions(
                crossval_probabilities, records
            ),
            "prediction_repetitions_by_class": {
                vehicle_class: sorted(
                    {
                        int(prediction_count[index])
                        for index, record in enumerate(records)
                        if record["vehicle_class"] == vehicle_class
                    }
                )
                for vehicle_class in CLASS_NAMES
            },
        },
    }
    comparison = []
    for method, value in results.items():
        metrics = value["corrupted"]
        comparison.append(
            {
                "method": method,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "tracked_recall": metrics["per_class_recall"]["tracked"],
                "wheeled_recall": metrics["per_class_recall"]["wheeled"],
                "sessions_correct": sum(
                    bool(value["correct"])
                    for value in value["session_predictions"].values()
                ),
                "session_count": len(value["session_predictions"]),
            }
        )
    payload = {
        "low_snr_adaptation_version": LOW_SNR_ADAPTATION_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "configuration": asdict(config),
        "configuration_path": str(config_path),
        "configuration_sha256": _sha256(config_path),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "dataset_version": manifest_sha256,
        "corpus_summary": corpus_summary,
        "base_probe_bundle": str(bundle_path),
        "base_probe_bundle_sha256": _sha256(bundle_path),
        "base_probe_training_metadata": {
            field: bundle[field]
            for field in (
                "training_config",
                "synthetic_manifest_sha256",
                "real_manifest_sha256",
                "pretrained_encoder",
            )
            if field in bundle
        },
        "pretrained_encoder": encoder,
        "test_domain": "controlled low-SNR augmentation of held-out real recording sessions",
        "protocol_status": "development_cross_validation_not_fresh_native_real_test",
        "fold_count": len(folds),
        "fold_results": fold_results,
        "results": results,
        "comparison": comparison,
        "limitations": [
            "The observations are synthetic corruptions of real recordings, not native low-SNR field recordings.",
            "Only three tracked and four wheeled recording sessions are available.",
            "The final all-session state is a candidate for a future fresh test and has no independent score here.",
        ],
        "artifacts": {
            "models": "models.pt",
            "feature_cache": "panns_features.pt",
            "splits": "splits.json",
            "comparison": "comparison.csv",
            "window_predictions": "window_predictions.csv",
        },
    }
    torch.save(
        {
            "final_all_development_sessions": final_state,
            "outer_fold_models": fold_states,
            "base_state_key": config.base_state_key,
            "base_probe_bundle_sha256": _sha256(bundle_path),
            "manifest_sha256": manifest_sha256,
            "configuration": asdict(config),
            "class_names": list(CLASS_NAMES),
            "final_history": final_history,
        },
        output / "models.pt",
    )
    split_payload = {
        "strategy": config.split_strategy,
        "group_field": "recording_session",
        "folds": [
            {
                **{key: value for key, value in fold.items() if not key.endswith("indices")},
                "train_sample_ids": [
                    str(records[index]["sample_id"])
                    for index in fold["train_indices"]
                ],
                "test_sample_ids": [
                    str(records[index]["sample_id"])
                    for index in fold["test_indices"]
                ],
            }
            for fold in folds
        ],
    }
    (output / "splits.json").write_text(
        json.dumps(split_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    with (output / "window_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "sample_id",
            "recording_session",
            "vehicle_class",
            "snr_db",
            "base_predicted_class",
            "base_tracked_probability",
            "base_wheeled_probability",
            "adapted_predicted_class",
            "adapted_tracked_probability",
            "adapted_wheeled_probability",
            "adapted_prediction_repetitions",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "recording_session": record["recording_session"],
                    "vehicle_class": record["vehicle_class"],
                    "snr_db": record["snr_db"],
                    "base_predicted_class": CLASS_NAMES[
                        int(base_corrupted_probabilities[index].argmax())
                    ],
                    "base_tracked_probability": float(
                        base_corrupted_probabilities[index, 0]
                    ),
                    "base_wheeled_probability": float(
                        base_corrupted_probabilities[index, 1]
                    ),
                    "adapted_predicted_class": CLASS_NAMES[
                        int(crossval_probabilities[index].argmax())
                    ],
                    "adapted_tracked_probability": float(
                        crossval_probabilities[index, 0]
                    ),
                    "adapted_wheeled_probability": float(
                        crossval_probabilities[index, 1]
                    ),
                    "adapted_prediction_repetitions": int(prediction_count[index]),
                }
            )
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload

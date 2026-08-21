"""Frozen PANNs AudioSet representation baseline for Milestone 6 transfer."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torchaudio.functional as audio_functional

from vehicle_audio.baseline import (
    CLASS_NAMES,
    SplitIndices,
    grouped_stratified_split,
)
from vehicle_audio.invariance_evaluation import (
    _evaluate,
    _evaluate_real_sessions,
    _git_commit,
    _load_jsonl,
    _split_payload,
    build_factorial_split_and_partitions,
)
from vehicle_audio.transfer_evaluation import (
    stratified_fraction_indices,
    validate_transfer_protocol,
)


PRETRAINED_EVALUATION_VERSION = 2
PANN_SAMPLE_RATE = 32_000
PANN_EMBEDDING_DIMENSION = 2_048
PANN_AUDIOSET_DIMENSION = 527
PANN_CHECKPOINT_SHA256 = (
    "0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31"
)
PANN_CHECKPOINT_URL = (
    "https://zenodo.org/records/3987831/files/Cnn14_mAP=0.431.pth"
)
PANN_SOURCE_URL = "https://github.com/qiuqiangkong/audioset_tagging_cnn"
PANN_LICENSE_URL = (
    "https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/LICENSE.MIT"
)
METHODS = (
    "panns_embedding_corrupted_linear",
    "panns_embedding_paired_linear",
    "panns_audioset_corrupted_linear",
    "panns_audioset_paired_linear",
)


class StandardizedLinearProbe(nn.Module):
    """A linear head whose training-set normalization is part of its state."""

    def __init__(
        self,
        feature_mean: torch.Tensor,
        feature_standard_deviation: torch.Tensor,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if feature_mean.ndim != 1 or feature_mean.shape != feature_standard_deviation.shape:
            raise ValueError("probe normalization tensors must be equal one-dimensional shapes")
        self.register_buffer("feature_mean", feature_mean.detach().clone())
        self.register_buffer(
            "feature_standard_deviation",
            feature_standard_deviation.detach().clone().clamp_min(1e-6),
        )
        self.classifier = nn.Linear(len(feature_mean), num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_standard_deviation
        return self.classifier(normalized)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_panns_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"PANNs checkpoint not found: {checkpoint}. Download it from "
            f"{PANN_CHECKPOINT_URL}"
        )
    observed_sha256 = _sha256(checkpoint)
    if observed_sha256 != PANN_CHECKPOINT_SHA256:
        raise ValueError(
            f"PANNs checkpoint SHA-256 mismatch: {observed_sha256} != "
            f"{PANN_CHECKPOINT_SHA256}"
        )
    return {
        "path": str(checkpoint),
        "sha256": observed_sha256,
        "size_bytes": checkpoint.stat().st_size,
        "download_url": PANN_CHECKPOINT_URL,
        "source_repository": PANN_SOURCE_URL,
        "license": "MIT",
        "license_url": PANN_LICENSE_URL,
        "pretraining_dataset": "AudioSet",
        "architecture": "PANNs Cnn14",
        "embedding_dimension": PANN_EMBEDDING_DIMENSION,
        "audioset_output_dimension": PANN_AUDIOSET_DIMENSION,
        "input_sample_rate": PANN_SAMPLE_RATE,
    }


def _read_resampled_channel(
    path: Path, channel: int, expected_sample_rate: int
) -> torch.Tensor:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"sample-rate mismatch for {path}: {sample_rate} != {expected_sample_rate}"
        )
    if not 0 <= channel < samples.shape[1]:
        raise ValueError(f"requested channel {channel} but {path} has {samples.shape[1]}")
    waveform = torch.from_numpy(samples[:, channel].copy())
    if expected_sample_rate != PANN_SAMPLE_RATE:
        waveform = audio_functional.resample(
            waveform, expected_sample_rate, PANN_SAMPLE_RATE
        )
    return waveform


def _extract_panns_features(
    records: Sequence[Mapping[str, Any]],
    manifest: Path,
    audio_key: str,
    tagging: Any,
    *,
    channel: int,
    sample_rate: int,
    batch_size: int,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    root = manifest.parent
    waveform_batch: list[torch.Tensor] = []
    embedding_batches: list[torch.Tensor] = []
    clipwise_batches: list[torch.Tensor] = []

    def flush() -> None:
        if not waveform_batch:
            return
        lengths = {len(waveform) for waveform in waveform_batch}
        if len(lengths) != 1:
            raise ValueError(f"{label} clips have unequal lengths: {sorted(lengths)}")
        audio = torch.stack(waveform_batch).numpy()
        clipwise_outputs, embeddings = tagging.inference(audio)
        embedding_batch = torch.from_numpy(np.asarray(embeddings)).to(torch.float32)
        clipwise_batch = torch.from_numpy(np.asarray(clipwise_outputs)).to(torch.float32)
        if embedding_batch.shape != (len(waveform_batch), PANN_EMBEDDING_DIMENSION):
            raise RuntimeError(
                f"unexpected PANNs embedding shape: {tuple(embedding_batch.shape)}"
            )
        if clipwise_batch.shape != (len(waveform_batch), PANN_AUDIOSET_DIMENSION):
            raise RuntimeError(
                f"unexpected PANNs AudioSet shape: {tuple(clipwise_batch.shape)}"
            )
        embedding_batches.append(embedding_batch)
        clipwise_batches.append(clipwise_batch)
        waveform_batch.clear()

    for index, record in enumerate(records):
        if audio_key not in record:
            raise ValueError(f"record {index} has no {audio_key!r} path")
        waveform_batch.append(
            _read_resampled_channel(
                root / str(record[audio_key]), channel, sample_rate
            )
        )
        if len(waveform_batch) == batch_size:
            flush()
        if (index + 1) % 100 == 0 or index + 1 == len(records):
            print(
                f"Extracted {label} PANNs features: {index + 1}/{len(records)}",
                flush=True,
            )
    flush()
    embeddings = torch.cat(embedding_batches)
    clipwise_outputs = torch.cat(clipwise_batches)
    if not torch.isfinite(embeddings).all() or not torch.isfinite(
        clipwise_outputs
    ).all():
        raise RuntimeError(f"{label} PANNs features contain NaN or Inf")
    return embeddings, clipwise_outputs


def _precompute_panns_embeddings(
    synthetic_records: Sequence[Mapping[str, Any]],
    real_records: Sequence[Mapping[str, Any]],
    synthetic_manifest: Path,
    real_manifest: Path,
    cache_path: Path,
    checkpoint: Mapping[str, Any],
    synthetic_sha256: str,
    real_sha256: str,
    *,
    channel: int,
    batch_size: int,
) -> dict[str, Any]:
    sample_rate = int(synthetic_records[0]["sample_rate"])
    expected = {
        "pretrained_evaluation_version": PRETRAINED_EVALUATION_VERSION,
        "synthetic_manifest_sha256": synthetic_sha256,
        "real_manifest_sha256": real_sha256,
        "panns_checkpoint_sha256": checkpoint["sha256"],
        "source_sample_rate": sample_rate,
        "panns_sample_rate": PANN_SAMPLE_RATE,
        "channel": channel,
    }
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if all(cached.get(key) == value for key, value in expected.items()):
            print(f"Using PANNs embedding cache: {cache_path}", flush=True)
            return cached

    try:
        from panns_inference import AudioTagging
    except ImportError as error:
        raise RuntimeError(
            "PANNs extraction requires `uv sync --extra pretrained`"
        ) from error
    tagging = AudioTagging(checkpoint_path=checkpoint["path"], device="cpu")
    clean_embeddings, clean_clipwise_outputs = _extract_panns_features(
        synthetic_records,
        synthetic_manifest,
        "clean_path",
        tagging,
        channel=channel,
        sample_rate=sample_rate,
        batch_size=batch_size,
        label="synthetic clean",
    )
    corrupted_embeddings, corrupted_clipwise_outputs = _extract_panns_features(
        synthetic_records,
        synthetic_manifest,
        "corrupted_path",
        tagging,
        channel=channel,
        sample_rate=sample_rate,
        batch_size=batch_size,
        label="synthetic corrupted",
    )
    real_embeddings, real_clipwise_outputs = _extract_panns_features(
        real_records,
        real_manifest,
        "audio_path",
        tagging,
        channel=channel,
        sample_rate=sample_rate,
        batch_size=batch_size,
        label="native real",
    )
    label_map = {name: index for index, name in enumerate(CLASS_NAMES)}
    result = {
        **expected,
        "clean_embeddings": clean_embeddings,
        "corrupted_embeddings": corrupted_embeddings,
        "real_embeddings": real_embeddings,
        "clean_clipwise_outputs": clean_clipwise_outputs,
        "corrupted_clipwise_outputs": corrupted_clipwise_outputs,
        "real_clipwise_outputs": real_clipwise_outputs,
        "synthetic_labels": torch.tensor(
            [label_map[str(record["vehicle_class"])] for record in synthetic_records],
            dtype=torch.long,
        ),
        "real_labels": torch.tensor(
            [label_map[str(record["vehicle_class"])] for record in real_records],
            dtype=torch.long,
        ),
        "synthetic_snrs": torch.tensor(
            [float(record["snr_db"]) for record in synthetic_records],
            dtype=torch.float32,
        ),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, cache_path)
    return result


def _fit_probe(
    method: str,
    clean_embeddings: torch.Tensor,
    corrupted_embeddings: torch.Tensor,
    labels: torch.Tensor,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int]:
    if method not in METHODS:
        raise ValueError(f"unknown PANNs probe method: {method}")
    train_corrupted = corrupted_embeddings[list(train_indices)]
    train_labels = labels[list(train_indices)]
    if method.endswith("_paired_linear"):
        train_features = torch.cat(
            (clean_embeddings[list(train_indices)], train_corrupted), dim=0
        )
        fitted_labels = torch.cat((train_labels, train_labels), dim=0)
    else:
        train_features = train_corrupted
        fitted_labels = train_labels
    feature_mean = train_features.mean(dim=0)
    feature_standard_deviation = train_features.std(dim=0).clamp_min(1e-6)

    torch.manual_seed(seed)
    model = StandardizedLinearProbe(
        feature_mean, feature_standard_deviation, len(CLASS_NAMES)
    ).to(device)
    class_counts = torch.bincount(fitted_labels, minlength=len(CLASS_NAMES)).float()
    if (class_counts == 0).any():
        raise ValueError("PANNs probe training partition must contain both classes")
    class_weights = len(fitted_labels) / (len(CLASS_NAMES) * class_counts)
    loss_function = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(
        model.parameters(), learning_rate, weight_decay=weight_decay
    )
    loader = DataLoader(
        TensorDataset(train_features, fitted_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for feature_batch, target_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(
                model(feature_batch.to(device)), target_batch.to(device)
            )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(target_batch)
            seen += len(target_batch)
        validation = _evaluate(
            model,
            corrupted_embeddings,
            labels,
            validation_indices,
            None,
            batch_size=batch_size,
            device=device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / seen,
                "validation_balanced_accuracy": validation["balanced_accuracy"],
                "validation_macro_f1": validation["macro_f1"],
            }
        )
        if validation["balanced_accuracy"] > best_score:
            best_score = validation["balanced_accuracy"]
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
    return best_state, history, best_epoch


def _probe_from_state(state: Mapping[str, torch.Tensor]) -> StandardizedLinearProbe:
    model = StandardizedLinearProbe(
        state["feature_mean"], state["feature_standard_deviation"], len(CLASS_NAMES)
    )
    model.load_state_dict(state)
    return model


def _adapt_probe(
    initial_state: Mapping[str, torch.Tensor],
    real_features: torch.Tensor,
    real_labels: torch.Tensor,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int, dict[str, Any]]:
    model = _probe_from_state(initial_state).to(device)
    selected_labels = real_labels[list(train_indices)]
    class_counts = torch.bincount(
        selected_labels, minlength=len(CLASS_NAMES)
    ).float()
    if (class_counts == 0).any():
        raise ValueError("real adaptation subset must contain both classes")
    class_weights = len(selected_labels) / (len(CLASS_NAMES) * class_counts)
    loss_function = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loader = DataLoader(
        TensorDataset(real_features[list(train_indices)], selected_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for feature_batch, target_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(
                model(feature_batch.to(device)), target_batch.to(device)
            )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(target_batch)
            seen += len(target_batch)
        validation = _evaluate(
            model,
            real_features,
            real_labels,
            validation_indices,
            None,
            batch_size=batch_size,
            device=device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / seen,
                "validation_balanced_accuracy": validation["balanced_accuracy"],
                "validation_macro_f1": validation["macro_f1"],
            }
        )
        if validation["balanced_accuracy"] > best_score:
            best_score = validation["balanced_accuracy"]
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
    best_model = _probe_from_state(best_state).to(device)
    validation = _evaluate(
        best_model,
        real_features,
        real_labels,
        validation_indices,
        None,
        batch_size=batch_size,
        device=device,
    )
    return best_state, history, best_epoch, validation


def evaluate_real_adaptation(
    initial_state: Mapping[str, torch.Tensor],
    real_features: torch.Tensor,
    real_labels: torch.Tensor,
    real_records: Sequence[Mapping[str, Any]],
    real_split: SplitIndices,
    fractions: Sequence[float],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    subset_seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, torch.Tensor]]]:
    """Fine-tune only the AudioSet probe using grouped real development data."""

    rows: list[dict[str, Any]] = []
    states: dict[str, dict[str, torch.Tensor]] = {}
    for fraction_index, fraction in enumerate(fractions):
        indices = stratified_fraction_indices(
            real_labels,
            real_split.train,
            fraction,
            seed=subset_seed,
        )
        state, history, best_epoch, validation = _adapt_probe(
            initial_state,
            real_features,
            real_labels,
            indices,
            real_split.validation,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed + 1000 + fraction_index,
            device=device,
        )
        key = f"{100 * fraction:g}_percent_real"
        states[key] = state
        model = _probe_from_state(state).to(device)
        test_metrics = _evaluate(
            model,
            real_features,
            real_labels,
            real_split.test,
            None,
            batch_size=batch_size,
            device=device,
        )
        rows.append(
            {
                "key": key,
                "requested_real_fraction": fraction,
                "actual_real_fraction": len(indices) / len(real_split.train),
                "real_training_support": len(indices),
                "real_training_class_support": {
                    name: sum(
                        int(real_records[index]["vehicle_class"] == name)
                        for index in indices
                    )
                    for name in CLASS_NAMES
                },
                "real_training_sample_ids": [
                    str(real_records[index]["sample_id"]) for index in indices
                ],
                "best_epoch": best_epoch,
                "history": history,
                "real_validation": validation,
                "native_real": test_metrics,
            }
        )
    return rows, states


def evaluate_precomputed_panns(
    clean_embeddings: torch.Tensor,
    corrupted_embeddings: torch.Tensor,
    clean_clipwise_outputs: torch.Tensor,
    corrupted_clipwise_outputs: torch.Tensor,
    synthetic_labels: torch.Tensor,
    synthetic_snrs: torch.Tensor,
    real_embeddings: torch.Tensor,
    real_clipwise_outputs: torch.Tensor,
    real_labels: torch.Tensor,
    synthetic_records: Sequence[Mapping[str, Any]],
    real_records: Sequence[Mapping[str, Any]],
    nuisance: Mapping[str, Sequence[int]],
    real_split: SplitIndices,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    """Fit frozen-embedding probes and evaluate the established partitions."""

    expected_synthetic = (len(synthetic_records), PANN_EMBEDDING_DIMENSION)
    if (
        clean_embeddings.shape != expected_synthetic
        or corrupted_embeddings.shape != expected_synthetic
    ):
        raise ValueError("synthetic PANNs embeddings have the wrong shape")
    if real_embeddings.shape != (len(real_records), PANN_EMBEDDING_DIMENSION):
        raise ValueError("native-real PANNs embeddings have the wrong shape")
    expected_clipwise = (len(synthetic_records), PANN_AUDIOSET_DIMENSION)
    if (
        clean_clipwise_outputs.shape != expected_clipwise
        or corrupted_clipwise_outputs.shape != expected_clipwise
    ):
        raise ValueError("synthetic PANNs AudioSet outputs have the wrong shape")
    if real_clipwise_outputs.shape != (len(real_records), PANN_AUDIOSET_DIMENSION):
        raise ValueError("native-real PANNs AudioSet outputs have the wrong shape")

    method_results: dict[str, Any] = {}
    model_states: dict[str, dict[str, torch.Tensor]] = {}
    for method in METHODS:
        if method.startswith("panns_embedding_"):
            representation = "embedding"
            clean_features = clean_embeddings
            corrupted_features = corrupted_embeddings
            real_features = real_embeddings
        else:
            representation = "audioset_clipwise_output"
            clean_features = clean_clipwise_outputs
            corrupted_features = corrupted_clipwise_outputs
            real_features = real_clipwise_outputs
        state, history, best_epoch = _fit_probe(
            method,
            clean_features,
            corrupted_features,
            synthetic_labels,
            nuisance["train_in_distribution"],
            nuisance["validation_in_distribution"],
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
            device=device,
        )
        model_states[method] = state
        model = _probe_from_state(state).to(device)
        evaluations = {
            "clean_synthetic": _evaluate(
                model,
                clean_features,
                synthetic_labels,
                nuisance["all_corruptions"],
                synthetic_snrs,
                batch_size=batch_size,
                device=device,
            ),
            **{
                condition: _evaluate(
                    model,
                    corrupted_features,
                    synthetic_labels,
                    nuisance[condition],
                    synthetic_snrs,
                    batch_size=batch_size,
                    device=device,
                )
                for condition in (
                    "seen_corruption",
                    "unseen_noise",
                    "unseen_microphone",
                    "unseen_environment",
                    "all_corruptions",
                )
            },
            "native_real": _evaluate(
                model,
                real_features,
                real_labels,
                real_split.test,
                None,
                batch_size=batch_size,
                device=device,
            ),
            "native_real_all_sessions": _evaluate_real_sessions(
                model,
                real_features,
                real_labels,
                real_records,
                batch_size=batch_size,
                device=device,
            ),
        }
        method_results[method] = {
            "representation": representation,
            "training_input": (
                "paired_clean_corrupted" if method.endswith("_paired_linear")
                else "corrupted_only"
            ),
            "encoder_trainable": False,
            "best_epoch": best_epoch,
            "history": history,
            "evaluations": evaluations,
        }
    return method_results, model_states


def _write_summaries(
    output: Path,
    methods: Mapping[str, Any],
    adaptation: Sequence[Mapping[str, Any]],
) -> None:
    condition_order = (
        "seen_corruption",
        "unseen_noise",
        "unseen_microphone",
        "unseen_environment",
        "all_corruptions",
        "native_real",
        "native_real_all_sessions",
    )
    rows: list[dict[str, Any]] = []
    for method, method_result in methods.items():
        for condition in condition_order:
            metrics = method_result["evaluations"][condition]
            rows.append(
                {
                    "method": method,
                    "condition": condition,
                    "support": metrics["support"],
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "session_balanced_accuracy": metrics.get(
                        "session_balanced_accuracy"
                    ),
                    "macro_f1": metrics["macro_f1"],
                    "tracked_recall": metrics["per_class_recall"]["tracked"],
                    "wheeled_recall": metrics["per_class_recall"]["wheeled"],
                }
            )
    with (output / "condition_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(10.5, 5.0))
    x_values = list(range(len(condition_order)))
    width = 0.8 / len(METHODS)
    for method_index, method in enumerate(METHODS):
        values = [
            methods[method]["evaluations"][condition]["balanced_accuracy"]
            for condition in condition_order
        ]
        axis.bar(
            [
                x + (method_index - (len(METHODS) - 1) / 2.0) * width
                for x in x_values
            ],
            values,
            width=width,
            label=method.replace("_", " "),
        )
    axis.set_xticks(
        x_values, [condition.replace("_", "\n") for condition in condition_order]
    )
    axis.set_ylabel("Balanced accuracy")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Frozen PANNs transfer comparison")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "condition_comparison.png", dpi=160)
    plt.close(figure)

    adaptation_rows = [
        {
            "requested_real_fraction": row["requested_real_fraction"],
            "actual_real_fraction": row["actual_real_fraction"],
            "real_training_support": row["real_training_support"],
            "balanced_accuracy": row["native_real"]["balanced_accuracy"],
            "macro_f1": row["native_real"]["macro_f1"],
            "tracked_recall": row["native_real"]["per_class_recall"]["tracked"],
            "wheeled_recall": row["native_real"]["per_class_recall"]["wheeled"],
        }
        for row in adaptation
    ]
    with (output / "real_adaptation_curve.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(adaptation_rows[0]))
        writer.writeheader()
        writer.writerows(adaptation_rows)

    figure, axis = plt.subplots(figsize=(7.5, 4.7))
    x_values = [100 * row["actual_real_fraction"] for row in adaptation_rows]
    axis.plot(
        x_values,
        [row["balanced_accuracy"] for row in adaptation_rows],
        marker="o",
        label="balanced accuracy",
    )
    axis.plot(
        x_values,
        [row["tracked_recall"] for row in adaptation_rows],
        marker="o",
        label="tracked recall",
    )
    axis.plot(
        x_values,
        [row["wheeled_recall"] for row in adaptation_rows],
        marker="o",
        label="wheeled recall",
    )
    axis.set_xlabel("Actual fraction of fixed real training split (%)")
    axis.set_ylabel("Held-out native-real metric")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("PANNs AudioSet probe real-data adaptation")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "real_adaptation_curve.png", dpi=160)
    plt.close(figure)


def evaluate_panns_transfer(
    synthetic_manifest_path: str | Path,
    real_manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    embedding_cache_path: str | Path | None = None,
    heldout_noise: str = "road traffic",
    heldout_geometry: str = "far_offset",
    epochs: int = 50,
    batch_size: int = 32,
    extraction_batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    real_fractions: Sequence[float] = (0.01, 0.05, 0.10, 0.25, 1.0),
    adaptation_epochs: int = 100,
    adaptation_learning_rate: float = 1e-4,
    seed: int = 42,
    split_seed: int = 42,
    channel: int = 0,
    device_name: str = "cpu",
) -> dict[str, Any]:
    if (
        epochs <= 0
        or adaptation_epochs <= 0
        or batch_size <= 0
        or extraction_batch_size <= 0
    ):
        raise ValueError("epoch and batch sizes must be positive")
    if learning_rate <= 0 or adaptation_learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay nonnegative")
    fractions = tuple(float(value) for value in real_fractions)
    if (
        not fractions
        or any(not 0 < value <= 1 for value in fractions)
        or tuple(sorted(set(fractions))) != fractions
    ):
        raise ValueError("real fractions must be unique, increasing, and in (0, 1]")
    if seed < 0 or split_seed < 0 or channel < 0:
        raise ValueError("seeds and channel must be nonnegative")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    synthetic_manifest = Path(synthetic_manifest_path)
    real_manifest = Path(real_manifest_path)
    synthetic_records, synthetic_sha256 = _load_jsonl(synthetic_manifest)
    real_records, real_sha256 = _load_jsonl(real_manifest)
    audit = validate_transfer_protocol(synthetic_records, real_records)
    if {record.get("factorial_dataset_version") for record in synthetic_records} == {None}:
        raise ValueError("PANNs comparison requires the balanced factorial corpus")
    synthetic_split, nuisance = build_factorial_split_and_partitions(
        synthetic_records,
        split_seed,
        heldout_noise=heldout_noise,
        heldout_geometry=heldout_geometry,
    )
    real_split = grouped_stratified_split(real_records, split_seed)
    checkpoint = validate_panns_checkpoint(checkpoint_path)
    cache_path = (
        Path(embedding_cache_path)
        if embedding_cache_path is not None
        else output / "panns_embeddings.pt"
    )
    cached = _precompute_panns_embeddings(
        synthetic_records,
        real_records,
        synthetic_manifest,
        real_manifest,
        cache_path,
        checkpoint,
        synthetic_sha256,
        real_sha256,
        channel=channel,
        batch_size=extraction_batch_size,
    )
    device = torch.device(device_name)
    method_results, model_states = evaluate_precomputed_panns(
        cached["clean_embeddings"],
        cached["corrupted_embeddings"],
        cached["clean_clipwise_outputs"],
        cached["corrupted_clipwise_outputs"],
        cached["synthetic_labels"],
        cached["synthetic_snrs"],
        cached["real_embeddings"],
        cached["real_clipwise_outputs"],
        cached["real_labels"],
        synthetic_records,
        real_records,
        nuisance,
        real_split,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
    )
    adaptation, adaptation_states = evaluate_real_adaptation(
        model_states["panns_audioset_paired_linear"],
        cached["real_clipwise_outputs"],
        cached["real_labels"],
        real_records,
        real_split,
        fractions,
        epochs=adaptation_epochs,
        batch_size=batch_size,
        learning_rate=adaptation_learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        subset_seed=split_seed,
        device=device,
    )
    _write_summaries(output, method_results, adaptation)
    split_payload = {
        "strategy": "balanced_factorial_recording_session_grouped",
        "seed": split_seed,
        "synthetic": _split_payload(synthetic_records, synthetic_split),
        "real": _split_payload(real_records, real_split),
        "heldout_noise": heldout_noise,
        "heldout_environment_proxy": heldout_geometry,
        "nuisance_partitions": {
            name: {
                "support": len(indices),
                "sample_ids": [
                    str(synthetic_records[index]["sample_id"]) for index in indices
                ],
            }
            for name, indices in nuisance.items()
        },
    }
    (output / "splits.json").write_text(
        json.dumps(split_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    dataset_version = hashlib.sha256(
        f"{synthetic_sha256}:{real_sha256}".encode("ascii")
    ).hexdigest()
    training_config = {
        "pretrained_evaluation_version": PRETRAINED_EVALUATION_VERSION,
        "methods": list(METHODS),
        "model": "frozen_panns_cnn14_linear_probe",
        "encoder_trainable": False,
        "embedding_dimension": PANN_EMBEDDING_DIMENSION,
        "audioset_output_dimension": PANN_AUDIOSET_DIMENSION,
        "heldout_noise": heldout_noise,
        "heldout_geometry": heldout_geometry,
        "epochs": epochs,
        "batch_size": batch_size,
        "extraction_batch_size": extraction_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "real_fractions": list(fractions),
        "adaptation_source_method": "panns_audioset_paired_linear",
        "adaptation_epochs": adaptation_epochs,
        "adaptation_learning_rate": adaptation_learning_rate,
        "real_subset_seed": split_seed,
        "seed": seed,
        "split_seed": split_seed,
        "channel": channel,
        "device": str(device),
    }
    results = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset_version": dataset_version,
        "synthetic_manifest": str(synthetic_manifest),
        "synthetic_manifest_sha256": synthetic_sha256,
        "real_manifest": str(real_manifest),
        "real_manifest_sha256": real_sha256,
        "test_domain": (
            "held-out procedural corruption conditions and held-out native-real "
            "recording sessions"
        ),
        "protocol_status": "complete",
        "real_corpus_audit": audit,
        "pretrained_encoder": checkpoint,
        "training_config": training_config,
        "split_counts": {
            "synthetic_nuisance": {
                name: len(indices) for name, indices in nuisance.items()
            },
            "real": {
                "train": len(real_split.train),
                "validation": len(real_split.validation),
                "test": len(real_split.test),
            },
        },
        "methods": method_results,
        "real_adaptation": adaptation,
        "artifacts": {
            "checkpoint": "probe_models.pt",
            "embedding_cache": str(cache_path),
            "splits": "splits.json",
            "condition_csv": "condition_summary.csv",
            "condition_plot": "condition_comparison.png",
            "adaptation_csv": "real_adaptation_curve.csv",
            "adaptation_plot": "real_adaptation_curve.png",
        },
    }
    torch.save(
        {
            "models": model_states,
            "real_adaptation_models": adaptation_states,
            "training_config": training_config,
            "pretrained_encoder": checkpoint,
            "class_names": list(CLASS_NAMES),
            "synthetic_manifest_sha256": synthetic_sha256,
            "real_manifest_sha256": real_sha256,
        },
        output / "probe_models.pt",
    )
    (output / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "experiment.json").write_text(
        json.dumps(
            {
                "git_commit": results["git_commit"],
                "configuration": training_config,
                "random_seed": seed,
                "split_seed": split_seed,
                "dataset_version": dataset_version,
                "train_test_split": "splits.json",
                "model_checkpoint": "probe_models.pt",
                "metrics": "metrics.json",
                "pretrained_encoder": checkpoint,
                "test_domain": results["test_domain"],
                "protocol_status": results["protocol_status"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", type=Path, required=True)
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout-noise", default="road traffic")
    parser.add_argument("--heldout-geometry", default="far_offset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--extraction-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--real-fraction", type=float, action="append", dest="real_fractions"
    )
    parser.add_argument("--adaptation-epochs", type=int, default=100)
    parser.add_argument("--adaptation-learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = evaluate_panns_transfer(
        args.synthetic_manifest,
        args.real_manifest,
        args.checkpoint,
        args.output,
        embedding_cache_path=args.embedding_cache,
        heldout_noise=args.heldout_noise,
        heldout_geometry=args.heldout_geometry,
        epochs=args.epochs,
        batch_size=args.batch_size,
        extraction_batch_size=args.extraction_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        real_fractions=(
            (0.01, 0.05, 0.10, 0.25, 1.0)
            if args.real_fractions is None
            else args.real_fractions
        ),
        adaptation_epochs=args.adaptation_epochs,
        adaptation_learning_rate=args.adaptation_learning_rate,
        seed=args.seed,
        split_seed=args.split_seed,
        channel=args.channel,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output),
                "protocol_status": results["protocol_status"],
                "methods": {
                    method: {
                        "synthetic_balanced_accuracy": result["evaluations"]
                        ["all_corruptions"]["balanced_accuracy"],
                        "real_balanced_accuracy": result["evaluations"]
                        ["native_real"]["balanced_accuracy"],
                    }
                    for method, result in results["methods"].items()
                },
                "real_adaptation": {
                    row["key"]: row["native_real"]["balanced_accuracy"]
                    for row in results["real_adaptation"]
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

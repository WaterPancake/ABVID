"""Milestone 6 paired clean/corrupted representation-invariance evaluation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from vehicle_audio.baseline import (
    CLASS_NAMES,
    FEATURE_IMPLEMENTATION_VERSION,
    FeatureConfig,
    LogMelExtractor,
    SmallAudioCNN,
    SplitIndices,
    classification_metrics,
    grouped_stratified_split,
)
from vehicle_audio.transfer_evaluation import validate_transfer_protocol


INVARIANCE_EVALUATION_VERSION = 3
METHODS = ("standard_supervised", "augmentation_only", "representation_invariance")


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = path.read_bytes()
    records = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records, hashlib.sha256(payload).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _read_channel(path: Path, channel: int, sample_rate: int) -> torch.Tensor:
    samples, observed_rate = sf.read(path, dtype="float32", always_2d=True)
    if observed_rate != sample_rate:
        raise ValueError(f"sample-rate mismatch for {path}: {observed_rate} != {sample_rate}")
    if not 0 <= channel < samples.shape[1]:
        raise ValueError(f"requested channel {channel} but {path} has {samples.shape[1]}")
    return torch.from_numpy(samples[:, channel].copy())


def _extract_features(
    records: Sequence[Mapping[str, Any]],
    manifest: Path,
    extractor: LogMelExtractor,
    audio_key: str,
    *,
    channel: int,
    batch_size: int,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    root = manifest.parent
    feature_batches: list[torch.Tensor] = []
    waveform_batch: list[torch.Tensor] = []
    labels: list[int] = []
    label_map = {name: index for index, name in enumerate(CLASS_NAMES)}

    def flush() -> None:
        if not waveform_batch:
            return
        lengths = {waveform.shape[-1] for waveform in waveform_batch}
        if len(lengths) != 1:
            raise ValueError(f"{label} clips have unequal lengths: {sorted(lengths)}")
        with torch.inference_mode():
            feature_batches.append(extractor(torch.stack(waveform_batch)).cpu())
        waveform_batch.clear()

    for index, record in enumerate(records):
        if audio_key not in record:
            raise ValueError(f"record {index} has no {audio_key!r} paired-audio path")
        waveform_batch.append(
            _read_channel(
                root / str(record[audio_key]),
                channel,
                extractor.config.sample_rate,
            )
        )
        labels.append(label_map[str(record["vehicle_class"])])
        if len(waveform_batch) == batch_size:
            flush()
        if (index + 1) % 100 == 0 or index + 1 == len(records):
            print(f"Extracted {label} features: {index + 1}/{len(records)}", flush=True)
    flush()
    features = torch.cat(feature_batches)
    if not torch.isfinite(features).all():
        raise RuntimeError(f"{label} feature extraction produced NaN or Inf")
    return features, torch.tensor(labels, dtype=torch.long)


def _precompute_features(
    synthetic_records: Sequence[Mapping[str, Any]],
    real_records: Sequence[Mapping[str, Any]],
    synthetic_manifest: Path,
    real_manifest: Path,
    output: Path,
    feature_config: FeatureConfig,
    synthetic_sha256: str,
    real_sha256: str,
    *,
    channel: int,
    batch_size: int,
) -> dict[str, Any]:
    cache_path = output / "paired_features.pt"
    expected = {
        "invariance_evaluation_version": INVARIANCE_EVALUATION_VERSION,
        "feature_implementation_version": FEATURE_IMPLEMENTATION_VERSION,
        "feature_config": asdict(feature_config),
        "synthetic_manifest_sha256": synthetic_sha256,
        "real_manifest_sha256": real_sha256,
        "channel": channel,
    }
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if all(cached.get(key) == value for key, value in expected.items()):
            print(f"Using paired feature cache: {cache_path}", flush=True)
            return cached

    extractor = LogMelExtractor(feature_config).eval()
    clean_features, labels = _extract_features(
        synthetic_records,
        synthetic_manifest,
        extractor,
        "clean_path",
        channel=channel,
        batch_size=batch_size,
        label="synthetic clean",
    )
    corrupted_features, corrupted_labels = _extract_features(
        synthetic_records,
        synthetic_manifest,
        extractor,
        "corrupted_path",
        channel=channel,
        batch_size=batch_size,
        label="synthetic corrupted",
    )
    real_features, real_labels = _extract_features(
        real_records,
        real_manifest,
        extractor,
        "audio_path",
        channel=channel,
        batch_size=batch_size,
        label="native real",
    )
    if not torch.equal(labels, corrupted_labels):
        raise RuntimeError("clean/corrupted pair labels are not aligned")
    cached = {
        **expected,
        "clean_features": clean_features,
        "corrupted_features": corrupted_features,
        "synthetic_labels": labels,
        "synthetic_snrs": torch.tensor(
            [float(record["snr_db"]) for record in synthetic_records],
            dtype=torch.float32,
        ),
        "real_features": real_features,
        "real_labels": real_labels,
    }
    torch.save(cached, cache_path)
    return cached


def _geometry_id(record: Mapping[str, Any]) -> str | None:
    geometry = record.get("source_listener_geometry")
    if not isinstance(geometry, Mapping) or geometry.get("geometry_id") is None:
        return None
    return str(geometry["geometry_id"])


def _microphone_applied(record: Mapping[str, Any]) -> bool:
    augmentations = record.get("augmentations")
    if not isinstance(augmentations, Mapping):
        return False
    microphone = augmentations.get("microphone_response")
    return bool(isinstance(microphone, Mapping) and microphone.get("applied"))


def _has_both_classes(
    records: Sequence[Mapping[str, Any]], indices: Sequence[int]
) -> bool:
    return {str(records[index]["vehicle_class"]) for index in indices} == set(CLASS_NAMES)


def build_nuisance_partitions(
    records: Sequence[Mapping[str, Any]],
    split: SplitIndices,
    *,
    heldout_noise: str,
    heldout_geometry: str,
) -> dict[str, tuple[int, ...]]:
    """Create controlled train/test nuisance masks without crossing source groups."""

    def in_distribution(index: int) -> bool:
        record = records[index]
        return (
            str(record.get("background_category")) != heldout_noise
            and _geometry_id(record) != heldout_geometry
            and not _microphone_applied(record)
        )

    train = tuple(index for index in split.train if in_distribution(index))
    validation = tuple(index for index in split.validation if in_distribution(index))
    test_seen = tuple(index for index in split.test if in_distribution(index))
    test_unseen_noise = tuple(
        index
        for index in split.test
        if str(records[index].get("background_category")) == heldout_noise
        and _geometry_id(records[index]) != heldout_geometry
        and not _microphone_applied(records[index])
    )
    test_unseen_microphone = tuple(
        index
        for index in split.test
        if str(records[index].get("background_category")) != heldout_noise
        and _geometry_id(records[index]) != heldout_geometry
        and _microphone_applied(records[index])
    )
    test_unseen_environment = tuple(
        index
        for index in split.test
        if str(records[index].get("background_category")) != heldout_noise
        and _geometry_id(records[index]) == heldout_geometry
        and not _microphone_applied(records[index])
    )
    partitions = {
        "train_in_distribution": train,
        "validation_in_distribution": validation,
        "seen_corruption": test_seen,
        "unseen_noise": test_unseen_noise,
        "unseen_microphone": test_unseen_microphone,
        "unseen_environment": test_unseen_environment,
        "all_corruptions": tuple(split.test),
    }
    for name, indices in partitions.items():
        if not indices:
            raise ValueError(f"nuisance partition {name!r} is empty")
        if not _has_both_classes(records, indices):
            raise ValueError(f"nuisance partition {name!r} does not contain both classes")
    return partitions


def hard_positive_partner_indices(
    records: Sequence[Mapping[str, Any]], train_indices: Sequence[int]
) -> tuple[int | None, ...]:
    """Choose deterministic same-vehicle partners under different conditions."""

    by_vehicle: dict[str, list[int]] = {}
    for index in train_indices:
        vehicle_id = str(records[index].get("vehicle_id") or "")
        if vehicle_id:
            by_vehicle.setdefault(vehicle_id, []).append(index)

    partners: list[int | None] = []
    for index in train_indices:
        record = records[index]
        vehicle_id = str(record.get("vehicle_id") or "")
        candidates = [candidate for candidate in by_vehicle.get(vehicle_id, []) if candidate != index]
        if not candidates:
            partners.append(None)
            continue

        def score(candidate: int) -> tuple[int, str]:
            other = records[candidate]
            differences = sum(
                (
                    str(record.get("recording_session"))
                    != str(other.get("recording_session")),
                    str(record.get("operating_condition"))
                    != str(other.get("operating_condition")),
                    str(record.get("background_category"))
                    != str(other.get("background_category")),
                    _geometry_id(record) != _geometry_id(other),
                    _microphone_applied(record) != _microphone_applied(other),
                )
            )
            return differences, str(other.get("sample_id"))

        partners.append(max(candidates, key=score))
    return tuple(partners)


def hard_negative_partner_indices(
    records: Sequence[Mapping[str, Any]], train_indices: Sequence[int]
) -> tuple[int | None, ...]:
    """Choose deterministic opposite-class vehicles under similar conditions."""

    partners: list[int | None] = []
    for index in train_indices:
        record = records[index]
        candidates = [
            candidate
            for candidate in train_indices
            if str(records[candidate]["vehicle_class"])
            != str(record["vehicle_class"])
            and str(records[candidate].get("vehicle_id") or "")
            != str(record.get("vehicle_id") or "")
        ]
        if not candidates:
            partners.append(None)
            continue

        def score(candidate: int) -> tuple[int, float, str]:
            other = records[candidate]
            metadata_matches = sum(
                (
                    str(record.get("operating_condition"))
                    == str(other.get("operating_condition")),
                    str(record.get("background_category"))
                    == str(other.get("background_category")),
                    _geometry_id(record) == _geometry_id(other),
                    _microphone_applied(record) == _microphone_applied(other),
                )
            )
            snr_distance = abs(
                float(record.get("snr_db") or 0.0)
                - float(other.get("snr_db") or 0.0)
            )
            return metadata_matches, -snr_distance, str(other.get("sample_id"))

        partners.append(max(candidates, key=score))
    return tuple(partners)


def _forward_embeddings(
    model: SmallAudioCNN, features: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = model.encoder(features).flatten(1)
    return model.classifier(embeddings), embeddings


def _evaluate(
    model: SmallAudioCNN,
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: Sequence[int],
    snrs: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    if not indices:
        raise ValueError("evaluation indices must not be empty")
    selected_snrs = (
        torch.zeros(len(indices), dtype=torch.float32)
        if snrs is None
        else snrs[list(indices)]
    )
    loader = DataLoader(
        TensorDataset(
            features[list(indices)],
            labels[list(indices)],
            selected_snrs,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    snr_values: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for feature_batch, target_batch, snr_batch in loader:
            probabilities.append(torch.softmax(model(feature_batch.to(device)), dim=1).cpu())
            targets.append(target_batch)
            snr_values.append(snr_batch)
    probability = torch.cat(probabilities)
    target = torch.cat(targets)
    metrics = classification_metrics(
        probability.argmax(dim=1),
        target,
        torch.cat(snr_values),
        probability,
    )
    if snrs is None:
        metrics.pop("per_snr")
    return metrics


def _fit_method(
    method: str,
    clean_features: torch.Tensor,
    corrupted_features: torch.Tensor,
    labels: torch.Tensor,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    hard_partners: Sequence[int | None],
    hard_negatives: Sequence[int | None],
    *,
    consistency_weight: float,
    hard_positive_weight: float,
    hard_negative_weight: float,
    triplet_margin: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int]:
    if method not in METHODS:
        raise ValueError(f"unknown invariance method: {method}")
    torch.manual_seed(seed)
    model = SmallAudioCNN(len(CLASS_NAMES)).to(device)
    train_labels = labels[list(train_indices)]
    class_counts = torch.bincount(train_labels, minlength=len(CLASS_NAMES)).float()
    if (class_counts == 0).any():
        raise ValueError("training partition must contain both classes")
    class_weights = len(train_labels) / (len(CLASS_NAMES) * class_counts)
    classification_loss = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    resolved_hard_indices = [
        index if partner is None else partner
        for index, partner in zip(train_indices, hard_partners, strict=True)
    ]
    hard_mask = torch.tensor(
        [partner is not None for partner in hard_partners], dtype=torch.bool
    )
    resolved_negative_indices = [
        index if partner is None else partner
        for index, partner in zip(train_indices, hard_negatives, strict=True)
    ]
    triplet_mask = torch.tensor(
        [
            positive is not None and negative is not None
            for positive, negative in zip(hard_partners, hard_negatives, strict=True)
        ],
        dtype=torch.bool,
    )
    loader = DataLoader(
        TensorDataset(
            clean_features[list(train_indices)],
            corrupted_features[list(train_indices)],
            corrupted_features[resolved_hard_indices],
            corrupted_features[resolved_negative_indices],
            hard_mask,
            triplet_mask,
            train_labels,
        ),
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
        total_sum = 0.0
        classification_sum = 0.0
        pair_sum = 0.0
        hard_sum = 0.0
        triplet_sum = 0.0
        seen = 0
        for (
            clean_batch,
            corrupted_batch,
            hard_batch,
            negative_batch,
            valid_hard,
            valid_triplet,
            target_batch,
        ) in loader:
            clean_batch = clean_batch.to(device)
            corrupted_batch = corrupted_batch.to(device)
            hard_batch = hard_batch.to(device)
            negative_batch = negative_batch.to(device)
            valid_hard = valid_hard.to(device)
            valid_triplet = valid_triplet.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            if method == "standard_supervised":
                logits, _ = _forward_embeddings(model, clean_batch)
                class_loss = classification_loss(logits, target_batch)
                pair_loss = torch.zeros((), device=device)
                hard_loss = torch.zeros((), device=device)
                triplet_loss = torch.zeros((), device=device)
            elif method == "augmentation_only":
                logits, _ = _forward_embeddings(model, corrupted_batch)
                class_loss = classification_loss(logits, target_batch)
                pair_loss = torch.zeros((), device=device)
                hard_loss = torch.zeros((), device=device)
                triplet_loss = torch.zeros((), device=device)
            else:
                clean_logits, clean_embeddings = _forward_embeddings(model, clean_batch)
                corrupted_logits, corrupted_embeddings = _forward_embeddings(
                    model, corrupted_batch
                )
                class_loss = 0.5 * (
                    classification_loss(clean_logits, target_batch)
                    + classification_loss(corrupted_logits, target_batch)
                )
                pair_loss = (
                    1.0
                    - F.cosine_similarity(clean_embeddings, corrupted_embeddings, dim=1)
                ).mean()
                if valid_hard.any():
                    _, hard_embeddings = _forward_embeddings(model, hard_batch)
                    hard_loss = (
                        1.0
                        - F.cosine_similarity(
                            clean_embeddings[valid_hard],
                            hard_embeddings[valid_hard],
                            dim=1,
                        )
                    ).mean()
                else:
                    hard_loss = torch.zeros((), device=device)
                if valid_triplet.any():
                    _, negative_embeddings = _forward_embeddings(
                        model, negative_batch
                    )
                    positive_distance = 1.0 - F.cosine_similarity(
                        clean_embeddings[valid_triplet],
                        hard_embeddings[valid_triplet],
                        dim=1,
                    )
                    negative_distance = 1.0 - F.cosine_similarity(
                        clean_embeddings[valid_triplet],
                        negative_embeddings[valid_triplet],
                        dim=1,
                    )
                    triplet_loss = F.relu(
                        positive_distance - negative_distance + triplet_margin
                    ).mean()
                else:
                    triplet_loss = torch.zeros((), device=device)
            loss = (
                class_loss
                + consistency_weight * pair_loss
                + hard_positive_weight * hard_loss
                + hard_negative_weight * triplet_loss
            )
            loss.backward()
            optimizer.step()
            count = len(target_batch)
            total_sum += float(loss.detach().cpu()) * count
            classification_sum += float(class_loss.detach().cpu()) * count
            pair_sum += float(pair_loss.detach().cpu()) * count
            hard_sum += float(hard_loss.detach().cpu()) * count
            triplet_sum += float(triplet_loss.detach().cpu()) * count
            seen += count

        validation = _evaluate(
            model,
            corrupted_features,
            labels,
            validation_indices,
            None,
            batch_size=batch_size,
            device=device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": total_sum / seen,
                "train_classification_loss": classification_sum / seen,
                "train_pair_consistency_loss": pair_sum / seen,
                "train_hard_positive_loss": hard_sum / seen,
                "train_hard_negative_triplet_loss": triplet_sum / seen,
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


def _embedding_consistency(
    state: Mapping[str, torch.Tensor],
    clean_features: torch.Tensor,
    corrupted_features: torch.Tensor,
    indices: Sequence[int],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    model = SmallAudioCNN(len(CLASS_NAMES)).to(device)
    model.load_state_dict(state)
    loader = DataLoader(
        TensorDataset(
            clean_features[list(indices)], corrupted_features[list(indices)]
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    cosine_values: list[torch.Tensor] = []
    l2_values: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for clean_batch, corrupted_batch in loader:
            _, clean_embeddings = _forward_embeddings(model, clean_batch.to(device))
            _, corrupted_embeddings = _forward_embeddings(
                model, corrupted_batch.to(device)
            )
            clean_normalized = F.normalize(clean_embeddings, dim=1)
            corrupted_normalized = F.normalize(corrupted_embeddings, dim=1)
            cosine_values.append(
                (clean_normalized * corrupted_normalized).sum(dim=1).cpu()
            )
            l2_values.append(
                (clean_normalized - corrupted_normalized).norm(dim=1).cpu()
            )
    cosine = torch.cat(cosine_values)
    l2 = torch.cat(l2_values)
    return {
        "support": len(cosine),
        "mean_cosine_similarity": float(cosine.mean()),
        "mean_normalized_l2_distance": float(l2.mean()),
    }


def _split_payload(
    records: Sequence[Mapping[str, Any]], split: SplitIndices
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, indices in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        payload[name] = {
            "groups": sorted(
                {str(records[index]["recording_session"]) for index in indices}
            ),
            "sample_ids": [str(records[index]["sample_id"]) for index in indices],
        }
    return payload


def _write_plots_and_csv(output: Path, methods: Mapping[str, Any]) -> None:
    condition_rows: list[dict[str, Any]] = []
    snr_rows: list[dict[str, Any]] = []
    condition_order = (
        "seen_corruption",
        "unseen_noise",
        "unseen_microphone",
        "unseen_environment",
        "all_corruptions",
        "native_real",
    )
    for method, result in methods.items():
        for condition in condition_order:
            metrics = result["evaluations"][condition]
            condition_rows.append(
                {
                    "method": method,
                    "condition": condition,
                    "support": metrics["support"],
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                }
            )
        for snr, metrics in result["evaluations"]["all_corruptions"]["per_snr"].items():
            snr_rows.append(
                {
                    "method": method,
                    "snr_db": float(snr),
                    "support": metrics["support"],
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                }
            )

    for filename, rows in (
        ("condition_summary.csv", condition_rows),
        ("snr_robustness.csv", snr_rows),
    ):
        with (output / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    for method in METHODS:
        rows = sorted(
            (row for row in snr_rows if row["method"] == method),
            key=lambda row: float(row["snr_db"]),
        )
        axis.plot(
            [row["snr_db"] for row in rows],
            [row["balanced_accuracy"] for row in rows],
            marker="o",
            label=method.replace("_", " "),
        )
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Balanced accuracy")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Milestone 6 robustness on held-out synthetic sessions")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "accuracy_vs_snr.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10.5, 5.0))
    width = 0.25
    x_values = list(range(len(condition_order)))
    for method_index, method in enumerate(METHODS):
        values = [
            next(
                row["balanced_accuracy"]
                for row in condition_rows
                if row["method"] == method and row["condition"] == condition
            )
            for condition in condition_order
        ]
        axis.bar(
            [x + (method_index - 1) * width for x in x_values],
            values,
            width=width,
            label=method.replace("_", " "),
        )
    axis.set_xticks(x_values, [value.replace("_", "\n") for value in condition_order])
    axis.set_ylabel("Balanced accuracy")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Controlled nuisance and real-transfer comparison")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "condition_comparison.png", dpi=160)
    plt.close(figure)


def evaluate_invariance(
    synthetic_manifest_path: str | Path,
    real_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    heldout_noise: str = "road traffic",
    heldout_geometry: str = "far_offset",
    consistency_weight: float = 0.5,
    hard_positive_weight: float = 0.1,
    hard_negative_weight: float = 0.1,
    triplet_margin: float = 0.2,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 42,
    channel: int = 0,
    device_name: str = "auto",
) -> dict[str, Any]:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if consistency_weight < 0 or hard_positive_weight < 0 or hard_negative_weight < 0:
        raise ValueError("consistency weights must be nonnegative")
    if triplet_margin <= 0:
        raise ValueError("triplet_margin must be positive")
    if seed < 0 or channel < 0:
        raise ValueError("seed and channel must be nonnegative")
    synthetic_manifest = Path(synthetic_manifest_path)
    real_manifest = Path(real_manifest_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    synthetic_records, synthetic_sha256 = _load_jsonl(synthetic_manifest)
    real_records, real_sha256 = _load_jsonl(real_manifest)
    audit = validate_transfer_protocol(synthetic_records, real_records)
    for index, record in enumerate(synthetic_records):
        if not record.get("clean_path") or not record.get("corrupted_path"):
            raise ValueError(f"synthetic record {index} is not a clean/corrupted pair")

    sample_rate = int(synthetic_records[0]["sample_rate"])
    feature_config = FeatureConfig(sample_rate=sample_rate, f_max=sample_rate / 2.0)
    cached = _precompute_features(
        synthetic_records,
        real_records,
        synthetic_manifest,
        real_manifest,
        output,
        feature_config,
        synthetic_sha256,
        real_sha256,
        channel=channel,
        batch_size=batch_size,
    )
    clean_features = cached["clean_features"]
    corrupted_features = cached["corrupted_features"]
    synthetic_labels = cached["synthetic_labels"]
    synthetic_snrs = cached["synthetic_snrs"]
    real_features = cached["real_features"]
    real_labels = cached["real_labels"]

    synthetic_split = grouped_stratified_split(synthetic_records, seed)
    real_split = grouped_stratified_split(real_records, seed)
    nuisance = build_nuisance_partitions(
        synthetic_records,
        synthetic_split,
        heldout_noise=heldout_noise,
        heldout_geometry=heldout_geometry,
    )
    hard_partners = hard_positive_partner_indices(
        synthetic_records, nuisance["train_in_distribution"]
    )
    hard_negatives = hard_negative_partner_indices(
        synthetic_records, nuisance["train_in_distribution"]
    )
    hard_pair_count = sum(partner is not None for partner in hard_partners)
    hard_negative_pair_count = sum(
        positive is not None and negative is not None
        for positive, negative in zip(hard_partners, hard_negatives, strict=True)
    )
    device = (
        torch.device(device_name)
        if device_name != "auto"
        else (
            torch.device("cuda")
            if torch.cuda.is_available()
            else (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        )
    )

    method_results: dict[str, Any] = {}
    model_states: dict[str, dict[str, torch.Tensor]] = {}
    for method in METHODS:
        state, history, best_epoch = _fit_method(
            method,
            clean_features,
            corrupted_features,
            synthetic_labels,
            nuisance["train_in_distribution"],
            nuisance["validation_in_distribution"],
            hard_partners,
            hard_negatives,
            consistency_weight=consistency_weight,
            hard_positive_weight=hard_positive_weight,
            hard_negative_weight=hard_negative_weight,
            triplet_margin=triplet_margin,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
        )
        model_states[method] = state
        model = SmallAudioCNN(len(CLASS_NAMES)).to(device)
        model.load_state_dict(state)
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
            "seen_corruption": _evaluate(
                model,
                corrupted_features,
                synthetic_labels,
                nuisance["seen_corruption"],
                synthetic_snrs,
                batch_size=batch_size,
                device=device,
            ),
            "unseen_noise": _evaluate(
                model,
                corrupted_features,
                synthetic_labels,
                nuisance["unseen_noise"],
                synthetic_snrs,
                batch_size=batch_size,
                device=device,
            ),
            "unseen_microphone": _evaluate(
                model,
                corrupted_features,
                synthetic_labels,
                nuisance["unseen_microphone"],
                synthetic_snrs,
                batch_size=batch_size,
                device=device,
            ),
            "unseen_environment": _evaluate(
                model,
                corrupted_features,
                synthetic_labels,
                nuisance["unseen_environment"],
                synthetic_snrs,
                batch_size=batch_size,
                device=device,
            ),
            "all_corruptions": _evaluate(
                model,
                corrupted_features,
                synthetic_labels,
                nuisance["all_corruptions"],
                synthetic_snrs,
                batch_size=batch_size,
                device=device,
            ),
            "native_real": _evaluate(
                model,
                real_features,
                real_labels,
                real_split.test,
                None,
                batch_size=batch_size,
                device=device,
            ),
        }
        method_results[method] = {
            "training_input": {
                "standard_supervised": "clean_only",
                "augmentation_only": "corrupted_only",
                "representation_invariance": "paired_clean_corrupted",
            }[method],
            "best_epoch": best_epoch,
            "history": history,
            "embedding_consistency": _embedding_consistency(
                state,
                clean_features,
                corrupted_features,
                nuisance["all_corruptions"],
                batch_size=batch_size,
                device=device,
            ),
            "evaluations": evaluations,
        }

    _write_plots_and_csv(output, method_results)
    split_payload = {
        "strategy": "recording_session_grouped_with_controlled_nuisance_holdouts",
        "seed": seed,
        "synthetic": _split_payload(synthetic_records, synthetic_split),
        "real": _split_payload(real_records, real_split),
        "heldout_noise": heldout_noise,
        "heldout_environment_proxy": {
            "field": "source_listener_geometry.geometry_id",
            "value": heldout_geometry,
            "limitation": "Propagation geometry proxy; no impulse-response corpus was available.",
        },
        "heldout_microphone_rule": "augmentations.microphone_response.applied == true",
        "nuisance_partitions": {
            name: {
                "sample_ids": [str(synthetic_records[index]["sample_id"]) for index in indices],
                "support": len(indices),
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
        "invariance_evaluation_version": INVARIANCE_EVALUATION_VERSION,
        "methods": list(METHODS),
        "model": "small_log_mel_cnn",
        "feature_config": asdict(feature_config),
        "heldout_noise": heldout_noise,
        "heldout_geometry": heldout_geometry,
        "consistency_objective": "one_minus_cosine_similarity",
        "consistency_weight": consistency_weight,
        "hard_positive_objective": "same_vehicle_one_minus_cosine_similarity",
        "hard_positive_weight": hard_positive_weight,
        "hard_positive_pair_count": hard_pair_count,
        "easy_negative_objective": "weighted_cross_entropy_between_coarse_classes",
        "hard_negative_objective": "cross_class_metadata_matched_cosine_triplet_margin",
        "hard_negative_weight": hard_negative_weight,
        "hard_negative_pair_count": hard_negative_pair_count,
        "triplet_margin": triplet_margin,
        "initialization_and_batch_order_seed_shared_across_methods": True,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
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
        "test_domain": "held-out procedural corruption conditions and held-out native real recording sessions",
        "protocol_status": "complete",
        "real_corpus_audit": audit,
        "training_config": training_config,
        "split_counts": {
            "synthetic_base": {
                "train": len(synthetic_split.train),
                "validation": len(synthetic_split.validation),
                "test": len(synthetic_split.test),
            },
            "synthetic_nuisance": {name: len(indices) for name, indices in nuisance.items()},
            "real": {
                "train": len(real_split.train),
                "validation": len(real_split.validation),
                "test": len(real_split.test),
            },
        },
        "methods": method_results,
        "artifacts": {
            "checkpoint": "models.pt",
            "splits": "splits.json",
            "condition_csv": "condition_summary.csv",
            "snr_csv": "snr_robustness.csv",
            "condition_plot": "condition_comparison.png",
            "snr_plot": "accuracy_vs_snr.png",
        },
    }
    torch.save(
        {
            "models": model_states,
            "training_config": training_config,
            "synthetic_manifest_sha256": synthetic_sha256,
            "real_manifest_sha256": real_sha256,
            "class_names": list(CLASS_NAMES),
        },
        output / "models.pt",
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
                "dataset_version": dataset_version,
                "train_test_split": "splits.json",
                "model_checkpoint": "models.pt",
                "metrics": "metrics.json",
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout-noise", default="road traffic")
    parser.add_argument("--heldout-geometry", default="far_offset")
    parser.add_argument("--consistency-weight", type=float, default=0.5)
    parser.add_argument("--hard-positive-weight", type=float, default=0.1)
    parser.add_argument("--hard-negative-weight", type=float, default=0.1)
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = evaluate_invariance(
        args.synthetic_manifest,
        args.real_manifest,
        args.output,
        heldout_noise=args.heldout_noise,
        heldout_geometry=args.heldout_geometry,
        consistency_weight=args.consistency_weight,
        hard_positive_weight=args.hard_positive_weight,
        hard_negative_weight=args.hard_negative_weight,
        triplet_margin=args.triplet_margin,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
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
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

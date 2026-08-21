"""Leakage-safe Milestone 5 synthetic-to-real transfer evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
import torch
from torch import nn
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
from vehicle_audio.real_corpus import audit_real_manifest, validate_real_manifest_record


TRANSFER_EVALUATION_VERSION = 3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable_not_git_checkout"


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = path.read_bytes()
    records = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    classes = {str(record.get("vehicle_class")) for record in records}
    if classes != set(CLASS_NAMES):
        raise ValueError(f"manifest must contain {list(CLASS_NAMES)}, found {sorted(classes)}")
    sample_rates = {int(record["sample_rate"]) for record in records}
    if len(sample_rates) != 1:
        raise ValueError(f"manifest has multiple sample rates: {sorted(sample_rates)}")
    return records, hashlib.sha256(payload).hexdigest()


def validate_transfer_protocol(
    synthetic_records: Sequence[Mapping[str, Any]],
    real_records: Sequence[Mapping[str, Any]],
    *,
    require_content_review: bool = True,
) -> dict[str, Any]:
    synthetic_domains = {str(record.get("source_domain")) for record in synthetic_records}
    if synthetic_domains != {"procedural_synthetic"}:
        raise ValueError(
            "synthetic manifest must contain only source_domain='procedural_synthetic', "
            f"found {sorted(synthetic_domains)}"
        )
    for record in real_records:
        validate_real_manifest_record(record)
    audit = audit_real_manifest(real_records)
    if not audit["session_split_ready"]:
        counts = audit["recording_session_counts"]
        raise ValueError(
            "Milestone 5 grouped evaluation requires at least three independent "
            "recording sessions per class; found "
            f"tracked={counts['tracked']}, wheeled={counts['wheeled']}"
        )
    if not audit["provenance_complete"]:
        raise ValueError(
            "real manifest has incomplete provenance for sources: "
            + ", ".join(audit["missing_provenance_source_ids"])
        )
    if require_content_review and not audit["content_review_complete"]:
        raise ValueError(
            "real manifest contains unreviewed whole-recording labels for sources: "
            + ", ".join(audit["unreviewed_source_ids"])
        )
    synthetic_shapes = {
        (int(record["sample_rate"]), int(record["num_samples"]))
        for record in synthetic_records
    }
    real_shapes = {
        (int(record["sample_rate"]), int(record["num_samples"]))
        for record in real_records
    }
    if len(synthetic_shapes) != 1 or len(real_shapes) != 1:
        raise ValueError(
            "each transfer domain must have one observation shape; found "
            f"synthetic={sorted(synthetic_shapes)}, real={sorted(real_shapes)}"
        )
    if synthetic_shapes != real_shapes:
        raise ValueError(
            "synthetic and real observation shapes differ; align sample rate and clip "
            f"duration before evaluation: {sorted(synthetic_shapes)} vs "
            f"{sorted(real_shapes)}"
        )
    sample_rate, num_samples = next(iter(synthetic_shapes))
    return {
        **audit,
        "observation_shape": {
            "sample_rate": sample_rate,
            "num_samples": num_samples,
            "duration_seconds": num_samples / sample_rate,
        },
    }


def stratified_fraction_indices(
    labels: torch.Tensor,
    candidate_indices: Sequence[int],
    fraction: float,
    *,
    seed: int,
) -> tuple[int, ...]:
    """Return a deterministic class-stratified subset of an existing train split."""

    if not 0 < fraction <= 1:
        raise ValueError("real-data fraction must be in (0, 1]")
    selected: list[int] = []
    for class_index in range(len(CLASS_NAMES)):
        class_indices = sorted(
            index for index in candidate_indices if int(labels[index]) == class_index
        )
        if not class_indices:
            raise ValueError("candidate real train split must contain both classes")
        random.Random(seed + 10_007 * class_index).shuffle(class_indices)
        count = max(1, round(len(class_indices) * fraction))
        selected.extend(class_indices[:count])
    random.Random(seed).shuffle(selected)
    return tuple(selected)


def _read_channel(path: Path, channel: int, sample_rate: int) -> torch.Tensor:
    samples, observed_rate = sf.read(path, dtype="float32", always_2d=True)
    if observed_rate != sample_rate:
        raise ValueError(f"sample-rate mismatch for {path}: {observed_rate} != {sample_rate}")
    if not 0 <= channel < samples.shape[1]:
        raise ValueError(f"requested channel {channel} but {path} has {samples.shape[1]}")
    return torch.from_numpy(samples[:, channel].copy())


def _extract_domain_features(
    records: Sequence[Mapping[str, Any]],
    manifest: Path,
    extractor: LogMelExtractor,
    *,
    channel: int,
    batch_size: int,
    domain_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    root = manifest.parent
    batches: list[torch.Tensor] = []
    waveforms: list[torch.Tensor] = []
    labels: list[int] = []
    label_map = {name: index for index, name in enumerate(CLASS_NAMES)}

    def flush() -> None:
        if not waveforms:
            return
        lengths = {waveform.shape[-1] for waveform in waveforms}
        if len(lengths) != 1:
            raise ValueError(f"{domain_name} clips have unequal lengths: {sorted(lengths)}")
        with torch.inference_mode():
            batches.append(extractor(torch.stack(waveforms)).cpu())
        waveforms.clear()

    for index, record in enumerate(records):
        audio_key = "audio_path" if domain_name == "real" else "corrupted_path"
        waveforms.append(
            _read_channel(root / str(record[audio_key]), channel, extractor.config.sample_rate)
        )
        labels.append(label_map[str(record["vehicle_class"])])
        if len(waveforms) == batch_size:
            flush()
        if (index + 1) % 100 == 0 or index + 1 == len(records):
            print(f"Extracted {domain_name} log-Mel features: {index + 1}/{len(records)}", flush=True)
    flush()
    features = torch.cat(batches)
    if not torch.isfinite(features).all():
        raise RuntimeError(f"{domain_name} feature extraction produced NaN or Inf")
    return features, torch.tensor(labels, dtype=torch.long)


def _precompute_features(
    synthetic_records: Sequence[Mapping[str, Any]],
    real_records: Sequence[Mapping[str, Any]],
    synthetic_manifest: Path,
    real_manifest: Path,
    synthetic_sha256: str,
    real_sha256: str,
    output: Path,
    feature_config: FeatureConfig,
    *,
    channel: int,
    batch_size: int,
) -> dict[str, Any]:
    cache_path = output / "domain_features.pt"
    expected = {
        "transfer_evaluation_version": TRANSFER_EVALUATION_VERSION,
        "feature_implementation_version": FEATURE_IMPLEMENTATION_VERSION,
        "feature_config": asdict(feature_config),
        "synthetic_manifest_sha256": synthetic_sha256,
        "real_manifest_sha256": real_sha256,
        "channel": channel,
    }
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if all(cached.get(key) == value for key, value in expected.items()):
            print(f"Using transfer feature cache: {cache_path}", flush=True)
            return cached
    extractor = LogMelExtractor(feature_config).eval()
    synthetic_features, synthetic_labels = _extract_domain_features(
        synthetic_records,
        synthetic_manifest,
        extractor,
        channel=channel,
        batch_size=batch_size,
        domain_name="synthetic",
    )
    real_features, real_labels = _extract_domain_features(
        real_records,
        real_manifest,
        extractor,
        channel=channel,
        batch_size=batch_size,
        domain_name="real",
    )
    cache = {
        **expected,
        "synthetic_features": synthetic_features,
        "synthetic_labels": synthetic_labels,
        "real_features": real_features,
        "real_labels": real_labels,
    }
    torch.save(cache, cache_path)
    return cache


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _evaluate(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: Sequence[int],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    if not indices:
        raise ValueError("evaluation indices must not be empty")
    dataset = TensorDataset(features[list(indices)], labels[list(indices)])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for feature_batch, target_batch in loader:
            probabilities.append(torch.softmax(model(feature_batch.to(device)), dim=1).cpu())
            targets.append(target_batch)
    probability = torch.cat(probabilities)
    target = torch.cat(targets)
    # Native real SNR is unknown. A temporary tensor satisfies the shared metric
    # implementation, and the synthetic per-SNR field is removed before output.
    metrics = classification_metrics(
        probability.argmax(dim=1),
        target,
        torch.zeros(len(target)),
        probability,
    )
    metrics.pop("per_snr")
    return metrics


def _fit(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    *,
    initial_state: Mapping[str, torch.Tensor] | None,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int, dict[str, Any]]:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("training parameters must be positive")
    torch.manual_seed(seed)
    model = SmallAudioCNN(len(CLASS_NAMES))
    if initial_state is not None:
        model.load_state_dict(initial_state)
    model.to(device)
    class_counts = torch.bincount(train_labels, minlength=len(CLASS_NAMES)).float()
    if (class_counts == 0).any():
        raise ValueError("training data must contain both classes")
    class_weights = len(train_labels) / (len(CLASS_NAMES) * class_counts)
    loss_function = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    validation_indices = tuple(range(len(validation_labels)))
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for feature_batch, target_batch in train_loader:
            feature_batch = feature_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(feature_batch), target_batch)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(target_batch)
            seen += len(target_batch)
        validation = _evaluate(
            model,
            validation_features,
            validation_labels,
            validation_indices,
            batch_size=batch_size,
            device=device,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "validation_accuracy": validation["accuracy"],
            "validation_balanced_accuracy": validation["balanced_accuracy"],
            "validation_macro_f1": validation["macro_f1"],
        }
        history.append(epoch_record)
        if validation["balanced_accuracy"] > best_score:
            best_score = validation["balanced_accuracy"]
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    model.to(device)
    validation = _evaluate(
        model,
        validation_features,
        validation_labels,
        validation_indices,
        batch_size=batch_size,
        device=device,
    )
    return best_state, history, best_epoch, validation


def _test_state(
    state: Mapping[str, torch.Tensor],
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: Sequence[int],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model = SmallAudioCNN(len(CLASS_NAMES)).to(device)
    model.load_state_dict(state)
    return _evaluate(
        model,
        features,
        labels,
        indices,
        batch_size=batch_size,
        device=device,
    )


def _split_metadata(
    records: Sequence[Mapping[str, Any]], split: SplitIndices, group_field: str
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, indices in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        values[name] = {
            "groups": sorted({str(records[index][group_field]) for index in indices}),
            "sample_ids": [str(records[index]["sample_id"]) for index in indices],
        }
    group_sets = [set(values[name]["groups"]) for name in ("train", "validation", "test")]
    if group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2]:
        raise RuntimeError("recording-session leakage in transfer split")
    return values


def _embedding_projection(
    pretrained_state: Mapping[str, torch.Tensor],
    synthetic_features: torch.Tensor,
    real_features: torch.Tensor,
    synthetic_labels: torch.Tensor,
    real_labels: torch.Tensor,
    synthetic_records: Sequence[Mapping[str, Any]],
    real_records: Sequence[Mapping[str, Any]],
    synthetic_indices: Sequence[int],
    real_indices: Sequence[int],
    output: Path,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model = SmallAudioCNN(len(CLASS_NAMES)).to(device)
    model.load_state_dict(pretrained_state)
    model.eval()

    def encode(features: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
        values: list[torch.Tensor] = []
        loader = DataLoader(
            TensorDataset(features[list(indices)]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        with torch.inference_mode():
            for (feature_batch,) in loader:
                values.append(model.encoder(feature_batch.to(device)).flatten(1).cpu())
        return torch.cat(values)

    synthetic_embeddings = encode(synthetic_features, synthetic_indices)
    real_embeddings = encode(real_features, real_indices)
    embeddings = torch.cat((synthetic_embeddings, real_embeddings)).to(torch.float64)
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    _, singular_values, right = torch.linalg.svd(centered, full_matrices=False)
    projection = (centered @ right[:2].T).to(torch.float32)
    variance = singular_values.square()
    explained = (variance[:2] / variance.sum().clamp_min(1e-12)).tolist()

    rows: list[dict[str, Any]] = []
    offset = 0
    for domain, indices, labels, records in (
        ("synthetic", synthetic_indices, synthetic_labels, synthetic_records),
        ("real", real_indices, real_labels, real_records),
    ):
        for local_index, record_index in enumerate(indices):
            rows.append(
                {
                    "sample_id": str(records[record_index]["sample_id"]),
                    "recording_session": str(records[record_index]["recording_session"]),
                    "domain": domain,
                    "vehicle_class": CLASS_NAMES[int(labels[record_index])],
                    "pc1": float(projection[offset + local_index, 0]),
                    "pc2": float(projection[offset + local_index, 1]),
                }
            )
        offset += len(indices)
    with (output / "embedding_projection.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    styles = {
        ("synthetic", "tracked"): ("tab:blue", "o"),
        ("synthetic", "wheeled"): ("tab:orange", "o"),
        ("real", "tracked"): ("tab:blue", "x"),
        ("real", "wheeled"): ("tab:orange", "x"),
    }
    for (domain, vehicle_class), (color, marker) in styles.items():
        selected = [row for row in rows if row["domain"] == domain and row["vehicle_class"] == vehicle_class]
        axis.scatter(
            [row["pc1"] for row in selected],
            [row["pc2"] for row in selected],
            color=color,
            marker=marker,
            alpha=0.75,
            label=f"{domain} {vehicle_class}",
        )
    axis.set(
        title="Pretrained encoder embeddings (PCA visualization only)",
        xlabel=f"PC1 ({100 * explained[0]:.1f}% variance)",
        ylabel=f"PC2 ({100 * explained[1]:.1f}% variance)",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output / "embedding_projection.png", dpi=160)
    plt.close(figure)
    return {
        "method": "PCA via exact centered SVD",
        "input_representation": "64-dimensional pooled synthetic-pretrained CNN encoder",
        "explained_variance_ratio": explained,
        "synthetic_support": len(synthetic_indices),
        "real_support": len(real_indices),
        "interpretation_boundary": "visualization_only_not_representation_quality_evidence",
        "csv": "embedding_projection.csv",
        "plot": "embedding_projection.png",
    }


def _write_learning_curve(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with (output / "learning_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    adaptation = [row for row in rows if row["series"] == "synthetic_pretrained"]
    real_only = [row for row in rows if row["series"] == "real_only"]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.plot(
        [100 * float(row["requested_real_fraction"]) for row in adaptation],
        [float(row["balanced_accuracy"]) for row in adaptation],
        marker="o",
        label="Synthetic pretraining + real adaptation",
    )
    axis.scatter(
        [100.0],
        [float(real_only[0]["balanced_accuracy"])],
        marker="s",
        s=70,
        label="Real only (100% real train split)",
    )
    axis.set(
        title="Synthetic-to-real adaptation learning curve",
        xlabel="Requested fraction of fixed real training split (%)",
        ylabel="Balanced accuracy on fixed held-out real sessions",
        ylim=(0.0, 1.02),
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(output / "learning_curve.png", dpi=160)
    plt.close(figure)


def evaluate_transfer(
    synthetic_manifest_path: str | Path,
    real_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    real_fractions: Sequence[float] = (0.01, 0.05, 0.10, 0.25),
    seed: int = 42,
    split_seed: int | None = None,
    channel: int = 0,
    pretrain_epochs: int = 10,
    finetune_epochs: int = 5,
    real_only_epochs: int = 10,
    batch_size: int = 32,
    pretrain_learning_rate: float = 1e-3,
    finetune_learning_rate: float = 2e-4,
    device_name: str = "auto",
    group_field: str = "recording_session",
    require_content_review: bool = True,
) -> dict[str, Any]:
    resolved_split_seed = seed if split_seed is None else split_seed
    if seed < 0 or resolved_split_seed < 0 or channel < 0:
        raise ValueError("seed, split_seed, and channel must be nonnegative")
    fractions = tuple(float(value) for value in real_fractions)
    if not fractions or any(not 0 < value <= 1 for value in fractions):
        raise ValueError("real_fractions must contain values in (0, 1]")
    if tuple(sorted(set(fractions))) != fractions:
        raise ValueError("real_fractions must be unique and increasing")
    synthetic_manifest = Path(synthetic_manifest_path)
    real_manifest = Path(real_manifest_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    synthetic_records, synthetic_sha256 = _load_jsonl(synthetic_manifest)
    real_records, real_sha256 = _load_jsonl(real_manifest)
    audit = validate_transfer_protocol(
        synthetic_records,
        real_records,
        require_content_review=require_content_review,
    )
    if any(group_field not in record for record in (*synthetic_records, *real_records)):
        raise ValueError(f"all records must contain group field {group_field!r}")
    random.seed(seed)
    torch.manual_seed(seed)
    synthetic_split = grouped_stratified_split(
        synthetic_records, resolved_split_seed, group_field=group_field
    )
    real_split = grouped_stratified_split(
        real_records, resolved_split_seed, group_field=group_field
    )
    synthetic_split_metadata = _split_metadata(
        synthetic_records, synthetic_split, group_field
    )
    real_split_metadata = _split_metadata(real_records, real_split, group_field)
    feature_config = FeatureConfig(
        sample_rate=int(synthetic_records[0]["sample_rate"]),
        f_max=int(synthetic_records[0]["sample_rate"]) / 2.0,
    )
    cached = _precompute_features(
        synthetic_records,
        real_records,
        synthetic_manifest,
        real_manifest,
        synthetic_sha256,
        real_sha256,
        output,
        feature_config,
        channel=channel,
        batch_size=batch_size,
    )
    synthetic_features = cached["synthetic_features"]
    synthetic_labels = cached["synthetic_labels"]
    real_features = cached["real_features"]
    real_labels = cached["real_labels"]
    device = _resolve_device(device_name)

    pretrained_state, pretrained_history, pretrained_best_epoch, pretrained_validation = _fit(
        synthetic_features[list(synthetic_split.train)],
        synthetic_labels[list(synthetic_split.train)],
        synthetic_features[list(synthetic_split.validation)],
        synthetic_labels[list(synthetic_split.validation)],
        initial_state=None,
        seed=seed,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        learning_rate=pretrain_learning_rate,
        device=device,
    )
    synthetic_only_test = _test_state(
        pretrained_state,
        real_features,
        real_labels,
        real_split.test,
        batch_size=batch_size,
        device=device,
    )
    real_only_state, real_only_history, real_only_best_epoch, real_only_validation = _fit(
        real_features[list(real_split.train)],
        real_labels[list(real_split.train)],
        real_features[list(real_split.validation)],
        real_labels[list(real_split.validation)],
        initial_state=None,
        seed=seed,
        epochs=real_only_epochs,
        batch_size=batch_size,
        learning_rate=pretrain_learning_rate,
        device=device,
    )
    real_only_test = _test_state(
        real_only_state,
        real_features,
        real_labels,
        real_split.test,
        batch_size=batch_size,
        device=device,
    )

    adaptation: dict[str, Any] = {}
    model_states: dict[str, Any] = {
        "synthetic_pretrained": pretrained_state,
        "real_only": real_only_state,
    }
    learning_rows: list[dict[str, Any]] = [
        {
            "series": "synthetic_pretrained",
            "experiment": "B_synthetic_only",
            "requested_real_fraction": 0.0,
            "actual_real_fraction": 0.0,
            "real_train_samples": 0,
            "balanced_accuracy": synthetic_only_test["balanced_accuracy"],
            "accuracy": synthetic_only_test["accuracy"],
            "macro_f1": synthetic_only_test["macro_f1"],
            "test_support": synthetic_only_test["support"],
        }
    ]
    for fraction_index, fraction in enumerate(fractions):
        indices = stratified_fraction_indices(
            real_labels,
            real_split.train,
            fraction,
            seed=resolved_split_seed,
        )
        state, history, best_epoch, validation = _fit(
            real_features[list(indices)],
            real_labels[list(indices)],
            real_features[list(real_split.validation)],
            real_labels[list(real_split.validation)],
            initial_state=pretrained_state,
            seed=seed + fraction_index + 1,
            epochs=finetune_epochs,
            batch_size=batch_size,
            learning_rate=finetune_learning_rate,
            device=device,
        )
        test = _test_state(
            state,
            real_features,
            real_labels,
            real_split.test,
            batch_size=batch_size,
            device=device,
        )
        key = f"{100 * fraction:g}_percent_real"
        actual_fraction = len(indices) / len(real_split.train)
        adaptation[key] = {
            "requested_real_fraction": fraction,
            "actual_real_fraction": actual_fraction,
            "real_train_samples": len(indices),
            "real_train_class_counts": {
                CLASS_NAMES[index]: int(count)
                for index, count in enumerate(
                    torch.bincount(real_labels[list(indices)], minlength=len(CLASS_NAMES))
                )
            },
            "real_train_sample_ids": [str(real_records[index]["sample_id"]) for index in indices],
            "real_train_groups": sorted({str(real_records[index][group_field]) for index in indices}),
            "best_epoch": best_epoch,
            "history": history,
            "validation": validation,
            "test": test,
        }
        model_states[f"adaptation_{key}"] = state
        learning_rows.append(
            {
                "series": "synthetic_pretrained",
                "experiment": f"C_{key}",
                "requested_real_fraction": fraction,
                "actual_real_fraction": actual_fraction,
                "real_train_samples": len(indices),
                "balanced_accuracy": test["balanced_accuracy"],
                "accuracy": test["accuracy"],
                "macro_f1": test["macro_f1"],
                "test_support": test["support"],
            }
        )
    learning_rows.append(
        {
            "series": "real_only",
            "experiment": "A_real_only",
            "requested_real_fraction": 1.0,
            "actual_real_fraction": 1.0,
            "real_train_samples": len(real_split.train),
            "balanced_accuracy": real_only_test["balanced_accuracy"],
            "accuracy": real_only_test["accuracy"],
            "macro_f1": real_only_test["macro_f1"],
            "test_support": real_only_test["support"],
        }
    )
    _write_learning_curve(output, learning_rows)
    embedding_analysis = _embedding_projection(
        pretrained_state,
        synthetic_features,
        real_features,
        synthetic_labels,
        real_labels,
        synthetic_records,
        real_records,
        synthetic_split.test,
        real_split.test,
        output,
        batch_size=batch_size,
        device=device,
    )
    splits_payload = {
        "strategy": "recording_session_grouped",
        "group_field": group_field,
        "seed": resolved_split_seed,
        "synthetic": synthetic_split_metadata,
        "real": real_split_metadata,
    }
    (output / "splits.json").write_text(
        json.dumps(splits_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    dataset_version = hashlib.sha256(
        f"{synthetic_sha256}:{real_sha256}".encode("ascii")
    ).hexdigest()
    training_config = {
        "transfer_evaluation_version": TRANSFER_EVALUATION_VERSION,
        "model": "small_log_mel_cnn",
        "channel": channel,
        "seed": seed,
        "split_seed": resolved_split_seed,
        "device": str(device),
        "group_field": group_field,
        "real_fractions": list(fractions),
        "pretrain_epochs": pretrain_epochs,
        "finetune_epochs": finetune_epochs,
        "real_only_epochs": real_only_epochs,
        "batch_size": batch_size,
        "pretrain_learning_rate": pretrain_learning_rate,
        "finetune_learning_rate": finetune_learning_rate,
        "require_content_review": require_content_review,
    }
    results = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset_version": dataset_version,
        "synthetic_manifest": str(synthetic_manifest),
        "synthetic_manifest_sha256": synthetic_sha256,
        "real_manifest": str(real_manifest),
        "real_manifest_sha256": real_sha256,
        "test_domain": "held-out native real recording sessions",
        "protocol_status": "complete" if require_content_review else "engineering_smoke_unreviewed",
        "real_corpus_audit": audit,
        "training_config": training_config,
        "feature_config": asdict(feature_config),
        "split_counts": {
            "synthetic": {
                "train": len(synthetic_split.train),
                "validation": len(synthetic_split.validation),
                "test": len(synthetic_split.test),
            },
            "real": {
                "train": len(real_split.train),
                "validation": len(real_split.validation),
                "test": len(real_split.test),
            },
        },
        "experiments": {
            "A_real_only": {
                "train_domain": "real",
                "test_domain": "real",
                "best_epoch": real_only_best_epoch,
                "history": real_only_history,
                "validation": real_only_validation,
                "test": real_only_test,
            },
            "B_synthetic_only": {
                "train_domain": "synthetic",
                "test_domain": "real",
                "best_epoch": pretrained_best_epoch,
                "history": pretrained_history,
                "synthetic_validation": pretrained_validation,
                "test": synthetic_only_test,
            },
            "C_synthetic_pretraining_limited_real": adaptation,
        },
        "embedding_analysis": embedding_analysis,
        "artifacts": {
            "checkpoint": "models.pt",
            "splits": "splits.json",
            "metrics": "metrics.json",
            "learning_curve_csv": "learning_curve.csv",
            "learning_curve_plot": "learning_curve.png",
            "embedding_csv": "embedding_projection.csv",
            "embedding_plot": "embedding_projection.png",
        },
    }
    torch.save(
        {
            "models": model_states,
            "class_names": list(CLASS_NAMES),
            "feature_config": asdict(feature_config),
            "training_config": training_config,
            "synthetic_manifest_sha256": synthetic_sha256,
            "real_manifest_sha256": real_sha256,
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
                "split_seed": resolved_split_seed,
                "dataset_version": dataset_version,
                "synthetic_manifest_sha256": synthetic_sha256,
                "real_manifest_sha256": real_sha256,
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
    parser.add_argument("--real-fraction", type=float, action="append", dest="real_fractions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        help="fixed grouped-split/subset seed; defaults to --seed",
    )
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--real-only-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
    parser.add_argument("--finetune-learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--group-field", default="recording_session")
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="engineering smoke only; output is marked non-reportable",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = evaluate_transfer(
        args.synthetic_manifest,
        args.real_manifest,
        args.output,
        real_fractions=(0.01, 0.05, 0.10, 0.25) if args.real_fractions is None else args.real_fractions,
        seed=args.seed,
        split_seed=args.split_seed,
        channel=args.channel,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        real_only_epochs=args.real_only_epochs,
        batch_size=args.batch_size,
        pretrain_learning_rate=args.pretrain_learning_rate,
        finetune_learning_rate=args.finetune_learning_rate,
        device_name=args.device,
        group_field=args.group_field,
        require_content_review=not args.allow_unreviewed,
    )
    print(
        json.dumps(
            {
                "protocol_status": results["protocol_status"],
                "test_domain": results["test_domain"],
                "real_only_balanced_accuracy": results["experiments"]["A_real_only"]["test"]["balanced_accuracy"],
                "synthetic_only_balanced_accuracy": results["experiments"]["B_synthetic_only"]["test"]["balanced_accuracy"],
                "output_dir": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

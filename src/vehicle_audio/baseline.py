"""Leakage-safe classical and neural baselines for tracked vs. wheeled audio."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any, Mapping, Sequence

import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchaudio.transforms as audio_transforms


CLASS_NAMES = ("tracked", "wheeled")
FEATURE_IMPLEMENTATION_VERSION = 2


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16_000
    n_fft: int = 1_024
    hop_length: int = 256
    n_mels: int = 64
    n_mfcc: int = 20
    f_min: float = 20.0
    f_max: float = 8_000.0


@dataclass(frozen=True)
class SplitIndices:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]


class LogMelExtractor(nn.Module):
    """Create per-example standardized log-Mel features from one channel."""

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__()
        self.config = config
        self.mel = audio_transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            power=2.0,
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 2:
            raise ValueError("waveforms must have shape [batch, samples]")
        features = torch.log(self.mel(waveforms).clamp_min(1e-10))
        mean = features.mean(dim=(-2, -1), keepdim=True)
        standard_deviation = features.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        return ((features - mean) / standard_deviation).unsqueeze(1)


class ClassicalFeatureExtractor(nn.Module):
    """MFCC and compact spectral/temporal statistics for logistic regression."""

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__()
        self.config = config
        self.mfcc = audio_transforms.MFCC(
            sample_rate=config.sample_rate,
            n_mfcc=config.n_mfcc,
            log_mels=True,
            melkwargs={
                "n_fft": config.n_fft,
                "hop_length": config.hop_length,
                "n_mels": config.n_mels,
                "f_min": config.f_min,
                "f_max": config.f_max,
                "power": 2.0,
            },
        )
        self.register_buffer("window", torch.hann_window(config.n_fft), persistent=False)
        self.register_buffer(
            "frequency_bins",
            torch.linspace(0.0, config.sample_rate / 2.0, config.n_fft // 2 + 1),
            persistent=False,
        )

    @staticmethod
    def _mean_std(values: torch.Tensor) -> torch.Tensor:
        return torch.stack((values.mean(dim=-1), values.std(dim=-1)), dim=1)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 2:
            raise ValueError("waveforms must have shape [batch, samples]")
        mfcc = self.mfcc(waveforms)
        mfcc_statistics = torch.cat((mfcc.mean(dim=-1), mfcc.std(dim=-1)), dim=1)

        spectrum = torch.stft(
            waveforms,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            window=self.window,
            return_complex=True,
        ).abs().square().clamp_min(1e-12)
        total_power = spectrum.sum(dim=1).clamp_min(1e-12)
        frequencies = self.frequency_bins.view(1, -1, 1)
        centroid = (spectrum * frequencies).sum(dim=1) / total_power
        bandwidth = torch.sqrt(
            (spectrum * (frequencies - centroid.unsqueeze(1)).square()).sum(dim=1)
            / total_power
        )
        cumulative_power = spectrum.cumsum(dim=1)
        rolloff_indices = (cumulative_power < 0.85 * total_power.unsqueeze(1)).sum(dim=1)
        rolloff_indices = rolloff_indices.clamp_max(len(self.frequency_bins) - 1)
        rolloff = self.frequency_bins[rolloff_indices]
        flatness = torch.exp(torch.log(spectrum).mean(dim=1)) / spectrum.mean(dim=1)
        # Concentration in the five strongest bins is a compact harmonic-structure proxy.
        peak_concentration = spectrum.topk(k=5, dim=1).values.sum(dim=1) / total_power
        spectral_statistics = torch.cat(
            [
                self._mean_std(value)
                for value in (centroid, bandwidth, rolloff, flatness, peak_concentration)
            ],
            dim=1,
        )

        rms = waveforms.square().mean(dim=1).sqrt().unsqueeze(1)
        zero_crossing_rate = (
            (waveforms[:, 1:] * waveforms[:, :-1]) < 0
        ).to(torch.float32).mean(dim=1, keepdim=True)
        crest_factor = waveforms.abs().amax(dim=1, keepdim=True) / rms.clamp_min(1e-8)
        return torch.cat(
            (mfcc_statistics, spectral_statistics, rms, zero_crossing_rate, crest_factor),
            dim=1,
        )


class SmallAudioCNN(nn.Module):
    """Compact log-Mel CNN with an explicit encoder/classifier boundary."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(features).flatten(1))


class LogisticRegressionBaseline(nn.Module):
    """Single linear layer over standardized handcrafted features."""

    def __init__(self, num_features: int, num_classes: int = 2) -> None:
        super().__init__()
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def load_manifest(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    manifest_path = Path(path)
    manifest_bytes = manifest_path.read_bytes()
    records = [json.loads(line) for line in manifest_bytes.splitlines() if line.strip()]
    if not records:
        raise ValueError(f"manifest is empty: {manifest_path}")
    observed_classes = {str(record["vehicle_class"]) for record in records}
    if observed_classes != set(CLASS_NAMES):
        raise ValueError(
            f"baseline requires exactly {list(CLASS_NAMES)}, found {sorted(observed_classes)}"
        )
    sample_rates = {int(record["sample_rate"]) for record in records}
    if len(sample_rates) != 1:
        raise ValueError(f"all samples must use one sample rate, found {sorted(sample_rates)}")
    return records, hashlib.sha256(manifest_bytes).hexdigest()


def _validate_split(records: Sequence[Mapping[str, Any]], split: SplitIndices) -> None:
    partitions = (set(split.train), set(split.validation), set(split.test))
    if any(not partition for partition in partitions):
        raise ValueError("train, validation, and test splits must all be nonempty")
    if partitions[0] & partitions[1] or partitions[0] & partitions[2] or partitions[1] & partitions[2]:
        raise RuntimeError("split partitions overlap")
    if set.union(*partitions) != set(range(len(records))):
        raise RuntimeError("split partitions do not cover the manifest")
    for name, indices in zip(("train", "validation", "test"), partitions, strict=True):
        classes = {str(records[index]["vehicle_class"]) for index in indices}
        if classes != set(CLASS_NAMES):
            raise ValueError(f"{name} split does not contain both classes: {sorted(classes)}")


def grouped_stratified_split(
    records: Sequence[Mapping[str, Any]],
    seed: int,
    *,
    group_field: str = "recording_session",
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> SplitIndices:
    """Stratify class-pure source groups without splitting any group."""

    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than one")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        value = record.get(group_field)
        if value is None or str(value) == "":
            raise ValueError(f"record {index} has no nonempty {group_field!r}")
        groups[str(value)].append(index)

    groups_by_class: dict[str, list[str]] = defaultdict(list)
    for group, indices in groups.items():
        classes = {str(records[index]["vehicle_class"]) for index in indices}
        if len(classes) != 1:
            raise ValueError(f"group {group!r} contains multiple classes: {sorted(classes)}")
        groups_by_class[next(iter(classes))].append(group)

    random_generator = random.Random(seed)
    split_groups: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    test_fraction = 1.0 - train_fraction - validation_fraction
    for vehicle_class in CLASS_NAMES:
        class_groups = sorted(groups_by_class.get(vehicle_class, []))
        if len(class_groups) < 3:
            raise ValueError(
                f"grouped split requires at least three {vehicle_class!r} groups in "
                f"{group_field!r}, found {len(class_groups)}"
            )
        random_generator.shuffle(class_groups)
        validation_count = max(1, round(len(class_groups) * validation_fraction))
        test_count = max(1, round(len(class_groups) * test_fraction))
        while validation_count + test_count >= len(class_groups):
            if validation_count >= test_count and validation_count > 1:
                validation_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                break
        split_groups["validation"].extend(class_groups[:validation_count])
        split_groups["test"].extend(
            class_groups[validation_count : validation_count + test_count]
        )
        split_groups["train"].extend(class_groups[validation_count + test_count :])

    def indices_for(name: str) -> tuple[int, ...]:
        indices = [
            index
            for group in sorted(split_groups[name])
            for index in groups[group]
        ]
        random_generator.shuffle(indices)
        return tuple(indices)

    split = SplitIndices(
        train=indices_for("train"),
        validation=indices_for("validation"),
        test=indices_for("test"),
    )
    _validate_split(records, split)
    return split


def holdout_split(
    records: Sequence[Mapping[str, Any]],
    *,
    field: str,
    validation_values: Sequence[str],
    test_values: Sequence[str],
) -> SplitIndices:
    """Hold out complete field values for controlled generalization tests."""

    validation_set = {str(value) for value in validation_values}
    test_set = {str(value) for value in test_values}
    if not validation_set or not test_set:
        raise ValueError("holdout split requires validation and test values")
    if validation_set & test_set:
        raise ValueError("validation and test holdout values overlap")
    observed = {str(record.get(field)) for record in records}
    missing = (validation_set | test_set) - observed
    if missing:
        raise ValueError(f"holdout values not present in {field!r}: {sorted(missing)}")

    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for index, record in enumerate(records):
        value = str(record.get(field))
        if value in test_set:
            test.append(index)
        elif value in validation_set:
            validation.append(index)
        else:
            train.append(index)
    split = SplitIndices(tuple(train), tuple(validation), tuple(test))
    _validate_split(records, split)
    return split


def stratified_split(
    records: Sequence[Mapping[str, Any]],
    seed: int,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> SplitIndices:
    """Compatibility name for the required recording-session grouped split."""

    return grouped_stratified_split(
        records,
        seed,
        group_field="recording_session",
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )


def _read_one_channel(
    audio_path: Path,
    channel: int,
    expected_sample_rate: int,
) -> torch.Tensor:
    samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"sample-rate mismatch for {audio_path}: {sample_rate} != {expected_sample_rate}"
        )
    if not 0 <= channel < samples.shape[1]:
        raise ValueError(
            f"requested channel {channel} but {audio_path} has {samples.shape[1]} channels"
        )
    return torch.from_numpy(samples[:, channel].copy())


def precompute_features(
    records: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    manifest_sha256: str,
    output_dir: Path,
    channel: int,
    feature_config: FeatureConfig,
    model_kind: str,
    *,
    batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load one explicit channel and cache either classical or log-Mel features."""

    cache_path = output_dir / f"features_{model_kind}_channel_{channel}.pt"
    expected_metadata = {
        "manifest_sha256": manifest_sha256,
        "channel": channel,
        "feature_config": asdict(feature_config),
        "model_kind": model_kind,
        "feature_implementation_version": FEATURE_IMPLEMENTATION_VERSION,
    }
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if all(cache.get(key) == value for key, value in expected_metadata.items()):
            print(f"Using feature cache: {cache_path}", flush=True)
            return cache["features"].float(), cache["labels"].long(), cache["snrs"].float()

    if model_kind == "cnn":
        extractor: nn.Module = LogMelExtractor(feature_config).eval()
    elif model_kind == "classical":
        extractor = ClassicalFeatureExtractor(feature_config).eval()
    else:
        raise ValueError(f"unknown model kind: {model_kind!r}")
    root = manifest_path.parent
    label_by_name = {name: index for index, name in enumerate(CLASS_NAMES)}
    feature_batches: list[torch.Tensor] = []
    label_values: list[int] = []
    snr_values: list[float] = []
    waveform_batch: list[torch.Tensor] = []

    def flush_batch() -> None:
        if not waveform_batch:
            return
        lengths = {waveform.shape[-1] for waveform in waveform_batch}
        if len(lengths) != 1:
            raise ValueError(f"all audio clips must have equal length, found {sorted(lengths)}")
        with torch.inference_mode():
            feature_batches.append(extractor(torch.stack(waveform_batch)).cpu())
        waveform_batch.clear()

    for index, record in enumerate(records):
        waveform_batch.append(
            _read_one_channel(
                root / str(record["corrupted_path"]),
                channel,
                feature_config.sample_rate,
            )
        )
        label_values.append(label_by_name[str(record["vehicle_class"])])
        if record.get("snr_db") is None:
            raise ValueError("SNR evaluation requires a numeric snr_db in every record")
        snr_values.append(float(record["snr_db"]))
        if len(waveform_batch) == batch_size:
            flush_batch()
        if (index + 1) % 100 == 0 or index + 1 == len(records):
            print(f"Extracted {model_kind} features: {index + 1}/{len(records)}", flush=True)
    flush_batch()

    features = torch.cat(feature_batches)
    labels = torch.tensor(label_values, dtype=torch.long)
    snrs = torch.tensor(snr_values, dtype=torch.float32)
    torch.save(
        {
            "features": features,
            "labels": labels,
            "snrs": snrs,
            **expected_metadata,
        },
        cache_path,
    )
    return features, labels, snrs


def classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    snrs: torch.Tensor,
    probabilities: torch.Tensor | None = None,
    *,
    class_names: Sequence[str] = CLASS_NAMES,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Return class metrics, confusion matrices, SNR curves, and calibration."""

    predictions = predictions.to(torch.long).cpu()
    targets = targets.to(torch.long).cpu()
    snrs = snrs.to(torch.float32).cpu()
    if not (len(predictions) == len(targets) == len(snrs)) or len(targets) == 0:
        raise ValueError("predictions, targets, and SNRs must have equal nonzero length")
    if probabilities is not None:
        probabilities = probabilities.to(torch.float32).cpu()
        if probabilities.shape != (len(targets), len(class_names)):
            raise ValueError("probabilities have the wrong shape")

    def summarize(mask: torch.Tensor) -> dict[str, Any]:
        selected_predictions = predictions[mask]
        selected_targets = targets[mask]
        selected_probabilities = None if probabilities is None else probabilities[mask]
        confusion = torch.zeros((len(class_names), len(class_names)), dtype=torch.long)
        for target, prediction in zip(selected_targets, selected_predictions, strict=True):
            confusion[int(target), int(prediction)] += 1

        per_class: dict[str, dict[str, float | int | None]] = {}
        precision_values: list[float] = []
        recall_values: list[float] = []
        f1_values: list[float] = []
        for class_index, name in enumerate(class_names):
            true_positive = int(confusion[class_index, class_index])
            support = int(confusion[class_index].sum())
            predicted_support = int(confusion[:, class_index].sum())
            precision = 0.0 if predicted_support == 0 else true_positive / predicted_support
            recall = None if support == 0 else true_positive / support
            f1 = (
                None
                if recall is None
                else (
                    0.0
                    if precision + recall == 0
                    else 2.0 * precision * recall / (precision + recall)
                )
            )
            per_class[name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "predicted_support": predicted_support,
            }
            if precision is not None:
                precision_values.append(precision)
            if recall is not None:
                recall_values.append(recall)
            if f1 is not None:
                f1_values.append(f1)

        calibration: dict[str, float] | None = None
        if selected_probabilities is not None:
            confidence, probability_predictions = selected_probabilities.max(dim=1)
            correctness = (probability_predictions == selected_targets).to(torch.float32)
            expected_calibration_error = 0.0
            for bin_index in range(calibration_bins):
                lower = bin_index / calibration_bins
                upper = (bin_index + 1) / calibration_bins
                bin_mask = (confidence > lower) & (confidence <= upper)
                if bin_mask.any():
                    weight = float(bin_mask.to(torch.float32).mean())
                    expected_calibration_error += weight * abs(
                        float(correctness[bin_mask].mean()) - float(confidence[bin_mask].mean())
                    )
            selected_true_probabilities = selected_probabilities.gather(
                1, selected_targets.unsqueeze(1)
            ).squeeze(1)
            one_hot = torch.nn.functional.one_hot(
                selected_targets, num_classes=len(class_names)
            ).to(torch.float32)
            calibration = {
                "expected_calibration_error": expected_calibration_error,
                "negative_log_likelihood": float(
                    -torch.log(selected_true_probabilities.clamp_min(1e-12)).mean()
                ),
                "brier_score": float((selected_probabilities - one_hot).square().sum(dim=1).mean()),
            }

        return {
            "support": int(mask.sum()),
            "accuracy": float((selected_predictions == selected_targets).to(torch.float32).mean()),
            "balanced_accuracy": sum(recall_values) / len(recall_values),
            "macro_precision": sum(precision_values) / len(precision_values),
            "macro_recall": sum(recall_values) / len(recall_values),
            "macro_f1": sum(f1_values) / len(f1_values),
            "per_class": per_class,
            "per_class_recall": {
                name: values["recall"] for name, values in per_class.items()
            },
            "class_support": {
                name: int(confusion[index].sum()) for index, name in enumerate(class_names)
            },
            "confusion_matrix": confusion.tolist(),
            "calibration": calibration,
        }

    metrics = summarize(torch.ones(len(targets), dtype=torch.bool))
    metrics["per_snr"] = {
        f"{float(snr):g}": summarize(snrs == snr)
        for snr in sorted(torch.unique(snrs).tolist(), reverse=True)
    }
    return metrics


def _evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    predictions: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    snrs: list[torch.Tensor] = []
    with torch.inference_mode():
        for feature_batch, target_batch, snr_batch in data_loader:
            logits = model(feature_batch.to(device))
            batch_probabilities = torch.softmax(logits, dim=1).cpu()
            probabilities.append(batch_probabilities)
            predictions.append(batch_probabilities.argmax(dim=1))
            targets.append(target_batch.cpu())
            snrs.append(snr_batch.cpu())
    return classification_metrics(
        torch.cat(predictions),
        torch.cat(targets),
        torch.cat(snrs),
        torch.cat(probabilities),
    )


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def _split_groups(
    records: Sequence[Mapping[str, Any]], split: SplitIndices, field: str
) -> dict[str, list[str]]:
    return {
        name: sorted({str(records[index].get(field)) for index in indices})
        for name, indices in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        )
    }


def _write_snr_csv(path: Path, test_metrics: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "snr_db",
                "accuracy",
                "balanced_accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "expected_calibration_error",
                "support",
                "tracked_support",
                "wheeled_support",
            ]
        )
        for snr, metrics in test_metrics["per_snr"].items():
            calibration = metrics["calibration"] or {}
            writer.writerow(
                [
                    snr,
                    metrics["accuracy"],
                    metrics["balanced_accuracy"],
                    metrics["macro_precision"],
                    metrics["macro_recall"],
                    metrics["macro_f1"],
                    calibration.get("expected_calibration_error"),
                    metrics["support"],
                    metrics["class_support"]["tracked"],
                    metrics["class_support"]["wheeled"],
                ]
            )


def train_baseline(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    model_kind: str = "cnn",
    channel: int = 0,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device_name: str = "auto",
    split_strategy: str = "grouped",
    group_field: str = "recording_session",
    validation_groups: Sequence[str] = (),
    test_groups: Sequence[str] = (),
    test_domain: str | None = None,
) -> dict[str, Any]:
    if model_kind not in {"classical", "cnn"}:
        raise ValueError("model_kind must be 'classical' or 'cnn'")
    if channel < 0 or epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("channel must be nonnegative and training parameters must be positive")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest = Path(manifest_path)
    records, manifest_sha256 = load_manifest(manifest)
    sample_rate = int(records[0]["sample_rate"])
    feature_config = FeatureConfig(sample_rate=sample_rate, f_max=sample_rate / 2.0)

    random.seed(seed)
    torch.manual_seed(seed)
    if split_strategy == "grouped":
        splits = grouped_stratified_split(records, seed, group_field=group_field)
    elif split_strategy == "holdout":
        splits = holdout_split(
            records,
            field=group_field,
            validation_values=validation_groups,
            test_values=test_groups,
        )
    else:
        raise ValueError("split_strategy must be 'grouped' or 'holdout'")

    features, labels, snrs = precompute_features(
        records,
        manifest,
        manifest_sha256,
        output_path,
        channel,
        feature_config,
        model_kind,
        batch_size=batch_size,
    )
    feature_normalization: dict[str, torch.Tensor] | None = None
    if model_kind == "classical":
        train_features = features[list(splits.train)]
        feature_mean = train_features.mean(dim=0)
        feature_std = train_features.std(dim=0).clamp_min(1e-6)
        features = (features - feature_mean) / feature_std
        feature_normalization = {"mean": feature_mean, "std": feature_std}

    dataset = TensorDataset(features, labels, snrs)
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        Subset(dataset, splits.train),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        Subset(dataset, splits.validation), batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        Subset(dataset, splits.test), batch_size=batch_size, shuffle=False, num_workers=0
    )

    device = _resolve_device(device_name)
    model: nn.Module
    if model_kind == "cnn":
        model = SmallAudioCNN(len(CLASS_NAMES))
    else:
        model = LogisticRegressionBaseline(features.shape[1], len(CLASS_NAMES))
    model = model.to(device)
    train_labels = labels[list(splits.train)]
    class_counts = torch.bincount(train_labels, minlength=len(CLASS_NAMES)).to(torch.float32)
    if (class_counts == 0).any():
        raise ValueError("training split must contain both classes")
    class_weights = len(train_labels) / (len(CLASS_NAMES) * class_counts)
    loss_function = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_balanced_accuracy = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for feature_batch, target_batch, _ in train_loader:
            feature_batch = feature_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(feature_batch), target_batch)
            loss.backward()
            optimizer.step()
            batch_count = len(target_batch)
            total_loss += float(loss.detach().cpu()) * batch_count
            seen += batch_count
        validation_metrics = _evaluate(model, validation_loader, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, sort_keys=True), flush=True)
        if validation_metrics["balanced_accuracy"] > best_balanced_accuracy:
            best_balanced_accuracy = validation_metrics["balanced_accuracy"]
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    model.to(device)
    validation_metrics = _evaluate(model, validation_loader, device)
    test_metrics = _evaluate(model, test_loader, device)
    split_sample_ids = {
        "train": [str(records[index]["sample_id"]) for index in splits.train],
        "validation": [str(records[index]["sample_id"]) for index in splits.validation],
        "test": [str(records[index]["sample_id"]) for index in splits.test],
    }
    split_groups = _split_groups(records, splits, group_field)
    if set(split_groups["train"]) & set(split_groups["validation"]):
        raise RuntimeError("group leakage between train and validation")
    if set(split_groups["train"]) & set(split_groups["test"]):
        raise RuntimeError("group leakage between train and test")
    if set(split_groups["validation"]) & set(split_groups["test"]):
        raise RuntimeError("group leakage between validation and test")
    (output_path / "splits.json").write_text(
        json.dumps(
            {
                "strategy": split_strategy,
                "group_field": group_field,
                "seed": seed,
                "groups": split_groups,
                "sample_ids": split_sample_ids,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    observed_domains = sorted({str(record.get("source_domain", "unspecified")) for record in records})
    resolved_test_domain = test_domain or " + ".join(observed_domains)
    training_config = {
        "model_kind": model_kind,
        "channel": channel,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": str(device),
        "split_strategy": split_strategy,
        "group_field": group_field,
        "validation_groups": list(validation_groups),
        "test_groups": list(test_groups),
        "test_domain": resolved_test_domain,
    }
    results = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "manifest": str(manifest),
        "dataset_version": manifest_sha256,
        "manifest_sha256": manifest_sha256,
        "output_dir": str(output_path),
        "test_domain": resolved_test_domain,
        "source_domains": observed_domains,
        "single_microphone": True,
        "class_names": list(CLASS_NAMES),
        "training_config": training_config,
        "feature_config": asdict(feature_config),
        "feature_implementation_version": FEATURE_IMPLEMENTATION_VERSION,
        "split_strategy": split_strategy,
        "split_group_field": group_field,
        "split_groups": split_groups,
        "split_counts": {name: len(values) for name, values in split_sample_ids.items()},
        "best_epoch": best_epoch,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_class_counts": {
            CLASS_NAMES[index]: int(count) for index, count in enumerate(class_counts)
        },
        "class_weights": {
            CLASS_NAMES[index]: float(weight) for index, weight in enumerate(class_weights)
        },
        "history": history,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    torch.save(
        {
            "model_state_dict": best_state,
            "model_kind": model_kind,
            "class_names": list(CLASS_NAMES),
            "channel": channel,
            "feature_config": asdict(feature_config),
            "feature_implementation_version": FEATURE_IMPLEMENTATION_VERSION,
            "feature_normalization": feature_normalization,
            "manifest_sha256": manifest_sha256,
            "best_epoch": best_epoch,
            "training_config": training_config,
            "test_metrics": test_metrics,
        },
        output_path / "model.pt",
    )
    (output_path / "experiment.json").write_text(
        json.dumps(
            {
                "git_commit": results["git_commit"],
                "dataset_version": manifest_sha256,
                "manifest": str(manifest),
                "manifest_sha256": manifest_sha256,
                "training_config": training_config,
                "feature_config": asdict(feature_config),
                "feature_implementation_version": FEATURE_IMPLEMENTATION_VERSION,
                "split_groups": split_groups,
                "checkpoint": "model.pt",
                "metrics": "metrics.json",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_path / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_snr_csv(output_path / "accuracy_by_snr.csv", test_metrics)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("classical", "cnn"), default="cnn")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--split-strategy", choices=("grouped", "holdout"), default="grouped")
    parser.add_argument("--group-field", default="recording_session")
    parser.add_argument("--validation-group", action="append", default=[])
    parser.add_argument("--test-group", action="append", default=[])
    parser.add_argument("--test-domain", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = train_baseline(
        args.manifest,
        args.output,
        model_kind=args.model,
        channel=args.channel,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_name=args.device,
        split_strategy=args.split_strategy,
        group_field=args.group_field,
        validation_groups=args.validation_group,
        test_groups=args.test_group,
        test_domain=args.test_domain,
    )
    print(
        json.dumps(
            {
                "model_kind": args.model,
                "best_epoch": results["best_epoch"],
                "test_accuracy": results["test"]["accuracy"],
                "test_balanced_accuracy": results["test"]["balanced_accuracy"],
                "test_macro_f1": results["test"]["macro_f1"],
                "accuracy_by_snr": {
                    snr: metrics["accuracy"]
                    for snr, metrics in results["test"]["per_snr"].items()
                },
                "output_dir": results["output_dir"],
                "test_domain": results["test_domain"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

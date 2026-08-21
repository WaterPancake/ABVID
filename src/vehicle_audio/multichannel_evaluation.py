"""Leakage-safe Milestone 4 localization, fusion, and beamforming evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from vehicle_audio.audio import load_audio
from vehicle_audio.baseline import (
    CLASS_NAMES,
    FEATURE_IMPLEMENTATION_VERSION,
    ClassicalFeatureExtractor,
    FeatureConfig,
    LogisticRegressionBaseline,
    SplitIndices,
    classification_metrics,
    grouped_stratified_split,
)
from vehicle_audio.multichannel import (
    angular_error_degrees,
    delay_and_sum_beamform,
    gcc_phat_azimuth,
    srp_phat_azimuth,
)


ARRAY_EVALUATION_VERSION = 2


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


def _load_array_manifest(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = path.read_bytes()
    records = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if not records:
        raise ValueError(f"array manifest is empty: {path}")
    if {str(record["vehicle_class"]) for record in records} != set(CLASS_NAMES):
        raise ValueError("array evaluation requires tracked and wheeled records")
    sample_rates = {int(record["sample_rate"]) for record in records}
    channel_counts = {int(record["num_channels"]) for record in records}
    sample_counts = {int(record["num_samples"]) for record in records}
    if len(sample_rates) != 1 or len(channel_counts) != 1 or len(sample_counts) != 1:
        raise ValueError("array observations must have consistent rates and shapes")
    return records, hashlib.sha256(payload).hexdigest()


def _extract_features_in_batches(
    waveforms: torch.Tensor,
    extractor: nn.Module,
    batch_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(waveforms), batch_size):
            outputs.append(extractor(waveforms[start : start + batch_size]).cpu())
    return torch.cat(outputs)


def _precompute_array_features(
    records: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    manifest_sha256: str,
    output_dir: Path,
    feature_config: FeatureConfig,
    microphone_counts: Sequence[int],
    *,
    batch_size: int,
    localization_samples: int = 8_192,
) -> dict[str, Any]:
    cache_path = output_dir / "array_features.pt"
    expected = {
        "manifest_sha256": manifest_sha256,
        "feature_config": asdict(feature_config),
        "feature_implementation_version": FEATURE_IMPLEMENTATION_VERSION,
        "array_evaluation_version": ARRAY_EVALUATION_VERSION,
        "microphone_counts": list(microphone_counts),
        "localization_samples": localization_samples,
    }
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if all(cached.get(key) == value for key, value in expected.items()):
            print(f"Using array feature cache: {cache_path}", flush=True)
            return cached

    root = manifest_path.parent
    observations: list[torch.Tensor] = []
    for index, record in enumerate(records):
        waveform, sample_rate = load_audio(root / str(record["observation_path"]))
        if sample_rate != feature_config.sample_rate:
            raise ValueError(f"sample-rate mismatch in {record['observation_path']}")
        if waveform.shape != (int(record["num_channels"]), int(record["num_samples"])):
            raise ValueError(f"shape mismatch in {record['observation_path']}")
        observations.append(waveform)
        if (index + 1) % 100 == 0 or index + 1 == len(records):
            print(f"Loaded array observations: {index + 1}/{len(records)}", flush=True)
    array_waveforms = torch.stack(observations)
    extractor = ClassicalFeatureExtractor(feature_config).eval()
    representations: dict[str, torch.Tensor] = {
        "single_mic_1": _extract_features_in_batches(
            array_waveforms[:, 0], extractor, batch_size
        )
    }
    azimuth_targets = torch.tensor(
        [float(record["azimuth_deg"]) for record in records], dtype=torch.float32
    )
    azimuth_estimates: dict[str, torch.Tensor] = {}
    positions = records[0]["array_config"]["microphone_positions_m"]
    speed_of_sound = float(records[0]["array_config"]["speed_of_sound_mps"])

    for microphone_count in microphone_counts:
        if microphone_count == 1:
            continue
        flattened = array_waveforms[:, :microphone_count].reshape(
            len(records) * microphone_count,
            array_waveforms.shape[-1],
        )
        channel_features = _extract_features_in_batches(flattened, extractor, batch_size)
        representations[f"feature_fusion_{microphone_count}"] = channel_features.reshape(
            len(records), microphone_count, -1
        ).mean(dim=1)

        srp_values: list[float] = []
        gcc_values: list[float] = []
        srp_beamformed: list[torch.Tensor] = []
        gcc_beamformed: list[torch.Tensor] = []
        microphone_positions = positions[:microphone_count]
        for index, observation in enumerate(array_waveforms[:, :microphone_count]):
            start = max(0, (observation.shape[-1] - localization_samples) // 2)
            localization_window = observation[
                :, start : start + min(localization_samples, observation.shape[-1])
            ]
            srp_estimate, _ = srp_phat_azimuth(
                localization_window,
                microphone_positions,
                feature_config.sample_rate,
                speed_of_sound_mps=speed_of_sound,
                interpolation=2,
            )
            gcc_estimate = gcc_phat_azimuth(
                localization_window,
                microphone_positions,
                feature_config.sample_rate,
                speed_of_sound_mps=speed_of_sound,
                interpolation=2,
            )
            srp_values.append(srp_estimate)
            gcc_values.append(gcc_estimate)
            srp_beamformed.append(
                delay_and_sum_beamform(
                    observation,
                    microphone_positions,
                    srp_estimate,
                    feature_config.sample_rate,
                    speed_of_sound_mps=speed_of_sound,
                ).squeeze(0)
            )
            gcc_beamformed.append(
                delay_and_sum_beamform(
                    observation,
                    microphone_positions,
                    gcc_estimate,
                    feature_config.sample_rate,
                    speed_of_sound_mps=speed_of_sound,
                ).squeeze(0)
            )
            if (index + 1) % 100 == 0 or index + 1 == len(records):
                print(
                    f"Localized/beamformed {microphone_count} microphones: "
                    f"{index + 1}/{len(records)}",
                    flush=True,
                )
        azimuth_estimates[f"srp_phat_{microphone_count}"] = torch.tensor(srp_values)
        azimuth_estimates[f"gcc_phat_{microphone_count}"] = torch.tensor(gcc_values)
        representations[f"srp_phat_beamforming_{microphone_count}"] = (
            _extract_features_in_batches(torch.stack(srp_beamformed), extractor, batch_size)
        )
        representations[f"gcc_phat_beamforming_{microphone_count}"] = (
            _extract_features_in_batches(torch.stack(gcc_beamformed), extractor, batch_size)
        )

    cache: dict[str, Any] = {
        **expected,
        "representations": representations,
        "azimuth_targets": azimuth_targets,
        "azimuth_estimates": azimuth_estimates,
        "labels": torch.tensor(
            [CLASS_NAMES.index(str(record["vehicle_class"])) for record in records],
            dtype=torch.long,
        ),
        "snrs": torch.tensor([float(record["snr_db"]) for record in records]),
    }
    torch.save(cache, cache_path)
    return cache


def _evaluate_model(
    model: nn.Module,
    dataset: TensorDataset,
    indices: Sequence[int],
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    predictions: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    snrs: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for features, labels, snr_values in loader:
            probability = torch.softmax(model(features), dim=1)
            probabilities.append(probability)
            predictions.append(probability.argmax(dim=1))
            targets.append(labels)
            snrs.append(snr_values)
    return classification_metrics(
        torch.cat(predictions),
        torch.cat(targets),
        torch.cat(snrs),
        torch.cat(probabilities),
    )


def _train_linear_classifier(
    features: torch.Tensor,
    labels: torch.Tensor,
    snrs: torch.Tensor,
    split: SplitIndices,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch.manual_seed(seed)
    train_features = features[list(split.train)]
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0).clamp_min(1e-6)
    normalized = (features - feature_mean) / feature_std
    dataset = TensorDataset(normalized, labels, snrs)
    train_loader = DataLoader(
        Subset(dataset, split.train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    model = LogisticRegressionBaseline(normalized.shape[1], len(CLASS_NAMES))
    class_counts = torch.bincount(labels[list(split.train)], minlength=len(CLASS_NAMES)).float()
    class_weights = len(split.train) / (len(CLASS_NAMES) * class_counts)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_balanced_accuracy = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for feature_batch, label_batch, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(feature_batch), label_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(label_batch)
            seen += len(label_batch)
        validation = _evaluate_model(model, dataset, split.validation, batch_size)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / seen,
                "validation_accuracy": validation["accuracy"],
                "validation_balanced_accuracy": validation["balanced_accuracy"],
                "validation_macro_f1": validation["macro_f1"],
            }
        )
        if validation["balanced_accuracy"] > best_balanced_accuracy:
            best_balanced_accuracy = validation["balanced_accuracy"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    results = {
        "best_epoch": best_epoch,
        "history": history,
        "validation": _evaluate_model(model, dataset, split.validation, batch_size),
        "test": _evaluate_model(model, dataset, split.test, batch_size),
    }
    checkpoint = {
        "model_state_dict": best_state,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "best_epoch": best_epoch,
        "class_names": list(CLASS_NAMES),
    }
    return results, checkpoint


def _localization_summary(
    estimates: torch.Tensor,
    targets: torch.Tensor,
    snrs: torch.Tensor,
    indices: Sequence[int],
) -> dict[str, Any]:
    selected = torch.tensor(list(indices), dtype=torch.long)
    selected_estimates = estimates[selected]
    selected_targets = targets[selected]
    selected_snrs = snrs[selected]
    errors = torch.tensor(
        [
            angular_error_degrees(float(estimate), float(target))
            for estimate, target in zip(
                selected_estimates,
                selected_targets,
                strict=True,
            )
        ]
    )

    def summarize(mask: torch.Tensor) -> dict[str, Any]:
        values = errors[mask]
        return {
            "support": int(mask.sum()),
            "mean_absolute_error_deg": float(values.mean()),
            "median_absolute_error_deg": float(values.median()),
            "p90_absolute_error_deg": float(torch.quantile(values, 0.9)),
            "within_5_deg": float((values <= 5.0).float().mean()),
            "within_10_deg": float((values <= 10.0).float().mean()),
        }

    aggregate = summarize(torch.ones(len(errors), dtype=torch.bool))
    aggregate["per_snr"] = {
        f"{float(snr):g}": summarize(selected_snrs == snr)
        for snr in sorted(torch.unique(selected_snrs).tolist(), reverse=True)
    }
    return aggregate


def _write_classification_csv(path: Path, metrics: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("representation", "microphone_count", "snr_db", "accuracy", "support"))
        for key, result in metrics.items():
            microphone_count = int(key.rsplit("_", 1)[1])
            for snr, values in result["test"]["per_snr"].items():
                writer.writerow((key, microphone_count, snr, values["accuracy"], values["support"]))


def _write_localization_csv(path: Path, metrics: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("estimator", "microphone_count", "snr_db", "mean_error_deg", "median_error_deg", "support"))
        for key, result in metrics.items():
            microphone_count = int(key.rsplit("_", 1)[1])
            for snr, values in result["per_snr"].items():
                writer.writerow(
                    (
                        key,
                        microphone_count,
                        snr,
                        values["mean_absolute_error_deg"],
                        values["median_absolute_error_deg"],
                        values["support"],
                    )
                )


def _make_plots(
    output_dir: Path,
    classification: Mapping[str, Any],
    localization: Mapping[str, Any],
    microphone_counts: Sequence[int],
) -> None:
    import matplotlib.pyplot as plt

    maximum = max(microphone_counts)
    snr_series = {
        "Single microphone": classification["single_mic_1"]["test"]["per_snr"],
        f"{maximum}-mic feature fusion": classification[f"feature_fusion_{maximum}"]["test"]["per_snr"],
        f"{maximum}-mic GCC-PHAT beamforming": classification[
            f"gcc_phat_beamforming_{maximum}"
        ]["test"]["per_snr"],
    }
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for label, values in snr_series.items():
        x = sorted(float(value) for value in values)
        y = [values[f"{value:g}"]["accuracy"] for value in x]
        axis.plot(x, y, marker="o", label=label)
    axis.set(xlabel="SNR (dB)", ylabel="Accuracy", ylim=(0, 1.02), title="Classification accuracy vs SNR")
    axis.set_xticks(x)
    axis.tick_params(axis="x", labelrotation=30, labelsize=9)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(output_dir / "classification_accuracy_vs_snr.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for estimator in (f"gcc_phat_{maximum}", f"srp_phat_{maximum}"):
        values = localization[estimator]["per_snr"]
        x = sorted(float(value) for value in values)
        y = [values[f"{value:g}"]["mean_absolute_error_deg"] for value in x]
        axis.plot(x, y, marker="o", label=estimator.replace("_", " ").upper())
    axis.set(xlabel="SNR (dB)", ylabel="Mean absolute error (degrees)", title="Localization error vs SNR")
    axis.set_xticks(x)
    axis.tick_params(axis="x", labelrotation=30, labelsize=9)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(output_dir / "localization_error_vs_snr.png", dpi=160)
    plt.close(figure)

    feature_fusion = []
    beamforming = []
    single_accuracy = classification["single_mic_1"]["test"]["accuracy"]
    for microphone_count in microphone_counts:
        feature_fusion.append(
            single_accuracy
            if microphone_count == 1
            else classification[f"feature_fusion_{microphone_count}"]["test"]["accuracy"]
        )
        beamforming.append(
            single_accuracy
            if microphone_count == 1
            else classification[f"gcc_phat_beamforming_{microphone_count}"]["test"]["accuracy"]
        )
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.plot(microphone_counts, feature_fusion, marker="o", label="Feature fusion")
    axis.plot(microphone_counts, beamforming, marker="o", label="GCC-PHAT beamforming")
    axis.set(
        xlabel="Microphone count",
        ylabel="Accuracy",
        ylim=(0, 1.02),
        xticks=list(microphone_counts),
        title="Classification accuracy vs microphone count",
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(output_dir / "classification_accuracy_vs_microphone_count.png", dpi=160)
    plt.close(figure)


def evaluate_multichannel(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    group_field: str = "recording_session",
    test_domain: str = "procedural synthetic -> simulated multichannel synthetic",
) -> dict[str, Any]:
    if seed < 0 or epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("seed must be nonnegative and training parameters must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = Path(manifest_path)
    records, manifest_sha256 = _load_array_manifest(manifest)
    split = grouped_stratified_split(records, seed, group_field=group_field)
    microphone_counts = tuple(
        int(value) for value in records[0]["array_config"]["microphone_counts"]
    )
    sample_rate = int(records[0]["sample_rate"])
    feature_config = FeatureConfig(sample_rate=sample_rate, f_max=sample_rate / 2.0)
    cached = _precompute_array_features(
        records,
        manifest,
        manifest_sha256,
        output,
        feature_config,
        microphone_counts,
        batch_size=batch_size,
    )
    labels = cached["labels"]
    snrs = cached["snrs"]
    classification: dict[str, Any] = {}
    model_checkpoints: dict[str, Any] = {}
    for representation, features in cached["representations"].items():
        print(f"Training representation: {representation}", flush=True)
        result, checkpoint = _train_linear_classifier(
            features,
            labels,
            snrs,
            split,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        classification[representation] = result
        model_checkpoints[representation] = checkpoint

    localization = {
        name: _localization_summary(
            estimates,
            cached["azimuth_targets"],
            snrs,
            split.test,
        )
        for name, estimates in cached["azimuth_estimates"].items()
    }
    split_groups = {
        name: sorted({str(records[index][group_field]) for index in indices})
        for name, indices in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        )
    }
    split_ids = {
        name: [str(records[index]["sample_id"]) for index in indices]
        for name, indices in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        )
    }
    if any(
        set(split_groups[first]) & set(split_groups[second])
        for first, second in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise RuntimeError("recording-session leakage in multichannel evaluation")

    training_config = {
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "group_field": group_field,
        "model": "classical_logistic_regression",
        "feature_implementation_version": FEATURE_IMPLEMENTATION_VERSION,
        "array_evaluation_version": ARRAY_EVALUATION_VERSION,
        "test_domain": test_domain,
    }
    results = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "dataset_version": manifest_sha256,
        "test_domain": test_domain,
        "training_config": training_config,
        "feature_config": asdict(feature_config),
        "microphone_counts": list(microphone_counts),
        "array_config": records[0]["array_config"],
        "split_groups": split_groups,
        "split_counts": {name: len(values) for name, values in split_ids.items()},
        "classification": classification,
        "localization": localization,
    }
    (output / "splits.json").write_text(
        json.dumps(
            {
                "strategy": "grouped",
                "group_field": group_field,
                "seed": seed,
                "groups": split_groups,
                "sample_ids": split_ids,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "models": model_checkpoints,
            "manifest_sha256": manifest_sha256,
            "feature_config": asdict(feature_config),
            "training_config": training_config,
        },
        output / "models.pt",
    )
    (output / "experiment.json").write_text(
        json.dumps(
            {
                "git_commit": results["git_commit"],
                "configuration": training_config,
                "random_seed": seed,
                "dataset_version": manifest_sha256,
                "manifest": str(manifest),
                "train_test_split": "splits.json",
                "model_checkpoint": "models.pt",
                "metrics": "metrics.json",
                "test_domain": test_domain,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_classification_csv(output / "classification_by_snr.csv", classification)
    _write_localization_csv(output / "localization_by_snr.csv", localization)
    _make_plots(output, classification, localization, microphone_counts)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--group-field", default="recording_session")
    parser.add_argument(
        "--test-domain",
        default="procedural synthetic -> simulated multichannel synthetic",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = evaluate_multichannel(
        args.manifest,
        args.output,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        group_field=args.group_field,
        test_domain=args.test_domain,
    )
    maximum = max(results["microphone_counts"])
    print(
        json.dumps(
            {
                "single_mic_accuracy": results["classification"]["single_mic_1"]["test"]["accuracy"],
                "feature_fusion_accuracy": results["classification"][f"feature_fusion_{maximum}"]["test"]["accuracy"],
                "gcc_beamforming_accuracy": results["classification"][f"gcc_phat_beamforming_{maximum}"]["test"]["accuracy"],
                "srp_beamforming_accuracy": results["classification"][f"srp_phat_beamforming_{maximum}"]["test"]["accuracy"],
                "srp_phat_mean_error_deg": results["localization"][f"srp_phat_{maximum}"]["mean_absolute_error_deg"],
                "output_dir": str(args.output),
                "test_domain": results["test_domain"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

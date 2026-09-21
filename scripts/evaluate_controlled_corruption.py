"""Evaluate fixed native-real folds on reproducible real-background corruption."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import yaml

from vehicle_audio.audio import load_audio, resample_audio
from vehicle_audio.augment import apply_frequency_response
from vehicle_audio.baseline import ClassicalFeatureExtractor, FeatureConfig
from vehicle_audio.benchmark_followup import (
    FrozenBenchmark,
    MANIFEST,
    sha256,
    summarize_predictions,
    write_json,
)
from vehicle_audio.mixing import mix_at_snr
from vehicle_audio.pretrained_evaluation import validate_panns_checkpoint


def stable_offset(seed, sample_id, background_id, background_samples, window_samples):
    if background_samples < window_samples:
        raise ValueError("background must be at least one complete window")
    key = f"{seed}:{sample_id}:{background_id}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (
        background_samples - window_samples + 1
    )


def corrupt_window(signal, noise=None, snr_db=None, response=None):
    meta = {}
    if noise is not None:
        mixed = mix_at_snr(signal, noise, snr_db)
        waveform = mixed.mixture
        meta.update(requested_snr_db=snr_db, measured_snr_db=mixed.measured_snr_db)
    elif response is not None:
        waveform = apply_frequency_response(
            signal, 16000, response["frequencies_hz"], response["gains_db"]
        )
        meta["frequency_response"] = response
    else:
        waveform = signal.clone()
    gain = min(1.0, 0.99 / max(float(waveform.abs().max()), 1e-12))
    waveform = waveform * gain
    if not torch.isfinite(waveform).all():
        raise ValueError("nonfinite corrupted audio")
    meta["common_gain"] = gain
    meta["float32_waveform_sha256"] = hashlib.sha256(
        waveform.numpy().tobytes()
    ).hexdigest()
    return waveform, meta


def extract(waveforms, tagging, extractor, semantic_indices, batch_size):
    classical, semantic = [], []
    with torch.inference_mode():
        for start in range(0, len(waveforms), batch_size):
            batch = waveforms[start : start + batch_size]
            classical.append(extractor(batch).cpu())
            audio = resample_audio(batch, 16000, 32000).numpy()
            clipwise, _ = tagging.inference(audio)
            semantic.append(torch.from_numpy(clipwise[:, semantic_indices].copy()))
    c, s = torch.cat(classical), torch.cat(semantic)
    if not torch.isfinite(c).all() or not torch.isfinite(s).all():
        raise ValueError("nonfinite extracted features")
    return s, c


def prepare_backgrounds(config, b):
    backgrounds, snapshot = {}, []
    raw_hashes = set()
    target_raw_hashes = {r["raw_sha256"] for r in b.records}
    catalog = yaml.safe_load(Path("configs/audio_sources.yaml").read_text())
    catalog = {s["id"]: s for s in catalog["sources"]}
    for item in config["backgrounds"]:
        path = Path(item["path"])
        sidecar = json.loads(path.with_suffix(".json").read_text())
        source = catalog[item["source_id"]]
        if (
            sidecar["source_id"] != item["source_id"]
            or source["kind"] != "background"
            or source["status"] != "approved"
        ):
            raise ValueError("background provenance/admission mismatch")
        if (
            source["expected_license"] != sidecar["license"]
            or Path("data") / source["output_path"] != path
        ):
            raise ValueError("background license/path mismatch")
        raw = sidecar["raw_sha256"]
        if raw in raw_hashes or raw in target_raw_hashes:
            raise ValueError(
                "duplicate original background or target/background overlap"
            )
        raw_hashes.add(raw)
        audio, sr = load_audio(path)
        background = resample_audio(audio[0:1], sr, config["sample_rate"])
        backgrounds[item["source_id"]] = background
        snapshot.append(
            {
                **item,
                "normalized_sha256": sha256(path),
                "sidecar_sha256": sha256(path.with_suffix(".json")),
                "original_sample_rate": sr,
                "original_channels": audio.shape[0],
                "sample_count_16khz": background.shape[-1],
                "provenance": sidecar,
            }
        )
    if config["background_training_source_ids"]:
        raise ValueError("native probes must not use added-background training sources")
    return backgrounds, snapshot


def evaluate(config_path, output, batch_size):
    start = time.perf_counter()
    config = yaml.safe_load(config_path.read_text())
    if config["sample_rate"] != 16000 or config["channel"] != 0:
        raise ValueError("protocol requires 16 kHz channel 0")
    b = FrozenBenchmark()
    if b.version != config["native_manifest_sha256"]:
        raise ValueError("config manifest mismatch")
    native_predictions = b.predict(b.semantic, b.classical)
    b.verify_native(native_predictions)
    backgrounds, background_snapshot = prepare_backgrounds(config, b)
    window_hashes = {
        r["sample_id"]: sha256(MANIFEST.parent / r["audio_path"]) for r in b.records
    }
    checkpoint = validate_panns_checkpoint(b.metrics["pretrained_encoder"]["path"])
    if checkpoint["sha256"] != b.metrics["pretrained_encoder"]["sha256"]:
        raise ValueError("encoder checkpoint changed")
    run_lock = {
        "config": config,
        "config_sha256": sha256(config_path),
        "protocol_sha256": sha256("docs/benchmark_v0_1_corruption_protocol.md"),
        "inputs": b.input_hashes,
        "backgrounds": background_snapshot,
        "native_window_sha256": window_hashes,
        "code_sha256": {
            str(p): sha256(p)
            for p in [Path(__file__), *sorted(Path("src/vehicle_audio").glob("*.py"))]
        },
    }
    lock_path = output / "run_lock.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != run_lock:
            raise ValueError(
                "input/config/code changed: start a new version, do not resume"
            )
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("nonempty output has no run lock")
        write_json(lock_path, run_lock)
        write_json(
            output / "experiment.json",
            {
                **b.metadata(config),
                "status": "running",
                "test_domain": "real -> held-out real with controlled added recorded background or simulated response",
                "hardware": platform.uname()._asdict(),
                "torch_version": str(torch.__version__),
                "numpy_version": np.__version__,
                "threads": torch.get_num_threads(),
                "background_holdout_scope": "current ABVID native probes only; historical use and encoder exposure not excluded",
            },
        )
    local = Path("runs") / output.name
    local.mkdir(parents=True, exist_ok=True)
    original = []
    for r in b.records:
        samples, sr = sf.read(
            MANIFEST.parent / r["audio_path"], dtype="float32", always_2d=True
        )
        if samples.shape != (32000, 1) or sr != 16000:
            raise ValueError("unexpected native observation shape/rate")
        original.append(torch.from_numpy(samples[:, 0].copy()))
    original = torch.stack(original)
    extractor = ClassicalFeatureExtractor(FeatureConfig()).eval()
    with torch.inference_mode():
        replay = extractor(original[:8]).numpy()
    if not np.allclose(replay, b.classical[:8], rtol=1e-5, atol=1e-5):
        raise ValueError("classical extractor does not reproduce native cache")
    from panns_inference import AudioTagging

    tagging = AudioTagging(checkpoint_path=checkpoint["path"], device="cpu")
    # Match native single-window preprocessing and validate the encoder before corruption.
    semantic_check, _ = extract(
        original[:8], tagging, extractor, b.semantic_indices, batch_size
    )
    if not np.allclose(semantic_check.numpy(), b.semantic[:8], rtol=2e-4, atol=2e-5):
        raise ValueError("encoder preprocessing does not reproduce native cache")
    conditions = []
    for bg in config["backgrounds"]:
        for snr in config["snr_db"]:
            conditions.append(
                {
                    "id": f"{bg['source_id']}_snr_{snr}",
                    "background_id": bg["source_id"],
                    "snr_db": snr,
                    "role": bg["role"],
                }
            )
    conditions += [
        {
            "id": response["id"],
            "response": response,
            "role": "simulated_microphone_response",
        }
        for response in config["microphone_responses"]
    ]
    all_results = {
        "native": {
            "condition": {"id": "native", "role": "native_reference"},
            "metrics": summarize_predictions(b, native_predictions),
        }
    }
    for position, condition in enumerate(conditions):
        condition_start = time.perf_counter()
        path = output / "conditions" / (condition["id"] + ".json")
        prediction_path = local / (condition["id"] + "_predictions.pt")
        if path.exists():
            completed = json.loads(path.read_text())
            if sha256(prediction_path) != completed["prediction_sha256"]:
                raise ValueError("prediction artifact changed")
            all_results[condition["id"]] = completed
            print(
                f"Reused {position + 1}/{len(conditions)} {condition['id']}", flush=True
            )
            continue
        transforms, waveforms = [], []
        for i, record in enumerate(b.records):
            meta = {
                "sample_id": record["sample_id"],
                "recording_session": record["recording_session"],
                "native_window_sha256": window_hashes[record["sample_id"]],
            }
            if "background_id" in condition:
                background = backgrounds[condition["background_id"]]
                offset = stable_offset(
                    config["seed"],
                    record["sample_id"],
                    condition["background_id"],
                    background.shape[-1],
                    32000,
                )
                noise = background[:, offset : offset + 32000]
                waveform, params = corrupt_window(
                    original[i : i + 1], noise, condition["snr_db"]
                )
                meta.update(
                    background_source_id=condition["background_id"],
                    background_offset_samples_16khz=offset,
                )
            else:
                waveform, params = corrupt_window(
                    original[i : i + 1], response=condition["response"]
                )
            transforms.append({**meta, **params})
            waveforms.append(waveform[0])
        semantic, classical = extract(
            torch.stack(waveforms), tagging, extractor, b.semantic_indices, batch_size
        )
        predictions = b.predict(semantic.numpy(), classical.numpy())
        # Tensors only: future loading can use weights_only=True.
        payload = [
            {
                k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                for k, v in fold.items()
            }
            for fold in predictions
        ]
        torch.save(
            {
                "condition": condition,
                "fold_predictions": payload,
                "semantic_features": semantic,
                "classical_features": classical,
                "manifest_sha256": b.version,
            },
            prediction_path,
        )
        result = {
            "condition": condition,
            "metrics": summarize_predictions(b, predictions),
            "runtime_seconds": time.perf_counter() - condition_start,
            "prediction_path": str(prediction_path),
            "prediction_sha256": sha256(prediction_path),
            "transforms": transforms,
        }
        write_json(path, result)
        all_results[condition["id"]] = result
        print(
            f"Completed {position + 1}/{len(conditions)} {condition['id']}: "
            f"fusion BA={result['metrics']['fusion']['balanced_accuracy']:.3f}, {result['runtime_seconds']:.1f}s",
            flush=True,
        )
    compact = {
        k: {key: value for key, value in v.items() if key != "transforms"}
        for k, v in all_results.items()
    }
    write_json(output / "results.json", compact)
    lines = [
        "# Controlled-corruption 7/5 v1",
        "",
        "Fixed native-trained models; held-out real sessions with added real recordings or simulated microphone response. No retraining or threshold tuning.",
        "",
        "SNR below is native-window-to-added-background ratio, not engine-only SNR. Traffic is competing-vehicle interference.",
        "",
        "| Condition | Classical BA | Semantic BA | Fusion BA | Fusion tracked recall | Fusion wheeled recall | Worst fusion session-context |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in compact.items():
        metrics = row["metrics"]
        f = metrics["fusion"]
        lines.append(
            f"| {key} | {metrics['classical']['balanced_accuracy']:.2%} | {metrics['semantic']['balanced_accuracy']:.2%} | {f['balanced_accuracy']:.2%} | {f['tracked_recall']:.2%} | {f['wheeled_recall']:.2%} | {f['worst_session_context_recall']:.2%} |"
        )
    lines.extend(
        [
            "",
            "Every condition retains per-session/per-fold metrics, confusion matrices, descriptive session-bootstrap intervals and transformations. See results.json and conditions/.",
            "",
            "Intervals resample 12 sessions, not dependent windows/folds; omit refit and new-background uncertainty. No calibrated-confidence claim. Current-probe background holdout does not mean historical or external-pretraining independence.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, role, title in zip(
        axes,
        ("environmental_noise", "competing_vehicle_interference"),
        (
            "3 recorded environmental backgrounds (equal mean)",
            "1 recorded traffic interferer",
        ),
    ):
        for model in ("classical", "semantic", "fusion"):
            values = []
            for snr in config["snr_db"]:
                rows = [
                    r
                    for r in compact.values()
                    if r["condition"].get("role") == role
                    and r["condition"].get("snr_db") == snr
                ]
                values.append(
                    np.mean([r["metrics"][model]["balanced_accuracy"] for r in rows])
                )
            ax.plot(config["snr_db"], values, "o-", label=model)
            ax.axhline(
                compact["native"]["metrics"][model]["balanced_accuracy"],
                linestyle=":",
                alpha=0.35,
                color=ax.lines[-1].get_color(),
            )
        ax.set(
            title=title,
            xlabel="Native-window / added-background ratio (dB)",
            ylim=(0, 1),
        )
        ax.invert_xaxis()
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean session-pair balanced accuracy")
    axes[1].legend()
    figure.suptitle(
        "Real-trained fixed models → corrupted held-out real sessions; dotted = native"
    )
    figure.tight_layout()
    figure.savefig(output / "accuracy_vs_added_background_snr.png", dpi=150)
    plt.close(figure)
    experiment = json.loads((output / "experiment.json").read_text())
    experiment.update(
        status="complete",
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        latest_invocation_seconds=time.perf_counter() - start,
        sum_condition_runtime_seconds=sum(
            r.get("runtime_seconds", 0) for r in compact.values()
        ),
        condition_count=len(conditions),
        corrupted_observation_count=len(conditions) * len(b.records),
        metrics="results.json",
        results_sha256=sha256(output / "results.json"),
    )
    write_json(output / "experiment.json", experiment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark_v0_1_corruption_7t5w.yaml"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/v0.1/corruption_7t5w_v1")
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    evaluate(args.config, args.output, args.batch_size)

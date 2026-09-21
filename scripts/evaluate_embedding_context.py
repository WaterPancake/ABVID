"""Preregistered frozen-embedding and matched-context native-real experiments."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import time
from types import SimpleNamespace

import numpy as np
import sklearn
import soundfile as sf
from threadpoolctl import threadpool_limits
import torch
import yaml

from vehicle_audio.audio import resample_audio
from vehicle_audio.benchmark_followup import (
    FrozenBenchmark,
    RUN,
    sha256,
    validate_split,
    write_json,
)
from vehicle_audio.pretrained_evaluation import validate_panns_checkpoint
from vehicle_audio.real_corpus import RealCorpusConfig, _read_window
from vehicle_audio.semantic_session_evaluation import (
    SEMANTIC_AUDIOSET_FEATURES,
    nested_leave_session_pair_out,
)

CONFIG = Path("configs/benchmark_embedding_context_v1.yaml")
PROTOCOL = Path("docs/embedding_context_protocol.md")


def matched_indices(records, catalog, seconds):
    """Select by geometry only, never predictions; stay in one reviewed interval."""
    selected = []
    for i, record in enumerate(records):
        center = (record["window_start_seconds"] + record["window_end_seconds"]) / 2
        matches = [
            s
            for s in catalog[record["source_id"]]["condition_segments"]
            if s["operating_condition"] == record["operating_condition"]
            and s["start_seconds"] <= center - seconds / 2
            and center + seconds / 2 <= s["end_seconds"]
        ]
        if len(matches) > 1:
            raise ValueError("ambiguous reviewed interval")
        if matches:
            selected.append(i)
    return selected


def bootstrap(values, seed=42, repetitions=10000):
    rng = np.random.default_rng(seed)
    draws = []
    for name in ("tracked", "wheeled"):
        a = np.asarray(values[name], dtype=float)
        draws.append(rng.choice(a, (repetitions, len(a)), replace=True).mean(axis=1))
    return np.quantile((draws[0] + draws[1]) / 2, [0.025, 0.975]).tolist()


def summary(evaluation):
    output = {}
    for name, field, aggregate in (
        (
            "ensemble",
            "regularization_ensemble_metrics",
            "regularization_ensemble_aggregate",
        ),
        ("selected", "metrics", "aggregate"),
    ):
        rows = defaultdict(list)
        classes = {}
        for fold in evaluation["folds"]:
            for label in ("tracked", "wheeled"):
                session = fold[f"{label}_test_session"]
                rows[session].append(fold[field]["per_class_recall"][label])
                classes[session] = label
        sessions = {
            s: {
                "class": classes[s],
                "mean_recall": float(np.mean(v)),
                "minimum_recall": float(min(v)),
                "contexts": len(v),
            }
            for s, v in sorted(rows.items())
        }
        values = {
            c: [r["mean_recall"] for r in sessions.values() if r["class"] == c]
            for c in ("tracked", "wheeled")
        }
        output[name] = {
            "balanced_accuracy": evaluation[aggregate]["balanced_accuracy"]["mean"],
            "recall": {c: float(np.mean(v)) for c, v in values.items()},
            "worst_session_context_recall": min(min(v) for v in rows.values()),
            "session_bootstrap_95_percent": bootstrap(values),
            "sessions": sessions,
        }
    return output


def paired_summary(before, after):
    result = {}
    for head in ("ensemble", "selected"):
        a, b = before[head]["sessions"], after[head]["sessions"]
        if a.keys() != b.keys():
            raise ValueError("paired comparison must have identical sessions")
        values = {
            c: [
                b[s]["mean_recall"] - a[s]["mean_recall"]
                for s in a
                if a[s]["class"] == c
            ]
            for c in ("tracked", "wheeled")
        }
        result[head] = {
            "balanced_accuracy_difference": float(
                np.mean([np.mean(v) for v in values.values()])
            ),
            "paired_session_bootstrap_95_percent": bootstrap(values),
        }
    return result


def evaluate(features, labels, records, output, config):
    if output.exists():
        raise FileExistsError(f"Refusing to replace experiment: {output}")
    output.mkdir(parents=True)
    start = time.perf_counter()
    with threadpool_limits(limits=config["blas_threads"]):
        evaluation, models, splits = nested_leave_session_pair_out(
            features,
            labels,
            records,
            c_values=config["c_values"],
            seed=config["seed"],
            cache_fits=True,
        )
    for split in splits["folds"]:
        validate_split(records, split)
    write_json(output / "metrics.json", evaluation)
    write_json(output / "splits.json", splits)
    torch.save({"models": models}, output / "models.pt")
    report = summary(evaluation)
    write_json(output / "summary.json", report)
    write_json(
        output / "experiment.json",
        {
            "domain": config["domain"],
            "seed": config["seed"],
            "elapsed_seconds": time.perf_counter() - start,
            "configuration": config,
            "feature_shape": list(features.shape),
            "feature_sha256": hashlib.sha256(features.numpy().tobytes()).hexdigest(),
            "dataset_sha256": hashlib.sha256(
                json.dumps(records, sort_keys=True).encode()
            ).hexdigest(),
            "record_count": len(records),
            "parent_experiment": "../experiment.json",
            "artifact_sha256": {
                p.name: sha256(p) for p in output.iterdir() if p.is_file()
            },
        },
    )
    print(
        output.name,
        json.dumps(
            {
                k: {p: v for p, v in row.items() if p != "sessions"}
                for k, row in report.items()
            }
        ),
        flush=True,
    )
    return report


def context_features(b, indices, seconds, tagging, output, config, sample_rate=16000):
    """Extract only reviewed original-source windows; no audio-cleaning changes."""
    output.mkdir(parents=True)
    records, waveforms = [], []
    embeddings, scores = [], []
    cfg = RealCorpusConfig(sample_rate=sample_rate, window_seconds=seconds, channel=0)

    def flush():
        if not waveforms:
            return
        resampled = resample_audio(torch.cat(waveforms), sample_rate, 32000).numpy()
        with torch.inference_mode():
            score, embedding = tagging.inference(resampled)
        embeddings.append(torch.from_numpy(embedding))
        scores.append(torch.from_numpy(score))
        waveforms.clear()

    for i in indices:
        original = b.records[i]
        center = (original["window_start_seconds"] + original["window_end_seconds"]) / 2
        start, end = center - seconds / 2, center + seconds / 2
        source = Path("data/targets") / original["target_source"]
        if end > sf.info(source).duration or start < 0:
            raise ValueError("context would need padding")
        candidate = SimpleNamespace(
            source=SimpleNamespace(path=source), start_sample=round(start * sample_rate)
        )
        waveform, _, _ = _read_window(candidate, cfg)
        if (
            waveform.shape != (1, seconds * sample_rate)
            or not torch.isfinite(waveform).all()
        ):
            raise ValueError("invalid context waveform")
        record = dict(original)
        # This metadata is a source-window manifest, not a generated WAV manifest.
        for key in (
            "audio_path",
            "corrupted_path",
            "metadata_path",
            "real_corpus_config",
        ):
            record.pop(key, None)
        record.update(
            native_index=i,
            anchor_sample_id=original["sample_id"],
            sample_id=f"{original['sample_id']}_context_{seconds}s",
            duration_seconds=seconds,
            num_samples=seconds * sample_rate,
            sample_rate=sample_rate,
            window_start_seconds=start,
            window_end_seconds=end,
            window_start_sample=round(start * sample_rate),
            waveform_sha256=hashlib.sha256(waveform.numpy().tobytes()).hexdigest(),
            rms=float(waveform.square().mean().sqrt()),
        )
        records.append(record)
        waveforms.append(waveform)
        if len(waveforms) == config["batch_size"]:
            flush()
    flush()
    features = {
        "embedding2048": torch.cat(embeddings).float(),
        "semantic35": torch.cat(scores)[
            :, [i for i, _ in SEMANTIC_AUDIOSET_FEATURES]
        ].float(),
    }
    if any(not torch.isfinite(value).all() for value in features.values()):
        raise ValueError("nonfinite context features")
    write_json(output / "windows.json", records)
    torch.save(features, output / "features.pt")
    return records, features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["embedding", "context", "bandwidth"], required=True
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/embedding_context_v1")
    )
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text())
    torch.set_num_threads(config["torch_threads"])
    b = FrozenBenchmark()
    b.verify_native(b.predict(b.semantic, b.classical))
    if b.version != config["native_manifest_sha256"]:
        raise ValueError("preregistered manifest mismatch")
    checkpoint = validate_panns_checkpoint(config["checkpoint"])
    original = torch.load(
        RUN / "panns_features.pt", weights_only=True, map_location="cpu"
    )
    if original["panns_checkpoint_sha256"] != checkpoint["sha256"]:
        raise ValueError("encoder checkpoint mismatch")
    original_features = {
        "semantic35": torch.tensor(b.semantic, dtype=torch.float32),
        "embedding2048": original["real_embeddings"].float(),
    }
    labels = original["real_labels"]
    root = args.output
    config_hashes = {str(p): sha256(p) for p in (CONFIG, PROTOCOL, Path(__file__))}
    if args.stage == "embedding":
        if root.exists():
            raise FileExistsError(root)
        root.mkdir(parents=True)
        metadata = b.metadata(config)
        metadata.update(
            configuration_sha256=config_hashes,
            model_checkpoint=checkpoint,
            train_test_split="per-experiment splits.json",
            versions={
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
            },
            runtime={"machine": platform.machine(), "platform": platform.platform()},
        )
        write_json(root / "experiment.json", metadata)
        reports = {}
        for representation, features in original_features.items():
            reports[representation] = evaluate(
                features, labels, b.records, root / f"native_{representation}", config
            )
        # Reproduce the comparator's original complete fold metrics exactly.
        new = json.loads((root / "native_semantic35/metrics.json").read_text())
        old = json.loads((RUN / "semantic/metrics.json").read_text())
        for new_fold, old_fold in zip(new["folds"], old["folds"], strict=True):
            for field in ("metrics", "regularization_ensemble_metrics"):
                if new_fold[field] != old_fold[field]:
                    raise ValueError("semantic comparator regression")
        reports["paired_embedding_minus_semantic"] = paired_summary(
            reports["semantic35"], reports["embedding2048"]
        )
        write_json(root / "embedding_results.json", reports)
    else:
        metadata = json.loads((root / "experiment.json").read_text())
        if metadata["configuration_sha256"] != config_hashes:
            raise ValueError("configuration/code changed since stage 1")
        if not (root / "embedding_results.json").exists():
            raise ValueError("complete embedding experiment first")
        from panns_inference import AudioTagging

        tagging = AudioTagging(checkpoint_path=checkpoint["path"], device="cpu")
        if args.stage == "bandwidth":
            if not (root / "context_results.json").exists():
                raise ValueError("complete context experiment first")
            records, features = context_features(
                b,
                list(range(len(b.records))),
                2,
                tagging,
                root / "direct32_features",
                config,
                sample_rate=config["bandwidth_ablation_sample_rate"],
            )
            baseline = json.loads((root / "embedding_results.json").read_text())
            results = {
                k: evaluate(v, labels, records, root / f"direct32_{k}", config)
                for k, v in features.items()
            }
            results["paired_vs_original"] = {
                k: paired_summary(baseline[k], results[k]) for k in original_features
            }
            write_json(root / "bandwidth_results.json", results)
            write_json(
                root / "bandwidth_artifact_sha256.json",
                {
                    str(p.relative_to(root)): sha256(p)
                    for p in sorted(root.rglob("*"))
                    if p.is_file() and not p.name.endswith("artifact_sha256.json")
                },
            )
            return
        results = {}
        for spec in config["context_sets"]:
            maximum = spec["max_seconds"]
            indices = matched_indices(b.records, b.catalog, maximum)
            counts = {
                c: len(
                    {
                        b.records[i]["recording_session"]
                        for i in indices
                        if b.records[i]["vehicle_class"] == c
                    }
                )
                for c in ("tracked", "wheeled")
            }
            if counts != spec["expected_sessions"]:
                raise ValueError(f"unexpected context session counts: {counts}")
            prefix = f"matched{maximum}"
            result = {
                "counts": counts,
                "windows": len(indices),
                "per_session_windows": dict(
                    Counter(b.records[i]["recording_session"] for i in indices)
                ),
                "excluded_sessions": sorted(
                    {r["recording_session"] for r in b.records}
                    - {b.records[i]["recording_session"] for i in indices}
                ),
                "durations": {},
            }
            for duration in spec["durations"]:
                if duration == 2:
                    records = [b.records[i] for i in indices]
                    features = {k: v[indices] for k, v in original_features.items()}
                else:
                    records, features = context_features(
                        b,
                        indices,
                        duration,
                        tagging,
                        root / f"{prefix}_{duration}s_features",
                        config,
                    )
                write_json(root / f"{prefix}_{duration}s_windows.json", records)
                result["durations"][duration] = {
                    k: evaluate(
                        v,
                        labels[indices],
                        records,
                        root / f"{prefix}_{duration}s_{k}",
                        config,
                    )
                    for k, v in features.items()
                }
            result["paired_vs_2s"] = {
                duration: {
                    k: paired_summary(
                        result["durations"][2][k], result["durations"][duration][k]
                    )
                    for k in original_features
                }
                for duration in spec["durations"]
                if duration != 2
            }
            results[prefix] = result
            write_json(root / "context_results.json", results)
    write_json(
        root / f"{args.stage}_artifact_sha256.json",
        {
            str(p.relative_to(root)): sha256(p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and not p.name.endswith("artifact_sha256.json")
        },
    )


if __name__ == "__main__":
    main()

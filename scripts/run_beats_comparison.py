"""Run the preregistered frozen BEATs comparison on unchanged development windows."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
import sklearn
import torch
import yaml

from evaluate_embedding_context import evaluate, paired_summary
from summarize_preprocessing_grid import audit_models
from vehicle_audio.beats_adapter import extract_embeddings, load_encoder
from vehicle_audio.benchmark_followup import sha256, write_json
from vehicle_audio.development_corpus import (
    CONTROL,
    MANIFEST,
    MANIFEST_SHA256,
    load_development,
)

CONFIG = Path("configs/benchmark_beats_v1.yaml")
PROTOCOL = Path("docs/beats_comparison_protocol.md")


def hashes(root):
    return {
        str(p.relative_to(root)): sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != ".DS_Store"
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/beats_comparison_v2"))
    parser.add_argument(
        "--package", type=Path, default=Path("benchmarks/v0.1/beats_comparison_v2")
    )
    args = parser.parse_args()
    if args.output.exists() or args.package.exists():
        raise FileExistsError("refusing to overwrite experiment/package")
    config = yaml.safe_load(CONFIG.read_text())
    if config["manifest_sha256"] != MANIFEST_SHA256:
        raise ValueError("protocol manifest mismatch")
    torch.set_num_threads(config["torch_threads"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    records, sources, raw, audit = load_development()
    encoder, provenance = load_encoder(config)
    controls_hashes = json.loads(
        Path("runs/preprocessing_grid_v1/artifact_sha256.json").read_text()
    )
    for name in ("features.pt", "semantic35/metrics.json", "semantic35/splits.json"):
        path = CONTROL / name
        if (
            sha256(path)
            != controls_hashes[str(path.relative_to("runs/preprocessing_grid_v1"))]
        ):
            raise ValueError("PANNs control artifacts changed")
    cache = torch.load(CONTROL / "features.pt", weights_only=True, map_location="cpu")
    start = time.perf_counter()
    root = args.output
    root.mkdir(parents=True)
    code_paths = [
        CONFIG,
        PROTOCOL,
        Path(__file__),
        Path("scripts/evaluate_embedding_context.py"),
        Path("scripts/summarize_preprocessing_grid.py"),
        *sorted(Path("src/vehicle_audio").glob("*.py")),
    ]
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "domain": config["domain"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ),
        "seed": config["seed"],
        "configuration": config,
        "dataset_sha256": MANIFEST_SHA256,
        "code_sha256": {str(p): sha256(p) for p in code_paths},
        "model_checkpoint": provenance,
        "panns_control_features_sha256": sha256(CONTROL / "features.pt"),
        "train_test_split": "per-representation splits.json",
        "audit": audit,
        "versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu": subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip(),
        },
        "parameters": sum(p.numel() for p in encoder.parameters()),
        "encoder_trainable_parameters": 0,
    }
    write_json(root / "experiment.json", metadata)
    for source, name in (
        (CONFIG, "config.yaml"),
        (PROTOCOL, "protocol.md"),
        (MANIFEST, "manifest.jsonl"),
        (Path("configs/audio_sources.yaml"), "audio_sources.yaml"),
        (Path("configs/benchmark_v0_1.yaml"), "split_policy.yaml"),
        (
            Path("configs/archived_startup_intervals.yaml"),
            "archived_startup_intervals.yaml",
        ),
    ):
        shutil.copy2(source, root / name)
    waves = torch.from_numpy(raw)
    inference_start = time.perf_counter()
    features = []
    for offset in range(0, len(waves), config["batch_size"]):
        features.append(
            extract_embeddings(encoder, waves[offset : offset + config["batch_size"]])
        )
        if offset % 80 == 0:
            print(
                f"BEATs embeddings: {min(offset + config['batch_size'], len(waves))}/{len(waves)}",
                flush=True,
            )
    embeddings = torch.cat(features)
    inference_seconds = time.perf_counter() - inference_start
    repeated = extract_embeddings(encoder, waves[:8])
    single = torch.cat(
        [extract_embeddings(encoder, waves[i : i + 1]) for i in range(8)]
    )
    repeat_error = float((embeddings[:8] - repeated).abs().max())
    batch_error = float((embeddings[:8] - single).abs().max())
    if repeat_error > 1e-5 or batch_error > 1e-4:
        raise ValueError(
            f"embedding reproducibility failed: {repeat_error}, {batch_error}"
        )
    torch.save(
        {
            "features": embeddings,
            "sample_ids": [r["sample_id"] for r in records],
            "manifest_sha256": MANIFEST_SHA256,
            "checkpoint_sha256": config["checkpoint_sha256"],
        },
        root / "beats_features.pt",
    )
    labels = torch.tensor([int(r["vehicle_class"] == "wheeled") for r in records])
    features_by_name = {
        "beats768": embeddings,
        "panns_semantic35": cache["semantic35"],
        "panns_embedding2048": cache["embedding2048"],
    }
    reports = {}
    replays = 0
    reference_splits = json.loads((CONTROL / "semantic35/splits.json").read_text())
    for name, feature in features_by_name.items():
        report = evaluate(feature, labels, records, root / name, config)
        splits = json.loads((root / name / "splits.json").read_text())
        if splits != reference_splits:
            raise ValueError("matched comparison split mismatch")
        models = torch.load(root / name / "models.pt", weights_only=True)["models"]
        metrics = json.loads((root / name / "metrics.json").read_text())
        replays += audit_models(feature.numpy(), records, metrics, models, splits)
        reports[name] = report
    original_metrics = json.loads((CONTROL / "semantic35/metrics.json").read_text())
    reproduced = json.loads((root / "panns_semantic35/metrics.json").read_text())
    if reproduced != original_metrics:
        raise ValueError("PANNs control did not reproduce exactly")
    results = {
        "domain": config["domain"],
        "primary_head": config["primary_head"],
        "models": reports,
        "paired_vs_panns_semantic35": paired_summary(
            reports["panns_semantic35"], reports["beats768"]
        ),
        "paired_vs_panns_embedding2048": paired_summary(
            reports["panns_embedding2048"], reports["beats768"]
        ),
        "runtime": {
            "feature_extraction_seconds": inference_seconds,
            "total_experiment_seconds": time.perf_counter() - start,
            "note": "CPU batch throughput, not streaming latency; excludes download/model loading",
        },
        "verification": {
            "saved_head_fold_replays": replays,
            "split_equality": True,
            "panns_control_exact_reproduction": True,
            "repeat_embedding_max_abs_error": repeat_error,
            "single_vs_batch_max_abs_error": batch_error,
            "protected_overlap": [],
            "startup_overlap": [],
        },
        "checkpoint_provenance_limitation": provenance["limitation"],
    }
    write_json(root / "results.json", results)
    write_json(root / "artifact_sha256.json", hashes(root))
    args.package.mkdir(parents=True)
    for p in root.rglob("*"):
        if (
            p.is_file()
            and p.suffix != ".pt"
            and p.name not in {"artifact_sha256.json", ".DS_Store"}
        ):
            destination = args.package / p.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, destination)
    write_json(
        args.package / "local_artifact_sha256.json",
        json.loads((root / "artifact_sha256.json").read_text()),
    )
    write_json(args.package / "artifact_sha256.json", hashes(args.package))
    print(f"Verified and packaged: {args.package}", flush=True)


if __name__ == "__main__":
    main()

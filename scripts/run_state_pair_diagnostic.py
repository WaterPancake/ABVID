"""Run the approved fixed-pair cross-state diagnostic, outside the benchmark."""

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
from threadpoolctl import threadpool_limits
import yaml

from vehicle_audio.benchmark_followup import sha256, write_json
from vehicle_audio.development_corpus import (
    CONTROL,
    MANIFEST,
    MANIFEST_SHA256,
    load_development,
)
from vehicle_audio.state_identity_audit import audit_inventory, worksheet_segments
from vehicle_audio.state_pair_diagnostic import (
    fit_head,
    make_tasks,
    metrics,
    replay_head,
)

CONFIG = Path("configs/state_pair_ab_v1.yaml")
PROTOCOL = Path("docs/state_pair_ab_protocol.md")
BEATS = Path("runs/beats_comparison_v2")
GRID = Path("runs/preprocessing_grid_v1")


def verified_artifact(root, relative):
    index = json.loads((root / "artifact_sha256.json").read_text())
    path = root / relative
    if sha256(path) != index[relative]:
        raise ValueError(f"artifact changed: {path}")
    return path


def load_features(records):
    beat_path = verified_artifact(BEATS, "beats_features.pt")
    control_path = verified_artifact(
        GRID, str((CONTROL / "features.pt").relative_to(GRID))
    )
    beat_meta = json.loads(verified_artifact(BEATS, "experiment.json").read_text())
    grid_meta = json.loads(verified_artifact(GRID, "experiment.json").read_text())
    if sha256(verified_artifact(BEATS, "manifest.jsonl")) != MANIFEST_SHA256:
        raise ValueError("BEATs manifest mismatch")
    if sha256(GRID / "dataset/real_manifest.jsonl") != MANIFEST_SHA256:
        raise ValueError("PANNs manifest mismatch")
    if beat_meta["dataset_sha256"] != MANIFEST_SHA256:
        raise ValueError("feature experiment dataset mismatch")
    if beat_meta["panns_control_features_sha256"] != sha256(control_path):
        raise ValueError("PANNs cache does not match prior verified comparison")
    beat = torch.load(beat_path, map_location="cpu", weights_only=True)
    control = torch.load(control_path, map_location="cpu", weights_only=True)
    if beat["sample_ids"] != [r["sample_id"] for r in records]:
        raise ValueError("BEATs sample ordering mismatch")
    if beat["manifest_sha256"] != MANIFEST_SHA256:
        raise ValueError("BEATs cache manifest mismatch")
    if beat["checkpoint_sha256"] != beat_meta["model_checkpoint"]["checkpoint_sha256"]:
        raise ValueError("BEATs checkpoint identity mismatch")
    checkpoints = {
        "beats": (
            Path(beat_meta["configuration"]["checkpoint"]),
            beat["checkpoint_sha256"],
        ),
        "panns": (
            Path(grid_meta["model_checkpoint"]["path"]),
            grid_meta["model_checkpoint"]["sha256"],
        ),
    }
    for path, digest in checkpoints.values():
        if sha256(path) != digest:
            raise ValueError(f"checkpoint changed: {path}")
    features = {
        "beats768": beat["features"].numpy().astype(np.float64),
        "panns_embedding2048": control["embedding2048"].numpy().astype(np.float64),
        "panns_semantic35": control["semantic35"].numpy().astype(np.float64),
    }
    for name, dimension in (
        ("beats768", 768),
        ("panns_embedding2048", 2048),
        ("panns_semantic35", 35),
    ):
        if (
            features[name].shape != (len(records), dimension)
            or not np.isfinite(features[name]).all()
        ):
            raise ValueError(f"feature shape/values mismatch: {name}")
    provenance = {
        "feature_files_sha256": {str(p): sha256(p) for p in (beat_path, control_path)},
        "beats_checkpoint": beat_meta["model_checkpoint"],
        "panns_checkpoint": grid_meta["model_checkpoint"],
        "ordering": "BEATs explicit IDs; PANNs pinned manifest and cache hash from matched-control comparison",
        "pretraining_overlap": "AudioSet pretraining overlap unknown for both encoders",
    }
    return features, provenance


def render_results(results):
    lines = [
        "# Fixed-pair idle/moving diagnostic results",
        "",
        "**Domain: real -> real within-session cross-state diagnostic.**",
        "Train/test audio is non-overlapping but source sessions are shared by approved exception.",
        "Not independent vehicle recognition, unseen-session evaluation, or milestone evidence.",
        "Moving means steady_speed only. A/B refer to the fixed pair-member order.",
        "",
        "| Pair (A / B) | Direction | Features | Train A/B | Test A/B | Train BA | Test BA | Recall A | Recall B |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for r in results["fits"]:
        tr, te = r["train_metrics"], r["test_metrics"]
        pair = " / ".join(m["model"] for m in r["pair"]["members"])
        direction = f"{r['train_state']} → {r['test_state']}"
        lines.append(
            f"| {pair} | {direction} | {r['representation']} | {tr['n_A']}/{tr['n_B']} "
            f"| {te['n_A']}/{te['n_B']} | {100 * tr['balanced_accuracy']:.2f}% "
            f"| {100 * te['balanced_accuracy']:.2f}% | {100 * te['recall_A']:.2f}% | {100 * te['recall_B']:.2f}% |"
        )
    lines += [
        "",
        "Training BA is resubstitution, not held-out evidence. A constant decision gives 50% BA.",
        "No confidence intervals: windows overlap and there is only one source session per member.",
        "Stryker is convoy audio; T-72B3 has only eight idle windows. Physical vehicle continuity is unverified.",
        "Class identity is confounded with source, microphone and scene. No component was isolated as engine or tracks.",
        "See results.json for confusion counts and numerical replay checks; splits.json for exact records;",
        "models.json for float64 heads; predictions.json for every window; experiment.json for provenance.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/state_pair_ab_v1"))
    parser.add_argument("--allow-within-session-diagnostic", action="store_true")
    args = parser.parse_args()
    if not args.allow_within_session_diagnostic:
        parser.error(
            "requires --allow-within-session-diagnostic; not a session-disjoint benchmark"
        )
    if args.output.exists():
        raise FileExistsError(args.output)
    config = yaml.safe_load(CONFIG.read_text())
    if (
        config["manifest_sha256"] != MANIFEST_SHA256
        or config["benchmark_eligible"] is not False
    ):
        raise ValueError("diagnostic configuration mismatch")
    start = time.perf_counter()
    records, sources, _, audit = load_development()
    worksheet = Path("docs/audio_segment_review.md")
    reviews = worksheet_segments(
        worksheet.read_text(), {r["source_id"] for r in records}
    )
    inventory = audit_inventory(records, sources, reviews)
    tasks = make_tasks(records, inventory, config, allow_within_session=True)
    features, provenance = load_features(records)
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    paths = [
        CONFIG,
        PROTOCOL,
        MANIFEST,
        worksheet,
        Path("configs/audio_sources.yaml"),
        Path("configs/benchmark_v0_1.yaml"),
        Path("configs/archived_startup_intervals.yaml"),
        Path(__file__),
        Path("src/vehicle_audio/state_pair_diagnostic.py"),
        Path("src/vehicle_audio/state_identity_audit.py"),
        Path("src/vehicle_audio/development_corpus.py"),
    ]
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "domain": config["domain"],
        "benchmark_eligible": False,
        "configuration": config,
        "seed": config["seed"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "input_code_sha256": {str(p): sha256(p) for p in paths},
        "model_checkpoint_and_feature_provenance": provenance,
        "corpus_audit": audit,
        "train_test_split": "splits.json",
        "encoder_trainable_parameters": 0,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "sklearn": sklearn.__version__,
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_model_note": "Exact CPU model not queried; sandbox restricts sysctl.",
        },
    }
    write_json(root / "experiment.json", metadata)
    write_json(root / "splits.json", tasks)
    write_json(root / "identity_audit.json", inventory)
    shutil.copy2(CONFIG, root / "config.yaml")
    shutil.copy2(PROTOCOL, root / "protocol.md")
    results, models, predictions = [], {}, []
    fitting_start = time.perf_counter()
    with threadpool_limits(limits=config["blas_threads"]):
        for task in tasks:
            train_indices = [r["parent_index"] for r in task["train"]]
            test_indices = [r["parent_index"] for r in task["test"]]
            y_train = np.array([r["pair_label"] for r in task["train"]])
            y_test = np.array([r["pair_label"] for r in task["test"]])
            for name in config["representations"]:
                x = features[name]
                state, scaler, model = fit_head(x[train_indices], y_train, config)
                fit_id = f"{task['id']}__{name}"
                models[fit_id] = state
                # JSON round-trip guarantees the replay uses serialized float64 values.
                replay_state = json.loads(json.dumps(state, allow_nan=False))
                errors = []
                scored = {}
                for partition, indices, labels in (
                    ("train", train_indices, y_train),
                    ("test", test_indices, y_test),
                ):
                    expected = model.predict_proba(scaler.transform(x[indices]))[:, 1]
                    probability = replay_head(x[indices], replay_state)
                    error = float(np.max(np.abs(expected - probability)))
                    if error > 1e-10 or not np.array_equal(
                        expected >= config["threshold"],
                        probability >= config["threshold"],
                    ):
                        raise ValueError("saved-head replay mismatch")
                    errors.append(error)
                    scored[partition] = metrics(
                        labels, probability, config["threshold"]
                    )
                    for row, p in zip(task[partition], probability, strict=True):
                        predictions.append(
                            {
                                "fit_id": fit_id,
                                "partition": partition,
                                "sample_id": row["sample_id"],
                                "true_pair_label": row["pair_label"],
                                "probability_B": float(p),
                                "predicted_pair_label": int(p >= config["threshold"]),
                            }
                        )
                results.append(
                    {
                        "fit_id": fit_id,
                        "pair": task["pair"],
                        "representation": name,
                        "train_state": task["train_state"],
                        "test_state": task["test_state"],
                        "domain": config["domain"],
                        "benchmark_eligible": False,
                        "shared_sessions": task["shared_sessions"],
                        "train_metrics": scored["train"],
                        "test_metrics": scored["test"],
                        "replay_max_abs_error": max(errors),
                    }
                )
                print(f"Completed {fit_id}", flush=True)
    output = {
        "domain": config["domain"],
        "benchmark_eligible": False,
        "fits": results,
        "verification": {
            "fits": len(results),
            "train_test_replays": 2 * len(results),
            "source_sample_overlap": False,
            "protected_overlap": [],
            "startup_overlap": [],
        },
        "runtime": {
            "fitting_and_replay_seconds": time.perf_counter() - fitting_start,
            "total_seconds": time.perf_counter() - start,
            "note": "Cached features; no encoder inference. Includes data/hash validation, not streaming latency.",
        },
    }
    write_json(root / "models.json", models)
    write_json(root / "predictions.json", predictions)
    write_json(root / "results.json", output)
    (root / "results.md").write_text(render_results(output))
    write_json(
        root / "artifact_sha256.json",
        {
            p.name: sha256(p)
            for p in sorted(root.iterdir())
            if p.is_file() and p.name != ".DS_Store"
        },
    )
    print(root / "results.md")


if __name__ == "__main__":
    main()

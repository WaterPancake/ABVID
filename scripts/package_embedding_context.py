"""Verify saved follow-up probes and package all results without selecting a winner."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from vehicle_audio.benchmark_followup import (
    FrozenBenchmark,
    RUN,
    ensemble_probability,
    sha256,
    validate_split,
    write_json,
)
from vehicle_audio.real_corpus import RealCorpusConfig, _read_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/embedding_context_v1"))
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/v0.1/embedding_context_v1")
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    root, output = args.run, args.output
    if output.exists():
        raise FileExistsError(output)
    b = FrozenBenchmark()
    original = torch.load(RUN / "panns_features.pt", weights_only=True)
    baseline = {
        "embedding2048": original["real_embeddings"],
        "semantic35": torch.tensor(b.semantic),
    }
    for relative, expected in json.loads(
        (root / "bandwidth_artifact_sha256.json").read_text()
    ).items():
        # Finder writes this metadata asynchronously; it is not experiment data.
        if Path(relative).name == ".DS_Store":
            continue
        if sha256(root / relative) != expected:
            raise ValueError(f"artifact changed: {relative}")
    waveform_checks = 0
    for manifest in sorted(root.glob("*_features/windows.json")):
        for record in json.loads(manifest.read_text()):
            waveform, _, _ = _read_window(
                SimpleNamespace(
                    source=SimpleNamespace(
                        path=Path("data/targets") / record["target_source"]
                    ),
                    start_sample=record["window_start_sample"],
                ),
                RealCorpusConfig(
                    sample_rate=record["sample_rate"],
                    window_seconds=record["duration_seconds"],
                    channel=0,
                ),
            )
            if (
                hashlib.sha256(waveform.numpy().tobytes()).hexdigest()
                != record["waveform_sha256"]
            ):
                raise ValueError("source-window regeneration mismatch")
            waveform_checks += 1
    checks, rows = [], []
    training_diagnostic = {}
    for directory in sorted(root.iterdir()):
        if not (directory / "models.pt").exists():
            continue
        representation = directory.name.split("_")[-1]
        prefix = directory.name.removesuffix("_" + representation)
        if prefix == "native":
            records, features = b.records, baseline[representation]
        elif prefix == "direct32":
            records = json.loads((root / "direct32_features/windows.json").read_text())
            features = torch.load(
                root / "direct32_features/features.pt", weights_only=True
            )[representation]
        else:
            records = json.loads((root / f"{prefix}_windows.json").read_text())
            if prefix.endswith("_2s"):
                index = {r["sample_id"]: i for i, r in enumerate(b.records)}
                features = baseline[representation][
                    [index[r["sample_id"]] for r in records]
                ]
            else:
                features = torch.load(
                    root / f"{prefix}_features/features.pt", weights_only=True
                )[representation]
        features = features.numpy()
        if not np.isfinite(features).all():
            raise ValueError("nonfinite features")
        for r in records:
            source = b.catalog[r["source_id"]]
            if not source["admitted_to_corpus"]:
                raise ValueError("protected/unadmitted source")
            if not any(
                s["start_seconds"] <= r["window_start_seconds"]
                and r["window_end_seconds"] <= s["end_seconds"]
                and s["operating_condition"] == r["operating_condition"]
                for s in source["condition_segments"]
            ):
                raise ValueError("unreviewed context")
        models = torch.load(directory / "models.pt", weights_only=True)["models"]
        splits = json.loads((directory / "splits.json").read_text())["folds"]
        metrics = json.loads((directory / "metrics.json").read_text())
        training_scores = []
        for split, fold in zip(splits, metrics["folds"], strict=True):
            validate_split(records, split)
            test_ids = set(split["test_sample_ids"])
            indices = [i for i, r in enumerate(records) if r["sample_id"] in test_ids]
            y = np.array(
                [int(records[i]["vehicle_class"] == "wheeled") for i in indices]
            )
            state = models[fold["fold_id"]]
            if prefix == "native":
                # Post-hoc resubstitution diagnostic, never a generalization score.
                train_ids = set(split["train_sample_ids"])
                train_indices = [
                    i for i, r in enumerate(records) if r["sample_id"] in train_ids
                ]
                predictions = (
                    ensemble_probability(
                        features[train_indices], state["regularization_ensemble"]
                    )
                    > 0.5
                )
                class_scores = []
                for name in ("tracked", "wheeled"):
                    sessions = {
                        records[i]["recording_session"]
                        for i in train_indices
                        if records[i]["vehicle_class"] == name
                    }
                    class_scores.append(
                        np.mean(
                            [
                                np.mean(
                                    predictions[
                                        [
                                            records[i]["recording_session"] == session
                                            for i in train_indices
                                        ]
                                    ]
                                    == int(name == "wheeled")
                                )
                                for session in sessions
                            ]
                        )
                    )
                training_scores.append(float(np.mean(class_scores)))
            for head, field in (
                ("regularization_ensemble", "regularization_ensemble_metrics"),
                ("nested_selected", "metrics"),
            ):
                states = (
                    state[head]
                    if head == "regularization_ensemble"
                    else {"selected": state[head]}
                )
                prediction = ensemble_probability(features[indices], states) > 0.5
                for k, name in enumerate(("tracked", "wheeled")):
                    if (
                        float(np.mean(prediction[y == k] == k))
                        != fold[field]["per_class_recall"][name]
                    ):
                        raise ValueError(
                            f"saved checkpoint replay mismatch: {directory}/{fold['fold_id']}/{head}"
                        )
                checks.append(
                    {
                        "experiment": directory.name,
                        "fold": fold["fold_id"],
                        "head": head,
                    }
                )
        if training_scores:
            training_diagnostic[representation] = {
                "mean_training_session_balanced_accuracy": float(
                    np.mean(training_scores)
                ),
                "per_outer_fit": training_scores,
                "interpretation": "post-hoc training resubstitution only; not held-out performance",
            }
        report = json.loads((directory / "summary.json").read_text())
        rows.append({"experiment": directory.name, "windows": len(records), **report})
    if len(rows) != 14:
        raise ValueError(f"incomplete experiment matrix: {len(rows)}")
    output.mkdir(parents=True)
    for path in root.rglob("*.json"):
        target = output / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write_json(
        output / "verification.json",
        {
            "saved_probe_replays": len(checks),
            "regenerated_source_windows_exact": waveform_checks,
            "ignored_nonexperimental_metadata": [".DS_Store"],
            "checks": checks,
            "protected_source_overlap": [],
            "reviewed_interval_bounds": "all pass",
            "configuration": "copied experiment.json",
            "checkpoints_local_root": str(root),
            "verifier_sha256": sha256(Path(__file__)),
        },
    )
    shutil.copyfile(
        "configs/benchmark_embedding_context_v1.yaml", output / "protocol.yaml"
    )
    shutil.copyfile("docs/embedding_context_protocol.md", output / "protocol.md")
    write_json(output / "results.json", rows)
    write_json(output / "training_fit_diagnostic.json", training_diagnostic)
    header = [
        "# Embedding and context follow-up: all results",
        "",
        "Native real → real, nested unseen-session **development**. Primary fixed-C probability ensemble.",
        "",
        "| Experiment | Windows | Balanced accuracy | Tracked recall | Wheeled recall | Worst recall | Descriptive 95% interval |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        r = row["ensemble"]
        lo, hi = r["session_bootstrap_95_percent"]
        header.append(
            f"| {row['experiment']} | {row['windows']} | {100 * r['balanced_accuracy']:.2f}% | "
            f"{100 * r['recall']['tracked']:.2f}% | {100 * r['recall']['wheeled']:.2f}% | "
            f"{100 * r['worst_session_context_recall']:.2f}% | {100 * lo:.2f}–{100 * hi:.2f}% |"
        )
    header += [
        "",
        "Compare durations **within** a matched set only. Matched4 has 7/5 sessions; matched8 has 6/5.",
        "Native/direct32 both use all 593 original windows and 7/5 sessions. Original fusion/classical controls remain in the frozen native benchmark.",
        "Full selected-C secondary results, splits, per-fold metrics, per-session recalls, paired intervals and checkpoint hashes are in the adjacent JSON files.",
        "Checkpoints and feature tensors remain under `runs/embedding_context_v1/`; no audio is redistributed here.",
        "No condition passes the worst-session gate. No reserved-pair predictions were generated.",
    ]
    (output / "README.md").write_text("\n".join(header) + "\n")
    context = json.loads((root / "context_results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, (key, result) in zip(axes, context.items(), strict=True):
        for representation, label in (
            ("semantic35", "35 AudioSet scores"),
            ("embedding2048", "2,048 embedding features"),
        ):
            durations = sorted(map(int, result["durations"]))
            values = [
                100
                * result["durations"][str(d)][representation]["ensemble"][
                    "balanced_accuracy"
                ]
                for d in durations
            ]
            ax.plot(durations, values, "o-", label=label)
            for d, v in zip(durations, values, strict=True):
                ax.annotate(
                    f"{v:.1f}",
                    (d, v),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
        ax.axhline(50, color="gray", ls=":", label="50% reference")
        ax.set(
            ylim=(25, 75),
            xticks=durations,
            xlabel="Window duration (seconds)",
            ylabel="Mean balanced accuracy (%)",
            title=f"{key}: {result['counts']['tracked']} tracked / 5 wheeled\n{result['windows']} matched centers",
        )
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Native real → real development; compare within each panel")
    fig.savefig(output / "context_comparison.png", dpi=170)
    plt.close(fig)
    print(
        json.dumps(
            {
                "experiments": len(rows),
                "saved_probe_replays": len(checks),
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()

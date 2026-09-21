"""Inner-only preprocessing selection, saved-head audit and versioned grid report."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import yaml

from evaluate_embedding_context import paired_summary, summary
from run_preprocessing_grid import file_hashes, verify_hashes
from vehicle_audio.benchmark_followup import (
    ensemble_probability,
    sha256,
    validate_split,
    write_json,
)
from vehicle_audio.preprocessing_grid import (
    preprocess,
    preprocessing_grid,
    selection_key,
)
from vehicle_audio.semantic_session_evaluation import (
    _aggregate_folds,
    SEMANTIC_AUDIOSET_FEATURES,
)


def audit_models(features, records, evaluation, models, splits):
    count = 0
    for fold, split in zip(evaluation["folds"], splits["folds"], strict=True):
        validate_split(records, split)
        if fold["fold_id"] != split["fold_id"]:
            raise ValueError("fold identity mismatch")
        ids = set(split["test_sample_ids"])
        indices = [i for i, r in enumerate(records) if r["sample_id"] in ids]
        labels = np.array(
            [int(records[i]["vehicle_class"] == "wheeled") for i in indices]
        )
        state = models[fold["fold_id"]]
        for head, field in (
            ("nested_selected", "metrics"),
            ("regularization_ensemble", "regularization_ensemble_metrics"),
        ):
            states = (
                {"selected": state[head]} if head == "nested_selected" else state[head]
            )
            predicted = ensemble_probability(features[indices], states) > 0.5
            for k, name in enumerate(("tracked", "wheeled")):
                if (
                    float(np.mean(predicted[labels == k] == k))
                    != fold[field]["per_class_recall"][name]
                ):
                    raise ValueError("saved probe recall mismatch")
            count += 1
    return count


def build(root, output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    complete = json.loads((root / "complete.json").read_text())
    if complete["variants"] != 24:
        raise ValueError("incomplete grid")
    verify_hashes(root, json.loads((root / "artifact_sha256.json").read_text()))
    config = yaml.safe_load((root / "protocol.yaml").read_text())
    variants = preprocessing_grid(
        config["grid"]["remove_dc"],
        config["grid"]["highpass_hz"],
        config["grid"]["rms_dbfs"],
    )
    records = [
        json.loads(line)
        for line in (root / "dataset/real_manifest.jsonl").read_text().splitlines()
    ]
    raw = np.stack(
        [
            sf.read(root / "dataset" / r["audio_path"], dtype="float32")[0]
            for r in records
        ]
    )
    catalog = {
        s["id"]: s
        for s in yaml.safe_load((root / "audio_sources.yaml").read_text())["sources"]
    }
    archive = yaml.safe_load((root / "archived_startup_intervals.yaml").read_text())[
        "intervals"
    ]
    policy = yaml.safe_load(Path("configs/benchmark_v0_1.yaml").read_text())
    forbidden = set(
        policy["consumed_locked_source_ids"] + policy["future_confirmation_source_ids"]
    )
    for r in records:
        if r["source_id"] in forbidden or r["operating_condition"] == "startup":
            raise ValueError("protected/startup sample")
        source = catalog[r["source_id"]]
        if source.get("admitted_to_corpus") is not True or not any(
            s["start_seconds"] <= r["window_start_seconds"]
            and r["window_end_seconds"] <= s["end_seconds"]
            and s["operating_condition"] == r["operating_condition"]
            for s in source["condition_segments"]
        ):
            raise ValueError("window is not reviewed and admitted")
        if any(
            a["source_id"] == r["source_id"]
            and r["window_start_seconds"] < a["end_seconds"]
            and a["start_seconds"] < r["window_end_seconds"]
            for a in archive
        ):
            raise ValueError("overlap with archived startup")
    for r in {r["source_id"]: r for r in records}.values():
        if (
            sha256(Path("data/targets") / r["target_source"])
            != r["normalized_source_sha256"]
        ):
            raise ValueError("original audio changed")
    evaluations, fixed_reports = {}, {}
    # Independent unchanged-frontend check on windows shared with the old corpus.
    old_manifest = Path("data/benchmark_v0_1_native_real_7t5w/real_manifest.jsonl")
    old_records = [json.loads(line) for line in old_manifest.read_text().splitlines()]
    new_by_id = {r["sample_id"]: i for i, r in enumerate(records)}
    common = [
        (i, new_by_id[r["sample_id"]])
        for i, r in enumerate(old_records)
        if r["sample_id"] in new_by_id
    ]
    old_cache = torch.load(
        "runs/benchmark_v0_1_native_real_7t5w/panns_features.pt", weights_only=True
    )
    new_cache = torch.load(
        root / "variants" / variants[0].name / "features.pt", weights_only=True
    )
    frontend_check = {"common_window_count": len(common), "max_absolute_difference": {}}
    for name, old_features in (
        (
            "semantic35",
            old_cache["real_clipwise_outputs"][
                :, [i for i, _ in SEMANTIC_AUDIOSET_FEATURES]
            ],
        ),
        ("embedding2048", old_cache["real_embeddings"]),
    ):
        difference = float(
            (
                old_features[[i for i, j in common]]
                - new_cache[name][[j for i, j in common]]
            )
            .abs()
            .max()
        )
        if difference > 5e-5:
            raise ValueError(
                "unchanged frontend differs beyond floating-point batch tolerance"
            )
        frontend_check["max_absolute_difference"][name] = difference
    replay_count, waveform_count = 0, 0
    for variant in variants:
        directory = root / "variants" / variant.name
        processed, _ = preprocess(raw, variant)
        diagnostics = json.loads((directory / "waveform_diagnostics.json").read_text())
        for r, y, d in zip(records, processed, diagnostics, strict=True):
            if (
                d["sample_id"] != r["sample_id"]
                or hashlib.sha256(y.tobytes()).hexdigest() != d["waveform_sha256"]
            ):
                raise ValueError("preprocessing regeneration mismatch")
            waveform_count += 1
        features = torch.load(directory / "features.pt", weights_only=True)
        evaluations[variant.name], fixed_reports[variant.name] = {}, {}
        for representation in ("semantic35", "classical53"):
            model_dir = directory / representation
            evaluation = json.loads((model_dir / "metrics.json").read_text())
            splits = json.loads((model_dir / "splits.json").read_text())
            models = torch.load(model_dir / "models.pt", weights_only=True)["models"]
            replay_count += audit_models(
                features[representation].numpy(), records, evaluation, models, splits
            )
            evaluations[variant.name][representation] = evaluation
            fixed_reports[variant.name][representation] = summary(evaluation)
    output.mkdir(parents=True)
    # Keep all fixed pipelines, including failures; do not promote the outer maximum.
    for name in (
        "experiment.json",
        "protocol.yaml",
        "protocol.md",
        "audio_sources.yaml",
        "archived_startup_intervals.yaml",
    ):
        shutil.copyfile(root / name, output / name)
    (output / "dataset").mkdir()
    for path in (root / "dataset").glob("*.json*"):
        shutil.copyfile(path, output / "dataset" / path.name)
    for path in (root / "variants").glob("*/*/*.json"):
        target = output / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write_json(
        output / "local_artifact_sha256.json",
        json.loads((root / "artifact_sha256.json").read_text()),
    )
    write_json(output / "fixed_pipeline_results.json", fixed_reports)
    results = {}
    for representation in ("semantic35", "classical53"):
        chosen_folds, selected_models = [], {}
        selections = []
        control = evaluations[variants[0].name][representation]
        splits = json.loads(
            (
                root / "variants" / variants[0].name / representation / "splits.json"
            ).read_text()
        )
        for i, control_fold in enumerate(control["folds"]):
            winner_index = max(
                range(len(variants)),
                key=lambda j: selection_key(
                    variants[j],
                    evaluations[variants[j].name][representation]["folds"][i],
                    j,
                ),
            )
            winner = variants[winner_index]
            fold = evaluations[winner.name][representation]["folds"][i]
            if fold["fold_id"] != control_fold["fold_id"]:
                raise ValueError("grid folds reordered")
            chosen_folds.append(
                {
                    **fold,
                    "selected_preprocessing": winner.to_dict(),
                    "selected_pipeline": winner.name,
                }
            )
            state = torch.load(
                root / "variants" / winner.name / representation / "models.pt",
                weights_only=True,
            )["models"][fold["fold_id"]]
            selected_models[fold["fold_id"]] = {
                **state,
                "preprocessing": winner.to_dict(),
            }
            selections.append(
                {
                    "fold_id": fold["fold_id"],
                    "pipeline": winner.name,
                    "regularization_c": fold["selected_regularization_c"],
                    "inner_score": fold["inner_selection_mean_balanced_accuracy"],
                }
            )
            # Verify the selected checkpoint uses the winning frontend at inference.
            features = torch.load(
                root / "variants" / winner.name / "features.pt", weights_only=True
            )[representation].numpy()
            replay_count += audit_models(
                features,
                records,
                {"folds": [fold]},
                {fold["fold_id"]: state},
                {"folds": [splits["folds"][i]]},
            )
        evaluation = {
            "folds": chosen_folds,
            "aggregate": _aggregate_folds(
                chosen_folds,
                metrics_field="metrics",
                session_predictions_field="session_predictions",
            ),
            "regularization_ensemble_aggregate": _aggregate_folds(
                chosen_folds,
                metrics_field="regularization_ensemble_metrics",
                session_predictions_field="regularization_ensemble_session_predictions",
            ),
        }
        report = summary(evaluation)
        directory = output / f"selected_{representation}"
        write_json(directory / "metrics.json", evaluation)
        write_json(directory / "splits.json", splits)
        write_json(directory / "selections.json", selections)
        saved = root / f"selected_{representation}_models.pt"
        torch.save(
            {
                "models": selected_models,
                "configuration": config,
                "dataset_version": sha256(root / "dataset/real_manifest.jsonl"),
            },
            saved,
        )
        write_json(
            directory / "checkpoint.json",
            {"local_path": str(saved), "sha256": sha256(saved)},
        )
        results[representation] = {
            "control": fixed_reports[variants[0].name][representation],
            "nested_grid": report,
            "paired_difference": paired_summary(
                fixed_reports[variants[0].name][representation], report
            ),
            "selection_counts": dict(Counter(s["pipeline"] for s in selections)),
        }
    write_json(output / "results.json", results)
    write_json(
        output / "verification.json",
        {
            "saved_head_fold_replays": replay_count,
            "regenerated_preprocessing_windows": waveform_count,
            "startup_overlap": [],
            "protected_overlap": [],
            "source_hashes": "unchanged",
            "unchanged_frontend_check": frontend_check,
            "script_sha256": sha256(Path(__file__)),
            "selection_rule": "inner validation only; outer maximum is exploratory",
        },
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)
    labels = [v.name for v in variants]
    for ax, representation in zip(axes, ("semantic35", "classical53"), strict=True):
        values = (
            np.array(
                [
                    [
                        fixed_reports[v.name][representation]["selected"][
                            "balanced_accuracy"
                        ],
                        fixed_reports[v.name][representation]["selected"]["recall"][
                            "tracked"
                        ],
                        fixed_reports[v.name][representation]["selected"]["recall"][
                            "wheeled"
                        ],
                    ]
                    for v in variants
                ]
            )
            * 100
        )
        ax.imshow(values, vmin=0, vmax=100, cmap="viridis", aspect="auto")
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_xticks(
            range(3), ["Balanced acc.", "Tracked recall", "Wheeled recall"], fontsize=8
        )
        ax.set_title(representation)
        for i in range(len(variants)):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if values[i, j] < 55 else "black",
                )
    fig.suptitle(
        "Fixed-pipeline outer results (%) — exploratory, NOT the nested grid-search result"
    )
    fig.savefig(output / "grid_diagnostics.png", dpi=160)
    plt.close(fig)
    lines = [
        "# Preprocessing grid v1",
        "",
        "Native real → real unseen-session development; 785 non-startup windows, 7 tracked / 5 wheeled sessions.",
        "",
        "Headline: preprocessing and C chosen exclusively inside inner session folds.",
        "",
        "| Representation | Rule | Balanced accuracy | Tracked recall | Wheeled recall | Worst recall |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        for rule in ("control", "nested_grid"):
            r = result[rule]["selected"]
            lines.append(
                f"| {name} | {rule} | {r['balanced_accuracy']:.2%} | {r['recall']['tracked']:.2%} | {r['recall']['wheeled']:.2%} | {r['worst_session_context_recall']:.2%} |"
            )
    lines += [
        "",
        "The maximum in fixed_pipeline_results.json is NOT an unbiased estimate for selecting that pipeline.",
        "All per-fold/per-session metrics and selection records are adjacent. Local feature caches/checkpoints and hashes are under runs/preprocessing_grid_v1.",
        "The unchanged control uses the same uncapped non-startup dataset; historical capped/startup-inclusive scores are not a matched comparison.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    write_json(output / "artifact_sha256.json", file_hashes(output))
    print(
        json.dumps(
            {
                name: {
                    rule: result[rule]["selected"]["balanced_accuracy"]
                    for rule in ("control", "nested_grid")
                }
                for name, result in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/preprocessing_grid_v1"))
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/v0.1/preprocessing_grid_v1")
    )
    args = parser.parse_args()
    build(args.run, args.output)

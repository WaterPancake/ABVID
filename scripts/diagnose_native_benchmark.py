"""Post-hoc development diagnostics; never fit, select, or change a model."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import torch

from vehicle_audio.benchmark_followup import (
    FrozenBenchmark,
    summarize_predictions,
    write_json,
)
from vehicle_audio.semantic_session_evaluation import SEMANTIC_AUDIOSET_FEATURES


def diagnose(output):
    if output.exists():
        raise FileExistsError(f"use a new output directory: {output}")
    b = FrozenBenchmark()
    predictions = b.predict(b.semantic, b.classical)
    b.verify_native(predictions)
    summary = summarize_predictions(b, predictions, include_fixed_fusion=True)
    per_window = defaultdict(lambda: defaultdict(list))
    attribution = defaultdict(list)
    thresholds = Counter()
    pair_rows = []
    for fold in predictions:
        thresholds[str(fold["threshold"])] += 1
        y = b.labels[fold["indices"]]
        s, c, f = fold["semantic"], fold["classical"], fold["fusion"]
        pair_rows.append(
            {
                "fold_id": fold["fold_id"],
                "branch_disagreement_fraction": float(
                    np.mean((s >= 0.5) != (c >= 0.5))
                ),
                "both_branches_wrong_fraction": float(
                    np.mean(((s >= 0.5) != y) & ((c >= 0.5) != y))
                ),
                "threshold_flipped_fraction": float(
                    np.mean((f >= 0.5) != (f >= fold["threshold"]))
                ),
                "inner_ba": next(
                    x["inner_selection_mean_balanced_accuracy"]
                    for x in b.metrics["folds"]
                    if x["fold_id"] == fold["fold_id"]
                ),
            }
        )
        for pos, i in enumerate(fold["indices"]):
            for model in ("semantic", "classical", "fusion"):
                threshold = fold["threshold"] if model == "fusion" else 0.5
                per_window[int(i)][model].append(
                    float((fold[model][pos] >= threshold) == y[pos])
                )
        states = b.models[fold["fold_id"]]["semantic_regularization_ensemble"]
        for session in {b.records[i]["recording_session"] for i in fold["indices"]}:
            indices = [
                i
                for i in fold["indices"]
                if b.records[i]["recording_session"] == session
            ]
            contributions, ood = [], []
            for state in states.values():
                z = (
                    b.semantic[indices].astype(float) - state["feature_mean"].numpy()
                ) / state["feature_scale"].numpy()
                contributions.append((z * state["coefficient"].numpy()).mean(axis=0))
                ood.append(float(np.mean(abs(z) > 3)))
            attribution[session].append(
                {
                    "contributions": np.mean(contributions, axis=0),
                    "z_above_3_fraction": np.mean(ood),
                }
            )
    windows = []
    for i, r in enumerate(b.records):
        x = b.classical[i]
        windows.append(
            {
                "sample_id": r["sample_id"],
                "recording_session": r["recording_session"],
                "vehicle_class": r["vehicle_class"],
                "condition": r["operating_condition"],
                "source_start": r["window_start_seconds"],
                "source_end": r["window_end_seconds"],
                "rms_dbfs": float(20 * np.log10(max(float(x[50]), 1e-12))),
                "spectral_centroid_hz": float(x[40]),
                "spectral_bandwidth_hz": float(x[42]),
                "rolloff_85_hz": float(x[44]),
                "crest_factor": float(x[52]),
                **{
                    f"{m}_mean_correct": float(np.mean(v))
                    for m, v in per_window[i].items()
                },
            }
        )
    sessions = {}
    for session in sorted({r["recording_session"] for r in b.records}):
        records = [r for r in b.records if r["recording_session"] == session]
        rows = [w for w in windows if w["recording_session"] == session]
        source = b.catalog[records[0]["source_id"]]
        contributions = np.mean(
            [v["contributions"] for v in attribution[session]], axis=0
        )
        order = np.argsort(contributions)
        conditions = {}
        for condition in sorted({w["condition"] for w in rows}):
            subset = [w for w in rows if w["condition"] == condition]
            conditions[condition] = {
                "windows": len(subset),
                **{
                    m: float(np.mean([w[f"{m}_mean_correct"] for w in subset]))
                    for m in ("classical", "semantic", "fusion")
                },
            }
        correlations = {}
        for feature in ("rms_dbfs", "spectral_centroid_hz", "spectral_bandwidth_hz"):
            a = [w[feature] for w in rows]
            y = [w["fusion_mean_correct"] for w in rows]
            correlations[feature] = (
                float(spearmanr(a, y).statistic)
                if np.std(a) > 0 and np.std(y) > 0
                else None
            )
        sessions[session] = {
            "source_id": source["id"],
            "vehicle_model": source.get("vehicle_model"),
            "vehicle_class": records[0]["vehicle_class"],
            "window_count": len(rows),
            "provider": source["provider"],
            "source_page": source["source_page"],
            "recording_device": source.get("recording_device", "unknown"),
            "distance": source.get("distance", "unknown"),
            "simultaneous_vehicle_count": "multiple, exact count unknown"
            if "stryker-convoy" in source["id"]
            else "not systematically annotated",
            "source_notes": source.get("notes"),
            "conditions": conditions,
            "acoustics": {
                feature: {
                    "median": float(np.median([r[feature] for r in rows])),
                    "minimum": min(r[feature] for r in rows),
                    "maximum": max(r[feature] for r in rows),
                }
                for feature in (
                    "rms_dbfs",
                    "spectral_centroid_hz",
                    "spectral_bandwidth_hz",
                    "rolloff_85_hz",
                    "crest_factor",
                )
            },
            "within_session_spearman_vs_fusion_correctness": correlations,
            "semantic_standardized_features_abs_gt_3_fraction": float(
                np.mean([v["z_above_3_fraction"] for v in attribution[session]])
            ),
            "semantic_logit_contributions": {
                direction: [
                    {
                        "feature": SEMANTIC_AUDIOSET_FEATURES[int(j)][1],
                        "mean_contribution": float(contributions[j]),
                    }
                    for j in indices
                ]
                for direction, indices in (
                    ("toward_tracked", order[:5]),
                    ("toward_wheeled", order[-5:][::-1]),
                )
            },
        }
    result = {
        **b.metadata(
            {"analysis": "post-hoc diagnostic only", "fixed_fusion_threshold": 0.5}
        ),
        "test_domain": "real -> real, frozen nested unseen-session development predictions",
        "summary": summary,
        "threshold_counts": dict(thresholds),
        "sessions": sessions,
        "pair_diagnostics": pair_rows,
        "limitations": [
            "No causal attribution from these observational associations.",
            "Overlapping windows and pairing contexts are dependent; no window-level significance tests.",
            "Device, distance and simultaneous vehicle counts are not systematically annotated.",
            "Mean logit contributions describe individual linear probes, not calibrated probabilities or causal sound components.",
            "Fixed-0.5 fusion is a post-hoc diagnostic, not a replacement selected model.",
        ],
    }
    write_json(output / "results.json", result)
    write_json(output / "window_diagnostics.json", windows)
    lines = [
        "# Native-real failure diagnostics",
        "",
        "Post-hoc development analysis; no retraining or selection. Protected recordings were not loaded.",
        "",
        "All three model branches reproduce the frozen per-class recalls in all 35 outer folds.",
        "",
        "| Model | Balanced accuracy | Tracked recall | Wheeled recall |",
        "|---|---:|---:|---:|",
    ]
    for model, values in summary.items():
        lines.append(
            f"| {model} | {values['balanced_accuracy']:.2%} | {values['tracked_recall']:.2%} | {values['wheeled_recall']:.2%} |"
        )
    lines.extend(
        [
            "",
            f"Selected wheeled thresholds (35 folds): `{dict(thresholds)}`.",
            "",
            "## Session audit",
            "",
            "Recall is the mean across that session's pairing contexts. RMS is digital level, not source sound pressure or SNR.",
            "",
            "| Session | Class | Windows | RMS median dBFS | Bandwidth median Hz | Classical recall | Semantic recall | Fusion recall |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for session, values in sessions.items():
        a = values["acoustics"]
        recalls = [
            summary[m]["per_session"][session]["mean_recall"]
            for m in ("classical", "semantic", "fusion")
        ]
        lines.append(
            f"| {values['vehicle_model']} | {values['vehicle_class']} | {values['window_count']} | {a['rms_dbfs']['median']:.1f} | {a['spectral_bandwidth_hz']['median']:.0f} | "
            + " | ".join(f"{r:.1%}" for r in recalls)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Operating states",
            "",
            "| Session / state | Windows | Classical recall | Semantic recall | Fusion recall |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for values in sessions.values():
        for condition, row in values["conditions"].items():
            lines.append(
                f"| {values['vehicle_model']} / {condition} | {row['windows']} | {row['classical']:.1%} | {row['semantic']:.1%} | {row['fusion']:.1%} |"
            )
    lines.extend(["", "## Limits", "", *[f"- {x}" for x in result["limitations"]]])
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(
        {
            m: {
                k: v[k]
                for k in ("balanced_accuracy", "tracked_recall", "wheeled_recall")
            }
            for m, v in summary.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/v0.1/diagnostics_7t5w_v1")
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    diagnose(args.output)

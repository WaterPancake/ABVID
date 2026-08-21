"""Validate and aggregate repeated Milestone 6 training-seed runs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vehicle_audio.invariance_evaluation import METHODS


AGGREGATION_VERSION = 2
SCALAR_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
OPTIONAL_SCALAR_METRICS = ("session_macro_accuracy", "session_balanced_accuracy")
PLOT_CONDITIONS = (
    "seen_corruption",
    "unseen_noise",
    "unseen_microphone",
    "unseen_environment",
    "all_corruptions",
    "native_real",
    "native_real_all_sessions",
)


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty metric series")
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "sample_standard_deviation": stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def _normalized_training_config(metrics: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(metrics["training_config"])
    config.pop("seed", None)
    return config


def _load_runs(paths: Sequence[str | Path]) -> list[tuple[Path, dict[str, Any]]]:
    if len(paths) < 2:
        raise ValueError("multi-seed aggregation requires at least two metrics files")
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_status") != "complete":
            raise ValueError(f"run is not complete: {path}")
        loaded.append((path, payload))
    return loaded


def _validate_runs(
    runs: Sequence[tuple[Path, Mapping[str, Any]]],
) -> tuple[list[str], list[str]]:
    reference_path, reference = runs[0]
    dataset_version = reference.get("dataset_version")
    commit = reference.get("git_commit")
    config = _normalized_training_config(reference)
    methods = sorted(reference["methods"])
    conditions = sorted(
        set.intersection(
            *(set(reference["methods"][method]["evaluations"]) for method in methods)
        )
    )
    seeds: set[int] = set()

    for path, metrics in runs:
        seed = int(metrics["training_config"]["seed"])
        if seed in seeds:
            raise ValueError(f"duplicate training seed {seed}: {path}")
        seeds.add(seed)
        if metrics.get("dataset_version") != dataset_version:
            raise ValueError(f"dataset version mismatch: {path}")
        if metrics.get("git_commit") != commit:
            raise ValueError(
                f"git commit mismatch between {reference_path} and {path}"
            )
        if _normalized_training_config(metrics) != config:
            raise ValueError(f"controlled training configuration mismatch: {path}")
        if sorted(metrics["methods"]) != methods:
            raise ValueError(f"method set mismatch: {path}")
        for method in methods:
            observed_conditions = sorted(metrics["methods"][method]["evaluations"])
            if observed_conditions != conditions:
                raise ValueError(f"evaluation condition mismatch for {method}: {path}")

    for method in methods:
        for condition in conditions:
            supports = {
                int(metrics["methods"][method]["evaluations"][condition]["support"])
                for _, metrics in runs
            }
            if len(supports) != 1:
                raise ValueError(
                    f"support mismatch for {method}/{condition}: {sorted(supports)}"
                )
    return methods, conditions


def aggregate_invariance_runs(
    metrics_paths: Sequence[str | Path], output_dir: str | Path
) -> dict[str, Any]:
    """Aggregate controlled seed repeats after validating their experiment identity."""

    runs = _load_runs(metrics_paths)
    methods, conditions = _validate_runs(runs)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference = runs[0][1]

    method_results: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for method in methods:
        evaluation_results: dict[str, Any] = {}
        for condition in conditions:
            reference_evaluation = reference["methods"][method]["evaluations"][condition]
            scalar_metrics = SCALAR_METRICS + tuple(
                metric
                for metric in OPTIONAL_SCALAR_METRICS
                if metric in reference_evaluation
            )
            metric_results = {
                metric: _summary(
                    [
                        float(run["methods"][method]["evaluations"][condition][metric])
                        for _, run in runs
                    ]
                )
                for metric in scalar_metrics
            }
            class_results = {
                vehicle_class: _summary(
                    [
                        float(
                            run["methods"][method]["evaluations"][condition]
                            ["per_class_recall"][vehicle_class]
                        )
                        for _, run in runs
                    ]
                )
                for vehicle_class in sorted(reference_evaluation["per_class_recall"])
            }
            class_session_results = {
                vehicle_class: _summary(
                    [
                        float(
                            run["methods"][method]["evaluations"][condition]
                            ["class_session_mean_recall"][vehicle_class]
                        )
                        for _, run in runs
                    ]
                )
                for vehicle_class in sorted(
                    reference_evaluation.get("class_session_mean_recall", {})
                )
            }
            evaluation_results[condition] = {
                "support_per_run": int(reference_evaluation["support"]),
                "metrics": metric_results,
                "per_class_recall": class_results,
            }
            if class_session_results:
                evaluation_results[condition][
                    "class_session_mean_recall"
                ] = class_session_results
            for metric, summary in metric_results.items():
                csv_rows.append(
                    {
                        "method": method,
                        "condition": condition,
                        "metric": metric,
                        "vehicle_class": "",
                        **summary,
                    }
                )
            for vehicle_class, summary in class_results.items():
                csv_rows.append(
                    {
                        "method": method,
                        "condition": condition,
                        "metric": "recall",
                        "vehicle_class": vehicle_class,
                        **summary,
                    }
                )
            for vehicle_class, summary in class_session_results.items():
                csv_rows.append(
                    {
                        "method": method,
                        "condition": condition,
                        "metric": "session_mean_recall",
                        "vehicle_class": vehicle_class,
                        **summary,
                    }
                )
        method_results[method] = {"evaluations": evaluation_results}

    source_runs = [
        {
            "metrics_path": str(path),
            "training_seed": int(metrics["training_config"]["seed"]),
            "split_seed": int(metrics["training_config"]["split_seed"]),
            "git_commit": metrics.get("git_commit"),
        }
        for path, metrics in sorted(
            runs, key=lambda item: int(item[1]["training_config"]["seed"])
        )
    ]
    result = {
        "aggregation_version": AGGREGATION_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_status": "complete",
        "dataset_version": reference["dataset_version"],
        "git_commit": reference.get("git_commit"),
        "split_seed": int(reference["training_config"]["split_seed"]),
        "training_config_except_seed": _normalized_training_config(reference),
        "source_runs": source_runs,
        "run_count": len(runs),
        "definitions": {
            "mean": "arithmetic mean across training seeds",
            "median": "median across training seeds",
            "sample_standard_deviation": "sample standard deviation across training seeds (n-1 denominator)",
        },
        "methods": method_results,
        "artifacts": {
            "summary_csv": "aggregate_summary.csv",
            "condition_plot": "aggregate_condition_comparison.png",
        },
    }
    (output / "aggregate_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output / "aggregate_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    plotted_conditions = [
        condition for condition in PLOT_CONDITIONS if condition in conditions
    ]
    figure, axis = plt.subplots(figsize=(11.5, 5.4))
    x_values = list(range(len(plotted_conditions)))
    ordered_methods = [method for method in METHODS if method in methods]
    ordered_methods.extend(method for method in methods if method not in ordered_methods)
    width = 0.8 / len(ordered_methods)
    for method_index, method in enumerate(ordered_methods):
        summaries = [
            method_results[method]["evaluations"][condition]["metrics"]
            ["balanced_accuracy"]
            for condition in plotted_conditions
        ]
        axis.bar(
            [
                x + (method_index - (len(ordered_methods) - 1) / 2.0) * width
                for x in x_values
            ],
            [summary["mean"] for summary in summaries],
            yerr=[summary["sample_standard_deviation"] for summary in summaries],
            width=width,
            capsize=2,
            label=method.replace("_", " "),
        )
    axis.set_xticks(
        x_values, [condition.replace("_", "\n") for condition in plotted_conditions]
    )
    axis.set_ylabel("Balanced accuracy (mean +/- sample SD)")
    axis.set_ylim(0.0, 1.02)
    axis.set_title(f"Milestone 6 controlled comparison across {len(runs)} seeds")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "aggregate_condition_comparison.png", dpi=160)
    plt.close(figure)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = aggregate_invariance_runs(args.metrics, args.output)
    print(
        json.dumps(
            {
                "output_dir": str(args.output),
                "protocol_status": result["protocol_status"],
                "run_count": result["run_count"],
                "training_seeds": [
                    run["training_seed"] for run in result["source_runs"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

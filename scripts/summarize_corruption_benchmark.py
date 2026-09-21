"""Equal-background summaries and paired session-bootstrap deltas; no selection."""

import argparse
import json
from pathlib import Path

import numpy as np

from vehicle_audio.benchmark_followup import sha256, write_json


def summarize(output):
    results = json.loads((output / "results.json").read_text())
    native = results["native"]["metrics"]
    summaries = []
    for role in (
        "environmental_noise",
        "competing_vehicle_interference",
        "simulated_microphone_response",
    ):
        values = [v for v in results.values() if v["condition"]["role"] == role]
        keys = sorted(
            {v["condition"].get("snr_db", v["condition"]["id"]) for v in values},
            reverse=True,
        )
        for key in keys:
            selected = [
                v
                for v in values
                if v["condition"].get("snr_db", v["condition"]["id"]) == key
            ]
            for model in ("classical", "semantic", "fusion"):
                sessions = native[model]["per_session"]
                rng = np.random.default_rng(42)
                bootstrap, delta_bootstrap, class_means = [], [], {}
                for name in ("tracked", "wheeled"):
                    ids = [s for s in sessions if sessions[s]["vehicle_class"] == name]
                    means = np.array(
                        [
                            np.mean(
                                [
                                    v["metrics"][model]["per_session"][s]["mean_recall"]
                                    for v in selected
                                ]
                            )
                            for s in ids
                        ]
                    )
                    base = np.array([sessions[s]["mean_recall"] for s in ids])
                    indices = rng.integers(0, len(ids), (10000, len(ids)))
                    bootstrap.append(means[indices].mean(axis=1))
                    delta_bootstrap.append((means - base)[indices].mean(axis=1))
                    class_means[name] = float(means.mean())
                ba = sum(class_means.values()) / 2
                summaries.append(
                    {
                        "role": role,
                        "condition": key,
                        "model": model,
                        "background_count": len(selected)
                        if role != "simulated_microphone_response"
                        else 0,
                        "balanced_accuracy": ba,
                        "per_class_recall": class_means,
                        "delta_ba_vs_native": ba - native[model]["balanced_accuracy"],
                        "descriptive_session_bootstrap_ba_95": np.quantile(
                            (bootstrap[0] + bootstrap[1]) / 2, [0.025, 0.975]
                        ).tolist(),
                        "paired_descriptive_session_bootstrap_delta_ba_95": np.quantile(
                            (delta_bootstrap[0] + delta_bootstrap[1]) / 2,
                            [0.025, 0.975],
                        ).tolist(),
                    }
                )
    write_json(
        output / "aggregate_summary.json",
        {
            "results_sha256": sha256(output / "results.json"),
            "summarizer_sha256": sha256(__file__),
            "seed": 42,
            "aggregation": "equal original background within condition, equal session within class, equal classes",
            "uncertainty_scope": "12 original vehicle sessions only; no refit or new-background uncertainty",
            "rows": summaries,
        },
    )
    for row in summaries:
        if row["model"] == "fusion":
            print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/v0.1/corruption_7t5w_v1")
    )
    summarize(parser.parse_args().output)

"""Read-only replay of the frozen 7/5 development folds for follow-up studies."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.special import expit
import torch
import yaml

from vehicle_audio.semantic_session_evaluation import SEMANTIC_AUDIOSET_FEATURES

SNAPSHOT = Path("benchmarks/v0.1/native_real_7t5w")
RUN = Path("runs/benchmark_v0_1_native_real_7t5w")
MANIFEST = Path("data/benchmark_v0_1_native_real_7t5w/real_manifest.jsonl")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def ensemble_probability(features, states):
    """Restore sklearn binary logistic probabilities from saved float32 state."""
    features = np.asarray(features, dtype=np.float64)
    probabilities = []
    for state in states.values():
        z = (features - state["feature_mean"].numpy()) / state["feature_scale"].numpy()
        logit = z @ state["coefficient"].numpy().T + state["intercept"].numpy()
        probabilities.append(expit(logit[:, 0]))
    return np.mean(probabilities, axis=0)


def validate_split(records, split):
    by_id = {r["sample_id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate sample IDs")
    train, test = set(split["train_sample_ids"]), set(split["test_sample_ids"])
    if train & test or train | test != set(by_id):
        raise ValueError("split membership is overlapping or incomplete")
    train_sessions = {by_id[i]["recording_session"] for i in train}
    test_sessions = {by_id[i]["recording_session"] for i in test}
    if train_sessions & test_sessions:
        raise ValueError("recording-session leakage")
    expected = {s for values in split["test_sessions"].values() for s in values}
    if test_sessions != expected:
        raise ValueError("test session metadata mismatch")


class FrozenBenchmark:
    def __init__(self):
        self.lock = json.loads((SNAPSHOT / "dataset_lock.json").read_text())
        self.version = sha256(MANIFEST)
        if self.version != self.lock["manifest_sha256"]:
            raise ValueError("native manifest changed")
        self.records = [json.loads(line) for line in MANIFEST.read_text().splitlines()]
        policy = yaml.safe_load(Path("configs/benchmark_v0_1.yaml").read_text())
        forbidden = set(
            policy["consumed_locked_source_ids"]
            + policy["future_confirmation_source_ids"]
        )
        if forbidden & {r["source_id"] for r in self.records}:
            raise ValueError("protected source in development manifest")
        if sha256(SNAPSHOT / "audio_sources.yaml") != self.lock["catalog_sha256"]:
            raise ValueError("frozen catalog changed")
        self.catalog = {
            s["id"]: s
            for s in yaml.safe_load((SNAPSHOT / "audio_sources.yaml").read_text())[
                "sources"
            ]
        }
        for r in self.records:
            source = self.catalog[r["source_id"]]
            if (
                source.get("status") != "approved"
                or source.get("admitted_to_corpus") is not True
            ):
                raise ValueError("unapproved source in manifest")
            matches = [
                s
                for s in source["condition_segments"]
                if s["start_seconds"] <= r["window_start_seconds"]
                and s["end_seconds"] >= r["window_end_seconds"]
                and s["operating_condition"] == r["operating_condition"]
            ]
            if len(matches) != 1:
                raise ValueError("window outside reviewed condition interval")
        self.input_hashes = {}
        for r in {r["source_id"]: r for r in self.records}.values():
            path = Path("data/targets") / r["target_source"]
            if sha256(path) != r["normalized_source_sha256"]:
                raise ValueError(f"normalized source changed: {path}")
        verification = json.loads((SNAPSHOT / "verification.json").read_text())[
            "artifact_sha256"
        ]
        for path, expected in verification.items():
            if path.startswith(str(RUN)):
                actual = sha256(path)
                if actual != expected:
                    raise ValueError(f"frozen artifact changed: {path}")
                self.input_hashes[path] = actual
        self.metrics = json.loads((RUN / "fusion/metrics.json").read_text())
        self.splits = json.loads((RUN / "fusion/splits.json").read_text())["folds"]
        self.models = torch.load(
            RUN / "fusion/fusion_probe_models.pt", weights_only=True, map_location="cpu"
        )["models"]
        for split in self.splits:
            validate_split(self.records, split)
        self.semantic_indices = [i for i, _ in SEMANTIC_AUDIOSET_FEATURES]
        panns = torch.load(
            RUN / "panns_features.pt", weights_only=True, map_location="cpu"
        )
        classical = torch.load(
            RUN / "fusion/real_classical_features.pt",
            weights_only=True,
            map_location="cpu",
        )
        if classical["manifest_sha256"] != self.version:
            raise ValueError("classical feature cache belongs to a different manifest")
        self.semantic = panns["real_clipwise_outputs"][:, self.semantic_indices].numpy()
        self.classical = classical["features"].numpy()
        self.labels = np.asarray(
            [int(r["vehicle_class"] == "wheeled") for r in self.records]
        )
        if not np.array_equal(
            self.labels, classical["labels"].numpy()
        ) or not np.array_equal(self.labels, panns["real_labels"].numpy()):
            raise ValueError("feature labels do not match manifest")
        self.input_hashes[str(RUN / "fusion/real_classical_features.pt")] = sha256(
            RUN / "fusion/real_classical_features.pt"
        )
        self.input_hashes[str(MANIFEST)] = self.version

    def predict(self, semantic, classical):
        """Only apply each fold's model to that fold's held-out original sessions."""
        output = []
        for split in self.splits:
            selected = set(split["test_sample_ids"])
            indices = np.asarray(
                [i for i, r in enumerate(self.records) if r["sample_id"] in selected]
            )
            state = self.models[split["fold_id"]]
            s = ensemble_probability(
                semantic[indices], state["semantic_regularization_ensemble"]
            )
            c = ensemble_probability(
                classical[indices], state["classical_regularization_ensemble"]
            )
            # Use the JSON-selected double threshold, as in original evaluation.
            threshold = next(
                f["selected_wheeled_threshold"]
                for f in self.metrics["folds"]
                if f["fold_id"] == split["fold_id"]
            )
            output.append(
                {
                    "fold_id": split["fold_id"],
                    "indices": indices,
                    "semantic": s,
                    "classical": c,
                    "fusion": (s + c) / 2,
                    "threshold": threshold,
                }
            )
        return output

    def verify_native(self, predictions):
        """Fail before new evaluation unless saved probes reproduce native scores."""
        for model in ("fusion", "semantic", "classical"):
            original = json.loads((RUN / model / "metrics.json").read_text())
            for fold in predictions:
                sessions = {
                    self.records[i]["recording_session"] for i in fold["indices"]
                }
                old = next(
                    f
                    for f in original["folds"]
                    if {f["tracked_test_session"], f["wheeled_test_session"]}
                    == sessions
                )
                metrics = old[
                    "metrics"
                    if model == "fusion"
                    else "regularization_ensemble_metrics"
                ]
                threshold = fold["threshold"] if model == "fusion" else 0.5
                actual = fold[model] >= threshold
                y = self.labels[fold["indices"]]
                for k, name in enumerate(("tracked", "wheeled")):
                    recall = float(np.mean(actual[y == k] == k))
                    if abs(recall - metrics["per_class_recall"][name]) > 1e-9:
                        raise ValueError(
                            f"native reproduction failed: {model}/{fold['fold_id']}/{name}"
                        )

    def metadata(self, configuration):
        return {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "git_worktree_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], text=True)
            ),
            "dataset_version": self.version,
            "random_seed": 42,
            "configuration": configuration,
            "input_sha256": self.input_hashes,
            "code_sha256": {
                str(p): sha256(p)
                for p in sorted(Path("src/vehicle_audio").glob("*.py"))
            },
            "train_test_split": str(RUN / "fusion/splits.json"),
            "model_checkpoint": str(RUN / "fusion/fusion_probe_models.pt"),
            "protected_source_overlap": [],
            "native_reproduction": "all 35 folds, all three models, exact window recalls",
        }


def summarize_predictions(benchmark, predictions, include_fixed_fusion=False):
    summaries = {}
    models = ["classical", "semantic", "fusion"] + (
        ["fusion_fixed_0.5"] if include_fixed_fusion else []
    )
    for model in models:
        sessions = defaultdict(list)
        folds = []
        for fold in predictions:
            p = fold["fusion" if model == "fusion_fixed_0.5" else model]
            threshold = fold["threshold"] if model == "fusion" else 0.5
            y = benchmark.labels[fold["indices"]]
            predicted = (p >= threshold).astype(int)
            recalls = [float(np.mean(predicted[y == k] == k)) for k in (0, 1)]
            confusion = np.bincount(2 * y + predicted, minlength=4).reshape(2, 2)
            precision = np.diag(confusion) / np.maximum(confusion.sum(axis=0), 1)
            recall = np.asarray(recalls)
            f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
            row = {
                "fold_id": fold["fold_id"],
                "balanced_accuracy": sum(recalls) / 2,
                "tracked_recall": recalls[0],
                "wheeled_recall": recalls[1],
                "threshold": threshold,
                "confusion_matrix": confusion.tolist(),
                "class_order": ["tracked", "wheeled"],
                "accuracy": float(np.mean(predicted == y)),
                "macro_precision": float(precision.mean()),
                "macro_recall": float(recall.mean()),
                "macro_f1": float(f1.mean()),
            }
            folds.append(row)
            for k in (0, 1):
                session = benchmark.records[fold["indices"][np.flatnonzero(y == k)[0]]][
                    "recording_session"
                ]
                sessions[session].append(
                    {
                        "recall": recalls[k],
                        "mean_p_wheeled": float(p[y == k].mean()),
                        "vehicle_class": ("tracked", "wheeled")[k],
                    }
                )
        per_session = {
            s: {
                "mean_recall": float(np.mean([x["recall"] for x in values])),
                "minimum_recall": min(x["recall"] for x in values),
                "mean_p_wheeled": float(np.mean([x["mean_p_wheeled"] for x in values])),
                "vehicle_class": values[0]["vehicle_class"],
            }
            for s, values in sessions.items()
        }
        rng = np.random.default_rng(42)
        bootstrap = []
        for name in ("tracked", "wheeled"):
            values = [
                v["mean_recall"]
                for v in per_session.values()
                if v["vehicle_class"] == name
            ]
            bootstrap.append(
                rng.choice(values, (10000, len(values)), replace=True).mean(axis=1)
            )
        summaries[model] = {
            **{
                key: float(np.mean([f[key] for f in folds]))
                for key in (
                    "balanced_accuracy",
                    "tracked_recall",
                    "wheeled_recall",
                    "accuracy",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                )
            },
            "worst_session_context_recall": min(
                v["minimum_recall"] for v in per_session.values()
            ),
            "descriptive_session_bootstrap_ba_95": np.quantile(
                (bootstrap[0] + bootstrap[1]) / 2, [0.025, 0.975]
            ).tolist(),
            "per_session": per_session,
            "folds": folds,
        }
    return summaries

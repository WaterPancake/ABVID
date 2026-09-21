#!/usr/bin/env python3
"""Verify and package the completed 7/5 native benchmark without rerunning models.

Run from the project root. Existing snapshot files may only be reused identically.
Audio and checkpoints remain local; metadata, metrics and splits are packaged.
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"Refusing to replace changed snapshot: {path}")
    if not path.exists():
        path.write_bytes(content)


def main():
    destination = Path("benchmarks/v0.1/native_real_7t5w")
    run = Path("runs/benchmark_v0_1_native_real_7t5w")
    lock = json.loads((destination / "dataset_lock.json").read_text())
    summary = json.loads((destination / "results.json").read_text())
    manifest = Path(lock["manifest_path"])
    if sha(manifest) != lock["manifest_sha256"]:
        raise ValueError("Manifest differs from dataset lock")
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    by_id = {r["sample_id"]: r for r in records}
    assert len(by_id) == len(records) == 593
    catalog_path = Path("configs/audio_sources.yaml")
    assert sha(catalog_path) == lock["catalog_sha256"]
    protocol_path = Path("configs/benchmark_v0_1_native_real.yaml")
    assert sha(protocol_path) == summary["protocol_sha256"]
    catalog = yaml.safe_load(catalog_path.read_text())
    protocol = yaml.safe_load(protocol_path.read_text())
    protected_ids = set(sum(protocol["protected_sources"].values(), []))
    protected = [s for s in catalog["sources"] if s["id"] in protected_ids]
    for r in records:
        assert r["source_id"] not in protected_ids
        for source in protected:
            assert r["recording_session"] != source["recording_session"]
            assert r["source_page"] != source["source_page"]
            sidecar = Path("data") / Path(source["output_path"]).with_suffix(".json")
            if sidecar.exists():
                metadata = json.loads(sidecar.read_text())
                assert r["raw_sha256"] != metadata["raw_sha256"]
    snapshot(destination / "real_manifest.jsonl", manifest.read_bytes())
    snapshot(destination / "audio_sources.yaml", catalog_path.read_bytes())
    snapshot(destination / "protocol.yaml", protocol_path.read_bytes())

    def sessions(ids):
        return {by_id[i]["recording_session"] for i in ids}

    reference_pairs = None
    per_session = {}
    artifacts = {}
    for model, key in (
        ("classical", "classical_regularization_ensemble"),
        ("semantic", "semantic_panns_regularization_ensemble"),
        ("fusion", "semantic_classical_equal_fusion_primary"),
    ):
        metrics_path = run / model / "metrics.json"
        assert sha(metrics_path) == summary["models"][key]["metrics_sha256"]
        metrics = json.loads(metrics_path.read_text())
        splits = json.loads((run / model / "splits.json").read_text())
        pairs = set()
        for fold in splits["folds"]:
            train, test = set(fold["train_sample_ids"]), set(fold["test_sample_ids"])
            assert not train & test and train | test == set(by_id)
            assert not sessions(train) & sessions(test)
            pairs.add(tuple(sorted(sessions(test))))
            for inner in fold.get("inner_folds", []):
                fit, val = set(inner["train_sample_ids"]), set(inner["validation_sample_ids"])
                assert not fit & val and fit | val == train
                assert not sessions(fit) & sessions(val)
                assert not sessions(val) & sessions(test)
        assert len(pairs) == 35
        if reference_pairs is None:
            reference_pairs = pairs
        assert pairs == reference_pairs
        values = defaultdict(list)
        field = "metrics" if model == "fusion" else "regularization_ensemble_metrics"
        for fold in metrics["folds"]:
            for cls in ("tracked", "wheeled"):
                values[fold[f"{cls}_test_session"]].append(fold[field]["per_class_recall"][cls])
        per_session[model] = {
            session: {"mean_recall": sum(v) / len(v), "minimum_recall": min(v),
                      "maximum_recall": max(v), "pairing_contexts": len(v)}
            for session, v in sorted(values.items())
        }
        for name in ("metrics.json", "splits.json", "experiment.json"):
            source = run / model / name
            snapshot(destination / model / name, source.read_bytes())
            artifacts[str(source)] = sha(source)
        checkpoint = run / model / metrics["artifacts"]["checkpoint"]
        artifacts[str(checkpoint)] = sha(checkpoint)
    for path in sorted(Path("src/vehicle_audio").glob("*.py")):
        artifacts[str(path)] = sha(path)
    artifacts[str(run / "panns_features.pt")] = sha(run / "panns_features.pt")
    payload = {"protected_overlap": [], "outer_pairs_verified": 35,
               "fusion_inner_pairs_verified": 35 * 24,
               "per_session_results": per_session, "artifact_sha256": artifacts,
               "code_hash_scope": "source files at packaging time; worktree was dirty"}
    snapshot(destination / "verification.json", (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
    print("Verified 593 windows, 35 common outer pairs, 840 fusion inner pairs, and all locked hashes.")


if __name__ == "__main__":
    main()

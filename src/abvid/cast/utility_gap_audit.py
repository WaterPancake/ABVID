"""Descriptive CAST-5A audit using hash-checked cached features and frozen heads.

No fitting, waveform access, feature extraction or held prediction occurs here.
See experiments/cast_utility/GAP_AUDIT.md for the post hoc analysis specification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess

import joblib
import numpy as np

from abvid.cast.grouped_storage import read_bundle, verify_bundle
from abvid.learning.utility import head_metrics
from abvid.paths import repository_root, resolve
from abvid.protocol import load
from abvid.provenance.hashing import canonical, sha256


RUN = "artifact:cast/utility/cast5a_run_20261011_v1"
RESULTS_SHA = "ce2df9b35e58e19b5a4326c8d9e636f3b5b9ac197cc07b4988ddaf7e9d4f2bf3"
REPS = ("BEATs768", "MFCC26")


def geometry(real, real_y, other, other_y):
    """Centroid geometry after the same real-training-fitted standardization."""
    centers = [real[real_y == c].mean(0) for c in (0, 1)]
    alt = [other[other_y == c].mean(0) for c in (0, 1)]
    axis, alt_axis = centers[1] - centers[0], alt[1] - alt[0]
    length, alt_length = np.linalg.norm(axis), np.linalg.norm(alt_axis)
    if min(length, alt_length) <= 0:
        raise ValueError("Degenerate class-centroid direction")
    out = {"class_separation_ratio": float(alt_length / length),
           "class_direction_cosine": float(axis @ alt_axis / (length * alt_length))}
    for c, label in enumerate(("car", "truck")):
        real_rms = np.sqrt(np.mean(np.sum((real[real_y == c] - centers[c]) ** 2, axis=1)))
        alt_rms = np.sqrt(np.mean(np.sum((other[other_y == c] - alt[c]) ** 2, axis=1)))
        if real_rms <= 0:
            raise ValueError("Degenerate within-class spread")
        out[f"{label}_centroid_shift_over_real_class_separation"] = float(np.linalg.norm(alt[c] - centers[c]) / length)
        out[f"{label}_within_class_spread_ratio"] = float(alt_rms / real_rms)
    return out


def aggregate(rows):
    """Descriptive equal-role means/ranges, without independence assumptions."""
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    return {k: {"mean": float(np.mean([r[k] for r in rows])),
                "min": float(min(r[k] for r in rows)), "max": float(max(r[k] for r in rows))}
            for k in keys}


def audit():
    configs = {}
    for name in ("cast_grouped_coverage", "h1_baselines"):
        path = repository_root() / "configs" / "experiments" / f"{name}.json"
        cfg = load(path, check_files=True)
        configs[cfg["id"]] = sha256(path)
    root = resolve(RUN)
    checked = {}

    def checked_json(relative, expected=None):
        path = root / relative
        value = sha256(path)
        if expected is not None and value != expected:
            raise ValueError(f"Hash mismatch: {relative}")
        checked[relative] = value
        return json.loads(path.read_text())

    manifest = verify_bundle(root / "design")
    _, payload = read_bundle(root / "design")
    checked["design/_bundle.json"] = sha256(root / "design/_bundle.json")
    parts = json.loads(payload["partitions.json"])
    design = json.loads(payload["design.json"])
    if sha256(resolve("repo:experiments/cast_utility/PROTOCOL.md")) != design["protocol_sha256"]:
        raise ValueError("CAST-5A protocol identity changed")
    sources = {r["file_id"]: r for r in (json.loads(x) for x in payload["inputs/sources.jsonl"].splitlines())}
    base = checked_json("learning/features_base/summary.json")
    rc_summary = checked_json("learning/features_rc/summary.json")
    features = {}
    for name, folder, expected in (
        ("real", "features_base", base["hashes"]["real.npz"]),
        ("ra", "features_base", base["hashes"]["ra.npz"]),
        ("ru", "features_base", base["hashes"]["ru.npz"]),
        ("rc", "features_rc", rc_summary["rc_sha256"]),
    ):
        relative = f"learning/{folder}/{name}.npz"
        path = root / relative
        if sha256(path) != expected:
            raise ValueError(f"Feature cache changed: {name}")
        checked[relative] = expected
        with np.load(path, allow_pickle=False) as data:
            features[name] = {k: data[k] for k in ("ids", "y", *REPS)}
        if len(set(features[name]["ids"])) != len(features[name]["ids"]):
            raise ValueError("Duplicate feature IDs")
    real_index = {str(i): k for k, i in enumerate(features["real"]["ids"])}
    if set(real_index) != set(sources):
        raise ValueError("Real cache differs from frozen sources")
    for file_id, i in real_index.items():
        if int(features["real"]["y"][i]) != sources[file_id]["class_id"]:
            raise ValueError("Real feature labels differ from source labels")
    hs = checked_json("learning/heads/summary.json")
    heads = checked_json("learning/heads/manifest.json", hs["manifest_sha256"])
    heads_by_key = {(h["representation"], h["arm"], h["partition"]): h for h in heads}
    results = checked_json("learning/scores/results.json", RESULTS_SHA)
    all_rows, all_geometry = [], []
    for part in parts:
        if set(part["train_ids"]) & set(part["held_ids"]):
            raise ValueError("Training/held IDs intersect")
        picks = {"real": np.array([real_index[i] for i in part["train_ids"]])}
        for name in ("ra", "ru", "rc"):
            prefix = ("s" + part["analysis"].removeprefix("utility_s") + "|") if name == "ru" else part["id"] + "|"
            picks[name] = np.flatnonzero(np.char.startswith(features[name]["ids"].astype(str), prefix))
        for name, indices in picks.items():
            if np.bincount(features[name]["y"][indices], minlength=2).tolist() != [190, 190]:
                raise ValueError(f"Wrong budget: {part['id']}/{name}")
        for rep in REPS:
            models = {}
            for arm in ("R", "C"):
                entry = heads_by_key[rep, arm, part["id"]]
                relative = f"learning/heads/models/{entry['head']}.joblib"
                path = root / relative
                if sha256(path) != entry["sha256"]:
                    raise ValueError(f"Frozen head changed: {relative}")
                checked[relative] = entry["sha256"]
                models[arm] = joblib.load(path)
            arrays = {name: features[name][rep][indices].astype(np.float64) for name, indices in picks.items()}
            labels = {name: features[name]["y"][indices] for name, indices in picks.items()}
            if any(not np.isfinite(a).all() for a in arrays.values()):
                raise ValueError("Nonfinite cached features")
            for arm, names in (("R", ("real", "ra", "rc", "ru")), ("C", ("real", "rc"))):
                for name in names:
                    scores = models[arm].predict_proba(arrays[name])[:, 1]
                    metrics = head_metrics(labels[name], scores)
                    all_rows.append({"partition": part["id"], "fold": part["fold"], "representation": rep,
                                     "head": arm, "input": name,
                                     **{k: metrics[k] for k in ("auroc", "balanced_accuracy", "positive_recall", "negative_recall")}})
            scaled = {name: models["R"][0].transform(x) for name, x in arrays.items()}
            for name in ("ra", "rc", "ru"):
                all_geometry.append({"partition": part["id"], "fold": part["fold"], "representation": rep, "input": name,
                                     **geometry(scaled["real"], labels["real"], scaled[name], labels[name])})
    summary, geometry_summary = {}, {}
    for rep in REPS:
        summary[rep] = {f"{arm}_on_{name}": aggregate([r for r in all_rows if r["representation"] == rep and r["head"] == arm and r["input"] == name])
                        for arm, names in (("R", ("real", "ra", "rc", "ru")), ("C", ("real", "rc"))) for name in names}
        geometry_summary[rep] = {name: aggregate([r for r in all_geometry if r["representation"] == rep and r["input"] == name]) for name in ("ra", "rc", "ru")}
    # Recheck all inputs, including model bytes, after use; historical artifacts are read-only.
    for relative, expected in checked.items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Input changed during audit: {relative}")
    read_bundle(root / "design")
    return {"schema_version": 1, "id": "cast_gap_saved_evidence_20261025_v1",
            "status": "post_hoc_descriptive_no_new_model_fit_or_audio_access", "input_run": RUN,
            "canonical_configs_sha256": configs,
            "design_identity_sha256": hashlib.sha256(canonical(manifest)).hexdigest(),
            "design_bundle_identity_sha256": manifest["identity_sha256"],
            "protocol_sha256": design["protocol_sha256"], "checked_input_sha256": checked,
            "roles": len(parts), "heads_loaded": 2 * len(REPS) * len(parts),
            "summary": summary, "geometry_summary": geometry_summary, "role_metrics": all_rows,
            "role_geometry": all_geometry, "historical_statistics": results["statistics"],
            "limits": ["Post hoc; no new significance claim or causal attribution.",
                       "Training-domain projections use existing fitted heads and their own selected parents.",
                       "Overlapping roles/streams are not independent; ranges are descriptive.",
                       "Unconditional samples cannot separate reconstruction loss from population-sampling loss.",
                       "Standardized centroid geometry is representation- and scaler-dependent."],
            "provenance": {"python": platform.python_version(), "numpy": np.__version__,
                           "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                           "working_tree_status": subprocess.check_output(["git", "status", "--short"], text=True).splitlines(),
                           "script_sha256": sha256(resolve("repo:src/abvid/cast/utility_gap_audit.py")),
                           "specification_sha256": sha256(resolve("repo:experiments/cast_utility/GAP_AUDIT.md"))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New root-qualified JSON file")
    args = parser.parse_args()
    output = resolve(args.output, must_exist=False)
    if output.exists():
        raise FileExistsError(output)
    result = audit()
    with output.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": args.output, "sha256": sha256(output), "roles": result["roles"], "heads": result["heads_loaded"]}))


if __name__ == "__main__":
    main()

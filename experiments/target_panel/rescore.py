"""Re-score archived exposed predictions with prevalence-robust metrics.

Thin layer over abvid.evaluation.ranking. Reads hash-checked prediction files
only; never loads models, features or audio, and fits nothing.
Run: uv run --extra cast python experiments/target_panel/rescore.py
"""
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from abvid.evaluation.ranking import METRICS, evaluate, group_draws, interval, selection_draws
from abvid.paths import resolve
from abvid.provenance.hashing import sha256

CONFIG = Path(__file__).with_name("config.json")


def summarize(samples):
    return {k: {"estimate": float(v[0]), "ci95": interval(v[1:])} for k, v in samples.items()}


def contrast(a, b):
    return {k: {"estimate": float(a[k][0] - b[k][0]), "ci95": interval(a[k][1:] - b[k][1:])}
            for k in ("auroc", "average_precision", "ap_lift", "recall_at_fpr_05", "balanced_accuracy")}


def per_group(scores, y, group, n_groups, dims):
    """Point estimate for each target group alone (no interval)."""
    ones = [np.ones((1, d), int) for d in dims]
    return [{k: float(v[0]) for k, v in evaluate(scores, y, group, n_groups,
                                                  np.eye(n_groups, dtype=int)[[g]], ones).items()}
            for g in range(n_groups)]


def h1(cfg, rng_draws):
    results = resolve(cfg["inputs"]["h1_results"])
    rows = {r["file_id"]: r for r in map(json.loads, resolve(cfg["inputs"]["h1_admitted"]).open())}
    seeds, folds = cfg["h1"]["seeds"], cfg["h1"]["folds"]
    out, ids = {}, None

    def load(rep, arm, seed, fold):
        name = f"{rep}__{arm}__seed{seed}__fold{fold}"
        record = json.loads((results / "models" / f"{name}.json").read_text())
        path = results / "predictions" / f"{name}.npz"
        if sha256(path) != record["predictions_sha256"]:
            raise ValueError(f"Prediction hash mismatch: {name}")
        return np.load(path, allow_pickle=False)

    for rep in cfg["h1"]["representations"]:
        arms = {}
        for arm in cfg["h1"]["real_arms"] + cfg["h1"]["synthetic_arms"]:
            real = arm in cfg["h1"]["real_arms"]
            grid = [[load(rep, arm, s, f) for f in range(folds)] if real else [load(rep, arm, s, "bank")]
                    for s in seeds]
            first = grid[0][0]
            if ids is None:
                ids, y = first["target_ids"], first["target_true"]
                names = sorted({rows[i]["provenance_group_id"] for i in ids})
                group = np.array([names.index(rows[i]["provenance_group_id"]) for i in ids])
                support = np.stack([np.bincount(group[y == c], minlength=len(names)) for c in (0, 1)], 1)
                gw, rejected = group_draws(rng_draws["rng"], cfg["uncertainty"]["draws"], support)
                gw = np.vstack([np.ones(len(names), int), gw])
                sw = np.vstack([np.ones(len(seeds), int), selection_draws(rng_draws["rng"], cfg["uncertainty"]["draws"], len(seeds))])
                fw = np.vstack([np.ones(folds, int), selection_draws(rng_draws["rng"], cfg["uncertainty"]["draws"], folds)])
                out["target"] = {"groups": names, "support": support.tolist(), "rejected_draws": rejected}
            for row in grid:
                for d in row:
                    if not np.array_equal(d["target_ids"], ids) or not np.array_equal(d["target_true"], y):
                        raise ValueError("Target population differs between prediction files")
            scores = np.array([[d["target_probabilities"][:, 1] for d in row] for row in grid])
            dims, weights = (scores.shape[:-1], [sw, fw] if real else [sw])
            samples = evaluate(scores if real else scores[:, 0], y, group, len(names), gw, weights)
            arms[arm] = {"samples": samples, "summary": summarize(samples),
                         "per_group": per_group(scores if real else scores[:, 0], y, group, len(names),
                                                dims if real else dims[:1])}
            if arm == "real":
                # Source (IDMT) score: each fold's held-out group, averaged like H1.
                per = {k: np.empty((len(seeds), folds)) for k in METRICS}
                for si, row in enumerate(grid):
                    for fi, d in enumerate(row):
                        sy, sp = d["source_true"], d["source_probabilities"][:, 1]
                        values = evaluate(sp, sy, np.zeros(len(sy), int), 1, np.ones((1, 1), int), [])
                        for k in METRICS:
                            per[k][si, fi] = values[k][0]
                src = {}
                for k in METRICS:
                    mass = sw[:, :, None] * fw[:, None, :]
                    src[k] = (mass * per[k][None]).sum(axis=(1, 2)) / mass.sum(axis=(1, 2))
                arms["idmt_within"] = {"samples": src, "summary": summarize(src)}
        contrasts = {"idmt_within_minus_melaudis_real": contrast(arms["idmt_within"]["samples"], arms["real"]["samples"]),
                     "procedural_minus_real": contrast(arms["procedural"]["samples"], arms["real"]["samples"]),
                     "audioldm_minus_real": contrast(arms["audioldm"]["samples"], arms["real"]["samples"]),
                     "real_plus_procedural_minus_real_repeated": contrast(arms["real_plus_procedural"]["samples"], arms["real_repeated"]["samples"]),
                     "real_plus_audioldm_minus_real_repeated": contrast(arms["real_plus_audioldm"]["samples"], arms["real_repeated"]["samples"])}
        out[rep] = {"arms": {a: {k: v for k, v in r.items() if k != "samples"} for a, r in arms.items()},
                    "contrasts": contrasts}
    return out


def factorial(cfg, key, rng_draws):
    folder = resolve(cfg["inputs"][key + "_results"])
    summary = json.loads((folder / "summary.json").read_text())
    path = folder / "target_predictions.npz"
    if sha256(path) != summary["artifacts_sha256"]["target_predictions.npz"]:
        raise ValueError(f"{key}: prediction hash mismatch")
    data = np.load(path, allow_pickle=False)
    y, group = data["y"], data["group_index"]
    names = list(summary["target_groups"])
    support = np.stack([np.bincount(group[y == c], minlength=len(names)) for c in (0, 1)], 1)
    if support.tolist() != [list(v) for v in summary["target_groups"].values()]:
        raise ValueError(f"{key}: group support differs from the recorded summary")
    draws = cfg["uncertainty"]["draws"]
    gw, rejected = group_draws(rng_draws["rng"], draws, support)
    gw = np.vstack([np.ones(len(names), int), gw])
    bw = np.vstack([np.ones(5, int), selection_draws(rng_draws["rng"], draws, 5)])
    out = {"target": {"groups": names, "support": support.tolist(), "rejected_draws": rejected}}
    for rep in cfg[key]["representations"]:
        cells = {}
        if key == "h2":
            array = data[rep]
            for ci, cell in enumerate(cfg["h2"]["cells"]):
                s = evaluate(array[ci], y, group, len(names), gw, [bw])
                cells[cell] = {"samples": s, "summary": summarize(s),
                               "per_group": per_group(array[ci], y, group, len(names), (5,))}
            q = {c: cells[c]["samples"] for c in cells}
            marg = {"path_marginal": ({k: (q["F01"][k] + q["F11"][k]) / 2 for k in q["F00"]},
                                      {k: (q["F00"][k] + q["F10"][k]) / 2 for k in q["F00"]}),
                    "source_marginal": ({k: (q["F10"][k] + q["F11"][k]) / 2 for k in q["F00"]},
                                        {k: (q["F00"][k] + q["F01"][k]) / 2 for k in q["F00"]}),
                    "D_source_minus_path": (q["F10"], q["F01"])}
        else:
            array = data[f"{cfg['latency']['variant']}__{rep}"]
            for ci, cell in enumerate(cfg["latency"]["cells"]):
                per_source = [evaluate(array[si, ci], y, group, len(names), gw, [bw]) for si in range(2)]
                s = {k: (per_source[0][k] + per_source[1][k]) / 2 for k in METRICS}
                cells[cell] = {"samples": s, "summary": summarize(s)}
            q = {c: cells[c]["samples"] for c in cells}
            marg = {"latency_only_minus_attenuation_only": (q["A0L1"], q["A1L0"]),
                    "both_minus_latency_only": (q["A1L1"], q["A0L1"])}
        out[rep] = {"cells": {c: {k: v for k, v in r.items() if k != "samples"} for c, r in cells.items()},
                    "contrasts": {name: contrast(a, b) for name, (a, b) in marg.items()}}
    return out


def provenance(repo):
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty_paths": len(git("status", "--porcelain").splitlines()),
            "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}


def main():
    cfg = json.loads(CONFIG.read_text())
    output = resolve(cfg["output"], must_exist=False)
    output.mkdir(parents=True, exist_ok=False)
    rng_draws = {"rng": np.random.default_rng(cfg["uncertainty"]["seed"])}
    result = {"config": cfg, "config_sha256": sha256(CONFIG), "script_sha256": sha256(Path(__file__)),
              "provenance": provenance(resolve("repo:.")),
              "metrics": list(METRICS), "role": "post hoc descriptive re-analysis of exposed development predictions",
              "h1": h1(cfg, rng_draws), "h2": factorial(cfg, "h2", rng_draws),
              "latency": factorial(cfg, "latency", rng_draws)}
    with (output / "results.json").open("x") as stream:
        json.dump(result, stream, indent=1, sort_keys=True, allow_nan=False)
    print(json.dumps({"output": str(output), "sha256": sha256(output / "results.json")}))


if __name__ == "__main__":
    sys.exit(main())

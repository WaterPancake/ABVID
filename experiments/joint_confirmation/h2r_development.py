"""Assemble a latency-free H2 source x path comparison from existing predictions.

P0 cells come from H2 (F00/F10); corrected-path P1c cells come from the latency
control's A1L0 (ground + centred air amplitude). Both studies share templates,
geometry slots, banks and budgets. Reads hash-checked prediction files only.
Run: uv run --extra cast python experiments/joint_confirmation/h2r_development.py
"""
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from abvid.evaluation.ranking import METRICS, evaluate, group_draws, interval, selection_draws
from abvid.paths import resolve
from abvid.provenance.hashing import sha256

CONFIG = Path(__file__).with_name("h2r_config.json")
KEYS = ("macro_f1", "balanced_accuracy", "auroc", "average_precision", "recall_at_fpr_05")


def load(folder):
    summary = json.loads((folder / "summary.json").read_text())
    path = folder / "target_predictions.npz"
    if sha256(path) != summary["artifacts_sha256"]["target_predictions.npz"]:
        raise ValueError(f"Prediction hash mismatch: {folder}")
    return np.load(path, allow_pickle=False), summary


def provenance(repo):
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty_paths": len(git("status", "--porcelain").splitlines()),
            "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}


def main():
    cfg = json.loads(CONFIG.read_text())
    output = resolve(cfg["output"], must_exist=False)
    output.mkdir(parents=True, exist_ok=False)
    data = {k: load(resolve(v)) for k, v in cfg["inputs"].items()}
    first = data["h2"][0]
    for name, (d, s) in data.items():
        if not (np.array_equal(d["target_ids"], first["target_ids"]) and np.array_equal(d["y"], first["y"])
                and np.array_equal(d["group_index"], first["group_index"])):
            raise ValueError(f"{name}: target population or grouping differs")
    y, group = first["y"], first["group_index"]
    names = list(data["h2"][1]["target_groups"])
    support = np.stack([np.bincount(group[y == c], minlength=len(names)) for c in (0, 1)], 1)
    rng = np.random.default_rng(cfg["uncertainty"]["seed"])
    gw, rejected = group_draws(rng, cfg["uncertainty"]["draws"], support)
    gw = np.vstack([np.ones(len(names), int), gw])
    bw = np.vstack([np.ones(cfg["banks"], int), selection_draws(rng, cfg["uncertainty"]["draws"], cfg["banks"])])
    result = {"config": cfg, "config_sha256": sha256(CONFIG), "script_sha256": sha256(Path(__file__)),
              "provenance": provenance(resolve("repo:.")), "metrics": list(METRICS),
              "role": "post hoc development assembly on exposed MELAUDIS",
              "target": {"groups": names, "support": support.tolist(), "rejected_draws": rejected}}
    for rep in cfg["representations"]:
        q = {}
        for cell, spec in cfg["cells"].items():
            array = data[spec["study"]][0][spec["array"].format(rep=rep)][tuple(spec["index"])]
            if array.shape != (cfg["banks"], len(y)):
                raise ValueError(f"{cell}: unexpected prediction shape {array.shape}")
            q[cell] = evaluate(array, y, group, len(names), gw, [bw])
        contrasts = {
            "path_marginal": lambda k: (q["S0_P1c"][k] + q["S1_P1c"][k] - q["S0_P0"][k] - q["S1_P0"][k]) / 2,
            "source_marginal": lambda k: (q["S1_P0"][k] + q["S1_P1c"][k] - q["S0_P0"][k] - q["S0_P1c"][k]) / 2,
            "interaction": lambda k: q["S1_P1c"][k] - q["S1_P0"][k] - q["S0_P1c"][k] + q["S0_P0"][k],
            "D_source_minus_path": lambda k: q["S1_P0"][k] - q["S0_P1c"][k],
        }
        result[rep] = {
            "cells": {c: {k: {"estimate": float(v[k][0]), "ci95": interval(v[k][1:])} for k in METRICS}
                      for c, v in q.items()},
            "contrasts": {name: {k: {"estimate": float(f(k)[0]), "ci95": interval(f(k)[1:])} for k in KEYS}
                          for name, f in contrasts.items()}}
    with (output / "results.json").open("x") as stream:
        json.dump(result, stream, indent=1, sort_keys=True, allow_nan=False)
    print(json.dumps({"output": str(output), "sha256": sha256(output / "results.json")}))


if __name__ == "__main__":
    main()

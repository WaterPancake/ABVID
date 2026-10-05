"""Source-only figures and residual summaries; never opens outer observations."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import time

import numpy as np
import soundfile as sf

from cast.config import ROOT, save, sha
from cast_generalization.method import describe
from cast_generalization.pipeline import read
from .data import setup_cpu
from .spectrum_evaluation import HERE
from .evaluate import load_banks
from .spectrum_evaluation import check_source


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    return plt


def source_figures(source, dest):
    protocol, summary, selected = check_source(source)
    scores = read(source/"scores.json")
    classes = protocol["metric_config"]["class_order"]
    candidates = summary["candidates"]
    names = list(candidates)
    display = ["Spectral calibration" if n.startswith("spectrum_calibrated") else "V8 reference" for n in names]
    dest.mkdir(parents=True, exist_ok=False)
    plt = pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    x = np.arange(len(names))
    for col, c in enumerate(classes):
        for arm, color, offset in (("joint", "#096b72", -.24), ("marginals", "#bc741a", 0), ("prototype", "#6e658a", .24)):
            cov = [100*candidates[n]["classes"][c]["arms"][arm]["coverage"] for n in names]
            axes[0, col].bar(x+offset, cov, .22, label=arm, color=color)
        axes[0, col].axhline(80, ls="--", color="black", lw=1, label="Coverage target")
        axes[0, col].set(title=c.title(), ylabel="Mean marginal coverage (%)", ylim=(0, 100),
                         xticks=x, xticklabels=display)
        for control, color, offset in (("prototype", "#6e658a", -.16), ("marginals", "#bc741a", .16)):
            gain = [100*candidates[n]["classes"][c]["gains"][control] for n in names]
            axes[1, col].bar(x+offset, gain, .29, label=f"vs {control}", color=color)
        axes[1, col].axhline(2.5, ls="--", color="black", lw=1, label="Gain target")
        axes[1, col].axhline(0, color="grey", lw=.7)
        axes[1, col].set(ylabel="Joint relative W1 reduction (%)", xticks=x, xticklabels=display)
    axes[0, 0].legend(fontsize=8, ncols=2)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle("Source-group development: coverage and matched-control margins\nEqual fold/seed weights; no outer held audio", fontsize=12)
    fig.savefig(dest/"source_comparison.png", dpi=170)
    plt.close(fig)

    families = protocol["metric_config"]["primary_families"]
    folds = summary["evaluated_groups"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    fold_results = []
    for col, c in enumerate(classes):
        for arm, color in (("joint", "#096b72"), ("marginals", "#bc741a"), ("prototype", "#6e658a")):
            subset = [r for r in scores if r["class"] == c and r["arm"] == arm and
                      (r["variant"], r["temperature"]) == (selected["variant"], selected["temperature"])]
            distances = [float(np.mean([r["score"]["families"][f]["W1"] for r in subset])) for f in families]
            axes[0, col].plot(families, distances, "o-", label=arm, color=color)
            for fold in folds:
                rows = [r for r in subset if r["fold"] == fold]
                fold_results.append({"class": c, "arm": arm, "fold": fold,
                                     "W1": float(np.mean([r["score"]["W1"] for r in rows])),
                                     "coverage": float(np.mean([r["score"]["coverage"] for r in rows]))})
        axes[0, col].set(title=c.title(), ylabel="Scaled descriptor W1 (lower is better)")
        for arm, color in (("joint", "#096b72"), ("marginals", "#bc741a")):
            values = [100*r["coverage"] for r in fold_results if r["class"] == c and r["arm"] == arm]
            axes[1, col].plot(np.arange(1, len(folds)+1), values, "o-", color=color, label=arm)
        axes[1, col].axhline(80, ls="--", color="black", lw=1)
        axes[1, col].set(xlabel="Training validation group (key in summary.json)",
                         ylabel="Marginal coverage (%)", xticks=range(1, len(folds)+1), ylim=(0, 100))
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(f"Ranked source candidate: {summary['selected']} — source criteria {'PASS' if selected['passes'] else 'FAIL'}\nDevelopment folds are not independent confirmation", fontsize=12)
    fig.savefig(dest/"source_families_groups.png", dpi=170)
    plt.close(fig)
    save(dest/"summary.json", {"outer_held_access": False, "source_evaluation": str(source.relative_to(ROOT)),
         "source_summary_sha256": sha(source/"summary.json"), "fold_order": folds,
         "fold_results": fold_results, "selected": summary["selected"], "source_passed": selected["passes"],
         "counts_note": "Five real source groups; seeds are not new recording sessions",
         "plot_hashes": {p.name: sha(p) for p in dest.glob("*.png")}, "code_sha256": sha(Path(__file__))})



def main():
    p = argparse.ArgumentParser(description="Audited joint spectrum source figures")
    p.add_argument("--eval-id", required=True)
    p.add_argument("--output-id", required=True)
    a = p.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", x) for x in (a.eval_id, a.output_id)):
        p.error("Use simple IDs")
    source_figures(HERE/"evaluations"/a.eval_id, HERE/"diagnostics"/a.output_id)
    print(json.dumps({"output": str(HERE/"diagnostics"/a.output_id)}), flush=True)


if __name__ == "__main__":
    main()

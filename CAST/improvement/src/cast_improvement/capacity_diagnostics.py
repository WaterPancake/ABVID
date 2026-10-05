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
from .capacity_data import HERE, setup_cpu
from .capacity_evaluation import load_banks
from .capacity_evaluation import check_source


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
                         xticks=x, xticklabels=names)
        for control, color, offset in (("prototype", "#6e658a", -.16), ("marginals", "#bc741a", .16)):
            gain = [100*candidates[n]["classes"][c]["gains"][control] for n in names]
            axes[1, col].bar(x+offset, gain, .29, label=f"vs {control}", color=color)
        axes[1, col].axhline(2.5, ls="--", color="black", lw=1, label="Gain target")
        axes[1, col].axhline(0, color="grey", lw=.7)
        axes[1, col].set(ylabel="Joint relative W1 reduction (%)", xticks=x, xticklabels=names)
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


def fit_residuals(out, scope, dest):
    setup_cpu()
    started = time.perf_counter()
    configs, banks, observations, metric_cfg = load_banks(out, scope)
    cfg = configs["capacity16x9"]
    audit = read(out/("fit_"+scope+"_verification.json"))
    if not audit["passed"] or audit["fit_summary_sha256"] != sha(out/("fit_"+scope+"_summary.json")):
        raise ValueError("Fit replay audit missing or changed")
    dest.mkdir(parents=True, exist_ok=False)
    observation_by_id = {r["file_id"]: r for r in observations}
    records = []
    for row in banks["capacity16x9"]:
        folder = out/"fits"/row["file_id"]
        fit = read(folder/"fit.json")
        wave, rate = sf.read(folder/"reconstructed_shape.wav", dtype="float32")
        if rate != 16000:
            raise ValueError("Changed reconstruction sample rate")
        real = observation_by_id[row["file_id"]]["descriptors"]
        recon = {k: v.tolist() for k, v in describe(wave, cfg, metric_cfg).items()}
        records.append({"file_id": row["file_id"], "class": row["class"], "group": row["group"],
                        "real": real, "reconstruction": recon, "diagnostics": fit["diagnostics"],
                        "check_loss": fit["check_loss"], "baseline_check_loss": fit["baseline_check_loss"],
                        "fit_sha256": sha(folder/"fit.json"), "fit_seconds": fit["fit_seconds"]})
    save(dest/"per_fit_residuals.json", records)
    statistics = {}
    for c in metric_cfg["class_order"]:
        selected = sorted([r for r in records if r["class"] == c], key=lambda r: (r["check_loss"], r["file_id"]))
        n = len(selected)
        stats = {"count": n, "groups": dict(Counter(r["group"] for r in selected)),
                 "systematic_examples": {"best": selected[0]["file_id"], "median": selected[n//2]["file_id"], "worst": selected[-1]["file_id"]},
                 "scientific_flags": dict(Counter(f for r in selected for f in r["diagnostics"]["flags"])),
                 "boundary_hits": dict(Counter(f for r in selected for f in r["diagnostics"]["boundary_hits"]))}
        for field in ("check_loss", "baseline_check_loss", "fit_seconds"):
            a = np.array([r[field] for r in selected])
            stats[field] = {"median": float(np.median(a)), "p90": float(np.quantile(a, .9)), "maximum": float(a.max())}
        for field in ("spectral_relative_magnitude_error", "envelope_rmse", "band_energy_L1_error", "modulation_L1_error"):
            a = np.array([r["diagnostics"][field] for r in selected])
            stats[field] = {"median": float(np.median(a)), "p90": float(np.quantile(a, .9))}
        statistics[c] = stats
    plt = pyplot()
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    coordinates = {"log_spectrum": (np.arange(64)+.5)*62.5,
                   "bands": np.arange(1, 9), "envelope": (np.arange(40)+.5)*.05,
                   "modulation": np.arange(1, 21)*.5}
    labels = {"log_spectrum": ("Frequency (Hz)", "Log-power residual"), "bands": ("Noise band index", "Energy fraction residual"),
              "envelope": ("Time (s)", "Unit RMS envelope residual"), "modulation": ("Modulation frequency (Hz)", "Modulation magnitude residual")}
    for row_index, c in enumerate(metric_cfg["class_order"]):
        selected = [r for r in records if r["class"] == c]
        for col, family in enumerate(metric_cfg["primary_families"]):
            residual = np.array([r["reconstruction"][family] for r in selected])-np.array([r["real"][family] for r in selected])
            low, median, high = np.quantile(residual, [.1, .5, .9], axis=0)
            coordinate = coordinates[family]
            if len(coordinate) != residual.shape[1]:
                raise ValueError("Descriptor coordinate count changed")
            ax = axes[row_index, col]
            ax.axhline(0, color="black", lw=.7)
            ax.fill_between(coordinate, low, high, alpha=.18, color="#096b72", label="10–90% clips")
            ax.plot(coordinate, median, color="#096b72", label="Median")
            ax.set(xlabel=labels[family][0], ylabel=labels[family][1], title=f"{c.title()} — {family}")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Training reconstruction minus observation\nFirst fixed checking realization; intervals describe clips, not session uncertainty", fontsize=12)
    fig.savefig(dest/"training_residuals.png", dpi=170)
    plt.close(fig)
    save(dest/"summary.json", {"scope": scope, "fit_run": str(out.relative_to(ROOT)), "outer_held_access": False,
         "classes": statistics, "fit_audit_sha256": sha(out/("fit_"+scope+"_verification.json")),
         "residuals_sha256": sha(dest/"per_fit_residuals.json"), "plot_sha256": sha(dest/"training_residuals.png"),
         "seconds": time.perf_counter()-started, "code_sha256": sha(Path(__file__)),
         "interpretation": "Paired source reconstruction; no independent generation or generalization claim"})


def paired(out, scope, dest):
    from .capacity_evaluation import SMOOTH
    _, banks, _, metric = load_banks(out, scope, True)
    dest.mkdir(parents=True, exist_ok=False)
    records = []
    for r in banks["capacity16x9"]:
        new = read(out/"fits"/r["file_id"]/"fit.json")
        old = read(SMOOTH/"fits"/r["file_id"]/"fit.json")
        records.append({"file_id": r["file_id"], "class": r["class"], "group": r["group"],
                        "capacity_check_loss": new["check_loss"], "smooth_check_loss": old["check_loss"],
                        "relative_check_improvement": 1-new["check_loss"]/old["check_loss"],
                        "capacity_diagnostics": new["diagnostics"], "smooth_diagnostics": old["diagnostics"],
                        "capacity_fit_sha256": sha(out/"fits"/r["file_id"]/"fit.json"),
                        "smooth_fit_sha256": sha(SMOOTH/"fits"/r["file_id"]/"fit.json")})
    statistics = {}
    fields = ("spectral_relative_magnitude_error", "band_energy_L1_error", "envelope_rmse", "modulation_L1_error")
    plt = pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for col, label in enumerate(metric["class_order"]):
        subset = [r for r in records if r["class"] == label]
        gains = np.array([r["relative_check_improvement"] for r in subset])
        statistics[label] = {"count": len(subset), "improved_clips": int((gains > 0).sum()),
                             "median_paired_relative_check_improvement": float(np.median(gains)),
                             "minimum_paired_relative_check_improvement": float(gains.min()),
                             "per_group_median_paired_relative_check_improvement": {
                                 g: float(np.median([r["relative_check_improvement"] for r in subset if r["group"] == g]))
                                 for g in sorted({r["group"] for r in subset})}, "residuals": {}}
        a = np.array([r["smooth_check_loss"] for r in subset])
        b = np.array([r["capacity_check_loss"] for r in subset])
        ax = axes[0, col]; ax.scatter(a, b, color="#096b72", s=25, alpha=.7)
        low, high = min(a.min(), b.min())-.01, max(a.max(), b.max())+.01
        ax.plot([low, high], [low, high], "--", color="black", lw=1)
        ax.set(title=f"{label.title()}: {100*np.median(gains):.2f}% median paired gain",
               xlabel="Smooth8 checking loss", ylabel="Capacity16x9 checking loss", xlim=(low, high), ylim=(low, high))
        for model, offset, color in (("smooth", -.18, "#bc741a"), ("capacity", .18, "#096b72")):
            values = [float(np.median([r[model+"_diagnostics"][f] for r in subset])) for f in fields]
            axes[1, col].bar(np.arange(4)+offset, values, .34, label=model, color=color)
            for field, v in zip(fields, values):
                statistics[label]["residuals"].setdefault(field, {})[model+"_median"] = v
        axes[1, col].set(xticks=np.arange(4), xticklabels=["Spectrum", "Band L1", "Envelope", "Modulation"],
                         ylabel="Median diagnostic error (separate units)")
    axes[1, 0].legend()
    fig.suptitle("Paired training reconstruction: same inputs, objective and checking streams\nSource calibration only; no held-group coverage claim", fontsize=12)
    fig.savefig(dest/"paired_reconstruction.png", dpi=170); plt.close(fig)
    save(dest/"records.json", records)
    save(dest/"summary.json", {"scope": scope, "classes": statistics, "outer_held_access": False,
         "fit_run": str(out.relative_to(ROOT)), "smooth_run": str(SMOOTH.relative_to(ROOT)),
         "paired_record_sha256": sha(dest/"records.json"), "code_sha256": sha(Path(__file__)),
         "plot_sha256": sha(dest/"paired_reconstruction.png"),
         "fit_audit_sha256": sha(out/("fit_"+scope+"_verification.json"))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["source", "fits", "paired", "gallery"])
    parser.add_argument("--id", required=True)
    parser.add_argument("--output-id", required=True)
    parser.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    args = parser.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", s) for s in (args.id, args.output_id)):
        parser.error("Use simple IDs")
    dest = HERE/"diagnostics"/args.output_id
    if args.command == "source":
        source_figures(HERE/"evaluations"/args.id, dest)
    elif args.command == "paired":
        paired(HERE/"runs"/args.id, args.scope, dest)
    elif args.command == "gallery":
        from .capacity_gallery import build
        build(HERE/"runs"/args.id, args.scope, dest)
    else:
        fit_residuals(HERE/"runs"/args.id, args.scope, dest)
    print(json.dumps({"artifacts": str(dest.relative_to(ROOT)), "outer_held_access": False}), flush=True)


if __name__ == "__main__":
    main()

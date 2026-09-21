"""Generate real-data charts for the condensed ABVID study guide.

Reads numbers from the checked run CSVs where available, otherwise from the
documented reports, and writes PNGs into docs/notes/condensed-guide/assets/.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "notes" / "condensed-guide" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


def read_csv_lines(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def style_ax(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Chart 1: M2 classical vs CNN accuracy by SNR
# ---------------------------------------------------------------------------
def chart_m2_snr() -> None:
    cl = read_csv_lines(ROOT / "runs/m2_classical_grouped_seed42/accuracy_by_snr.csv")
    cn = read_csv_lines(ROOT / "runs/m2_cnn_grouped_seed42/accuracy_by_snr.csv")
    if not cl or not cn:
        return
    snr = [float(r["snr_db"]) for r in cl]
    a = [float(r["balanced_accuracy"]) for r in cl]
    b = [float(r["balanced_accuracy"]) for r in cn]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(snr, a, "o-", label="Classical (MFCC+LR)")
    ax.plot(snr, b, "s--", label="CNN (log-Mel)")
    ax.invert_xaxis()
    style_ax(ax, "M2: balanced accuracy vs SNR (grouped split, seed 42)",
             "SNR (dB)", "Balanced accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "m2_accuracy_vs_snr.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 2: M4 classification accuracy by SNR across representations
# ---------------------------------------------------------------------------
def chart_m4_classification() -> None:
    rows = read_csv_lines(ROOT / "runs/m4_multichannel_paired_sessions_seed42/classification_by_snr.csv")
    if not rows:
        return
    reps = ["single_mic_1", "feature_fusion_2", "feature_fusion_4", "gcc_phat_4", "srp_phat_4"]
    labels = ["1 mic", "2-mic fusion", "4-mic fusion", "4-mic GCC-PHAT BF", "4-mic SRP-PHAT BF"]
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for rep, lab in zip(reps, labels):
        xs = [(r["snr_db"], float(r["accuracy"])) for r in rows if r["representation"] == rep]
        xs.sort(key=lambda t: float(t[0]))
        ax.plot([float(x[0]) for x in xs], [x[1] * 100 for x in xs], "o-", label=lab)
    ax.invert_xaxis()
    ax.set_ylim(80, 102)
    style_ax(ax, "M4: classification accuracy vs SNR",
             "SNR (dB)", "Accuracy (%)")
    ax.legend(ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "m4_classification_vs_snr.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 3: M4 localization error vs SNR
# ---------------------------------------------------------------------------
def chart_m4_localization() -> None:
    rows = read_csv_lines(ROOT / "runs/m4_multichannel_paired_sessions_seed42/localization_by_snr.csv")
    if not rows:
        return
    est = ["gcc_phat_2", "gcc_phat_4", "srp_phat_2", "srp_phat_4"]
    labels = ["GCC-PHAT 2-mic", "GCC-PHAT 4-mic", "SRP-PHAT 2-mic", "SRP-PHAT 4-mic"]
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for e, lab in zip(est, labels):
        xs = [(r["snr_db"], float(r["mean_error_deg"])) for r in rows if r["estimator"] == e]
        xs.sort(key=lambda t: float(t[0]))
        ax.plot([float(x[0]) for x in xs], [x[1] for x in xs], "o-", label=lab)
    ax.invert_xaxis()
    style_ax(ax, "M4: localization mean angular error vs SNR",
             "SNR (dB)", "Mean angular error (deg)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "m4_localization_vs_snr.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 4: M4 confusion matrices (single mic vs 2-mic fusion)
# ---------------------------------------------------------------------------
def chart_m4_confusion() -> None:
    singles = np.array([[84, 0], [13, 71]])
    fusion = np.array([[82, 2], [1, 83]])
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    for ax, mat, title in zip(axes, (singles, fusion),
                              ("1 mic (92.26% BA)", "2-mic feature fusion (98.21% BA)")):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=84)
        ax.set_xticks([0, 1], ["tracked", "wheeled"])
        ax.set_yticks([0, 1], ["tracked", "wheeled"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title, fontsize=9)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                        color="white" if mat[i, j] > 42 else "black")
    fig.suptitle("M4: confusion matrices (rows=true, cols=predicted)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "m4_confusion_matrices.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 5: M5 transfer A/B/C balanced accuracy
# ---------------------------------------------------------------------------
def chart_m5_transfer() -> None:
    labels = ["A real-only", "B synth-only", "C +1% real", "C +5% real", "C +10% real", "C +25% real"]
    ba = [10.2, 32.8, 16.4, 26.6, 26.6, 17.2]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    colors = ["#c44", "#4a4", "#888", "#888", "#888", "#888"]
    bars = ax.bar(labels, ba, color=colors)
    ax.axhline(50, color="black", ls=":", lw=1, label="chance (balanced)")
    ax.set_ylim(0, 55)
    for b, v in zip(bars, ba):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v}%", ha="center", fontsize=8)
    style_ax(ax, "M5: real-test balanced accuracy by protocol",
             "", "Balanced accuracy (%)")
    ax.legend(loc="lower left")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUT / "m5_transfer_barchart.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 6: M6 Stage A — three methods across test domains
# ---------------------------------------------------------------------------
def chart_m6_stage_a() -> None:
    domains = ["Seen corrupt", "Unseen noise", "Unseen mic", "Unseen env", "All held-out synth", "Native real"]
    std = [70.5, 40.0, 70.3, 91.3, 77.7, 29.0]
    aug = [66.7, 50.0, 65.4, 73.9, 73.4, 48.4]
    inv = [72.7, 35.0, 76.9, 89.1, 78.9, 12.5]
    x = np.arange(len(domains))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.bar(x - w, std, w, label="Standard supervised", color="#7aa")
    ax.bar(x, aug, w, label="Augmentation only", color="#a87")
    ax.bar(x + w, inv, w, label="Representation invariance", color="#8a8")
    ax.set_xticks(x, domains, rotation=18, ha="right")
    ax.set_ylim(0, 105)
    style_ax(ax, "M6 Stage A: balanced accuracy across test domains",
             "", "Balanced accuracy (%)")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "m6_stage_a_comparison.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 7: M6 Stage B — five-seed methods, synthetic vs real
# ---------------------------------------------------------------------------
def chart_m6_stage_b() -> None:
    methods = ["Standard", "Augmentation", "Paired", "Direct inv", "Projected inv"]
    synth = [75.14, 76.24, 76.65, 75.83, 76.60]
    real = [35.97, 38.79, 36.75, 35.66, 37.05]
    serr = [2.30, 0.76, 1.39, 1.72, 0.96]
    rerr = [9.43, 9.54, 5.96, 4.76, 6.84]
    x = np.arange(len(methods))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(x - w / 2, synth, w, yerr=serr, capsize=3, label="Synthetic all-corruption", color="#7aa")
    ax.bar(x + w / 2, real, w, yerr=rerr, capsize=3, label="Fixed native real", color="#c88")
    ax.set_xticks(x, methods, rotation=15, ha="right")
    ax.set_ylim(0, 95)
    style_ax(ax, "M6 Stage B: five-seed mean balanced accuracy (mean ± SD)",
             "", "Balanced accuracy (%)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "m6_stage_b_synth_vs_real.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 8: M6 Stage E — fusion gate metrics (mean balanced accuracy)
# ---------------------------------------------------------------------------
def chart_m6_fusion() -> None:
    labels = ["Semantic-only\n(nested)", "Semantic-only\n(ensemble)", "Late fusion\n(ensemble)"]
    values = [70.72, 72.92, 77.35]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    bars = ax.bar(labels, values, color=["#9a9", "#8a8", "#6c6"])
    ax.axhline(75, color="black", ls=":", lw=1, label="Gate 2: ≥75%")
    ax.set_ylim(0, 90)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.2f}%", ha="center", fontsize=8)
    style_ax(ax, "M6 Stage E: nested real-session mean balanced accuracy",
             "", "Mean balanced accuracy (%)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "m6_fusion_gate.png")
    plt.close(fig)


def main() -> None:
    chart_m2_snr()
    chart_m4_classification()
    chart_m4_localization()
    chart_m4_confusion()
    chart_m5_transfer()
    chart_m6_stage_a()
    chart_m6_stage_b()
    chart_m6_fusion()
    print("Wrote charts to", OUT)
    for p in sorted(OUT.glob("*.png")):
        print(" -", p.name, f"{p.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()

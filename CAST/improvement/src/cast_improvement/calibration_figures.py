"""Audited source-grid figures for bounded band-bias calibration."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from cast.config import ROOT, save, sha
from cast_generalization.pipeline import read
from .calibration_evaluation import HERE
SPECTRAL = (.5, .65, .8, 1.)
STRENGTH = (0., 1.)
from .calibration_evaluation import frozen


def build(out, dest):
    protocol, summary, *_ = frozen(out)
    audit = read(out/"audit_verification.json")
    if not audit["passed"] or audit["summary_sha256"] != sha(out/"summary.json"):
        raise ValueError("A matching exact source replay audit is required")
    dest.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    selected = summary["candidates"][summary["selected"]]
    values = {(r["spectral_temperature"], r["correction_strength"]): r for r in summary["candidates"].values()}
    matrices = {}
    for c in ("car", "truck"):
        matrices[c] = [np.array([[100*values[s, e]["classes"][c]["arms"]["joint"]["coverage"] for e in STRENGTH] for s in SPECTRAL])]
        matrices[c] += [np.array([[100*values[s, e]["classes"][c]["gains"][control] for e in STRENGTH] for s in SPECTRAL]) for control in ("marginals", "prototype")]
    gain_limit = max(2.5, max(float(np.abs(m).max()) for a in matrices.values() for m in a[1:]))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row, c in enumerate(("car", "truck")):
        for col, title in enumerate(("Coverage (%)", "W1 gain vs marginals (%)", "W1 gain vs prototype (%)")):
            m = matrices[c][col]
            kwargs = {"cmap": "cividis", "vmin": 0, "vmax": 100} if col == 0 else {"cmap": "coolwarm_r", "vmin": -gain_limit, "vmax": gain_limit}
            ax = axes[row, col]
            chart = ax.imshow(m, aspect="auto", **kwargs)
            for i, s in enumerate(SPECTRAL):
                for j, e in enumerate(STRENGTH):
                    failed_spread = not values[s, e]["classes"][c]["passes"]["spread"]
                    text = f"{m[i, j]:.1f}"+(" *" if failed_spread else "")
                    ax.text(j, i, text, ha="center", va="center", fontsize=10,
                            bbox={"facecolor": "white", "alpha": .78, "edgecolor": "none", "pad": 2})
            i, j = SPECTRAL.index(selected["spectral_temperature"]), STRENGTH.index(selected["correction_strength"])
            ax.add_patch(Rectangle((j-.48, i-.48), .96, .96, fill=False, ec="black", lw=2))
            ax.set(xticks=range(len(STRENGTH)), xticklabels=STRENGTH, yticks=range(len(SPECTRAL)), yticklabels=SPECTRAL,
                   xlabel="Correction strength", ylabel="Spectral temperature", title=f"{c.title()}: {title}")
            fig.colorbar(chart, ax=ax, shrink=.8)
    status = "PASS" if selected["passes"] else "FAIL"
    fig.suptitle(f"Source-group development: selected candidate {status}; outer group not used\nTargets: coverage ≥80%, gain ≥2.5% against each control; border = selected, * = spread failure", fontsize=12)
    fig.savefig(dest/"source_grid.png", dpi=170)
    plt.close(fig)
    rows = []
    for s in SPECTRAL:
        for e in STRENGTH:
            r = values[s, e]
            rows.append({"spectral_temperature": s, "correction_strength": e, "passes": r["passes"],
                         "classes": {c: {"coverage": r["classes"][c]["arms"]["joint"]["coverage"],
                                         "W1": r["classes"][c]["arms"]["joint"]["W1"],
                                         "gains": r["classes"][c]["gains"], "passes": r["classes"][c]["passes"]}
                                     for c in ("car", "truck")}})
    save(dest/"summary.json", {"selected": summary["selected"], "selected_passes": selected["passes"], "candidates": rows,
         "source_summary_sha256": sha(out/"summary.json"), "audit_sha256": sha(out/"audit_verification.json"),
         "plot_sha256": sha(dest/"source_grid.png"), "code_sha256": sha(Path(__file__)), "outer_held_access": False})
    print(json.dumps({"artifacts": str(dest.relative_to(ROOT)), "source_candidate_passes": selected["passes"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--output-id", required=True)
    args = parser.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", x) for x in (args.eval_id, args.output_id)):
        parser.error("Use simple IDs")
    build(HERE/"evaluations"/args.eval_id, HERE/"diagnostics"/args.output_id)


if __name__ == "__main__":
    main()

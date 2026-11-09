"""Render small summaries of the frozen stage diagnosis; no model operations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from abvid.cast.grouped_storage import read_bundle
from abvid.paths import resolve
from abvid.provenance.hashing import sha256


RUN = "artifact:cast/diagnostics/cast_stage_diagnosis_20261025_v1"
root = resolve(RUN)
_, payload = read_bundle(root / "scores")
result = json.loads(payload["results.json"])
_, verification = read_bundle(root / "verification")
_, score_verification = read_bundle(root / "verification_scores")
families = {"original": ["original"], "ra": ["ra"], "smooth8": ["smooth8_check_0", "smooth8_check_1"],
            "v11_check": ["v11_check_0", "v11_check_1"], "v11_fresh": ["v11_fresh_0", "v11_fresh_1"],
            "population": ["population"],
            **{stage: [stage + "_check_0", stage + "_check_1"] for stage in ("v7", "v8", "v9", "v10")}}
summary = {"run": RUN, "results_sha256": sha256(root / "scores/results.json"),
           "protocol_sha256": sha256(root / "design/protocol.md"), "design_identity": result["design_identity"],
           "scope": "post_hoc_source_training_only", "role": result["role"],
           "realization_ranges_are_not_confidence_intervals": True,
           "original_feature_controls": result["original_feature_controls"],
           "waveform_verification": json.loads(verification["waveforms.json"]),
           "score_verification": json.loads(score_verification["scores.json"]), "rows": []}
for scope in ("all380", "balanced50"):
    for rep in ("BEATs768", "MFCC26"):
        for arm in ("R", "C"):
            for family, members in families.items():
                rows = [r for r in result["rows"] if r["scope"] == scope and r["representation"] == rep and r["head"] == arm and r["input"] in members]
                if not rows:
                    continue
                def triplet(values):
                    return {"mean": float(np.mean(values)), "min": float(min(values)), "max": float(max(values))}
                metrics = {k: triplet([r["metrics"][k] for r in rows]) for k in ("auroc", "balanced_accuracy", "positive_recall", "negative_recall")}
                geom = {k: triplet([r["geometry"][k] for r in rows]) for k in rows[0]["geometry"]}
                groups = []
                for group in rows[0]["groups"]:
                    matched = [g for r in rows for g in r["groups"] if g["group"] == group["group"]]
                    groups.append({"group": group["group"], "class_counts": group["class_counts"],
                                   "metrics": {k: triplet([g["metrics"][k] for g in matched]) for k in metrics}})
                summary["rows"].append({"scope": scope, "representation": rep, "head": arm, "stage": family,
                                         "count": rows[0]["count"], "class_counts": rows[0]["class_counts"],
                                         "realizations": len(rows), "metrics": metrics, "geometry": geom,
                                         "signed_margin_mean": triplet([r["signed_margin_mean"] for r in rows]), "groups": groups})
out = resolve("repo:reports/CAST_stage_diagnosis_20261025_summary.json", must_exist=False)
with out.open("x") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(14.3, 4.8))
stages = ["original", "smooth8", "v11_check", "v11_fresh", "population"]
labels = ["Original", "Smooth8", "Calibrated\nv11", "Fresh phase\n+ noise", "Population"]
specs = [("metrics", "auroc", "AUROC (%)", 100, (45, 104)),
         ("metrics", "balanced_accuracy", "Balanced accuracy (%)", 100, (45, 104)),
         ("geometry", "class_direction_cosine", "Alignment of car/truck direction", 1, (0, 1.05))]
for rep, color in (("BEATs768", "#2563a6"), ("MFCC26", "#258469")):
    rows = {r["stage"]: r for r in summary["rows"] if r["scope"] == "all380" and r["representation"] == rep and r["head"] == "R"}
    for ax, (section, key, label, scale, limits) in zip(axes, specs):
        means = np.array([rows[s][section][key]["mean"] for s in stages]) * scale
        low = np.array([rows[s][section][key]["min"] for s in stages]) * scale
        high = np.array([rows[s][section][key]["max"] for s in stages]) * scale
        ax.errorbar(np.arange(len(stages)), means, yerr=np.stack([means - low, high - means]),
                    color=color, marker="o", lw=2, capsize=4, label=rep)
        ax.axhline(rows["ra"][section][key]["mean"] * scale, color=color, ls=":", lw=1.5, alpha=.65)
        ax.set_xticks(np.arange(len(stages)), labels, fontsize=9)
        ax.set_ylabel(label)
        ax.set_ylim(*limits)
        ax.grid(axis="y", alpha=.18)
axes[0].legend(frameon=False, loc="lower left")
fig.suptitle("CAST stage diagnosis · one source-training role · 190 cars + 190 trucks", fontsize=14, x=.5, y=.98)
fig.text(.5, .025, "Saved real-only heads; no fitting. Whiskers: two realized waveforms, not confidence intervals. Dotted lines: conventional augmentation.", ha="center", fontsize=9, color="#444444")
fig.tight_layout(rect=(0, .08, 1, .93), w_pad=2.5)
png = resolve("repo:reports/figures/CAST_stage_diagnosis_20261025.png", must_exist=False)
svg = png.with_suffix(".svg")
if png.exists() or svg.exists():
    raise FileExistsError("Figure already exists")
fig.savefig(png, dpi=180, facecolor="white")
fig.savefig(svg, facecolor="white")
plt.close(fig)
print(json.dumps({"summary": str(out), "summary_sha256": sha256(out), "png": str(png), "svg": str(svg)}))

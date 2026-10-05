"""Source-only decomposition of group effects and generation-seed variation."""
import argparse
import json
from pathlib import Path
import re
import time
import numpy as np

from cast.config import ROOT, save, sha
from cast_generalization.pipeline import read, readl
from .resolution_evaluation import check_source


def components(arrays):
    means = np.stack([x.mean(axis=0) for x in arrays])
    within = np.mean([x.var(axis=0) for x in arrays], axis=0)
    between = means.var(axis=0)
    total = within + between
    fraction = between / np.maximum(total, 1e-20)
    return {"group_means": means.tolist(), "within_variance": within.tolist(),
            "between_variance": between.tolist(), "between_fraction": fraction.tolist(),
            "median_between_fraction": float(np.median(fraction)),
            "mean_between_fraction": float(np.mean(fraction)),
            "weighting": "equal groups, then equal parents within group; population variance"}


def parameter_blocks(p):
    def clr(values):
        x = np.log(np.maximum(values, 1e-8)); return x-x.mean()
    mixture = np.clip(p["harmonic_fraction"], 1e-4, 1-1e-4)
    return {"spacing_log": np.log(p["spacing_hz"]), "harmonic_clr": clr(p["harmonic_weights"]),
            "noise_clr": clr(p["noise_weights"]), "mixture_logit": np.atleast_1d(np.log(mixture/(1-mixture))),
            "envelope_clr": clr(p["envelope_knots"]),
            "envelope_log_gauge": np.atleast_1d(np.log(p["envelope_knots"]).mean())}


def build(source, out):
    start = time.perf_counter()
    protocol, summary, selected = check_source(source)
    observations = readl(source/"real_training_descriptors.jsonl")
    bank = readl(source/"spectrum16_bank.jsonl")
    scores = read(source/"scores.json")
    if len(bank) != 380 or {r["file_id"] for r in bank} != {r["file_id"] for r in observations}:
        raise ValueError("Changed training inventory")
    groups = sorted({r["group"] for r in bank})
    if len(groups) != 5 or protocol["metric_config"]["held_group"] in groups:
        raise ValueError("Unexpected or excluded group")
    out.mkdir(parents=True, exist_ok=False)
    result = {}
    for c in protocol["metric_config"]["class_order"]:
        obs = [r for r in observations if r["class"] == c]
        parents = [r for r in bank if r["class"] == c]
        descriptors = {f: components([np.stack([r["descriptors"][f] for r in obs if r["group"] == g]) for g in groups])
                       for f in protocol["metric_config"]["primary_families"]}
        parameters = {k: components([np.stack([parameter_blocks(r["parameters"])[k] for r in parents if r["group"] == g]) for g in groups])
                      for k in parameter_blocks(parents[0]["parameters"])}
        fold_scores = []
        for g in groups:
            rows = [r["score"] for r in scores if r["variant"] == "spectrum16" and r["arm"] == "joint" and r["class"] == c and r["fold"] == g]
            if len(rows) != 5:
                raise ValueError("Missing frozen source seed result")
            fold_scores.append({"group": g, "real_count": sum(r["group"] == g for r in obs),
                **{k: {"mean": float(np.mean([r[k] for r in rows])), "minimum": min(r[k] for r in rows),
                       "maximum": max(r[k] for r in rows), "seed_sd": float(np.std([r[k] for r in rows], ddof=1))}
                   for k in ("W1", "coverage")}})
        result[c] = {"descriptors": descriptors, "parameters": parameters, "fold_scores": fold_scores}
    save(out/"summary.json", {"classes": result, "group_order": groups, "outer_inputs_read": False,
        "raw_audio_read": False, "saved_training_observations_read": True,
        "source_summary_sha256": sha(source/"summary.json"), "source_audit_sha256": sha(source/"audit_verification.json"),
        "source_bank_sha256": sha(source/"spectrum16_bank.jsonl"), "code_sha256": sha(Path(__file__)),
        "seconds": time.perf_counter()-start, "notes": "Variance components are descriptive, not causal effects or proof of independent physical vehicles."})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11,4), constrained_layout=True)
    names = protocol["metric_config"]["primary_families"]
    for ax, c in zip(axes, protocol["metric_config"]["class_order"]):
        ax.bar(names, [100*result[c]["descriptors"][f]["mean_between_fraction"] for f in names], color="#096b72")
        ax.set(title=c.title(), ylabel="Mean fraction of variance between groups (%)", ylim=(0,100))
    fig.suptitle("Five source groups only: equal group and within-group parent weights")
    fig.savefig(out/"group_variance.png", dpi=160);plt.close(fig)
    print(json.dumps({c:{kind:{k:v["mean_between_fraction"] for k,v in result[c][kind].items()}
                         for kind in ("descriptors","parameters")} for c in result}),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-id", required=True)
    a=p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}",a.output_id):p.error("Use a simple ID")
    build(ROOT/"CAST/improvement/resolution_v11/evaluations/smooth16_temporal41_v11_full_source",
          ROOT/"CAST/improvement/diagnostics"/a.output_id)

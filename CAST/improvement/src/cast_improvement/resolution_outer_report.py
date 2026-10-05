"""Read-only presentation of a completed, verified outer development result."""
import argparse
from html import escape
import json
import os
from pathlib import Path
import re
import time

import numpy as np

from cast.config import ROOT, save, sha
from cast_generalization.pipeline import read, readl
from .data import PREVIOUS
from .resolution_evaluation import HERE


def build(out, dest):
    started = time.perf_counter()
    verification = read(out / "verification.json")
    if not verification["passed"]:
        raise ValueError("Outer presentation requires a completed replay audit")
    scores = read(out / "scores.json")
    cfg = read(out / "config.resolved.json")
    summary = scores["summary"]
    metric = cfg["sampling"]
    classes = metric["class_order"]
    dest.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    colors = {"joint": "#096b72", "marginals": "#bc741a", "prototype": "#6e658a"}
    seed_ranges = {}
    for col, c in enumerate(classes):
        result = summary["classes"][c]
        arms = ["joint", "marginals", "prototype"]
        coverage = [100 * result["arms"][a]["coverage"] for a in arms]
        axes[0, col].bar(arms, coverage, color=[colors[a] for a in arms])
        axes[0, col].axhline(80, color="black", ls="--", lw=1)
        axes[0, col].set(title=c.title(), ylabel="Mean marginal coverage (%)", ylim=(0, 100))
        for i, value in enumerate(coverage):
            if abs(value-80) < 4:
                axes[0, col].text(i, value-2, f"{value:.1f}", ha="center", va="top", color="white")
            else:
                axes[0, col].text(i, value+1, f"{value:.1f}", ha="center")
        controls = ["marginals", "prototype"]
        gains = [100 * result["gains"][a] for a in controls]
        axes[1, col].bar(controls, gains, color=[colors[a] for a in controls])
        axes[1, col].axhline(2.5, color="black", ls="--", lw=1)
        axes[1, col].axhline(0, color="grey", lw=.7)
        axes[1, col].set(ylabel="Joint relative W1 reduction (%)")
        seed_ranges[c] = {}
        for arm in arms:
            rows = [r["score"] for r in scores["records"] if r["class"] == c and r["arm"] == arm]
            seed_ranges[c][arm] = {key: {"min": min(r[key] for r in rows), "max": max(r[key] for r in rows)}
                                   for key in ("W1", "coverage")}
    fig.suptitle("Frozen spectrum16: exposed IDMT held-group development\nDashed lines: unchanged coverage and margin targets; one real group")
    fig.savefig(dest / "outer_comparison.png", dpi=170)
    plt.close(fig)

    families = metric["primary_families"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for col, c in enumerate(classes):
        for arm in ("joint", "marginals", "prototype"):
            rows = [r["score"] for r in scores["records"] if r["class"] == c and r["arm"] == arm]
            values = [np.mean([r["families"][f]["W1"] for r in rows]) for f in families]
            axes[col].plot(families, values, "o-", color=colors[arm], label=arm)
        axes[col].set(title=c.title(), ylabel="Scaled descriptor W1 (lower is better)")
    axes[0].legend()
    fig.suptitle("Four unchanged, equally weighted descriptor families")
    fig.savefig(dest / "outer_families.png", dpi=170)
    plt.close(fig)

    def link(path):
        if not path.is_file():
            raise ValueError(f"Missing presentation artifact: {path}")
        return escape(os.path.relpath(path, dest), quote=True)

    generated = readl(out / "generated_manifest.jsonl")
    held = readl(out / "held_descriptors.jsonl")
    examples = []
    cards = []
    for c in classes:
        original = min((r for r in held if r["class"] == c), key=lambda r: r["file_id"])
        original_path = PREVIOUS / original["path"] / "playback.wav"
        players = [f'<label>Held original · {escape(original["file_id"][:12])}<audio controls preload="none" src="{link(original_path)}"></audio></label>']
        examples.append({"class": c, "role": "held_original", "file_id": original["file_id"],
                         "path": str(original_path.relative_to(ROOT)), "sha256": sha(original_path)})
        for arm in ("joint", "marginals", "prototype"):
            row = next(r for r in generated if r["class"] == c and r["arm"] == arm and r["seed"] == metric["seeds"][0] and r["index"] == 0)
            path = out / row["path"] / "playback.wav"
            sidecar = out / row["path"] / "sample.json"
            players.append(f'<label>Generated · {arm}<audio controls preload="none" src="{link(path)}"></audio><a href="{link(sidecar)}">Parameters and ancestry</a></label>')
            examples.append({"class": c, "role": arm, "seed": row["seed"], "index": 0,
                             "path": str(path.relative_to(ROOT)), "sha256": sha(path)})
        cards.append(f'<article><h2>{c.title()}</h2><div>{"".join(players)}</div></article>')
    status = "PASS" if summary["passes"] else "FAIL"
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CAST outer development</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:30px auto;padding:20px;color:#16343b}article{border-top:1px solid #ccc;padding:18px 0}article div{display:flex;flex-wrap:wrap;gap:16px}label{display:grid;gap:6px}audio{max-width:100%}img{max-width:100%}</style><h1>CAST held-group development review</h1><p>Previously exposed IDMT development group; no fresh independent confirmation or classification result. Generated examples are unconditional draws, not reconstructions of the held recordings. Example selection is fixed: first held ID per class and seed 42/index 0 per generated arm. Playback gains are recorded in the linked sidecars; comparisons are of unit-RMS acoustic shape, not physical loudness.</p><p>The 69-coordinate marginal control breaks spectral and envelope dependence. Its margin is representation dependent; it is not a universal best competing generator.</p><p>Local research derivatives, CC BY-NC-ND 4.0; no redistribution authorization.</p>'''
    document += f'<p>Declared scientific criteria: <strong>{status}</strong>. <a href="{link(out/"scores.json")}">Complete scores</a>; <a href="{link(out/"failure.json")}">failure record</a>; <a href="{link(out/"verification.json")}">replay verification</a>.</p>'
    document += ''.join(f'<img src="{link(dest/name)}" alt="{escape(name)}">' for name in ("outer_comparison.png", "outer_families.png"))
    document += ''.join(cards) + '</html>'
    (dest / "index.html").write_text(document)
    save(dest / "summary.json", {"evaluation": str(out.relative_to(ROOT)), "goal_passed": summary["passes"],
        "scores_sha256": sha(out / "scores.json"), "verification_sha256": sha(out / "verification.json"),
        "seed_ranges": seed_ranges, "seed_range_is_not_group_uncertainty": True, "held_groups": 1,
        "examples": examples, "code_sha256": sha(Path(__file__)),
        "files": {p.name: sha(p) for p in dest.iterdir() if p.is_file()}, "seconds": time.perf_counter() - started})
    print(json.dumps({"output": str(dest.relative_to(ROOT)), "scientific_criteria": status, "examples": len(examples)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-id", required=True)
    p.add_argument("--output-id", required=True)
    a = p.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", v) for v in (a.eval_id, a.output_id)):
        p.error("Use simple IDs")
    build(HERE / "evaluations" / a.eval_id, HERE / "diagnostics" / a.output_id)

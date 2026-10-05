"""Inspect derived calibration parameters, limitations and matched local audio."""
import argparse
from collections import Counter
from html import escape
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import soundfile as sf
import torch

from cast.audio import shape
from cast.config import ROOT, save, sha
from cast.renderer import tensors
from cast_generalization.pipeline import read, readl
from .engine import SmoothRenderer
from .parent_mixture_evaluation import HERE, SMOOTH, check_source


def build(source, dest):
    started = time.perf_counter()
    protocol, summary, _ = check_source(source)
    rows = readl(source/"parent_calibrated_bank.jsonl")
    cfg = protocol["renderer_configs"]["parent_calibrated"]
    dest.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    stats = {}
    examples = []
    for col, label in enumerate(("car", "truck")):
        selected = [r for r in rows if r["class"] == label]
        a = [r["mixture_calibration"] for r in selected]
        old = np.array([r["original_harmonic_fraction"] for r in a])
        new = np.array([r["harmonic_fraction"] for r in a])
        axes[0,col].scatter(old, new, s=12, alpha=.7)
        axes[0,col].plot([0,1], [0,1], color="grey", ls="--")
        axes[0,col].set(title=label.title(), xlabel="Original effective fraction", ylabel="Calibrated effective fraction", xlim=(0,1), ylim=(0,1))
        identifiable = [r for r in a if "calibrated_expected_band_squared_error" in r]
        before = np.array([r["original_expected_band_squared_error"] for r in identifiable])
        after = np.array([r["calibrated_expected_band_squared_error"] for r in identifiable])
        axes[1,col].scatter(before, after, s=12, alpha=.7)
        limit = max(float(before.max()), float(after.max()))
        axes[1,col].plot([0,limit], [0,limit], color="grey", ls="--")
        axes[1,col].set(xlabel="Before: expected band squared error", ylabel="After: expected band squared error")
        stats[label] = {"parents": len(a), "identifiable": len(identifiable),
                        "median_original_fraction": float(np.median(old)), "median_calibrated_fraction": float(np.median(new)),
                        "flags": dict(Counter(f for r in a for f in r["flags"])),
                        "median_original_expected_squared_error": float(np.median(before)),
                        "median_calibrated_expected_squared_error": float(np.median(after))}
        for group in sorted({r["group"] for r in selected}):
            examples.append(min((r for r in selected if r["group"] == group), key=lambda r:r["file_id"]))
    fig.suptitle("Training-parent calibration only; no independent reconstruction claim\nComponent powers ignore stochastic cross terms; original fits remain unchanged")
    fig.savefig(dest/"calibration.png", dpi=160); plt.close(fig)

    def link(path):
        if not path.is_file(): raise ValueError("Missing diagnostic link")
        return escape(os.path.relpath(path, dest), quote=True)

    cards, audio_records = [], []
    for row in examples:
        folder = SMOOTH/"fits"/row["file_id"]
        fit = read(folder/"fit.json")
        renderer = SmoothRenderer(cfg, fit["input_id"])
        with torch.no_grad():
            wave = shape(renderer.render(tensors(row["parameters"]), "check"))[0,0].numpy()
        old, sr = sf.read(folder/"reconstructed_shape.wav", dtype="float32")
        original, original_sr = sf.read(folder/"original_shape.wav", dtype="float32")
        if sr != 16000 or original_sr != sr: raise ValueError("Changed audio sample rate")
        gain = .95/max(float(np.abs(x).max()) for x in (original, old, wave))
        target = dest/row["file_id"]; target.mkdir()
        for name, x in (("observed", original), ("original_fit", old), ("calibrated", wave)):
            sf.write(target/(name+".wav"), x*gain, sr, subtype="FLOAT")
        save(target/"parameters.json", row)
        rec = {"file_id":row["file_id"], "group": row["group"], "class":row["class"],
               "common_shape_playback_gain":gain, "calibration_sha256":row["mixture_calibration_sha256"],
               "files": {p.name:sha(p) for p in target.iterdir()}}
        audio_records.append(rec)
        players = ''.join(f'<label>{name}<audio controls preload="none" src="{link(target/(name+".wav"))}"></audio></label>' for name in ("observed","original_fit","calibrated"))
        flags = ', '.join(row["mixture_calibration"]["flags"]) or 'none'
        cards.append(f'<article><h2>{row["class"]} · {row["file_id"][:12]}</h2><p>{escape(row["group"])}</p><div>{players}</div><p>Calibration flags: {escape(flags)}. <a href="{link(target/"parameters.json")}">Parameters and complete ancestry</a></p></article>')
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CAST mixture calibration review</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:30px auto;padding:20px;color:#16343b}article{border-top:1px solid #ccc;padding:18px 0}article div{display:flex;flex-wrap:wrap;gap:16px}label{display:grid;gap:6px}audio{max-width:100%}img{max-width:100%}</style><h1>Per-parent mixture calibration</h1><p>First sorted ID per class and source group, independent of quality. Training reconstruction only: the saved checking components informed calibration, so this playback is not independent validation. Unit-RMS shapes share a common peak-safe playback gain in each card. Effective mixture fraction is not physical engine power.</p><p>Local research derivatives, CC BY-NC-ND 4.0; no redistribution authorization.</p>'''
    document += f'<img src="{link(dest/"calibration.png")}" alt="Original and calibrated fractions and expected band-power residuals">'+''.join(cards)+'</html>'
    (dest/"index.html").write_text(document)
    save(dest/"summary.json", {"classes":stats, "examples":audio_records,
        "source_summary_sha256":sha(source/"summary.json"), "source_audit_sha256":sha(source/"audit_verification.json"),
        "plot_sha256":sha(dest/"calibration.png"), "gallery_sha256":sha(dest/"index.html"),
        "code_sha256":sha(Path(__file__)), "elapsed_seconds":time.perf_counter()-started,
        "outer_access":False, "raw_source_audio_opened":False, "saved_training_audio_read":True})
    print(json.dumps({"output":str(dest.relative_to(ROOT)), "classes":stats}), flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-id", required=True); p.add_argument("--output-id", required=True)
    a=p.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", v) for v in (a.eval_id,a.output_id)):
        p.error("Use simple IDs")
    build(HERE/"evaluations"/a.eval_id, HERE/"diagnostics"/a.output_id)

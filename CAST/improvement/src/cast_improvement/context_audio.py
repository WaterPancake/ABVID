"""Fixed-example local audio review of audited source-only group priors."""
import argparse
import gzip
import hashlib
from html import escape
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import soundfile as sf

from cast.config import ROOT, digest, save, sha
from cast_generalization.pipeline import readl
from .context_evaluation import HERE, UPSTREAM_RUN, check_source
from .resolution_evaluation import render


def build(source, dest):
    started = time.perf_counter()
    protocol, summary, selected = check_source(source)
    cfg = protocol["renderer_configs"]["group_predictive"]
    metric = protocol["metric_config"]
    bank = {r["file_id"]:r for r in readl(source/"group_predictive_bank.jsonl")}
    examples = {}
    with gzip.open(source/"generated_records.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row["seed"] == metric["seeds"][0] and row["index"] == 0 and row["arm"] == "joint":
                key = row["fold"], row["class"]
                if row["variant"] in examples.setdefault(key, {}):
                    raise ValueError("Duplicated example")
                examples[key][row["variant"]] = row
    if len(examples) != 10 or any(set(rows) != set(protocol["variants"]) for rows in examples.values()):
        raise ValueError("Incomplete fixed-example schedule")
    dest.mkdir(parents=True, exist_ok=False)
    records, cards = [], []

    def link(path):
        if not path.is_file(): raise ValueError(f"Missing presentation artifact: {path}")
        return escape(os.path.relpath(path, dest), quote=True)

    original_run = ROOT/"CAST/improvement/runs/cast_smooth8_v1_20261004_r1"
    for number, ((fold, label), rows) in enumerate(sorted(examples.items()), 1):
        parents = {r["parent_ids"][0] for r in rows.values()}
        if len(parents) != 1: raise ValueError("Examples do not use matched residual donors")
        parent = parents.pop()
        folder = dest/f"example_{number:02d}"
        folder.mkdir()
        original = original_run/"fits"/parent/"original_shape.wav"
        reconstruction = UPSTREAM_RUN/"parents"/parent/"calibration_reconstruction_0.wav"
        waves = {}
        for name, path in (("original_parent", original), ("fitted_parent", reconstruction)):
            waves[name], sr = sf.read(path, dtype="float32")
            if sr != 16000: raise ValueError("Changed source example sample rate")
        for variant, row in rows.items():
            wave, streams = render(row, cfg, "spectrum16")
            if hashlib.sha256(wave.tobytes()).hexdigest() != row["waveform_samples_sha256"] or streams != row["streams"]:
                raise ValueError("Source audio example replay failed")
            waves[variant] = wave
        gain = .95/max(float(np.abs(w).max()) for w in waves.values())
        for name, wave in waves.items():
            sf.write(folder/(name+".wav"), wave*gain, 16000, subtype="FLOAT")
            actual, sr = sf.read(folder/(name+".wav"), dtype="float32")
            if sr != 16000 or not np.array_equal(actual, wave*gain):
                raise ValueError("Audio presentation export changed samples")
        record = {"validation_fold":fold, "class":label, "seed":metric["seeds"][0], "index":0,
                  "residual_parent":parent, "parent_group":bank[parent]["group"],
                  "original_shape_sha256":sha(original), "fitted_parent_sha256":sha(reconstruction),
                  "bank_row_sha256":digest(bank[parent]), "parent_flags":bank[parent]["flags"],
                  "playback_gain":gain, "generated_rows":rows,
                  "example_role":"Generated draws are not refits or reconstructions of this parent",
                  "audio":{name:{"path":str((folder/(name+".wav")).relative_to(ROOT)),
                                 "sha256":sha(folder/(name+".wav"))} for name in waves}}
        save(folder/"parameters_and_ancestry.json", record)
        records.append(record)
        labels = {"original_parent":"Original residual parent", "fitted_parent":"V11 fitted reconstruction",
                  "group_predictive":"V12 reference draw", "spectral_context":"Spectral scope draw",
                  "spectrotemporal_context":"Spectral and envelope scope draw"}
        players = ''.join(f'<label>{labels[name]}<audio controls preload="none" src="{link(folder/(name+".wav"))}"></audio></label>' for name in waves)
        cards.append(f'<article><h2>{escape(label)} · {escape(parent[:12])}</h2>'
                     f'<p>Source validation group: {escape(fold)}. Residual parent group: {escape(bank[parent]["group"])}.</p>'
                     f'<div>{players}</div><p><a href="{link(folder/"parameters_and_ancestry.json")}">Parameters, flags, all center ancestors and replay hashes</a></p></article>')
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CAST scoped group-effect review</title><style>body{font:16px/1.5 system-ui;max-width:1200px;margin:30px auto;padding:20px;color:#16343b}article{border-top:1px solid #ccc;padding:18px 0}article div{display:flex;flex-wrap:wrap;gap:16px}label{display:grid;gap:6px}audio{max-width:100%}</style><h1>CAST scoped group-effect audio review</h1><p>Ten fixed examples: seed 42, index 0, joint draw for each source validation fold and class. Every variant uses the same direct residual parent and stochastic waveform streams. Group variants also depend on every training parent in that class through group means. No outer audio is opened.</p><p>Original and fitted audio describe a training parent. The following three clips are generated draws, not refits or reconstructions of that recording. All five unit-RMS shapes share one peak-safe playback gain within each card. Acoustic parameters are effective observation controls, not measured vehicle physics.</p><p>Local research derivatives, CC BY-NC-ND 4.0; no redistribution authorization. Five source groups and arbitrary random seeds do not establish independent vehicle generalization.</p>'''
    document += f'<p>Source-selected candidate: {escape(summary["selected"])}; criteria {"PASS" if selected["passes"] else "FAIL"}. <a href="{link(source/"summary.json")}">All source results</a>; <a href="{link(source/"audit_verification.json")}">exact replay audit</a>.</p>'
    (dest/"index.html").write_text(document+''.join(cards)+"</html>")
    save(dest/"summary.json", {"outer_access":False, "source_summary_sha256":sha(source/"summary.json"),
         "source_audit_sha256":sha(source/"audit_verification.json"), "example_count":len(records),
         "examples":records, "code_sha256":sha(Path(__file__)), "seconds":time.perf_counter()-started,
         "files":{str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob("*")) if p.is_file()}})
    print(json.dumps({"output":str(dest.relative_to(ROOT)), "examples":len(records)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-id", required=True)
    p.add_argument("--output-id", required=True)
    a = p.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", x) for x in (a.eval_id,a.output_id)):
        p.error("Use simple IDs")
    build(HERE/"evaluations"/a.eval_id, HERE/"diagnostics"/a.output_id)

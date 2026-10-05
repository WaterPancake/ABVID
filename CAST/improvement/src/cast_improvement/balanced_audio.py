"""Fixed source examples with separately identified residual parents per sampler."""
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
from .balanced_evaluation import HERE, UPSTREAM_RUN, check_source
from .resolution_evaluation import render


def build(source, dest):
    started=time.perf_counter()
    protocol,summary,selected=check_source(source)
    cfg=protocol["renderer_configs"]["group_predictive"];metric=protocol["metric_config"]
    bank={r["file_id"]:r for r in readl(source/"group_predictive_bank.jsonl")}
    examples={}
    with gzip.open(source/"generated_records.jsonl.gz","rt") as stream:
        for line in stream:
            row=json.loads(line)
            if (row["seed"],row["index"],row["arm"])==(metric["seeds"][0],0,"joint"):
                key=row["fold"],row["class"]
                if row["variant"] in examples.setdefault(key,{}):raise ValueError("Duplicate example")
                examples[key][row["variant"]]=row
    if len(examples)!=10 or any(set(rows)!=set(protocol["variants"]) for rows in examples.values()):
        raise ValueError("Incomplete fixed source examples")
    dest.mkdir(parents=True,exist_ok=False)
    def link(path):
        if not path.is_file():raise ValueError(f"Missing presentation artifact: {path}")
        return escape(os.path.relpath(path,dest),quote=True)
    records,cards=[],[]
    original_run=ROOT/"CAST/improvement/runs/cast_smooth8_v1_20261004_r1"
    for number,((fold,label),rows) in enumerate(sorted(examples.items()),1):
        folder=dest/f"example_{number:02d}";folder.mkdir()
        waves,ancestry={},{}
        for variant,row in rows.items():
            parent=row["parent_ids"][0]
            original=original_run/"fits"/parent/"original_shape.wav"
            reconstruction=UPSTREAM_RUN/"parents"/parent/"calibration_reconstruction_0.wav"
            for role,path in (("original",original),("fitted",reconstruction)):
                waves[variant+"_"+role],sr=sf.read(path,dtype="float32")
                if sr!=16000:raise ValueError("Changed source sample rate")
            wave,streams=render(row,cfg,"spectrum16")
            if hashlib.sha256(wave.tobytes()).hexdigest()!=row["waveform_samples_sha256"] or streams!=row["streams"]:
                raise ValueError("Source generated example failed replay")
            waves[variant+"_generated"]=wave
            ancestry[variant]={"residual_parent":parent,"parent_group":bank[parent]["group"],
                "bank_row_sha256":digest(bank[parent]),"original_sha256":sha(original),
                "reconstruction_sha256":sha(reconstruction),"parent_flags":bank[parent]["flags"],"generated_row":row}
        gain=.95/max(float(np.abs(x).max()) for x in waves.values())
        for name,wave in waves.items():
            path=folder/(name+".wav");sf.write(path,wave*gain,16000,subtype="FLOAT")
            actual,sr=sf.read(path,dtype="float32")
            if sr!=16000 or not np.array_equal(actual,wave*gain):raise ValueError("Playback export failed exact readback")
        record={"validation_fold":fold,"class":label,"seed":metric["seeds"][0],"index":0,
                "playback_gain":gain,"variants":ancestry,
                "audio":{name:{"path":str((folder/(name+".wav")).relative_to(ROOT)),"sha256":sha(folder/(name+".wav"))} for name in waves}}
        save(folder/"parameters_and_ancestry.json",record);records.append(record)
        sections=[]
        for variant,a in ancestry.items():
            title="V12 random draws" if variant=="group_predictive" else "Balanced draws"
            players=''.join(f'<label>{role}<audio controls preload="none" src="{link(folder/(variant+"_"+role+".wav"))}"></audio></label>' for role in ("original","fitted","generated"))
            sections.append(f'<h3>{title} · residual parent {escape(a["residual_parent"][:12])}</h3><p>{escape(a["parent_group"])}</p><div>{players}</div>')
        cards.append(f'<article><h2>{escape(label)} · source validation group {escape(fold)}</h2>'+''.join(sections)+f'<p><a href="{link(folder/"parameters_and_ancestry.json")}">Parameters, complete parent/center ancestry and replay hashes</a></p></article>')
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CAST balanced sampling review</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:30px auto;padding:20px;color:#16343b}article{border-top:1px solid #ccc;padding:18px 0}article div{display:flex;flex-wrap:wrap;gap:16px}label{display:grid;gap:6px}audio{max-width:100%}</style><h1>CAST balanced sampling source review</h1><p>Fixed examples: seed 42, index 0, joint draws for each source validation fold and class. Allocation strategies generally select different residual parents. Each original and fitted reconstruction belongs to the parent explicitly identified above its generated draw. Generated draws also use all other training parents through group centers; they are not refits or reconstructions of the displayed original. No outer observations are opened.</p><p>All six unit-RMS shapes in a card share one peak-safe playback gain. Parameters describe effective recorded observations, not recovered physical sources. Local research derivatives retain CC BY-NC-ND 4.0 restrictions; no redistribution is authorized.</p>'''
    document+=f'<p>Source selection: {escape(summary["selected"])}; criteria {"PASS" if selected["passes"] else "FAIL"}. <a href="{link(source/"summary.json")}">All scores</a>; <a href="{link(source/"audit_verification.json")}">full replay</a>; <a href="{link(source/"prior_tables.json")}">frozen centers and allocation plans</a>.</p>'
    (dest/"index.html").write_text(document+''.join(cards)+"</html>")
    save(dest/"summary.json",{"outer_access":False,"source_summary_sha256":sha(source/"summary.json"),
        "source_audit_sha256":sha(source/"audit_verification.json"),"examples":records,"example_count":len(records),
        "code_sha256":sha(Path(__file__)),"seconds":time.perf_counter()-started,
        "files":{str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob("*")) if p.is_file()}})
    print(json.dumps({"output":str(dest.relative_to(ROOT)),"examples":len(records)}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--eval-id",required=True);p.add_argument("--output-id",required=True);a=p.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}",x) for x in (a.eval_id,a.output_id)):p.error("Use simple IDs")
    build(HERE/"evaluations"/a.eval_id,HERE/"diagnostics"/a.output_id)

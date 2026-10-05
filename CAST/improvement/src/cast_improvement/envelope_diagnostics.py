"""Same-parent waveform diagnostics for the audited envelope calibration."""
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
from cast_generalization.method import describe
from cast_generalization.pipeline import read, readl
from .engine import SmoothRenderer
from .envelope_evaluation import HERE, SMOOTH, check_source


def build(source,dest):
    started=time.perf_counter()
    protocol,summary,_=check_source(source)
    cfg=protocol["renderer_configs"]["mixture_envelope"]; metric=protocol["metric_config"]
    bank=readl(source/"mixture_envelope_bank.jsonl")
    before={r["file_id"]:r for r in readl(source/"mixture_only_bank.jsonl")}
    observations={r["file_id"]:r for r in readl(source/"real_training_descriptors.jsonl")}
    example_ids={min((r["file_id"] for r in bank if r["class"]==c and r["group"]==g))
                 for c in metric["class_order"] for g in sorted({r["group"] for r in bank})}
    dest.mkdir(parents=True,exist_ok=False)
    records=[]; cards=[]; audio=[]
    def link(path):
        if not path.is_file(): raise ValueError("Missing audio review link")
        return escape(os.path.relpath(path,dest),quote=True)
    for row in bank:
        folder=SMOOTH/"fits"/row["file_id"]
        fit=read(folder/"fit.json")
        renderer=SmoothRenderer(cfg,fit["input_id"])
        waves={}; descriptors={}
        for name,parent in (("mixture_only",before[row["file_id"]]),("mixture_envelope",row)):
            with torch.no_grad():
                wave=shape(renderer.render(tensors(parent["parameters"]),"check"))[0].numpy()
            waves[name]=wave[0]
            realizations=[describe(x,cfg,metric) for x in wave]
            descriptors[name]={k:np.mean([d[k] for d in realizations],axis=0).tolist() for k in metric["primary_families"]}
        observed=observations[row["file_id"]]["descriptors"]
        errors={name:{"envelope_RMSE":float(np.sqrt(np.square(np.array(desc["envelope"])-observed["envelope"]).mean())),
                      "modulation_L1":float(np.abs(np.array(desc["modulation"])-observed["modulation"]).mean())}
                for name,desc in descriptors.items()}
        records.append({"file_id":row["file_id"],"class":row["class"],"group":row["group"],"errors":errors,
                        "observed":observed,"rendered_descriptors_mean_two_checking_streams":descriptors,
                        "envelope_calibration":row["envelope_calibration"]})
        if row["file_id"] in example_ids:
            observed_wave,sr=sf.read(folder/"original_shape.wav",dtype="float32")
            if sr!=16000: raise ValueError("Unexpected sample rate")
            waves={"observed":observed_wave,**waves}
            gain=.95/max(float(np.abs(v).max()) for v in waves.values())
            target=dest/row["file_id"];target.mkdir()
            for name,wave in waves.items(): sf.write(target/(name+".wav"),gain*wave,sr,subtype="FLOAT")
            save(target/"parameters.json",row)
            players=''.join(f'<label>{name}<audio controls preload="none" src="{link(target/(name+".wav"))}"></audio></label>' for name in waves)
            cards.append(f'<article><h2>{row["class"]} · {row["file_id"][:12]}</h2><p>{row["group"]}</p><div>{players}</div><p><a href="{link(target/"parameters.json")}">Parameters, flags and ancestry</a></p></article>')
            audio.append({"file_id":row["file_id"],"common_gain":gain,"files":{p.name:sha(p) for p in target.iterdir()}})
    save(dest/"per_parent.json",records)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    stats={}
    for col,c in enumerate(metric["class_order"]):
        rs=[r for r in records if r["class"]==c]
        stats[c]={"parents":len(rs),"flags":dict(Counter(f for r in rs for f in r["envelope_calibration"]["flags"]))}
        for i,key in enumerate(("envelope_RMSE","modulation_L1")):
            a=np.array([r["errors"]["mixture_only"][key] for r in rs]);b=np.array([r["errors"]["mixture_envelope"][key] for r in rs])
            stats[c][key]={"median_before":float(np.median(a)),"median_after":float(np.median(b)),"improved":int((b<a).sum())}
            axes[i,col].scatter(a,b,s=12,alpha=.65);limit=max(float(a.max()),float(b.max()))
            axes[i,col].plot([0,limit],[0,limit],color="grey",ls="--")
            axes[i,col].set(title=f"{c.title()}: {key}",xlabel="Mixture-only reconstruction",ylabel="After envelope calibration")
            axes[i,col].xaxis.set_major_locator(MaxNLocator(nbins=4))
            axes[i,col].yaxis.set_major_locator(MaxNLocator(nbins=4))
    fig.suptitle("Same-parent calibration reconstruction; two saved checking streams\nThese observations/components informed calibration; no independent validation claim")
    fig.savefig(dest/"temporal_residuals.png",dpi=160);plt.close(fig)
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CAST temporal calibration</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:30px auto;padding:20px;color:#16343b}article{border-top:1px solid #ccc;padding:18px 0}article div{display:flex;flex-wrap:wrap;gap:16px}label{display:grid;gap:6px}audio{max-width:100%}img{max-width:100%}</style><h1>Temporal calibration review</h1><p>First ID per class and source group, independent of quality. Same-parent reconstruction only; observations and saved checking components informed calibration. Playback uses unit-RMS shapes with one common peak-safe gain per card. These are effective recorded-signal parameters, not recovered physical engine variables.</p><p>Local research derivatives, CC BY-NC-ND 4.0; no redistribution authorization.</p>'''
    document+=f'<img src="{link(dest/"temporal_residuals.png")}" alt="Temporal reconstruction errors before and after calibration">'+''.join(cards)+'</html>'
    (dest/"index.html").write_text(document)
    save(dest/"summary.json",{"classes":stats,"examples":audio,"code_sha256":sha(Path(__file__)),
         "source_summary_sha256":sha(source/"summary.json"),"audit_sha256":sha(source/"audit_verification.json"),
         "records_sha256":sha(dest/"per_parent.json"),"plot_sha256":sha(dest/"temporal_residuals.png"),
         "gallery_sha256":sha(dest/"index.html"),"elapsed_seconds":time.perf_counter()-started,
         "outer_access":False,"raw_source_audio_opened":False,"saved_training_audio_read":True})
    print(json.dumps({"output":str(dest.relative_to(ROOT)),"classes":stats}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--eval-id",required=True);p.add_argument("--output-id",required=True)
    a=p.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}",v) for v in (a.eval_id,a.output_id)):p.error("Use simple IDs")
    build(HERE/"evaluations"/a.eval_id,HERE/"diagnostics"/a.output_id)

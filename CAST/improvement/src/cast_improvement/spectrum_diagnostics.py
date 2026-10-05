"""Audited source calibration residuals, optimizer histories and audio review."""
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
from cast.config import ROOT,save,sha
from cast.renderer import tensors
from cast_generalization.method import describe
from cast_generalization.pipeline import read,readl
from .engine import SmoothRenderer
from .spectrum_evaluation import HERE,check_source
from .spectrum_bank import UPSTREAM


def build(source,dest):
    started=time.perf_counter();protocol,summary,_=check_source(source)
    run=ROOT/protocol["fit_run"];cfg=protocol["renderer_configs"]["spectrum_calibrated"];metric=protocol["metric_config"]
    bank=readl(run/"calibrated_bank.jsonl");obs={r["file_id"]:r for r in readl(run/"observations.jsonl")}
    original_run=ROOT/read(UPSTREAM/"protocol.json")["fit_run"]
    examples={min(r["file_id"] for r in bank if r["class"]==c and r["group"]==g)
              for c in metric["class_order"] for g in sorted({r["group"] for r in bank})}
    dest.mkdir(parents=True,exist_ok=False);records=[];audio=[];cards=[]
    def link(path):
        if not path.is_file():raise ValueError("Missing diagnostic artifact")
        return escape(os.path.relpath(path,dest),quote=True)
    for row in bank:
        fit=row["spectral_calibration"];folder=run/"parents"/row["file_id"]
        renderer=SmoothRenderer(cfg,fit["input_id"])
        with torch.no_grad():before=shape(renderer.render(tensors(fit["initial_parameters"]),"check"))[0].numpy()
        after=[]
        for i in range(2):
            x,sr=sf.read(folder/f"calibration_reconstruction_{i}.wav",dtype="float32")
            if sr!=16000:raise ValueError("Changed reconstruction sample rate")
            after.append(x)
        target=obs[row["file_id"]]["descriptors"];errors={}
        for name,waves in (("v8_reference",before),("spectrum_calibrated",after)):
            values=[describe(x,cfg,metric) for x in waves]
            ds={k:np.mean([d[k] for d in values],axis=0) for k in metric["primary_families"]}
            errors[name]={"log_spectrum_RMSE":float(np.sqrt(np.square(ds["log_spectrum"]-target["log_spectrum"]).mean())),
                          "band_L1":float(np.abs(ds["bands"]-target["bands"]).sum()),
                          "envelope_RMSE":float(np.sqrt(np.square(ds["envelope"]-target["envelope"]).mean())),
                          "modulation_L1":float(np.abs(ds["modulation"]-target["modulation"]).mean())}
        records.append({"file_id":row["file_id"],"class":row["class"],"group":row["group"],"errors":errors,
                        "initial_objective":fit["initial_objective"],"best_objective":fit["best_objective"],
                        "best_step":fit["best_step"],"history":fit["history"],"flags":fit["flags"],
                        "initial_harmonic_fraction":fit["initial_parameters"]["harmonic_fraction"],
                        "final_harmonic_fraction":fit["parameters"]["harmonic_fraction"]})
        if row["file_id"] in examples:
            original,sr=sf.read(original_run/"fits"/row["file_id"]/"original_shape.wav",dtype="float32")
            if sr!=16000:raise ValueError("Changed observation sample rate")
            waves={"observed":original,"v8_reference":before[0],"spectrum_calibrated":after[0]}
            gain=.95/max(float(np.abs(x).max()) for x in waves.values())
            target_folder=dest/row["file_id"];target_folder.mkdir()
            for name,x in waves.items():sf.write(target_folder/(name+".wav"),gain*x,sr,subtype="FLOAT")
            save(target_folder/"parameters.json",row)
            players=''.join(f'<label>{name}<audio controls preload="none" src="{link(target_folder/(name+".wav"))}"></audio></label>' for name in waves)
            flags=', '.join(fit["flags"]) or 'none'
            cards.append(f'<article><h2>{row["class"]} · {row["file_id"][:12]}</h2><p>{row["group"]}</p><div>{players}</div><p>Calibration flags: {escape(flags)}. <a href="{link(target_folder/"parameters.json")}">Parameters, all optimizer steps and ancestry</a></p></article>')
            audio.append({"file_id":row["file_id"],"common_gain":gain,"files":{p.name:sha(p) for p in target_folder.iterdir()}})
    save(dest/"per_parent.json",records)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True);stats={}
    for i,c in enumerate(metric["class_order"]):
        rs=[r for r in records if r["class"]==c]
        stats[c]={"parents":len(rs),"flags":dict(Counter(f for r in rs for f in r["flags"])),
                  "median_initial_fraction":float(np.median([r["initial_harmonic_fraction"] for r in rs])),
                  "median_calibrated_fraction":float(np.median([r["final_harmonic_fraction"] for r in rs])),
                  "median_initial_objective":float(np.median([r["initial_objective"] for r in rs])),
                  "median_best_objective":float(np.median([r["best_objective"] for r in rs]))}
        history=np.array([r["history"] for r in rs]);q=np.quantile(history,[.1,.5,.9],axis=0)
        axes[i,0].fill_between(np.arange(151),q[0],q[2],alpha=.2,color="#096b72")
        axes[i,0].plot(q[1],color="#096b72");axes[i,0].set(xlabel="Calibration step",ylabel="Expected log-spectrum objective",title=c.title(),yscale="log")
        for j,key in enumerate(("log_spectrum_RMSE","band_L1","envelope_RMSE","modulation_L1")):
            a=np.array([r["errors"]["v8_reference"][key] for r in rs]);b=np.array([r["errors"]["spectrum_calibrated"][key] for r in rs])
            stats[c][key]={"median_before":float(np.median(a)),"median_after":float(np.median(b)),"improved":int((b<a).sum())}
            if j<2:
                ax=axes[i,j+1];ax.scatter(a,b,s=12,alpha=.65);limit=max(float(a.max()),float(b.max()))
                ax.plot([0,limit],[0,limit],color="grey",ls="--");ax.set(title=key,xlabel="V8 reconstruction",ylabel="After spectral calibration")
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4));ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    fig.suptitle("Training calibration: expected-power objective and actual waveform residuals\nChecking streams and observations informed calibration; no independent validation claim")
    fig.savefig(dest/"spectral_residuals.png",dpi=160);plt.close(fig)
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CAST spectral calibration</title><style>body{font:16px/1.5 system-ui;max-width:1200px;margin:30px auto;padding:20px;color:#16343b}article{border-top:1px solid #ccc;padding:18px 0}article div{display:flex;flex-wrap:wrap;gap:16px}label{display:grid;gap:6px}audio{max-width:100%}img{max-width:100%}</style><h1>Joint spectral calibration review</h1><p>First sorted ID per class and source group, independent of quality. These are training reconstructions: observed descriptors and checking streams informed calibration. Playback uses unit-RMS shapes with a common peak-safe gain in each card. Parameters describe effective recorded observations, not physical engine/tire components.</p><p>Local research derivatives, CC BY-NC-ND 4.0; no redistribution authorization.</p>'''
    document+=f'<img src="{link(dest/"spectral_residuals.png")}" alt="Optimizer histories and spectral reconstruction residuals">'+''.join(cards)+'</html>'
    (dest/"index.html").write_text(document)
    save(dest/"summary.json",{"classes":stats,"examples":audio,"code_sha256":sha(Path(__file__)),
        "source_summary_sha256":sha(source/"summary.json"),"source_audit_sha256":sha(source/"audit_verification.json"),
        "calibration_audit_sha256":sha(run/"calibration_verification.json"),"records_sha256":sha(dest/"per_parent.json"),
        "plot_sha256":sha(dest/"spectral_residuals.png"),"gallery_sha256":sha(dest/"index.html"),
        "elapsed_seconds":time.perf_counter()-started,"outer_access":False,"raw_source_audio_opened":False,"saved_training_audio_read":True})
    print(json.dumps({"output":str(dest.relative_to(ROOT)),"classes":stats}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--eval-id",required=True);p.add_argument("--output-id",required=True)
    a=p.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}",v) for v in (a.eval_id,a.output_id)):p.error("Use simple IDs")
    build(HERE/"evaluations"/a.eval_id,HERE/"diagnostics"/a.output_id)

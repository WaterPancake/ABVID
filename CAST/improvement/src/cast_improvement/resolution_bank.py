"""Frozen calibration bank with full exact fit replay and local ancestry."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import soundfile as sf
import torch

from cast.audio import shape
from cast.config import ROOT,canonical,digest,save,sha
from cast.provenance import snapshot
from cast.renderer import tensors,validate_controls
from .resolution_sampling import sample
from cast_generalization.pipeline import read,readl,jsonl
from .data import setup_cpu
from .resolution_model import ResolutionRenderer as SmoothRenderer, model_config, lift_controls
from .resolution_calibration import fit,SETTINGS
from .temporal_evaluation import frozen as previous_frozen, sources as previous_sources
from .temporal_prior import draw as previous_draw

HERE=ROOT/"CAST/improvement/resolution_v11"
UPSTREAM=ROOT/"CAST/improvement/temporal_v10/evaluations/smooth8_temporal_v10_full_source"
MODES=("temporal41","spectrum16")


def sources():
    base=ROOT/"CAST/improvement"
    names=[base/"src/cast_improvement"/x for x in (
        "resolution_model.py","resolution_calibration.py","resolution_sampling.py","resolution_bank.py","resolution_evaluation.py","capacity_engine.py")]
    names += [base/"tests/test_resolution_calibration.py",base/"tests/test_resolution_bank.py",HERE/"PROTOCOL.md",HERE/"run.sh"]
    return {**previous_sources(),**{str(p.relative_to(ROOT)):sha(p) for p in names}}


def upstream():
    protocol,summary,cfgs,banks,obs,metric=previous_frozen(UPSTREAM)
    audit=read(UPSTREAM/"audit_verification.json")
    if (not audit["passed"] or audit["summary_sha256"]!=sha(UPSTREAM/"summary.json")
        or audit["generated_waveform_descriptor_donor_calibration_replays"]!=22500
        or audit["scores_recomputed"]!=450 or not summary["candidates"][summary["selected"]]["passes"] or summary["selected"]!="temporal41_T1"):
        raise ValueError("Complete matching passing v10 source experiment required")
    bank=banks["temporal41"]
    if len(bank)!=380 or {r["file_id"] for r in bank}!={r["file_id"] for r in obs}:
        raise ValueError("Changed source parent inventory")
    return model_config(),bank,obs,metric


def fit_record(row,observation,cfg,metric,input_id):
    if any(row[k]!=observation[k] for k in ("file_id","class","group","parent")) or row["group"]==metric["held_group"]:
        raise ValueError("Spectral calibration ancestry crosses input boundary")
    target={k:observation["descriptors"][k] for k in ("log_spectrum","bands")}
    result=fit(lift_controls(row["parameters"],cfg),target,cfg,input_id,metric)
    result.update(file_id=row["file_id"],group=row["group"],class_label=row["class"],
                  initial_bank_row_sha256=digest(row),observation_sha256=digest(observation),
                  original_fit_sha256=row["fit_sha256"])
    derived=deepcopy(row);derived["parameters"]=result["parameters"]
    derived["resolution_calibration"]=result;derived["resolution_calibration_sha256"]=digest(result)
    return derived


def checking_waveforms(row,cfg):
    renderer=SmoothRenderer(cfg,row["resolution_calibration"]["input_id"])
    with torch.no_grad():
        return shape(renderer.render(tensors(row["parameters"]),"check"))[0].numpy()


def worker(task):
    row,obs,cfg,metric,input_id,folder=task
    setup_cpu();started=time.perf_counter()
    result=fit_record(row,obs,cfg,metric,input_id)
    wave=checking_waveforms(result,cfg)
    if folder is not None:
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
        save(folder/"derived_row.json",result)
        for i,x in enumerate(wave):sf.write(folder/f"calibration_reconstruction_{i}.wav",x,16000,subtype="FLOAT")
        save(folder/"runtime.json",{"elapsed_seconds":time.perf_counter()-started,"maximum_worker_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        save(folder/"hashes.json",{p.name:sha(p) for p in folder.iterdir() if p.is_file()})
    return result,wave,time.perf_counter()-started


def prepare(out):
    setup_cpu();started=time.perf_counter()
    cfg,bank,obs,metric=upstream()
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,"-m","pytest","CAST/improvement/tests","CAST/generalization/tests","CAST/tests",
             "tests/test_source_simulation.py","-q","-p","no:cacheprovider","--junitxml",str(out/"tests.xml")]
    tests=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    (out/"tests.log").write_text(tests.stdout+tests.stderr)
    save(out/"tests.json",{"command":command,"passed":tests.returncode==0})
    if tests.returncode:raise RuntimeError("Numerical/provenance tests failed before calibration")
    hashes=sources()
    input_ids={r["file_id"]:r["spectral_calibration"]["input_id"] for r in bank}
    save(out/"protocol.json",{"version":"nested_spectrum16_temporal41_v11","renderer_config":cfg,"metric_config":metric,
         "settings":SETTINGS,"upstream":str(UPSTREAM.relative_to(ROOT)),"upstream_audit_sha256":sha(UPSTREAM/"audit_verification.json"),
         "upstream_summary_sha256":sha(UPSTREAM/"summary.json"),"snapshot":snapshot(),"source_hashes":hashes,
         "raw_source_audio_opened":False,"saved_training_audio_read":True,"outer_held_access":False,
         "verified_preparation_seconds":time.perf_counter()-started,"input_ids":input_ids})
    jsonl(out/"initial_bank.jsonl",bank);jsonl(out/"observations.jsonl",obs)
    for name in hashes:
        dest=out/"source_snapshot"/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    save(out/"lock.json",{"created_before_calibration":True,"source_hashes":hashes,
         "files":{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
    return cfg,bank,obs,metric,input_ids


def frozen_inputs(out,verify_parent=True):
    setup_cpu();lock=read(out/"lock.json");protocol=read(out/"protocol.json")
    if sources()!=lock["source_hashes"] or protocol["source_hashes"]!=sources():
        raise ValueError("Frozen spectrum calibration implementation changed")
    for name,h in lock["files"].items():
        if sha(out/name)!=h:raise ValueError("Frozen spectrum calibration input changed")
    if protocol["outer_held_access"] or protocol["settings"]!=SETTINGS:
        raise ValueError("Changed calibration access or optimizer contract")
    cfg,bank,obs,metric=protocol["renderer_config"],readl(out/"initial_bank.jsonl"),readl(out/"observations.jsonl"),protocol["metric_config"]
    if verify_parent:
        if (cfg,bank,obs,metric)!=upstream():raise ValueError("Upstream calibration data changed")
        if sha(UPSTREAM/"audit_verification.json")!=protocol["upstream_audit_sha256"] or sha(UPSTREAM/"summary.json")!=protocol["upstream_summary_sha256"]:
            raise ValueError("Upstream calibration audit changed")
    return protocol,cfg,bank,obs,metric


def calibrate_bank(out,workers):
    cfg,bank,obs,metric,ids=prepare(out)
    observed={r["file_id"]:r for r in obs};started=time.perf_counter()
    tasks=[(r,observed[r["file_id"]],cfg,metric,ids[r["file_id"]],str(out/"parents"/r["file_id"])) for r in bank]
    rows=[];failures=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures={pool.submit(worker,t):t[0]["file_id"] for t in tasks}
        for future in as_completed(futures):
            try:rows.append(future.result()[0])
            except Exception as exc:
                failure={"file_id":futures[future],"error":repr(exc),"traceback":traceback.format_exc()}
                failures.append(failure);save(out/f"failure_{futures[future]}.json",failure)
            if (len(rows)+len(failures))%25==0:print(f"Spectral calibration: {len(rows)} completed, {len(failures)} failed",flush=True)
    rows.sort(key=lambda r:r["file_id"]);jsonl(out/"calibrated_bank.jsonl",rows)
    result={"passed":len(rows)==380 and not failures,"count":len(rows),"failures":failures,
            "seconds":time.perf_counter()-started,"workers":workers,"bank_sha256":sha(out/"calibrated_bank.jsonl"),
            "lock_sha256":sha(out/"lock.json"),"outer_access":False}
    save(out/"calibration_summary.json",result);print(json.dumps(result),flush=True)
    if not result["passed"]:raise RuntimeError("Incomplete spectral calibration; failures retained")


def load_bank(out,require_audit=True):
    protocol,cfg,initial,obs,metric=frozen_inputs(out)
    summary=read(out/"calibration_summary.json")
    bank=readl(out/"calibrated_bank.jsonl")
    if (not summary["passed"] or summary["count"]!=380 or summary["failures"] or len(bank)!=380
        or {r["file_id"] for r in bank}!={r["file_id"] for r in initial}
        or summary["lock_sha256"]!=sha(out/"lock.json") or summary["bank_sha256"]!=sha(out/"calibrated_bank.jsonl")):
        raise ValueError("Complete unchanged spectral calibration bank required")
    initial_by_id={r["file_id"]:r for r in initial};obs_by_id={r["file_id"]:r for r in obs}
    for row in bank:
        folder=out/"parents"/row["file_id"]
        for name,h in read(folder/"hashes.json").items():
            if sha(folder/name)!=h:raise ValueError("Calibrated parent artifact changed")
        if row!=read(folder/"derived_row.json"):raise ValueError("Calibrated bank row differs from saved parent")
        record=row["resolution_calibration"]
        if record["initial_bank_row_sha256"]!=digest(initial_by_id[row["file_id"]]) or record["observation_sha256"]!=digest(obs_by_id[row["file_id"]]):
            raise ValueError("Changed spectral calibration source ancestry")
        validate_ancestry(row)
    if require_audit:
        audited=read(out/"calibration_verification.json")
        if not audited["passed"] or audited["calibrations_replayed"]!=380 or audited["summary_sha256"]!=sha(out/"calibration_summary.json"):
            raise ValueError("Matching complete spectral calibration replay audit required")
    return cfg,{"temporal41":initial,"spectrum16":bank},obs,metric


def audit_bank(out,workers):
    cfg,banks,obs,metric=load_bank(out,require_audit=False)
    protocol=read(out/"protocol.json");observed={r["file_id"]:r for r in obs}
    saved={r["file_id"]:r for r in banks["spectrum16"]};started=time.perf_counter();count=0
    tasks=[(r,observed[r["file_id"]],cfg,metric,protocol["input_ids"][r["file_id"]],None) for r in banks["temporal41"]]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        for row,wave,_ in pool.map(worker,tasks):
            if canonical(row)!=canonical(saved[row["file_id"]]):raise ValueError("Spectral optimizer, parameters, history or ancestry replay differs")
            folder=out/"parents"/row["file_id"]
            for i,x in enumerate(wave):
                y,sr=sf.read(folder/f"calibration_reconstruction_{i}.wav",dtype="float32")
                if sr!=16000 or not np.array_equal(x,y):raise ValueError("Calibration waveform sample replay differs")
            count+=1
            if count%50==0:print(f"Spectral bank audit: {count} exact fit and waveform replays",flush=True)
    if snapshot()["tracked_diff_sha256"]!=protocol["snapshot"]["tracked_diff_sha256"]:
        raise ValueError("Tracked ABVID state changed")
    result={"passed":True,"calibrations_replayed":count,"waveforms_replayed":2*count,
            "summary_sha256":sha(out/"calibration_summary.json"),"seconds":time.perf_counter()-started,
            "tracked_ABVID_diff_unchanged":True,"outer_access":False,"audit_code_sha256":sha(Path(__file__))}
    save(out/"calibration_verification.json",result);print(json.dumps(result),flush=True)


def validate_ancestry(row):
    result=row["resolution_calibration"]
    if (any(result[a]!=row[b] for a,b in (("file_id","file_id"),("class_label","class"),("group","group")))
        or digest(result)!=row["resolution_calibration_sha256"] or result["parameters"]!=row["parameters"]
        or any(result["initial_parameters"][k]!=row["parameters"][k] for k in ("spacing_hz","harmonic_weights","envelope_knots"))):
        raise ValueError("Changed spectrum calibration parameters or ancestry")


def draw(bank,cfg,metric,label,seed,index,arm,variant,fold):
    if variant not in MODES or any(r["group"] in (fold,metric["held_group"]) for r in bank):
        raise ValueError("Undeclared spectrum variant or excluded source ancestor")
    if variant=="temporal41":return previous_draw(bank,cfg,metric,label,seed,index,arm,"temporal41",fold)
    record=sample(bank,{**metric,"version":f"cast_improvement_sampling_v1:{fold}"},label,seed,index,arm,cfg)
    record.pop("parent_flags");by_id={r["file_id"]:r for r in bank};ancestry={}
    for p in record["parent_ids"]:
        row=by_id[p];validate_ancestry(row)
        ancestry[p]={"resolution":row["resolution_calibration_sha256"],"prior_bank_row":row["resolution_calibration"]["initial_bank_row_sha256"]}
    return {**record,"calibration_ancestors":ancestry}

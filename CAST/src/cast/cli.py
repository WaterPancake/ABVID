"""No notebook state, downloads, feature caches, or classifier dependencies."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
import traceback
import numpy as np
import torch

from . import synthetic
from .artifacts import save_fit
from .audio import load_allowed, normalize
from .config import ROOT, HERE, digest, load, save, sha
from .fit import fit
from .provenance import freeze, read_frozen, snapshot


def configure(cfg):
    for line in (HERE/"requirements-lock.txt").read_text().splitlines():
        if not line or line.startswith("#"):continue
        name, version = line.split("==")
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Pinned environment mismatch: {name} requires {version}")
    torch.set_num_threads(cfg["hardware"]["torch_threads"])
    torch.set_num_interop_threads(cfg["hardware"]["interop_threads"])
    torch.use_deterministic_algorithms(True)


def peak_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value*1024)


def numerical_tests(out, cfg):
    start = time.perf_counter()
    command = [sys.executable, "-m", "pytest", str(HERE/"tests"), "-q", "-p", "no:cacheprovider", "--junitxml="+str(out/"tests.xml")]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (out/"tests.log").write_text(result.stdout+result.stderr)
    record = {"passed": result.returncode == 0, "returncode": result.returncode,
              "command": command, "elapsed_seconds": time.perf_counter()-start,
              "config_sha256": digest(cfg), "tests_xml_sha256": sha(out/"tests.xml")}
    save(out/"tests.json", record)
    print(result.stdout, flush=True)
    if result.returncode:raise RuntimeError("Numerical/provenance tests failed; real audio remains unopened")


def validate_gate(out, cfg):
    for relative in ("tests.json", "synthetic/summary.json"):
        record = json.loads((out/relative).read_text())
        if not record["passed"] or record["config_sha256"] != digest(cfg):
            raise RuntimeError("CAST-1 acceptance failed; real audio access denied")
    tests = json.loads((out/"tests.json").read_text())
    if sha(out/"tests.xml") != tests["tests_xml_sha256"]:
        raise RuntimeError("Numerical test evidence changed")


def pilot(out):
    cfg, rows, lock = read_frozen(out)
    validate_gate(out, cfg)
    folder = out/"pilot"
    folder.mkdir(exist_ok=False)
    save(folder/"gate_at_audio_access.json", {"synthetic_summary_sha256": sha(out/"synthetic/summary.json"),
                                           "tests_sha256": sha(out/"tests.json"), "config_sha256": digest(cfg),
                                           "lock_sha256": sha(out/"lock.json"), "allowed_ids": lock["selected_ids"]})
    started = time.perf_counter()
    records = []
    for index, row in enumerate(rows):
        begin = time.perf_counter()
        clip = folder/row["file_id"]
        record = {"input_id": row["file_id"], "class": row["canonical_class"], "group": row["provenance_group_id"]}
        try:
            original = load_allowed(row, cfg, rows)
            target, stats = normalize(original, cfg)
            result = fit(target, cfg, row["file_id"])
            diagnostics = save_fit(clip, original, result, cfg, {"origin": "real_calibrated_observation_reconstruction", "role": cfg["data"]["role"], "parent": row,
                                   "parent_lock_sha256": cfg["data"]["parent_lock_sha256"], "run_lock_sha256": sha(out/"lock.json")})
            record.update({"status": "completed", "fit_loss": result["fit_loss"], "check_loss": result["check_loss"],
                           "baseline_check_loss": result["baseline_check_loss"], "fit_seconds": result["fit_seconds"], **diagnostics})
        except (ValueError, RuntimeError, FloatingPointError, OSError) as e:
            clip.mkdir(exist_ok=True)
            failure = {"numerical_failure": True, "error_type": type(e).__name__, "error": str(e),
                       "traceback": traceback.format_exc(), "ancestry": row, "config_sha256": digest(cfg)}
            if not (clip/"failure.json").exists(): save(clip/"failure.json", failure)
            else: save(clip/"artifact_failure.json", failure)
            record.update({"status": "failed", "error": str(e)})
        record["elapsed_seconds"] = time.perf_counter()-begin
        record["process_peak_rss_bytes"] = peak_bytes()
        save(clip/"timing.json", {k:record[k] for k in ("elapsed_seconds", "process_peak_rss_bytes")})
        records.append(record)
        print(json.dumps({"completed": index+1, "total": len(rows), **{k:record[k] for k in ("input_id", "class", "status", "elapsed_seconds")},
                          "check_loss": record.get("check_loss"), "flags": record.get("flags")}), flush=True)
        if index == 0:
            projection = {"measured_first_clip_seconds": record["elapsed_seconds"],
                          "remaining_clips": len(rows)-1, "projected_remaining_seconds": record["elapsed_seconds"]*(len(rows)-1),
                          "projection_only": True, "synthetic_seconds": json.loads((out/"synthetic/summary.json").read_text())["elapsed_seconds"]}
            save(folder/"runtime_projection.json", projection)
            print(json.dumps(projection), flush=True)
    summary = {"records": records, "count": len(records), "completed": sum(r["status"]=="completed" for r in records),
               "failed": sum(r["status"]=="failed" for r in records), "elapsed_seconds": time.perf_counter()-started,
               "process_peak_rss_bytes": peak_bytes(), "rss_scope": "lifetime process high-water mark including synthetic stage when workflow is used",
               "fitted_audio_seconds": len(rows)*2, "config_sha256": digest(cfg),
               "domain": "source_real_in_sample_observation_reconstruction_no_classifier"}
    summary["seconds_per_fitted_second"] = summary["elapsed_seconds"]/summary["fitted_audio_seconds"]
    save(folder/"summary.json", summary)
    return summary


def verify(out):
    import soundfile as sf
    from .audio import shape
    from .fit import Objective
    from .renderer import Renderer, tensors, validate_controls
    from .provenance import checked_path
    cfg, rows, lock = read_frozen(out)
    validate_gate(out, cfg)
    summary = json.loads((out/"pilot/summary.json").read_text())
    if [r["input_id"] for r in summary["records"]] != lock["selected_ids"]:
        raise ValueError("Incomplete or altered pilot coverage")
    checked = []
    for row, rec in zip(rows, summary["records"]):
        checked_path(row, cfg, rows)  # rehash only the selected source files
        folder = out/"pilot"/row["file_id"]
        if rec["status"] == "failed":
            if not (folder/"failure.json").exists():raise ValueError("Missing failure record")
            checked.append({"id": row["file_id"], "failed_fit_preserved": True})
            continue
        hashes = json.loads((folder/"hashes.json").read_text())
        for path, expected in hashes.items():
            if sha(folder/path) != expected:raise ValueError(f"Artifact changed: {folder/path}")
        p = json.loads((folder/"parameters.json").read_text())
        if p["ancestry"]["parent"] != row or p["config_sha256"] != digest(cfg):raise ValueError("Broken reconstruction ancestry")
        validate_controls(p["effective_controls"],cfg)
        renderer = Renderer(cfg,row["file_id"])
        if renderer.streams()!=p["streams"]:raise ValueError("Broken random streams")
        with torch.no_grad(): replay = renderer.render(tensors(p["effective_controls"]),"check")
        saved,sr = sf.read(folder/"reconstructed_shape.wav",dtype="float32")
        error = float(np.max(np.abs(saved-shape(replay)[0,0].numpy())))
        if error > cfg["acceptance"]["replay_max_abs_tolerance"] or sr!=16000 or saved.shape!=(32000,):raise ValueError("Replay mismatch")
        original,_ = sf.read(folder/"original.wav",dtype="float32")
        target,_=normalize(original,cfg)
        with torch.no_grad(): loss=float(Objective(target,cfg)(replay).mean())
        if abs(loss-rec["check_loss"])>1e-5*max(loss,1):raise ValueError("Loss replay mismatch")
        gain=p["common_playback_gain"]
        for name, expected in (("original_playback",original*gain),("reconstructed_playback",saved*p["original_level"]["ac_rms"]*gain)):
            x,_=sf.read(folder/(name+".wav"),dtype="float32")
            np.testing.assert_array_equal(x,expected)
            if max(abs(x))>cfg["normalization"]["playback_peak_limit"]+1e-6:raise ValueError("Playback clipping")
        checked.append({"id":row["file_id"],"waveform_replay_max_abs":error,"loss_recomputed":loss,"artifacts_checked":len(hashes)})
    before=json.loads((out/"snapshot.json").read_text());after=snapshot()
    if before["tracked_diff_sha256"]!=after["tracked_diff_sha256"]:raise ValueError("Existing tracked ABVID work changed during CAST")
    result={"passed":True,"config_sha256":digest(cfg),"selected_sources_rehashed":len(rows),"records":checked,
            "tracked_ABVID_diff_unchanged":True,"excluded_audio_access":False,"pilot_summary_sha256":sha(out/"pilot/summary.json")}
    save(out/"verification.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["workflow","inventory","synthetic","pilot","verify","report"])
    parser.add_argument("--run-id",required=True)
    args=parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*",args.run_id):parser.error("run ID must be a simple directory name")
    out=HERE/"runs"/args.run_id
    cfg=load(HERE/"configs/pilot_v0.json")
    configure(cfg)
    started=time.perf_counter()
    if args.command in ("workflow","inventory"):
        rows=freeze(out,cfg)
        print(f"CAST-0: frozen {len(rows)} allowed IDs; no audio decoded",flush=True)
    if args.command in ("workflow","synthetic"):
        read_frozen(out)
        numerical_tests(out,cfg)
        result=synthetic.run(out/"synthetic",cfg)
        if not result["passed"]:raise RuntimeError("Synthetic acceptance failed; real audio remains unopened")
    if args.command in ("workflow","pilot"):
        pilot(out)
    if args.command in ("workflow","verify"):
        verify(out)
    if args.command in ("workflow","report"):
        from .report import write_report
        write_report(out)
    if args.command=="workflow":
        save(out/"workflow_timing.json",{"elapsed_seconds":time.perf_counter()-started,"process_peak_rss_bytes":peak_bytes()})
    print(f"CAST {args.command}: {out}",flush=True)


if __name__=="__main__":main()

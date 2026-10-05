"""Frozen nested-spectrum16 source comparison and exact replay."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from cast.config import ROOT, canonical, digest, save, sha
from cast.provenance import snapshot
from cast_generalization.method import compare, describe, scales, stack
from cast_generalization.pipeline import jsonl, read, readl
from .data import setup_cpu
from .resolution_bank import MODES, draw, load_bank, sources, calibrate_bank, audit_bank
from .temporal_evaluation import frozen as preceding_frozen
from .evaluate import aggregate, rendered as smooth_rendered
from .resolution_model import ResolutionRenderer
from cast.audio import shape
from cast.renderer import tensors, seed_for
import torch

SMOOTH = ROOT/"CAST/improvement/runs/cast_smooth8_v1_20261004_r1"
HERE = ROOT/"CAST/improvement/resolution_v11"


def load_banks(out, scope, include_reference=True):
    if scope != "full": raise ValueError("Spectrum resolution requires the complete source bank")
    cfg,banks,obs,metric=load_bank(out)
    old=read(ROOT/"CAST/improvement/temporal_v10/evaluations/smooth8_temporal_v10_full_source/protocol.json")["renderer_configs"]["temporal41"]
    return {"temporal41":old,"spectrum16":cfg},banks,obs,metric


def preceding_audit():
    out = ROOT/"CAST/improvement/temporal_v10/evaluations/smooth8_temporal_v10_full_source"
    _, summary, *_ = preceding_frozen(out)
    audit = read(out/"audit_verification.json")
    if (not audit["passed"] or audit["summary_sha256"] != sha(out/"summary.json")
        or audit["generated_waveform_descriptor_donor_calibration_replays"] != 22500
        or audit["scores_recomputed"] != 450
        or not summary["candidates"][summary["selected"]]["passes"] or summary["selected"] != "temporal41_T1"):
        raise ValueError("Require the completed and audited v10 source pass")
    return {"path": str(out.relative_to(ROOT)), "audit_sha256": sha(out/"audit_verification.json"),
            "summary_sha256": sha(out/"summary.json")}


def render(record, cfg, variant):
    if variant not in MODES:
        raise ValueError("Undeclared prior mode")
    if variant=="temporal41":return smooth_rendered(record,cfg,"smooth8")
    renderer=ResolutionRenderer(cfg,record["input_id"],sampling=True)
    with torch.no_grad():wave=shape(renderer.render(tensors(record["parameters"]),"sampling"))[0,0].numpy()
    return wave,{"phase":seed_for(cfg,record["input_id"],"phase","sampling"),
                 "noise":[seed_for(cfg,record["input_id"],"noise","sampling",0)],"purpose":"sampling"}


def generated_rows(train, cfg, metric, variant, fold):
    if any(r["group"] in (fold, metric["held_group"]) for r in train):
        raise ValueError("Source donor crosses validation boundary")
    scfg = deepcopy(metric)
    scfg["version"] = f"cast_improvement_sampling_v1:{fold}"
    for seed in scfg["seeds"]:
        for label in scfg["class_order"]:
            for arm in scfg["arms"]:
                for index in range(scfg["samples_per_class_per_seed_per_arm"]):
                    row = draw(train, cfg, metric, label, seed, index, arm, variant, fold)
                    x, streams = render(row, cfg, variant)
                    yield {**row, "variant": variant, "temperature": 1., "fold": fold, "streams": streams,
                           "waveform_samples_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                           "descriptors": {k: v.tolist() for k, v in describe(x, cfg, metric).items()}}


def fold_data(obs, fold, metric):
    valid, scale = {}, {}
    for label in metric["class_order"]:
        valid[label] = stack([r["descriptors"] for r in obs if r["group"] == fold and r["class"] == label], metric)
        scale[label] = scales(stack([r["descriptors"] for r in obs if r["group"] != fold and r["class"] == label], metric), metric)
    return valid, scale


def source(run, out, scope):
    setup_cpu()
    preparation_started = time.perf_counter()
    cfgs, banks, obs, metric = load_banks(run, scope, True)
    previous = preceding_audit()
    preparation_seconds = time.perf_counter()-preparation_started
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    command = [sys.executable, "-m", "pytest", "CAST/improvement/tests", "CAST/generalization/tests", "CAST/tests",
               "tests/test_source_simulation.py", "-q", "-p", "no:cacheprovider", "--junitxml", str(out/"tests.xml")]
    tested = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (out/"tests.log").write_text(tested.stdout+tested.stderr)
    save(out/"tests.json", {"command": command, "passed": tested.returncode == 0})
    if tested.returncode:
        raise RuntimeError("Test gate failed")
    protocol = {"domain": "leave_one_training_group_out_source_development", "outer_held_access": False,
                "fit_run": str(run.relative_to(ROOT)), "scope": scope, "renderer_configs": cfgs,
                "variants": list(banks), "temperatures": [1.], "metric_config": metric,
                "criteria": {"coverage": .8, "gain_each_control": .025, "spread": [.5, 2]},
                "fit_lock_sha256": sha(run/"lock.json"), "source_hashes": sources(), "snapshot": snapshot(),
                "fit_audit_sha256": sha(run/"calibration_verification.json"),
                "selection": "unchanged aggregate ranking; source-only sixteen-point spectral calibration versus exact v10 replication"}
    protocol.update(raw_source_audio_opened=False, saved_training_audio_read=True,
                    checking_components_role="calibration inputs for derived bank only")
    protocol["verified_bank_preparation_seconds"] = preparation_seconds
    save(out/"preceding_audit.json", previous)
    save(out/"protocol.json", protocol)
    for name in protocol["source_hashes"]:
        dest = out/"source_snapshot"/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
    for variant, bank in banks.items():
        jsonl(out/(variant+"_bank.jsonl"), bank)
    jsonl(out/"real_training_descriptors.jsonl", obs)
    save(out/"lock.json", {"files": {p.name: sha(p) for p in out.iterdir() if p.is_file()},
                           "created_before_generation": True, "source_hashes": sources()})
    groups = sorted({r["group"] for r in obs})
    scores, all_scales = [], {}
    with (out/"generated_records.jsonl.gz").open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
        for fold in groups:
            valid, scale = fold_data(obs, fold, metric); all_scales[fold] = scale
            for variant, bank in banks.items():
                bucket = []
                for r in generated_rows([p for p in bank if p["group"] != fold], cfgs[variant], metric, variant, fold):
                    stream.write(canonical(r)+b"\n"); bucket.append(r["descriptors"])
                    if len(bucket) == metric["samples_per_class_per_seed_per_arm"]:
                        scores.append({**{k: r[k] for k in ("fold", "variant", "temperature", "class", "arm", "seed")},
                                       "score": compare(stack(bucket, metric), valid[r["class"]], scale[r["class"]], metric)})
                        bucket = []
                if bucket:
                    raise ValueError("Incomplete sample block")
                print(json.dumps({"fold": fold, "variant": variant, "elapsed_s": time.perf_counter()-started}), flush=True)
    save(out/"scales_by_fold.json", all_scales); save(out/"scores.json", scores)
    candidates = aggregate(scores, metric)
    selected = min(candidates, key=lambda k: candidates[k]["selection_key"])
    summary = {"candidates": candidates, "selected": selected, "selected_by_source_only": True,
               "outer_held_access": False, "elapsed_seconds": time.perf_counter()-started,
               "generated_record_sha256": sha(out/"generated_records.jsonl.gz"), "evaluated_groups": groups}
    save(out/"summary.json", summary)
    save(out/"scientific_status.json", {"passes": candidates[selected]["passes"], "outer_held_access": False,
                                      "all_records_retained": True, "class_criteria": candidates[selected]["classes"]})
    print(json.dumps({"selected": selected, "candidates": candidates}), flush=True)


def frozen(out):
    setup_cpu()
    lock, protocol = read(out/"lock.json"), read(out/"protocol.json")
    if lock["source_hashes"] != sources() or protocol["source_hashes"] != sources():
        raise ValueError("Changed spectral-resolution evaluation implementation")
    for name, expected in lock["files"].items():
        if sha(out/name) != expected:
            raise ValueError("Changed frozen source input")
    cfgs, banks, obs, metric = load_banks(ROOT/protocol["fit_run"], protocol["scope"], True)
    if cfgs != protocol["renderer_configs"] or metric != protocol["metric_config"] or obs != readl(out/"real_training_descriptors.jsonl"):
        raise ValueError("Source-only comparison inputs changed")
    if any(bank != readl(out/(v+"_bank.jsonl")) for v, bank in banks.items()):
        raise ValueError("Fitted bank changed")
    if preceding_audit() != read(out/"preceding_audit.json"):
        raise ValueError("Preceding v8 evidence changed")
    summary = read(out/"summary.json")
    if sha(out/"generated_records.jsonl.gz") != summary["generated_record_sha256"]:
        raise ValueError("Generated records changed")
    if protocol["outer_held_access"] or summary["outer_held_access"] or protocol["temperatures"] != [1.]:
        raise ValueError("Source-only scope changed")
    candidates = aggregate(read(out/"scores.json"), metric)
    if candidates != summary["candidates"] or min(candidates, key=lambda k: candidates[k]["selection_key"]) != summary["selected"]:
        raise ValueError("Source ranking changed")
    return protocol, summary, cfgs, banks, obs, metric


def check_source(out):
    protocol, summary, *_ = frozen(out)
    audit = read(out/"audit_verification.json")
    if not audit["passed"] or audit["summary_sha256"] != sha(out/"summary.json"):
        raise ValueError("Source figures require matching exact audit")
    return protocol, summary, summary["candidates"][summary["selected"]]


def audit(out):
    started = time.perf_counter()
    protocol, summary, cfgs, banks, obs, metric = frozen(out)
    scores, all_scales, count = [], {}, 0
    groups = sorted({r["group"] for r in obs})
    with gzip.open(out/"generated_records.jsonl.gz", "rb") as stream:
        for fold in groups:
            valid, scale = fold_data(obs, fold, metric); all_scales[fold] = scale
            for variant in protocol["variants"]:
                bucket = []
                for expected in generated_rows([r for r in banks[variant] if r["group"] != fold], cfgs[variant], metric, variant, fold):
                    line = stream.readline()
                    if not line or canonical(json.loads(line)) != canonical(expected):
                        raise ValueError("Exact parameter/donor/waveform/descriptor or full schedule replay differs")
                    count += 1; bucket.append(expected["descriptors"])
                    if len(bucket) == metric["samples_per_class_per_seed_per_arm"]:
                        scores.append({**{k: expected[k] for k in ("fold", "variant", "temperature", "class", "arm", "seed")},
                                       "score": compare(stack(bucket, metric), valid[expected["class"]], scale[expected["class"]], metric)})
                        bucket = []
                print(f"Spectrum source audit: {count} exact replays", flush=True)
        if stream.readline():
            raise ValueError("Unexpected extra generated record")
    if scores != read(out/"scores.json") or all_scales != read(out/"scales_by_fold.json"):
        raise ValueError("Independent source scores/scales differ")
    if snapshot()["tracked_diff_sha256"] != protocol["snapshot"]["tracked_diff_sha256"]:
        raise ValueError("Tracked ABVID state changed")
    reference = ROOT/"CAST/improvement/temporal_v10/evaluations/smooth8_temporal_v10_full_source/scores.json"
    identity = [r for r in scores if r["variant"] == "temporal41"]
    original = [r for r in read(reference) if r["variant"] == "temporal41" and r["temperature"] == 1.]
    if identity != original or len(identity) != 150:
        raise ValueError("V10 identity does not reproduce the prior audited source scores")
    save(out/"identity_replication.json", {"passed": True, "scores_exact": len(identity),
         "reference": str(reference.relative_to(ROOT)), "reference_sha256": sha(reference)})
    result = {"passed": True, "generated_waveform_descriptor_donor_calibration_replays": count, "calibration_parent_inputs_and_banks_verified": len(banks["spectrum16"]), "exact_optimizer_replay_audit_sha256": sha(ROOT/protocol["fit_run"]/"calibration_verification.json"), "scores_recomputed": len(scores),
              "summary_sha256": sha(out/"summary.json"), "audit_code_sha256": sha(Path(__file__)),
              "all_fold_ancestry_disjoint": True, "scales_recomputed_from_fold_training_only": True,
              "complete_schedule_replayed": True, "outer_held_audio_access": False,
              "tracked_ABVID_diff_unchanged": True, "elapsed_seconds": time.perf_counter()-started}
    save(out/"audit_verification.json", result); print(json.dumps(result), flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["calibrate","audit-bank","source","audit-source"])
    p.add_argument("--run-id");p.add_argument("--eval-id");p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    if any(v and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}",v) for v in (a.run_id,a.eval_id)):
        p.error("Use simple IDs")
    if not 1<=a.workers<=4:p.error("Use one to four CPU workers")
    if a.command in ("calibrate","audit-bank","source") and not a.run_id:p.error("This stage requires --run-id")
    if a.command in ("source","audit-source") and not a.eval_id:p.error("This stage requires --eval-id")
    run=HERE/"runs"/a.run_id if a.run_id else None
    out=HERE/"evaluations"/a.eval_id if a.eval_id else None
    if a.command=="calibrate":calibrate_bank(run,a.workers)
    elif a.command=="audit-bank":
        if (run/"calibration_verification.json").exists():p.error("Preserve the existing calibration audit")
        audit_bank(run,a.workers)
    elif a.command=="source":source(run,out,"full")
    else:
        if (out/"audit_verification.json").exists():p.error("Preserve the existing source audit")
        audit(out)


if __name__=="__main__":main()

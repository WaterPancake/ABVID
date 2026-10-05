"""Conditional outer development check for the frozen spectral-resolution bank."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import resource
import shutil
import sys
import time

import numpy as np
import soundfile as sf
import torch

from cast.config import ROOT, canonical, digest, save, sha
from cast.provenance import snapshot
from cast_generalization.method import compare, describe, sample, stack
from cast_generalization.pipeline import file_hashes, jsonl, read, readl, frozen as prior_frozen, decode_held
from .data import PREVIOUS, PILOT, now, setup_cpu
from .resolution_evaluation import HERE, load_banks, frozen as source_frozen, sources as temporal_sources, render as rendered
from .resolution_bank import draw
FIT_ROOT = ROOT/"CAST/improvement"
from .evaluate import aggregate


def check_source(source_out):
    protocol, summary, *_ = source_frozen(source_out)
    return protocol, summary, summary["candidates"][summary["selected"]]


def score_records(generated, held, train, scale, cfg, variant, temperature):
    validate_schedule(generated, cfg)
    results = []
    references = {}
    for c in cfg["class_order"]:
        h = stack([r["descriptors"] for r in held if r["class"] == c], cfg)
        if len(h["bands"]) != cfg["held_counts"][c]:
            raise ValueError("Wrong held class count")
        references[c] = compare(stack([r["descriptors"] for r in train if r["class"] == c], cfg), h, scale[c], cfg)
        for arm in cfg["arms"]:
            for seed in cfg["seeds"]:
                selected = [r for r in generated if (r["class"], r["arm"], r["seed"]) == (c, arm, seed)]
                if len(selected) != cfg["samples_per_class_per_seed_per_arm"]:
                    raise ValueError("Wrong generated class/seed count")
                results.append({"variant": variant, "temperature": temperature, "class": c, "arm": arm, "seed": seed,
                    "score": compare(stack([r["descriptors"] for r in selected], cfg), h, scale[c], cfg)})
    summary = next(iter(aggregate(results, cfg).values()))
    return {"summary": summary, "records": results, "real_training_reference": references,
            "domain": "real_calibrated_synthetic_vs_exposed_held_IDMT_development_group", "independent_confirmation": False,
            "held_group": cfg["held_group"], "held_groups": 1, "group_result_equals_worst_group": True}


def validate_schedule(records, cfg):
    """A repeated index cannot stand in for a missing generated example."""
    expected = {(c, arm, seed, i) for c in cfg["class_order"] for arm in cfg["arms"]
                for seed in cfg["seeds"] for i in range(cfg["samples_per_class_per_seed_per_arm"])}
    actual = [(r["class"], r["arm"], r["seed"], r["index"]) for r in records]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Generated schedule incomplete, duplicated or outside frozen budget")


def require_source_pass(source_out, candidate):
    if not candidate["passes"]:
        raise ValueError("Selected source candidate fails declared criteria; inspect training residuals before outer access")
    audit = read(source_out/"audit_verification.json")
    if not audit["passed"] or audit["summary_sha256"] != sha(source_out/"summary.json") or audit["outer_held_audio_access"]:
        raise ValueError("Source audit missing, changed or invalid")


def run(source_out, destination):
    setup_cpu()
    started = time.perf_counter()
    protocol, source_summary, candidate = check_source(source_out)
    require_source_pass(source_out, candidate)
    if candidate["variant"] != "spectrum16":
        raise ValueError("Reference temporal41 retained; preserve its existing outer evaluation")
    prior_frozen(PREVIOUS)
    fit_out = ROOT/protocol["fit_run"]
    cfgs, banks, observations, metric_cfg = load_banks(fit_out, "full")
    variant, temperature = candidate["variant"], candidate["temperature"]
    cfg, bank = cfgs[variant], banks[variant]
    if bank != readl(source_out/(variant+"_bank.jsonl")):
        raise ValueError("Bank changed after source selection")
    destination.mkdir(parents=True, exist_ok=False)
    sampling_cfg = deepcopy(metric_cfg)
    sampling_cfg["version"] = "cast_improvement_sampling_v1:outer"
    sampling_cfg["criteria"]["relative_W1_gain_over_each_control_each_class"] = .025
    save(destination/"config.resolved.json", {"sampling": sampling_cfg, "renderer": cfg, "variant": variant, "temperature": temperature,
        "target": {"coverage_each_class": .8, "relative_W1_gain_each_control_each_class": .025, "family_spread": [.5, 2.]},
        "source_evaluation": str(source_out.relative_to(ROOT)), "source_selection_sha256": sha(source_out/"summary.json"),
        "source_audit_sha256": sha(source_out/"audit_verification.json"),
        "fit_run": str(fit_out.relative_to(ROOT)), "train_scope": protocol["scope"],
        "held_exposure": "H1 and CAST v0 development; no fresh confirmation", "no_held_fitting_or_parameter_selection": True,
        "control_scope": "Independent scalar coordinates in the 69-coordinate spectral representation; not a representation-invariant baseline"})
    jsonl(destination/"parameter_bank.jsonl", bank)
    jsonl(destination/"training_descriptors.jsonl", observations)
    shutil.copyfile(PREVIOUS/"scales.json", destination/"scales.json")
    shutil.copyfile(PREVIOUS/"held_manifest.jsonl", destination/"held_manifest.jsonl")
    save(destination/"snapshot.json", snapshot())
    source_files = [Path(__file__), HERE/"outer.sh", FIT_ROOT/"tests/test_resolution_outer.py"]
    own = {**temporal_sources(), **{str(p.relative_to(ROOT)):sha(p) for p in source_files}}
    for name in own:
        target = destination/"source_snapshot"/name;target.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(ROOT/name,target)
    complete = read(PREVIOUS/"held.complete.json")
    save(destination/"lock.json", {"created_utc": now(), "source_hashes": own,
        "files": {p.name: sha(p) for p in destination.iterdir() if p.is_file()},
        "expected_held_descriptor_sha256": complete["descriptor_sha256"], "old_held_complete_sha256": sha(PREVIOUS/"held.complete.json"),
        "old_generalization_verification_sha256": sha(PREVIOUS/"verification.json"), "source_selection_sha256": sha(source_out/"summary.json"),
        "old_generalization_scores_sha256": sha(PREVIOUS/"scores.json"), "generated_before_held_read": True})
    schedule = [draw(bank, cfg, sampling_cfg, c, seed, i, arm, variant, "outer")
                for arm in sampling_cfg["arms"] for seed in sampling_cfg["seeds"] for c in sampling_cfg["class_order"]
                for i in range(sampling_cfg["samples_per_class_per_seed_per_arm"])]
    validate_schedule(schedule, sampling_cfg)
    jsonl(destination/"schedule.jsonl", schedule)
    save(destination/"schedule.lock.json", {"sha256": sha(destination/"schedule.jsonl"), "run_lock_sha256": sha(destination/"lock.json"), "created_utc": now()})
    records = []
    generation_start = time.perf_counter()
    for index, r in enumerate(schedule):
        if sampling_cfg["held_group"] in r["parent_groups"]:
            raise ValueError("Held parent in sampler")
        wave, streams = rendered(r, cfg, variant)
        folder = destination/"generated"/r["arm"]/str(r["seed"])/r["class"]/f'{r["index"]:04d}'
        folder.mkdir(parents=True)
        gain = .95/max(float(np.abs(wave).max()), 1e-12)
        sf.write(folder/"shape.wav", wave, 16000, subtype="FLOAT")
        sf.write(folder/"playback.wav", wave*gain, 16000, subtype="FLOAT")
        rec = {**r, "variant": variant, "temperature": temperature, "bank_sha256": sha(destination/"parameter_bank.jsonl"),
               "run_lock_sha256": sha(destination/"lock.json"), "config_sha256": sha(destination/"config.resolved.json"),
               "flags_and_complete_ancestry": "parameter_bank.jsonl indexed by parent_ids; all ambiguity flags retained",
               "streams": streams, "path": str(folder.relative_to(destination)), "playback_gain": gain,
               "waveform_sha256": sha(folder/"shape.wav"), "playback_sha256": sha(folder/"playback.wav"),
               "descriptors": {k: v.tolist() for k,v in describe(wave, cfg, sampling_cfg).items()}}
        save(folder/"sample.json", rec);records.append(rec)
        if (index+1) % 250 == 0:
            print(f"Frozen candidate: generated {index+1}/1500 before held descriptor read", flush=True)
    jsonl(destination/"generated_manifest.jsonl", records)
    save(destination/"generation.complete.json", {"completed_utc": now(), "count": len(records), "manifest_sha256": sha(destination/"generated_manifest.jsonl"),
        "artifacts": file_hashes(destination/"generated"), "seconds": time.perf_counter()-generation_start})
    # First access to actual held descriptors in this new comparison.
    save(destination/"held_access_receipt.json", {"started_utc": now(), "generation_complete_sha256": sha(destination/"generation.complete.json"),
        "run_lock_sha256": sha(destination/"lock.json"), "source_selection_sha256": sha(source_out/"summary.json"), "role": "exposed development"})
    if sha(PREVIOUS/"held_descriptors.jsonl") != complete["descriptor_sha256"]:
        raise ValueError("Held descriptor cache drift")
    held = readl(PREVIOUS/"held_descriptors.jsonl")
    selected_held = readl(destination/"held_manifest.jsonl")
    if [r["file_id"] for r in held] != [r["file_id"] for r in selected_held]:
        raise ValueError("Held cache IDs not exact frozen group")
    jsonl(destination/"held_descriptors.jsonl", held)
    scores = score_records(records, held, observations, read(destination/"scales.json"), sampling_cfg, variant, temperature)
    save(destination/"scores.json", scores)
    save(destination/"failure.json", {"numerical_failure": False, "scientific_goal_passed": scores["summary"]["passes"],
        "class_criteria": {c:r["passes"] for c,r in scores["summary"]["classes"].items()}, "dropped_records": 0})
    save(destination/"timing.json", {"through_scoring_seconds": time.perf_counter()-started,
        "generation_seconds": read(destination/"generation.complete.json")["seconds"], "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)})
    print(json.dumps(scores["summary"]), flush=True)
    return destination


def verify(out):
    started = time.perf_counter()
    setup_cpu()
    lock, cfg = read(out/"lock.json"), read(out/"config.resolved.json")
    for name, expected in lock["source_hashes"].items():
        if sha(ROOT/name) != expected:
            raise ValueError("Outer comparison implementation changed")
    for name, expected in lock["files"].items():
        if sha(out/name) != expected:
            raise ValueError("Frozen input changed")
    source = ROOT/cfg["source_evaluation"]
    _, _, selected = check_source(source)
    require_source_pass(source, selected)
    if sha(source/"audit_verification.json") != cfg["source_audit_sha256"]:
        raise ValueError("Source replay evidence changed")
    if sha(source/"summary.json") != lock["source_selection_sha256"] or (selected["variant"],selected["temperature"]) != (cfg["variant"],cfg["temperature"]):
        raise ValueError("Source candidate changed")
    bank = readl(out/"parameter_bank.jsonl")
    current_cfgs,current_banks,_,_=load_banks(ROOT/cfg["fit_run"],"full")
    if current_banks[cfg["variant"]]!=bank or current_cfgs[cfg["variant"]]!=cfg["renderer"]:
        raise ValueError("Calibrated outer bank changed after source selection")
    gen = readl(out/"generated_manifest.jsonl")
    schedule = readl(out/"schedule.jsonl")
    done = read(out/"generation.complete.json")
    schedule_lock = read(out/"schedule.lock.json")
    receipt = read(out/"held_access_receipt.json")
    if schedule_lock["sha256"] != sha(out/"schedule.jsonl") or schedule_lock["run_lock_sha256"] != sha(out/"lock.json"):
        raise ValueError("Frozen schedule changed")
    if not lock["created_utc"] <= schedule_lock["created_utc"] <= done["completed_utc"] <= receipt["started_utc"] or receipt["generation_complete_sha256"] != sha(out/"generation.complete.json"):
        raise ValueError("Generation/access sequence not verified")
    if len(gen) != 1500 or len(schedule) != 1500 or done["manifest_sha256"] != sha(out/"generated_manifest.jsonl") or done["artifacts"] != file_hashes(out/"generated"):
        raise ValueError("Generation completeness/hashes failed")
    for scheduled, r in zip(schedule, gen):
        chosen = draw(bank, cfg["renderer"], cfg["sampling"], r["class"], r["seed"], r["index"], r["arm"], cfg["variant"], "outer")
        if canonical(chosen)!=canonical(scheduled) or any(canonical(r[k])!=canonical(v) for k,v in chosen.items()):
            raise ValueError("Sampling/ancestry replay failed")
        if r != read(out/r["path"]/"sample.json") or r["bank_sha256"] != sha(out/"parameter_bank.jsonl") or r["run_lock_sha256"] != sha(out/"lock.json") or r["config_sha256"] != sha(out/"config.resolved.json"):
            raise ValueError("Generated sidecar provenance failed")
        x, streams=rendered(r,cfg["renderer"],cfg["variant"])
        stored,sr=sf.read(out/r["path"]/"shape.wav",dtype="float32")
        played,sr2=sf.read(out/r["path"]/"playback.wav",dtype="float32")
        if sr!=16000 or sr2!=16000 or not np.array_equal(stored,x) or not np.array_equal(played,x*r["playback_gain"]) or streams!=r["streams"]:
            raise ValueError("Generated waveform/stream replay failed")
        if {k:v.tolist() for k,v in describe(x,cfg["renderer"],cfg["sampling"]).items()}!=r["descriptors"]:
            raise ValueError("Generated descriptors changed")
    held=readl(out/"held_descriptors.jsonl");rows=readl(out/"held_manifest.jsonl")
    if len(held) != 570 or [r["file_id"] for r in held] != [r["file_id"] for r in rows]:
        raise ValueError("Held IDs incomplete or reordered")
    base=read(PILOT/"config.resolved.json")
    for row, r in zip(rows,held):
        if r["class"] != row["canonical_class"] or r["group"] != row["provenance_group_id"] or r["source_file_sha256"] != row["source_file_sha256"]:
            raise ValueError("Held ancestry changed")
        _, x, _=decode_held(row,rows,base,cfg["sampling"])
        if {k:v.tolist() for k,v in describe(x,base,cfg["sampling"]).items()}!=r["descriptors"]:
            raise ValueError("Raw held preprocessing/descriptors replay failed")
    actual=score_records(gen,held,readl(out/"training_descriptors.jsonl"),read(out/"scales.json"),cfg["sampling"],cfg["variant"],cfg["temperature"])
    if actual!=read(out/"scores.json"):
        raise ValueError("Scores failed independent recomputation")
    if sha(PREVIOUS/"scores.json")!=lock["old_generalization_scores_sha256"] or sha(PREVIOUS/"verification.json")!=lock["old_generalization_verification_sha256"]:
        raise ValueError("Historical baseline changed")
    prior_frozen(PREVIOUS)
    if read(out/"snapshot.json")["tracked_diff_sha256"]!=snapshot()["tracked_diff_sha256"]:
        raise ValueError("Tracked ABVID work changed")
    return {"passed":True,"generated_bit_exact":len(gen),"held_raw_descriptor_replays":len(held),"scores_recomputed":True,
        "all_ancestry_training_only":True,"source_selection_recomputed":True,"historical_baseline_unchanged":True,
        "tracked_ABVID_diff_unchanged":True,"goal_passed":actual["summary"]["passes"],
        "elapsed_seconds":time.perf_counter()-started,
        "peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run','verify'])
    p.add_argument('--source-eval-id');p.add_argument('--eval-id',required=True)
    a=p.parse_args()
    if any(x and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}',x) for x in (a.source_eval_id,a.eval_id)):
        p.error('Use simple run IDs')
    out=HERE/'evaluations'/a.eval_id
    if a.command=='run':
        if not a.source_eval_id:p.error('--source-eval-id required')
        run(HERE/'evaluations'/a.source_eval_id,out)
    else:
        if (out/"verification.json").exists():p.error("Preserve the completed outer verification")
        result=verify(out)
        save(out/'verification.json',result)
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()

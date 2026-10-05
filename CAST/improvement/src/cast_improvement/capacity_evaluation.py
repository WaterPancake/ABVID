"""Source-only matched capacity comparison; no outer descriptor entry point."""
import argparse
from copy import deepcopy
import gzip
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch

from cast.audio import shape
from cast.config import ROOT, canonical, digest, save, sha
from cast.provenance import snapshot
from cast.renderer import seed_for, tensors
from cast_generalization.method import compare, describe, scales, stack, sample as sample_smooth
from cast_generalization.pipeline import file_hashes, jsonl, read, readl, frozen as previous_frozen
from .capacity_data import HERE, PREVIOUS, frozen as fit_frozen, setup_cpu
from .capacity_engine import CapacityRenderer, validate_controls
from .capacity_sampling import sample as sample_capacity
from .evaluate import aggregate, load_banks as smooth_banks, rendered as smooth_rendered

SMOOTH = ROOT/"CAST/improvement/runs/cast_smooth8_v1_20261004_r1"
VARIANT = "capacity16x9"


def sources():
    base = ROOT/"CAST/improvement"
    paths = [base/"src/cast_improvement"/s for s in (
        "capacity_evaluation.py", "capacity_sampling.py", "capacity_audit.py",
        "evaluate.py", "engine.py", "data.py")]
    paths += [base/"tests/test_capacity_sampling.py", HERE/"SOURCE_PROTOCOL.md"]
    paths += [ROOT/"CAST/generalization/src/cast_generalization"/s for s in ("method.py", "pipeline.py")]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def require_audit(out, scope, selected):
    stage = read(out/("fit_"+scope+"_summary.json"))
    audit = read(out/("fit_"+scope+"_verification.json"))
    if (not stage["passed"] or stage["failures"] or not audit["passed"]
        or audit["fit_count"] != len(selected)
        or audit["fit_summary_sha256"] != sha(out/("fit_"+scope+"_summary.json"))
        or audit["run_lock_sha256"] != sha(out/"lock.json")
        or stage["run_lock_sha256"] != sha(out/"lock.json")
        or {r["file_id"] for r in stage["records"]} != selected
        or len(stage["records"]) != len(selected)):
        raise ValueError("Complete matching fit audit required")


def load_banks(out, scope, include_smooth=False):
    cfg, rows, _ = fit_frozen(out)
    previous_frozen(PREVIOUS)
    metric = read(PREVIOUS/"config.resolved.json")
    ids = set(read(out/"pilot_ids.json")) if scope == "pilot" else {r["file_id"] for r in rows}
    require_audit(out, scope, ids)
    bank, obs = [], []
    for row in rows:
        if row["file_id"] not in ids:
            continue
        folder = out/"fits"/row["file_id"]
        actual = file_hashes(folder); actual.pop("hashes.json", None)
        if actual != read(folder/"hashes.json"):
            raise ValueError("Changed fit artifacts")
        fit = read(folder/"fit.json")
        if fit["ancestry"]["parent"] != row or fit["config_sha256"] != digest(cfg):
            raise ValueError("Changed fitted ancestry/config")
        validate_controls(fit["parameters"], cfg)
        common = {"file_id": row["file_id"], "class": row["canonical_class"],
                  "group": row["provenance_group_id"], "parent": row}
        bank.append({**common, "parameters": fit["parameters"], "flags": read(folder/"failure.json"),
                     "fit_sha256": sha(folder/"fit.json")})
        wave, rate = sf.read(folder/"original_shape.wav", dtype="float32")
        if rate != 16000 or wave.shape != (32000,):
            raise ValueError("Invalid source observation shape")
        obs.append({**common, "descriptors": {k: v.tolist() for k, v in describe(wave, cfg, metric).items()}})
    configs, banks = {VARIANT: cfg}, {VARIANT: bank}
    if include_smooth:
        require_audit(SMOOTH, scope, ids)
        old_configs, old_banks, old_obs, old_metric = smooth_banks(SMOOTH, scope)
        if old_obs != obs or old_metric != metric:
            raise ValueError("Matched variants must use identical observations and scoring")
        configs.update(old_configs); banks.update(old_banks)
    return configs, banks, obs, metric


def render(record, cfg, variant):
    if variant == "smooth8":
        return smooth_rendered(record, cfg, variant)
    if variant != VARIANT:
        raise ValueError("Undeclared renderer")
    r = CapacityRenderer(cfg, record["input_id"], sampling=True)
    with torch.no_grad():
        x = shape(r.render(tensors(record["parameters"]), "sampling"))[0, 0].numpy()
    if x.shape != (32000,) or not np.isfinite(x).all():
        raise ValueError("Invalid generated waveform")
    return x, {"phase": seed_for(cfg, record["input_id"], "phase", "sampling"),
               "noise": [seed_for(cfg, record["input_id"], "noise", "sampling", 0)], "purpose": "sampling"}


def generated_rows(train, cfg, metric, variant, fold):
    if any(r["group"] in (fold, metric["held_group"]) for r in train):
        raise ValueError("Source donor crosses validation boundary")
    scfg = deepcopy(metric)
    scfg["version"] = f"cast_improvement_sampling_v1:{fold}"
    sampler = sample_capacity if variant == VARIANT else sample_smooth
    for seed in scfg["seeds"]:
        for label in scfg["class_order"]:
            for arm in scfg["arms"]:
                for index in range(scfg["samples_per_class_per_seed_per_arm"]):
                    row = sampler(train, scfg, label, seed, index, arm, cfg)
                    row.pop("parent_flags")
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
    cfgs, banks, obs, metric = load_banks(run, scope, True)
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
                "capacity_fit_audit_sha256": sha(run/("fit_"+scope+"_verification.json")),
                "smooth_fit_audit_sha256": sha(SMOOTH/("fit_"+scope+"_verification.json")),
                "selection": "unchanged aggregate ranking; source-only matched fixed-capacity ablation"}
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
        raise ValueError("Changed capacity evaluation implementation")
    for name, expected in lock["files"].items():
        if sha(out/name) != expected:
            raise ValueError("Changed frozen source input")
    cfgs, banks, obs, metric = load_banks(ROOT/protocol["fit_run"], protocol["scope"], True)
    if cfgs != protocol["renderer_configs"] or metric != protocol["metric_config"] or obs != readl(out/"real_training_descriptors.jsonl"):
        raise ValueError("Source-only comparison inputs changed")
    if any(bank != readl(out/(v+"_bank.jsonl")) for v, bank in banks.items()):
        raise ValueError("Fitted bank changed")
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
                print(f"Capacity source audit: {count} exact replays", flush=True)
        if stream.readline():
            raise ValueError("Unexpected extra generated record")
    if scores != read(out/"scores.json") or all_scales != read(out/"scales_by_fold.json"):
        raise ValueError("Independent source scores/scales differ")
    if snapshot()["tracked_diff_sha256"] != protocol["snapshot"]["tracked_diff_sha256"]:
        raise ValueError("Tracked ABVID state changed")
    result = {"passed": True, "generated_waveform_descriptor_donor_replays": count, "scores_recomputed": len(scores),
              "summary_sha256": sha(out/"summary.json"), "audit_code_sha256": sha(Path(__file__)),
              "all_fold_ancestry_disjoint": True, "scales_recomputed_from_fold_training_only": True,
              "complete_schedule_replayed": True, "outer_held_audio_access": False,
              "tracked_ABVID_diff_unchanged": True, "elapsed_seconds": time.perf_counter()-started}
    save(out/"audit_verification.json", result); print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["source", "audit-source"])
    p.add_argument("--run-id")
    p.add_argument("--eval-id", required=True)
    p.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    a = p.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", x) for x in (a.eval_id, a.run_id or "unused")):
        p.error("Use simple IDs")
    out = HERE/"evaluations"/a.eval_id
    if a.command == "source":
        if not a.run_id:
            p.error("Source requires --run-id")
        source(HERE/"runs"/a.run_id, out, a.scope)
    else:
        if (out/"audit_verification.json").exists():
            p.error("Preserve completed source audit")
        audit(out)


if __name__ == "__main__":
    main()

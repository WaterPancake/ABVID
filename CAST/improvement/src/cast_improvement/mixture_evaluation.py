"""Frozen effective harmonic/noise allocation comparison and exact replay."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

from cast.config import ROOT, canonical, save, sha
from cast.provenance import snapshot
from cast_generalization.method import compare, describe, scales, stack
from cast_generalization.pipeline import jsonl, read, readl
from .data import setup_cpu
HERE = ROOT/"CAST/improvement/mixture_v6"
FIT_ROOT = ROOT/"CAST/improvement"
from .evaluate import load_banks, rendered
from .mixture_prior import GRID, method_id, transform_bank, draw, summarize
from .calibration_evaluation import frozen as preceding_frozen


def source_files():
    files = [Path(__file__), FIT_ROOT/"src/cast_improvement/mixture_prior.py",
             FIT_ROOT/"src/cast_improvement/block_prior.py", FIT_ROOT/"src/cast_improvement/evaluate.py",
             FIT_ROOT/"src/cast_improvement/engine.py", FIT_ROOT/"src/cast_improvement/data.py",
             HERE/"PROTOCOL.md", HERE/"run.sh", FIT_ROOT/"tests/test_mixture_prior.py",
             FIT_ROOT/"diagnostics/smooth8_component_allocation_r1/summary.json",
             FIT_ROOT/"diagnostics/smooth8_component_allocation_r1/per_parent.json",
             ROOT/"CAST/generalization/src/cast_generalization/method.py",
             ROOT/"CAST/generalization/src/cast_generalization/pipeline.py"]
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def preceding_audit():
    out = FIT_ROOT/"calibration_v5/evaluations/smooth8_bias_v5_full_source"
    _, summary, *_ = preceding_frozen(out)
    audit = read(out/"audit_verification.json")
    if (not audit["passed"] or audit["summary_sha256"] != sha(out/"summary.json")
        or audit["generated_waveform_descriptor_donor_calibration_replays"] != 60000
        or audit["scores_recomputed"] != 1200
        or any(c["passes"] for c in summary["candidates"].values())):
        raise ValueError("Require the completed, audited preceding scientific failure")
    return {"path": str(out.relative_to(ROOT)), "audit_sha256": sha(out/"audit_verification.json"),
            "summary_sha256": sha(out/"summary.json")}


def fitting_gate(out):
    configs, banks, observations, metric_cfg = load_banks(out, "full")
    verified = read(out/"fit_full_verification.json")
    if not verified["passed"] or verified["fit_count"] != 380 or verified["fit_summary_sha256"] != sha(out/"fit_full_summary.json"):
        raise ValueError("A complete matching 380-fit replay audit is required")
    return configs["smooth8"], banks["smooth8"], observations, metric_cfg


def fold_data(observations, fold, metric_cfg):
    validation, scale = {}, {}
    for label in metric_cfg["class_order"]:
        training = [r["descriptors"] for r in observations if r["group"] != fold and r["class"] == label]
        held = [r["descriptors"] for r in observations if r["group"] == fold and r["class"] == label]
        validation[label] = stack(held, metric_cfg)
        scale[label] = scales(stack(training, metric_cfg), metric_cfg)
    return validation, scale


def score_record(fold, spectral, offset, label, arm, seed, descriptors, valid, scale, metric_cfg):
    return {"fold": fold, "variant": method_id(spectral, offset), "temperature": 1.,
            "spectral_temperature": spectral, "harmonic_logit_offset": offset,
            "class": label, "arm": arm, "seed": seed,
            "score": compare(stack(descriptors, metric_cfg), valid[label], scale[label], metric_cfg)}


def source(fit_out, out):
    setup_cpu()
    cfg, bank, observations, metric_cfg = fitting_gate(fit_out)
    previous = preceding_audit()
    out.mkdir(parents=True, exist_ok=False)
    tests_cmd = [sys.executable, "-m", "pytest", "CAST/improvement/tests", "CAST/generalization/tests", "CAST/tests",
                 "tests/test_source_simulation.py", "-q", "-p", "no:cacheprovider", "--junitxml", str(out/"tests.xml")]
    tests = subprocess.run(tests_cmd, cwd=ROOT, capture_output=True, text=True)
    (out/"tests.log").write_text(tests.stdout+tests.stderr)
    save(out/"tests.json", {"command": tests_cmd, "passed": tests.returncode == 0})
    if tests.returncode:
        raise ValueError("Prior numerical/provenance test gate failed; no generation")
    started = time.perf_counter()
    hashes = source_files()
    protocol = {"version": "cast_mixture_allocation_v6", "domain": "leave_one_training_group_out_source_development",
                "fit_run": str(fit_out.relative_to(ROOT)), "scope": "full", "grid": [list(v) for v in GRID],
                "renderer_config": cfg, "metric_config": metric_cfg, "source_hashes": hashes,
                "criteria": {"coverage_each_class": .8, "gain_each_control_each_class": .025, "family_spread": [.5, 2.]},
                "selection": "spread failure, max class shortfall, mean W1, distance from identity, spectral temperature, harmonic logit offset",
                "snapshot": snapshot(), "outer_held_access": False,
                "fit_lock_sha256": sha(fit_out/"lock.json"), "fit_audit_sha256": sha(fit_out/"fit_full_verification.json")}
    save(out/"protocol.json", protocol)
    save(out/"preceding_audit.json", previous)
    jsonl(out/"parameter_bank.jsonl", bank)
    jsonl(out/"real_training_descriptors.jsonl", observations)
    for name in hashes:
        target = out/"source_snapshot"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    save(out/"lock.json", {"source_hashes": hashes,
         "files": {p.name: sha(p) for p in out.iterdir() if p.is_file()}, "created_before_generation": True})
    groups = sorted({r["group"] for r in observations})
    scores, fold_scales, prior_statistics = [], {}, {}
    with (out/"generated_records.jsonl.gz").open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            for fold in groups:
                valid, scale = fold_data(observations, fold, metric_cfg)
                fold_scales[fold], prior_statistics[fold] = scale, {}
                training = [r for r in bank if r["group"] != fold]
                for spectral, offset in GRID:
                    transformed_bank, statistics = transform_bank(training, cfg, spectral, offset)
                    prior_statistics[fold][method_id(spectral, offset)] = statistics
                    for seed in metric_cfg["seeds"]:
                        for label in metric_cfg["class_order"]:
                            for arm in metric_cfg["arms"]:
                                descriptors = []
                                for index in range(metric_cfg["samples_per_class_per_seed_per_arm"]):
                                    r = draw(transformed_bank, statistics, cfg, metric_cfg, label, seed, index, arm, fold, spectral, offset)
                                    wave, streams = rendered(r, cfg, "smooth8")
                                    desc = {k: v.tolist() for k, v in describe(wave, cfg, metric_cfg).items()}
                                    stream.write(canonical({**r, "streams": streams, "descriptors": desc,
                                         "waveform_samples_sha256": hashlib.sha256(wave.tobytes()).hexdigest()})+b"\n")
                                    descriptors.append(desc)
                                scores.append(score_record(fold, spectral, offset, label, arm, seed, descriptors, valid, scale, metric_cfg))
                    print(json.dumps({"fold": fold, "spectral_temperature": spectral, "harmonic_logit_offset": offset,
                                      "elapsed_seconds": round(time.perf_counter()-started, 2)}), flush=True)
    save(out/"scales_by_fold.json", fold_scales)
    save(out/"prior_statistics_by_fold.json", prior_statistics)
    save(out/"scores.json", scores)
    candidates = summarize(scores, metric_cfg)
    selected = min(candidates, key=lambda name: candidates[name]["selection_key"])
    save(out/"summary.json", {"candidates": candidates, "selected": selected,
         "selected_by_source_only": True, "outer_held_access": False, "evaluated_groups": groups,
         "elapsed_seconds": time.perf_counter()-started, "generated_record_sha256": sha(out/"generated_records.jsonl.gz")})
    save(out/"scientific_failure.json", {"technical_generation_completed": True, "all_records_retained": True,
         "selected_candidate_passes": candidates[selected]["passes"], "class_criteria": candidates[selected]["classes"],
         "outer_held_access": False})
    print(json.dumps({"selected": selected, "passes": candidates[selected]["passes"], "classes": candidates[selected]["classes"]}), flush=True)


def frozen(out):
    lock = read(out/"lock.json")
    if lock["source_hashes"] != source_files():
        raise ValueError("Mixture-prior implementation changed; preserve this version")
    for name, expected in lock["files"].items():
        if sha(out/name) != expected:
            raise ValueError("Mixture-prior frozen input changed")
    protocol, summary = read(out/"protocol.json"), read(out/"summary.json")
    cfg, bank, observations, metric_cfg = fitting_gate(ROOT/protocol["fit_run"])
    if preceding_audit() != read(out/"preceding_audit.json"):
        raise ValueError("Preceding calibration evidence changed")
    if cfg != protocol["renderer_config"] or metric_cfg != protocol["metric_config"] or bank != readl(out/"parameter_bank.jsonl") or observations != readl(out/"real_training_descriptors.jsonl"):
        raise ValueError("Mixture-prior bank or original descriptors changed")
    if protocol["grid"] != [list(x) for x in GRID] or protocol["outer_held_access"] or summary["outer_held_access"]:
        raise ValueError("Changed source-only candidate grid")
    candidates = summarize(read(out/"scores.json"), metric_cfg)
    if candidates != summary["candidates"] or min(candidates, key=lambda name: candidates[name]["selection_key"]) != summary["selected"]:
        raise ValueError("Mixture-prior source selection not reproducible")
    if sha(out/"generated_records.jsonl.gz") != summary["generated_record_sha256"]:
        raise ValueError("Generated source records changed")
    return protocol, summary, cfg, bank, observations, metric_cfg


def audit(out):
    setup_cpu()
    started = time.perf_counter()
    protocol, summary, cfg, bank, observations, metric_cfg = frozen(out)
    saved_scales = read(out/"scales_by_fold.json")
    saved_statistics = read(out/"prior_statistics_by_fold.json")
    groups = sorted({r["group"] for r in observations})
    actual_scores, count = [], 0
    with gzip.open(out/"generated_records.jsonl.gz", "rb") as stream:
        for fold in groups:
            valid, scale = fold_data(observations, fold, metric_cfg)
            if scale != saved_scales[fold]:
                raise ValueError("Fold scales changed or include validation inputs")
            training = [r for r in bank if r["group"] != fold]
            for spectral, offset in GRID:
                transformed_bank, statistics = transform_bank(training, cfg, spectral, offset)
                if statistics != saved_statistics[fold][method_id(spectral, offset)]:
                    raise ValueError("Class-center statistics or ancestors changed")
                for seed in metric_cfg["seeds"]:
                    for label in metric_cfg["class_order"]:
                        for arm in metric_cfg["arms"]:
                            descriptors = []
                            for index in range(metric_cfg["samples_per_class_per_seed_per_arm"]):
                                line = stream.readline()
                                if not line:
                                    raise ValueError("Missing generated source sample")
                                r = json.loads(line)
                                expected = draw(transformed_bank, statistics, cfg, metric_cfg, label, seed, index, arm, fold, spectral, offset)
                                if any(canonical(r[k]) != canonical(v) for k, v in expected.items()):
                                    raise ValueError("Source sample, schedule or direct/calibration ancestry changed")
                                wave, streams = rendered(expected, cfg, "smooth8")
                                if streams != r["streams"] or hashlib.sha256(wave.tobytes()).hexdigest() != r["waveform_samples_sha256"]:
                                    raise ValueError("Source waveform/stream replay failed")
                                desc = {k: v.tolist() for k, v in describe(wave, cfg, metric_cfg).items()}
                                if desc != r["descriptors"]:
                                    raise ValueError("Source descriptor replay failed")
                                descriptors.append(desc)
                                count += 1
                                if count % 10000 == 0:
                                    print(f"Mixture-prior audit: {count} exact waveform/descriptor/donor/calibration replays", flush=True)
                            actual_scores.append(score_record(fold, spectral, offset, label, arm, seed, descriptors, valid, scale, metric_cfg))
        if stream.readline():
            raise ValueError("Unexpected extra generated source sample")
    if actual_scores != read(out/"scores.json") or summarize(actual_scores, metric_cfg) != summary["candidates"]:
        raise ValueError("Source scores or selected candidate failed recomputation")
    if snapshot()["tracked_diff_sha256"] != protocol["snapshot"]["tracked_diff_sha256"]:
        raise ValueError("Tracked ABVID work changed")
    reference = FIT_ROOT/"evaluations/smooth8_block_v2_full_source/scores.json"
    original = [r for r in read(reference) if r["envelope_temperature"] == 1. and r["spectral_temperature"] in (.65,.8,1.)]
    identity = [{**{k:v for k,v in r.items() if k != "harmonic_logit_offset"},
                 "variant": f"smooth8_s{r['spectral_temperature']:g}_e1", "envelope_temperature": 1.}
                for r in actual_scores if r["harmonic_logit_offset"] == 0.]
    if identity != original or len(identity) != 450:
        raise ValueError("Zero-offset cases do not reproduce frozen prior-v2 reference")
    save(out/"identity_replication.json", {"passed": True, "exact_scores": len(identity),
         "reference": str(reference.relative_to(ROOT)), "reference_sha256": sha(reference)})
    result = {"passed": True, "generated_waveform_descriptor_donor_calibration_replays": count,
              "scores_recomputed": len(actual_scores), "all_fold_ancestry_disjoint": True,
              "class_centers_and_scales_recomputed_from_training_only": True, "outer_held_audio_access": False,
              "summary_sha256": sha(out/"summary.json"), "audit_code_sha256": sha(Path(__file__)),
              "tracked_ABVID_diff_unchanged": True, "elapsed_seconds": time.perf_counter()-started}
    save(out/"audit_verification.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["source", "audit"])
    parser.add_argument("--run-id")
    parser.add_argument("--eval-id", required=True)
    args = parser.parse_args()
    if any(x and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", x) for x in (args.run_id, args.eval_id)):
        parser.error("Use simple IDs")
    out = HERE/"evaluations"/args.eval_id
    if args.command == "source":
        if not args.run_id:
            parser.error("Source experiment requires --run-id")
        source(FIT_ROOT/"runs"/args.run_id, out)
    else:
        if (out/"audit_verification.json").exists():
            parser.error("Audit already recorded; preserve its original result")
        audit(out)


if __name__ == "__main__":
    main()

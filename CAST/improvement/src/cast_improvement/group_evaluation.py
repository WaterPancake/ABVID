"""Frozen source-group prior comparison and exact replay."""
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

from cast.config import ROOT, canonical, save, sha
from cast.provenance import snapshot
from cast_generalization.method import compare, describe, stack
from cast_generalization.pipeline import jsonl, read, readl
from .data import setup_cpu
from .group_prior import MODES, GroupPrior
from .resolution_evaluation import frozen as preceding_frozen, sources as preceding_sources, render
from .resolution_evaluation import fold_data
from .evaluate import aggregate

HERE = ROOT/"CAST/improvement/group_v12"
UPSTREAM = ROOT/"CAST/improvement/resolution_v11/evaluations/smooth16_temporal41_v11_full_source"
UPSTREAM_RUN = ROOT/"CAST/improvement/resolution_v11/runs/cast_resolution_v11_20261005"


def sources():
    base = ROOT/"CAST/improvement"
    names = [base/"src/cast_improvement"/x for x in ("group_prior.py", "group_evaluation.py", "group_diagnostics.py")]
    names += [base/"tests/test_group_prior.py", HERE/"PROTOCOL.md", HERE/"run.sh",
              base/"diagnostics/spectral16_source_group_variation/summary.json"]
    return {**preceding_sources(), **{str(p.relative_to(ROOT)): sha(p) for p in names}}


def preceding_audit():
    summary, audit = read(UPSTREAM/"summary.json"), read(UPSTREAM/"audit_verification.json")
    identity = read(UPSTREAM/"identity_replication.json")
    if (not audit["passed"] or audit["summary_sha256"] != sha(UPSTREAM/"summary.json")
        or audit["generated_waveform_descriptor_donor_calibration_replays"] != 15000
        or audit["scores_recomputed"] != 300 or not identity["passed"] or identity["scores_exact"] != 150
        or summary["selected"] != "spectrum16_T1" or not summary["candidates"][summary["selected"]]["passes"]):
        raise ValueError("Require the completed and audited v11 source pass")
    return {"path": str(UPSTREAM.relative_to(ROOT)), "audit_sha256": sha(UPSTREAM/"audit_verification.json"),
            "summary_sha256": sha(UPSTREAM/"summary.json"), "identity_sha256": sha(UPSTREAM/"identity_replication.json")}


def load_banks(out, scope, include_reference=True):
    if scope != "full" or Path(out) != UPSTREAM_RUN:
        raise ValueError("Group prior requires the fixed complete v11 source bank")
    _, _, cfgs, banks, obs, metric = preceding_frozen(UPSTREAM)
    preceding_audit()
    bank = banks["spectrum16"]
    if len(bank) != 380 or {r["file_id"] for r in bank} != {r["file_id"] for r in obs}:
        raise ValueError("Changed source parent inventory")
    return {v:cfgs["spectrum16"] for v in MODES}, {v:bank for v in MODES}, obs, metric


def fold_priors(cfgs, banks, obs, metric):
    return {(fold, variant): GroupPrior([r for r in bank if r["group"] != fold],
                cfgs[variant], metric, variant, fold)
            for fold in sorted({r["group"] for r in obs}) for variant, bank in banks.items()}


def prior_tables(priors):
    return {fold:{variant:model.statistics for (f,variant),model in priors.items() if f == fold}
            for fold in sorted({f for f,v in priors})}


def generated_rows(prior, cfg, metric, variant, fold):
    for seed in metric["seeds"]:
        for label in metric["class_order"]:
            for arm in metric["arms"]:
                for index in range(metric["samples_per_class_per_seed_per_arm"]):
                    row = prior.draw(label, seed, index, arm)
                    x, streams = render(row, cfg, "spectrum16")
                    yield {**row, "variant": variant, "temperature": 1., "fold": fold, "streams": streams,
                           "waveform_samples_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                           "descriptors": {k: v.tolist() for k, v in describe(x, cfg, metric).items()}}


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
                "selection": "unchanged aggregate ranking; fixed source-group priors versus exact v11 replication"}
    protocol.update(raw_source_audio_opened=False, saved_training_audio_read=True,
                    checking_components_role="upstream calibration ancestry only; no new fitting",
                    new_raw_audio_reads=False, prior_formulas="group_v12/PROTOCOL.md")
    protocol["verified_bank_preparation_seconds"] = preparation_seconds
    models = fold_priors(cfgs, banks, obs, metric)
    save(out/"prior_tables.json", prior_tables(models))
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
                for r in generated_rows(models[(fold, variant)], cfgs[variant], metric, variant, fold):
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
        raise ValueError("Changed group-prior evaluation implementation")
    for name, expected in lock["files"].items():
        if sha(out/name) != expected:
            raise ValueError("Changed frozen source input")
    cfgs, banks, obs, metric = load_banks(ROOT/protocol["fit_run"], protocol["scope"], True)
    if cfgs != protocol["renderer_configs"] or metric != protocol["metric_config"] or obs != readl(out/"real_training_descriptors.jsonl"):
        raise ValueError("Source-only comparison inputs changed")
    if any(bank != readl(out/(v+"_bank.jsonl")) for v, bank in banks.items()):
        raise ValueError("Fitted bank changed")
    if preceding_audit() != read(out/"preceding_audit.json"):
        raise ValueError("Preceding v11 evidence changed")
    if prior_tables(fold_priors(cfgs, banks, obs, metric)) != read(out/"prior_tables.json"):
        raise ValueError("Frozen training-only group statistics changed")
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
    if (not audit["passed"] or audit["summary_sha256"] != sha(out/"summary.json")
        or audit["generated_waveform_descriptor_donor_calibration_replays"] != 22500 or audit["scores_recomputed"] != 450):
        raise ValueError("Source figures require matching exact audit")
    return protocol, summary, summary["candidates"][summary["selected"]]


def audit(out):
    started = time.perf_counter()
    protocol, summary, cfgs, banks, obs, metric = frozen(out)
    scores, all_scales, count = [], {}, 0
    models = fold_priors(cfgs, banks, obs, metric)
    groups = sorted({r["group"] for r in obs})
    with gzip.open(out/"generated_records.jsonl.gz", "rb") as stream:
        for fold in groups:
            valid, scale = fold_data(obs, fold, metric); all_scales[fold] = scale
            for variant in protocol["variants"]:
                bucket = []
                for expected in generated_rows(models[(fold, variant)], cfgs[variant], metric, variant, fold):
                    line = stream.readline()
                    if not line or canonical(json.loads(line)) != canonical(expected):
                        raise ValueError("Exact parameter/donor/waveform/descriptor or full schedule replay differs")
                    count += 1; bucket.append(expected["descriptors"])
                    if len(bucket) == metric["samples_per_class_per_seed_per_arm"]:
                        scores.append({**{k: expected[k] for k in ("fold", "variant", "temperature", "class", "arm", "seed")},
                                       "score": compare(stack(bucket, metric), valid[expected["class"]], scale[expected["class"]], metric)})
                        bucket = []
                print(f"Group-prior source audit: {count} exact replays", flush=True)
        if stream.readline():
            raise ValueError("Unexpected extra generated record")
    if scores != read(out/"scores.json") or all_scales != read(out/"scales_by_fold.json"):
        raise ValueError("Independent source scores/scales differ")
    if snapshot()["tracked_diff_sha256"] != protocol["snapshot"]["tracked_diff_sha256"]:
        raise ValueError("Tracked ABVID state changed")
    reference = UPSTREAM/"scores.json"
    identity = [r for r in scores if r["variant"] == "spectrum16"]
    original = [r for r in read(reference) if r["variant"] == "spectrum16" and r["temperature"] == 1.]
    if identity != original or len(identity) != 150:
        raise ValueError("V11 identity does not reproduce the prior audited source scores")
    save(out/"identity_replication.json", {"passed": True, "scores_exact": len(identity),
         "reference": str(reference.relative_to(ROOT)), "reference_sha256": sha(reference)})
    result = {"passed": True, "generated_waveform_descriptor_donor_calibration_replays": count, "calibration_parent_inputs_and_banks_verified": len(banks["spectrum16"]), "exact_optimizer_replay_audit_sha256": sha(ROOT/protocol["fit_run"]/"calibration_verification.json"), "scores_recomputed": len(scores),
              "summary_sha256": sha(out/"summary.json"), "audit_code_sha256": sha(Path(__file__)),
              "all_fold_ancestry_disjoint": True, "group_statistics_recomputed_and_frozen_before_generation": True,
              "prior_tables_sha256": sha(out/"prior_tables.json"), "scales_recomputed_from_fold_training_only": True,
              "complete_schedule_replayed": True, "outer_held_audio_access": False,
              "tracked_ABVID_diff_unchanged": True, "elapsed_seconds": time.perf_counter()-started}
    save(out/"audit_verification.json", result); print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["source", "audit-source"])
    p.add_argument("--eval-id", required=True)
    a = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", a.eval_id):
        p.error("Use simple IDs")
    out = HERE/"evaluations"/a.eval_id
    if a.command == "source": source(UPSTREAM_RUN, out, "full")
    else:
        if (out/"audit_verification.json").exists(): p.error("Preserve the existing source audit")
        audit(out)


if __name__ == "__main__": main()

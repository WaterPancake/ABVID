"""Describe finite source-sampling variability without accessing outer inputs."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import time

import numpy as np

from cast.config import ROOT, save, sha
from cast_generalization.pipeline import read
from .group_evaluation import HERE


def build(source, dest):
    started = time.perf_counter()
    summary, audit = read(source/"summary.json"), read(source/"audit_verification.json")
    protocol = read(source/"protocol.json")
    if (not audit["passed"] or audit["summary_sha256"] != sha(source/"summary.json")
        or audit["generated_waveform_descriptor_donor_calibration_replays"] != 22500
        or audit["scores_recomputed"] != 450 or summary["outer_held_access"]
        or protocol["outer_held_access"]
        or sha(source/"generated_records.jsonl.gz") != summary["generated_record_sha256"]):
        raise ValueError("Require the complete matching source-only v12 audit")
    scores = read(source/"scores.json")
    variant = summary["candidates"][summary["selected"]]["variant"]
    metric = protocol["metric_config"]
    batches = {}
    count = 0
    with gzip.open(source/"generated_records.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line); count += 1
            if row["variant"] == variant and row["arm"] == "joint":
                key = row["fold"], row["class"], row["seed"]
                batches.setdefault(key, []).append(row)
    if count != 22500 or len(batches) != 50:
        raise ValueError("Incomplete source schedule")
    schedule = []
    for (fold, label, seed), rows in sorted(batches.items()):
        if len(rows) != 50 or {r["index"] for r in rows} != set(range(50)):
            raise ValueError("Missing source sample")
        groups = sorted(rows[0]["prior_calibration_groups"])
        direct = Counter(r["parent_groups"][0] for r in rows)
        centers = Counter(r["group_effect_choices"][0]["center_group"] for r in rows)
        signs = Counter(str(r["group_effect_choices"][0]["sign"]) for r in rows)
        schedule.append({"fold":fold, "class":label, "seed":seed,
            "distinct_direct_parents":len({r["parent_ids"][0] for r in rows}),
            "direct_group_counts":{g:direct[g] for g in groups},
            "center_group_counts":{g:centers[g] for g in groups}, "sign_counts":dict(signs),
            "expected_count_per_group":len(rows)/len(groups)})
    classes = {}
    for label in metric["class_order"]:
        folds = []
        for fold in summary["evaluated_groups"]:
            rows = [r["score"] for r in scores if (r["variant"],r["class"],r["fold"],r["arm"]) == (variant,label,fold,"joint")]
            if len(rows) != 5: raise ValueError("Missing source score seed")
            folds.append({"group":fold, **{key:{"mean":float(np.mean([r[key] for r in rows])),
                "seed_sd":float(np.std([r[key] for r in rows],ddof=1)),
                "range": [min(r[key] for r in rows),max(r[key] for r in rows)]} for key in ("W1","coverage")}})
        selected = [r for r in schedule if r["class"] == label]
        classes[label] = {"folds":folds,
            "mean_within_fold_coverage_seed_sd":float(np.mean([r["coverage"]["seed_sd"] for r in folds])),
            "direct_group_count_range":[min(v for r in selected for v in r["direct_group_counts"].values()),
                                        max(v for r in selected for v in r["direct_group_counts"].values())],
            "mean_distinct_parents_per_50":float(np.mean([r["distinct_direct_parents"] for r in selected]))}
    dest.mkdir(parents=True, exist_ok=False)
    result = {"source_evaluation":str(source.relative_to(ROOT)), "variant":variant, "classes":classes,
        "schedules":schedule, "raw_audio_read":False, "outer_inputs_read":False,
        "source_audit_sha256":sha(source/"audit_verification.json"), "source_summary_sha256":sha(source/"summary.json"),
        "generated_records_sha256":sha(source/"generated_records.jsonl.gz"), "code_sha256":sha(Path(__file__)),
        "seconds":time.perf_counter()-started,
        "limitation":"Seed variation combines donor, child, phase and noise draws; it is not uncertainty across new real groups."}
    save(dest/"summary.json", result)
    print(json.dumps({c:{k:v for k,v in x.items() if k != "folds"} for c,x in classes.items()}),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-id",required=True)
    a=p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}",a.output_id):p.error("Use a simple ID")
    build(HERE/"evaluations/spectrum16_group_v12_full_source",ROOT/"CAST/improvement/diagnostics"/a.output_id)

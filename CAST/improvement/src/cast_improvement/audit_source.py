"""Replay every source-selection sample and independently recompute all scores."""
from collections import defaultdict
from copy import deepcopy
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import time

from cast.config import ROOT, canonical, save, sha
from cast_generalization.method import compare, describe, sample, scales, stack
from cast_generalization.pipeline import read, readl
from .data import HERE, setup_cpu
from .evaluate import temper, rendered, aggregate
from .outer import check_source


def audit(out):
    setup_cpu()
    started = time.perf_counter()
    protocol, summary, _ = check_source(out)
    metric_cfg = protocol["metric_config"]
    banks = {v: readl(out/(v+"_bank.jsonl")) for v in protocol["variants"]}
    observations = readl(out/"real_training_descriptors.jsonl")
    expected_scales = read(out/"scales_by_fold.json")
    groups = sorted({r["group"] for r in observations})
    validation = {}
    for fold in groups:
        for c in metric_cfg["class_order"]:
            real_train = [r["descriptors"] for r in observations if r["class"] == c and r["group"] != fold]
            calculated = scales(stack(real_train, metric_cfg), metric_cfg)
            if calculated != expected_scales[fold][c]:
                raise ValueError("Fold scale uses different observations")
            validation[fold, c] = stack([r["descriptors"] for r in observations if r["class"] == c and r["group"] == fold], metric_cfg)
    priors, buckets, indices = {}, defaultdict(list), defaultdict(set)
    count = 0
    with gzip.open(out/"generated_records.jsonl.gz", "rb") as stream:
        for line in stream:
            r = json.loads(line)
            fold, variant, temperature = r["fold"], r["variant"], r["temperature"]
            cfg = protocol["renderer_configs"][variant]
            key = fold, variant, temperature
            if key not in priors:
                train = [p for p in banks[variant] if p["group"] != fold]
                if not train or any(p["group"] == metric_cfg["held_group"] for p in train):
                    raise ValueError("Invalid fold ancestry")
                priors[key] = temper(train, temperature, cfg)
            scfg = deepcopy(metric_cfg)
            scfg["version"] = f"cast_improvement_sampling_v1:{fold}"
            expected = sample(priors[key], scfg, r["class"], r["seed"], r["index"], r["arm"], cfg)
            expected.pop("parent_flags")
            if any(canonical(r[k]) != canonical(v) for k, v in expected.items()):
                raise ValueError("Source donor/parameter replay failed")
            if fold in r["parent_groups"] or metric_cfg["held_group"] in r["parent_groups"]:
                raise ValueError("Source donor crosses fold")
            wave, streams = rendered(r, cfg, variant)
            if hashlib.sha256(wave.tobytes()).hexdigest() != r["waveform_samples_sha256"] or streams != r["streams"]:
                raise ValueError("Source waveform/stream replay failed")
            desc = {k: v.tolist() for k, v in describe(wave, cfg, metric_cfg).items()}
            if desc != r["descriptors"]:
                raise ValueError("Source descriptor replay failed")
            score_key = fold, variant, temperature, r["class"], r["arm"], r["seed"]
            if r["index"] in indices[score_key]:
                raise ValueError("Duplicate source generated index")
            indices[score_key].add(r["index"])
            buckets[score_key].append(desc)
            count += 1
            if count % 5000 == 0:
                print(f"Source audit: {count} exact waveform/descriptor/donor replays", flush=True)
    scores = read(out/"scores.json")
    expected_keys = {(r["fold"],r["variant"],r["temperature"],r["class"],r["arm"],r["seed"]) for r in scores}
    if expected_keys != set(buckets) or len(expected_keys) != len(scores):
        raise ValueError("Source score groups incomplete or duplicated")
    for r in scores:
        key = r["fold"],r["variant"],r["temperature"],r["class"],r["arm"],r["seed"]
        if indices[key] != set(range(metric_cfg["samples_per_class_per_seed_per_arm"])):
            raise ValueError("Incomplete sample indices")
        actual = compare(stack(buckets[key],metric_cfg),validation[r["fold"],r["class"]],expected_scales[r["fold"]][r["class"]],metric_cfg)
        if actual != r["score"]:
            raise ValueError("Source score independent recomputation differs")
    if aggregate(scores, metric_cfg) != summary["candidates"]:
        raise ValueError("Source aggregate differs")
    result = {"passed":True,"generated_waveform_descriptor_donor_replays":count,"scores_recomputed":len(scores),
              "all_fold_ancestry_disjoint":True,"scales_recomputed_from_fold_training_only":True,"outer_held_audio_access":False,
              "summary_sha256":sha(out/"summary.json"),"audit_code_sha256":sha(Path(__file__)),"elapsed_seconds":time.perf_counter()-started}
    save(out/"audit_verification.json",result)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--eval-id',required=True);a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}',a.eval_id):p.error('Use a simple run ID')
    audit(HERE/'evaluations'/a.eval_id)

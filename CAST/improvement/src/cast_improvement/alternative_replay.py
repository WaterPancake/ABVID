"""Exact v4 replay with seed-invariant prototype means computed once per fold."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import re
import time

from cast.config import ROOT, canonical, save, sha
from cast.provenance import snapshot
from cast_generalization.method import compare, describe, draw_seed, stack
from cast_generalization.pipeline import read
from .alternative_prior import sample
from .alternative_evaluation import HERE, frozen, fold_data, render


class ReplaySampler:
    """A prototype depends on its fold/class bank, never seed or sample index."""
    def __init__(self, bank, cfg, renderer, mode):
        self.bank, self.cfg, self.renderer = deepcopy(bank), deepcopy(cfg), deepcopy(renderer)
        self.mode, self.prototypes = mode, {}

    def draw(self, label, seed, index, arm):
        cfg = self.cfg
        if (label not in cfg["class_order"] or seed not in cfg["seeds"] or arm not in cfg["arms"]
            or not 0 <= index < cfg["samples_per_class_per_seed_per_arm"]):
            raise ValueError("Outside frozen sampling schedule")
        if arm != "prototype":
            return sample(self.bank, cfg, label, seed, index, arm, self.renderer, self.mode)
        if label not in self.prototypes:
            self.prototypes[label] = sample(self.bank, cfg, label, cfg["seeds"][0], 0, arm, self.renderer, self.mode)
        result = deepcopy(self.prototypes[label])
        result.update(seed=seed, index=index,
                      input_id=f'{cfg["version"]}:sampling:{label}:{seed}:{index:04d}',
                      selection_seed=draw_seed(cfg["version"], label, seed, index, arm))
        return result


def rows(train, cfg, metric, variant, fold):
    if any(r["group"] in (fold, metric["held_group"]) for r in train):
        raise ValueError("Source ancestry crosses validation boundary")
    schedule = deepcopy(metric)
    schedule["version"] = f"cast_improvement_sampling_v1:{fold}"
    sampler = ReplaySampler(train, schedule, cfg, variant)
    for seed in schedule["seeds"]:
        for label in schedule["class_order"]:
            for arm in schedule["arms"]:
                for index in range(schedule["samples_per_class_per_seed_per_arm"]):
                    r = sampler.draw(label, seed, index, arm)
                    r.pop("parent_flags")
                    wave, streams = render(r, cfg, variant)
                    yield {**r, "variant": variant, "temperature": 1., "fold": fold, "streams": streams,
                           "waveform_samples_sha256": hashlib.sha256(wave.tobytes()).hexdigest(),
                           "descriptors": {k: v.tolist() for k, v in describe(wave, cfg, metric).items()}}


def audit(out):
    started = time.perf_counter()
    protocol, summary, configs, banks, obs, metric = frozen(out)
    expected_scores, all_scales, count = [], {}, 0
    with gzip.open(out/"generated_records.jsonl.gz", "rb") as stream:
        for fold in sorted({r["group"] for r in obs}):
            valid, scale = fold_data(obs, fold, metric); all_scales[fold] = scale
            for variant in protocol["variants"]:
                bucket = []
                train = [r for r in banks[variant] if r["group"] != fold]
                for expected in rows(train, configs[variant], metric, variant, fold):
                    line = stream.readline()
                    if not line or canonical(json.loads(line)) != canonical(expected):
                        raise ValueError("Parameter/start/ancestry/waveform/descriptor or schedule replay differs")
                    count += 1; bucket.append(expected["descriptors"])
                    if len(bucket) == metric["samples_per_class_per_seed_per_arm"]:
                        expected_scores.append({**{k: expected[k] for k in ("fold", "variant", "temperature", "class", "arm", "seed")},
                            "score": compare(stack(bucket, metric), valid[expected["class"]], scale[expected["class"]], metric)})
                        bucket = []
                print(f"Equivalent-fit replay: {count} exact samples", flush=True)
        if stream.readline():
            raise ValueError("Extra generated record")
    if expected_scores != read(out/"scores.json") or all_scales != read(out/"scales_by_fold.json"):
        raise ValueError("Independent scores/scales differ")
    if snapshot()["tracked_diff_sha256"] != protocol["snapshot"]["tracked_diff_sha256"]:
        raise ValueError("Tracked ABVID work changed")
    reference = ROOT/"CAST/improvement/evaluations/smooth8_full_source_grid/scores.json"
    identity = [{**r, "variant": "smooth8"} for r in expected_scores if r["variant"] == "winner"]
    original = [r for r in read(reference) if r["variant"] == "smooth8" and r["temperature"] == 1.]
    if identity != original or len(identity) != 150:
        raise ValueError("Winner-only identity did not reproduce the frozen source reference")
    save(out/"identity_replication.json", {"passed": True, "scores_exact": len(identity),
         "reference": str(reference.relative_to(ROOT)), "reference_sha256": sha(reference)})
    tests = ROOT/"CAST/improvement/tests/test_alternative_replay.py"
    result = {"passed": True, "generated_waveform_descriptor_donor_replays": count,
              "scores_recomputed": len(expected_scores), "summary_sha256": sha(out/"summary.json"),
              "audit_code_sha256": sha(Path(__file__)), "audit_tests_sha256": sha(tests),
              "all_fold_ancestry_disjoint": True, "complete_schedule_replayed": True,
              "scales_recomputed_from_fold_training_only": True, "outer_held_audio_access": False,
              "tracked_ABVID_diff_unchanged": True,
              "optimization": "Only seed-invariant class prototype controls/ancestry cached; every waveform rendered",
              "elapsed_seconds": time.perf_counter()-started}
    save(out/"audit_verification.json", result); print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-id", required=True)
    a = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", a.eval_id):
        p.error("Use a simple ID")
    out = HERE/"evaluations"/a.eval_id
    if (out/"audit_verification.json").exists():
        p.error("Preserve the completed source audit")
    audit(out)


if __name__ == "__main__":
    main()

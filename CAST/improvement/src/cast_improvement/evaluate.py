"""Training-group selection; held development data is inaccessible in source mode."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import re
import shutil
import time

import numpy as np
import soundfile as sf
import torch

from cast.audio import shape
from cast.config import ROOT, canonical, digest, save, sha
from cast.provenance import snapshot
from cast.renderer import seed_for, tensors, validate_controls
from cast_generalization.method import compare, describe, pack, render as render_v0, sample, scales, stack, unpack
from cast_generalization.pipeline import frozen as frozen_previous, file_hashes, jsonl, read, readl
from .data import HERE, PILOT, PREVIOUS, frozen, setup_cpu
from .engine import SmoothRenderer


def transformed(p):
    """Positive/log-simplex/logit coordinates; remove envelope scale gauge."""
    out = []
    out.extend(np.log(p["spacing_hz"]))
    for key in ("harmonic_weights", "noise_weights"):
        a = np.log(np.maximum(p[key], 1e-8));out.extend(a-a.mean())
    mix = np.clip(p["harmonic_fraction"], 1e-4, 1-1e-4)
    out.append(np.log(mix/(1-mix)))
    env = np.log(p["envelope_knots"]);out.extend(env-env.mean())
    return np.array(out)


def inverse(z, cfg):
    def softmax(x):
        x = np.exp(x-x.max());return (x/x.sum()).tolist()
    p = {"spacing_hz": np.exp(z[:3]).clip(*cfg["renderer"]["spacing_hz_bounds"]).tolist(),
         "harmonic_weights": softmax(z[3:11]), "noise_weights": softmax(z[11:19]),
         "harmonic_fraction": float(1/(1+np.exp(-np.clip(z[19], -15, 15)))),
         "envelope_knots": np.exp(z[20:25]).clip(*cfg["renderer"]["envelope_bounds"]).tolist()}
    validate_controls(p, cfg)
    return p


def temper(bank, temperature, cfg):
    if temperature not in (1., 1.1, 1.25, 1.5):
        raise ValueError("Temperature outside declared grid")
    if temperature == 1:
        return deepcopy(bank)
    result = []
    for c in ("car", "truck"):
        rows = [r for r in bank if r["class"] == c]
        groups = sorted({r["group"] for r in rows})
        center = np.mean([np.mean([transformed(r["parameters"]) for r in rows if r["group"] == g], axis=0) for g in groups], axis=0)
        for row in rows:
            z = center+temperature*(transformed(row["parameters"])-center)
            result.append({**row, "parameters": inverse(z, cfg)})
    return sorted(result, key=lambda r: r["file_id"])


def load_banks(out, scope, include_v0=False):
    cfg, rows, _ = frozen(out)
    frozen_previous(PREVIOUS)
    stage = read(out/("fit_"+scope+"_summary.json"))
    if not stage["passed"]:
        raise ValueError("Incomplete real-fit stage")
    ids = set(read(out/"pilot_ids.json")) if scope == "pilot" else {r["file_id"] for r in rows}
    selected = [r for r in rows if r["file_id"] in ids]
    old_cfg = read(PREVIOUS/"config.resolved.json")
    banks = {"smooth8": []}
    configs = {"smooth8": cfg}
    observations = []
    if include_v0:
        if scope != "pilot":
            raise ValueError("v0 bank available for original 50 parents only")
        banks["v0"] = []
        configs["v0"] = read(PILOT/"config.resolved.json")
    for row in selected:
        folder = out/"fits"/row["file_id"]
        for name, expected in read(folder/"hashes.json").items():
            if sha(folder/name) != expected:
                raise ValueError("Fitted artifact hash mismatch")
        result = read(folder/"fit.json")
        if result["ancestry"]["parent"] != row or result["config_sha256"] != digest(cfg):
            raise ValueError("Fitted source ancestry mismatch")
        common = {"file_id": row["file_id"], "class": row["canonical_class"], "group": row["provenance_group_id"], "parent": row}
        banks["smooth8"].append({**common, "parameters": result["parameters"], "flags": read(folder/"failure.json"), "fit_sha256": sha(folder/"fit.json")})
        wave, sr = sf.read(folder/"original_shape.wav", dtype="float32")
        if sr != 16000 or wave.shape != (32000,):
            raise ValueError("Incorrect stored observation")
        observations.append({**common, "descriptors": {k: v.tolist() for k, v in describe(wave, cfg, old_cfg).items()}})
        if include_v0:
            old = PILOT/"pilot"/row["file_id"]
            for name, expected in read(old/"hashes.json").items():
                if sha(old/name) != expected:
                    raise ValueError("v0 artifact changed")
            p = read(old/"parameters.json")
            banks["v0"].append({**common, "parameters": p["effective_controls"], "flags": read(old/"failure.json"), "fit_sha256": sha(old/"parameters.json")})
    return configs, banks, observations, old_cfg


def rendered(record, cfg, variant):
    if variant == "v0":
        return render_v0(record, cfg)
    r = SmoothRenderer(cfg, record["input_id"], sampling=True)
    with torch.no_grad():
        wave = shape(r.render(tensors(record["parameters"]), "sampling"))[0, 0].numpy()
    return wave, {"phase": seed_for(cfg, record["input_id"], "phase", "sampling"),
                  "noise": [seed_for(cfg, record["input_id"], "noise", "sampling", 0)], "purpose": "sampling"}


def generate_case(bank, cfg, metric_cfg, variant, temperature, fold, parent_stream):
    transformed_bank = temper(bank, temperature, cfg)
    sampling_cfg = deepcopy(metric_cfg)
    sampling_cfg["version"] = f"cast_improvement_sampling_v1:{fold}"
    for seed in sampling_cfg["seeds"]:
        for c in sampling_cfg["class_order"]:
            for arm in sampling_cfg["arms"]:
                descriptors = []
                for i in range(sampling_cfg["samples_per_class_per_seed_per_arm"]):
                    r = sample(transformed_bank, sampling_cfg, c, seed, i, arm, cfg)
                    if fold in r["parent_groups"] or sampling_cfg["held_group"] in r["parent_groups"]:
                        raise ValueError("Generated ancestry crosses validation boundary")
                    x, streams = rendered(r, cfg, variant)
                    d = {k: v.tolist() for k, v in describe(x, cfg, sampling_cfg).items()}
                    # Store every donor and descriptor; flags live once in the bank.
                    r.pop("parent_flags")
                    parent_stream.write(canonical({**r, "variant": variant, "temperature": temperature, "fold": fold,
                        "streams": streams, "waveform_samples_sha256": __import__("hashlib").sha256(x.tobytes()).hexdigest(),
                        "descriptors": d})+b"\n")
                    descriptors.append(d)
                yield c, arm, seed, stack(descriptors, sampling_cfg)


def aggregate(records, metric_cfg):
    result = {}
    keys = sorted({(r["variant"], r["temperature"]) for r in records})
    for variant, temperature in keys:
        classes = {}
        for c in metric_cfg["class_order"]:
            arms = {}
            for arm in metric_cfg["arms"]:
                selected = [r for r in records if (r["variant"], r["temperature"], r["class"], r["arm"]) == (variant, temperature, c, arm)]
                arms[arm] = {"W1": float(np.mean([r["score"]["W1"] for r in selected])),
                             "coverage": float(np.mean([r["score"]["coverage"] for r in selected])),
                             "family_spread": {f: float(np.mean([r["score"]["families"][f]["spread_ratio"] for r in selected])) for f in metric_cfg["primary_families"]}}
            gains = {a: 1-arms["joint"]["W1"]/arms[a]["W1"] for a in ("prototype", "marginals")}
            classes[c] = {"arms": arms, "gains": gains,
                          "passes": {"coverage": arms["joint"]["coverage"] >= .8, "gain": min(gains.values()) >= .025,
                                     "spread": all(.5 <= x <= 2 for x in arms["joint"]["family_spread"].values())}}
        spread_pass = all(r["passes"]["spread"] for r in classes.values())
        shortfall = max([0.]+[.8-r["arms"]["joint"]["coverage"] for r in classes.values()]+[.025-g for r in classes.values() for g in r["gains"].values()])
        w1 = float(np.mean([r["arms"]["joint"]["W1"] for r in classes.values()]))
        result[f"{variant}_T{temperature:g}"] = {"variant": variant, "temperature": temperature, "classes": classes,
            "selection_key": [not spread_pass, shortfall, w1, temperature], "passes": all(all(r["passes"].values()) for r in classes.values())}
    return result


def source(out, eval_id, scope="pilot", temperatures=(1.,), include_v0=True):
    setup_cpu()
    configs, banks, observations, metric_cfg = load_banks(out, scope, include_v0)
    destination = HERE/"evaluations"/eval_id
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sources = {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT/"CAST/generalization/src/cast_generalization/method.py", ROOT/"CAST/generalization/src/cast_generalization/pipeline.py"]}
    save(destination/"protocol.json", {"domain": "leave_one_training_group_out_source_development", "outer_held_access": False,
        "fit_run": str(out.relative_to(ROOT)), "scope": scope, "temperatures": list(temperatures), "variants": list(banks),
        "criteria": {"coverage": .8, "gain_each_control": .025, "spread": [.5, 2]},
        "selection": "minimize (any spread failure, max class coverage or gain shortfall, mean W1, temperature)",
        "source_hashes": sources, "metric_config": metric_cfg, "renderer_configs": configs,
        "snapshot": snapshot(), "fit_lock_sha256": sha(out/"lock.json")})
    for name in sources:
        dest = destination/"source_snapshot"/name;dest.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(ROOT/name, dest)
    for variant, bank in banks.items():
        jsonl(destination/(variant+"_bank.jsonl"), bank)
    jsonl(destination/"real_training_descriptors.jsonl", observations)
    save(destination/"lock.json", {"protocol_sha256": sha(destination/"protocol.json"), "bank_hashes": {v: sha(destination/(v+"_bank.jsonl")) for v in banks},
        "observations_sha256": sha(destination/"real_training_descriptors.jsonl"), "created_before_generation": True,
        "source_hashes": sources})
    groups = sorted({r["group"] for r in observations})
    records, fold_scales = [], {}
    with (destination/"generated_records.jsonl.gz").open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            for fold in groups:
                valid = {c: stack([r["descriptors"] for r in observations if r["group"] == fold and r["class"] == c], metric_cfg) for c in metric_cfg["class_order"]}
                scale = {c: scales(stack([r["descriptors"] for r in observations if r["group"] != fold and r["class"] == c], metric_cfg), metric_cfg) for c in metric_cfg["class_order"]}
                fold_scales[fold] = scale
                for variant, bank in banks.items():
                    train = [r for r in bank if r["group"] != fold]
                    for temperature in temperatures:
                        for c, arm, seed, generated in generate_case(train, configs[variant], metric_cfg, variant, temperature, fold, stream):
                            records.append({"fold": fold, "variant": variant, "temperature": temperature, "class": c, "arm": arm, "seed": seed,
                                            "score": compare(generated, valid[c], scale[c], metric_cfg)})
                        print(json.dumps({"fold": fold, "variant": variant, "temperature": temperature, "last_class": c, "last_arm": arm,
                                          "elapsed_s": round(time.perf_counter()-started, 2)}), flush=True)
    save(destination/"scales_by_fold.json", fold_scales)
    save(destination/"scores.json", records)
    summary = aggregate(records, metric_cfg)
    winner = min(summary, key=lambda k: summary[k]["selection_key"])
    save(destination/"summary.json", {"candidates": summary, "selected": winner, "selected_by_source_only": True,
         "outer_held_access": False, "elapsed_seconds": time.perf_counter()-started,
         "generated_record_sha256": sha(destination/"generated_records.jsonl.gz"), "evaluated_groups": groups})
    print(json.dumps({"selected": winner, "candidates": summary}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["source"])
    p.add_argument("--run-id", required=True)
    p.add_argument("--eval-id", required=True)
    p.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    p.add_argument("--temperatures", nargs="+", type=float, default=[1.])
    p.add_argument("--include-v0", action="store_true")
    a = p.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", x) for x in (a.run_id, a.eval_id)):
        p.error("Use simple run IDs")
    source(HERE/"runs"/a.run_id, a.eval_id, a.scope, tuple(a.temperatures), a.include_v0)


if __name__ == "__main__":
    main()

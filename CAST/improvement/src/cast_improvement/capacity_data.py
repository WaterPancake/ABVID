"""Separately frozen capacity-v3 fitting; preserve every preceding experiment."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch

from cast.audio import normalize, observation, shape
from cast.artifacts import diagnose, plot
from cast.config import ROOT, canonical, digest, save, sha
from cast.fit import Objective
from cast.provenance import checked_path, environment, read_frozen, snapshot, source_hashes, validate_row
from cast.renderer import tensors
from cast.synthetic import fixtures, assess
from cast_generalization.pipeline import disjoint, file_hashes, jsonl, read, readl, metadata
from cast_generalization.method import describe
from .capacity_engine import CapacityRenderer, fit, model_config, lift_controls, validate_controls
from .data import implementation as previous_implementation
from .block_evaluation import frozen as prior_frozen

HERE = ROOT/"CAST/improvement/capacity_v3"
SOURCE = ROOT/"CAST/improvement/src/cast_improvement"
PRIOR = ROOT/"CAST/improvement/evaluations/smooth8_block_v2_full_source"
PILOT = ROOT/"CAST/runs/cast_pilot_v0_20261004"
PREVIOUS = ROOT/"CAST/generalization/runs/cast_generalization_v0_20261004_r1"


def now():
    return datetime.now(timezone.utc).isoformat()


def implementation():
    files = [SOURCE/name for name in ("capacity_engine.py", "capacity_data.py", "capacity_cli.py")]
    files += [HERE/"run.sh", HERE/"PROTOCOL.md"]
    files += [ROOT/"CAST/improvement/tests"/name for name in ("test_capacity_engine.py", "test_capacity_guards.py")]
    return {**previous_implementation(), **{str(p.relative_to(ROOT)): sha(p) for p in files}}


def require_prior_audit(out=PRIOR):
    # Freeze the preceding scientific failure, including a complete exact replay.
    _, summary, *_ = prior_frozen(out)
    verified = read(out/"audit_verification.json")
    if (not verified["passed"] or verified["summary_sha256"] != sha(out/"summary.json")
        or verified["generated_waveform_descriptor_donor_calibration_replays"] != 90000
        or verified["scores_recomputed"] != 1800 or verified["outer_held_audio_access"]
        or not verified["all_fold_ancestry_disjoint"]
        or not verified["class_centers_and_scales_recomputed_from_training_only"]
        or any(c["passes"] for c in summary["candidates"].values())):
        raise ValueError("Require the intact, audited prior-v2 scientific failure")
    return {"path": str(out.relative_to(ROOT)), "summary_sha256": sha(out/"summary.json"),
            "audit_sha256": sha(out/"audit_verification.json")}


def assert_versions():
    old, current = read(PILOT/"snapshot.json")["environment"], environment()
    if old["versions"] != current["versions"] or old["python"].split()[0] != current["python"].split()[0]:
        raise ValueError("Pinned numerical environment changed")


def prepare(out, variant="capacity_noise16_env9_v3"):
    setup_cpu()
    prior = require_prior_audit()
    base, pilot, parent_lock = read_frozen(PILOT)
    assert_versions()
    old_cfg = read(PREVIOUS/"config.resolved.json")
    held = metadata(old_cfg, base, pilot)
    manifest = readl(ROOT/base["data"]["manifest"])
    by_id = {r["file_id"]: r for r in manifest}
    parent = read(ROOT/base["data"]["parent_lock"])
    fold = next(f for f in parent["folds"] if f["fold"] == 0)
    ids = fold["train_ids"]["42"]
    if len(ids) != 380 or len(set(ids)) != 380:
        raise ValueError("Changed frozen H1 training selection")
    train = sorted([by_id[i] for i in ids], key=lambda r: r["file_id"])
    for row in train:
        validate_row(row, base, set(ids))
    disjoint(train, held)
    if Counter(r["canonical_class"] for r in train) != {"car": 190, "truck": 190}:
        raise ValueError("Incorrect training class counts")
    out.mkdir(parents=True, exist_ok=False)
    cfg = model_config()
    save(out/"preceding_prior_audit.json", prior)
    save(out/"renderer.config.json", cfg)
    jsonl(out/"training_manifest.jsonl", train)
    jsonl(out/"held_metadata_only.jsonl", held)
    save(out/"pilot_ids.json", sorted(r["file_id"] for r in pilot))
    save(out/"target.json", {"coverage_per_class": .80, "relative_W1_gain_against_each_control_per_class": .025,
        "controls": ["prototype", "marginals"], "family_spread_bounds": [.5, 2.], "origin": "explicit user goal",
        "historical_criterion": .05, "historical_results_preserved": True, "held_is_exposed_development": True})
    save(out/"snapshot.json", snapshot())
    save(out/"previous_scores_hash.json", {"scores": sha(PREVIOUS/"scores.json"), "verification": sha(PREVIOUS/"verification.json"),
         "pilot_report": sha(ROOT/"reports/CAST_pilot.md"), "generalization_report": sha(ROOT/"reports/CAST_generalization.md")})
    hashes = {**implementation(), **source_hashes()}
    for name in hashes:
        dest = out/"source_snapshot"/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
    names = [p for p in out.iterdir() if p.is_file()]
    save(out/"lock.json", {"created_utc": now(), "variant": variant, "config_sha256": digest(cfg),
         "files": {p.name: sha(p) for p in sorted(names)}, "source_hashes": hashes,
         "real_audio_access_at_freeze": "none in this run", "parent_pilot_lock_sha256": sha(PILOT/"lock.json"),
         "prior_generalization_lock_sha256": sha(PREVIOUS/"lock.json"), "counts": {"train": 380, "pilot": 50, "held_metadata": 570}})
    print(f"Frozen {out}: 380 training IDs, 50 pilot IDs; no raw audio decoded", flush=True)


def frozen(out):
    lock = read(out/"lock.json")
    if {**implementation(), **source_hashes()} != lock["source_hashes"]:
        raise ValueError("Fitting implementation changed; preserve this run and create a new version")
    for name, expected in lock["files"].items():
        if sha(out/name) != expected:
            raise ValueError(f"Changed frozen file {name}")
    read_frozen(PILOT)
    prior = read(out/"preceding_prior_audit.json")
    if sha(ROOT/prior["path"]/"summary.json") != prior["summary_sha256"] or sha(ROOT/prior["path"]/"audit_verification.json") != prior["audit_sha256"]:
        raise ValueError("Preceding prior-v2 audit receipt changed")
    assert_versions()
    cfg = read(out/"renderer.config.json")
    if digest(cfg) != lock["config_sha256"]:
        raise ValueError("Changed renderer configuration")
    return cfg, readl(out/"training_manifest.jsonl"), lock


def setup_cpu():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def artifact(out, original, result, cfg, ancestry):
    out.mkdir(parents=True, exist_ok=False)
    target, level = normalize(original, cfg)
    renderer = CapacityRenderer(cfg, result["input_id"])
    with torch.no_grad():
        raw, components = renderer.render(tensors(result["parameters"]), "check", components=True)
        recon = shape(raw)[0, 0].numpy()
        initial = result["starts"][result["baseline_check_winner"]]["initial_parameters"]
        baseline = shape(renderer.render(tensors(initial), "check"))[0, 0].numpy()
        loss, terms = Objective(target, cfg)(raw, detail=True)
    if abs(float(loss.mean())-result["check_loss"]) > 1e-5:
        raise ValueError("Recomputed checking loss differs")
    gain = min(1., .95/max(float(np.abs(original).max()), float(np.abs(recon).max())*level["ac_rms"], 1e-12))
    waves = {"original": original, "original_shape": target.numpy(), "reconstructed_shape": recon,
             "reconstructed_raw": raw[0, 0].numpy(), "baseline_shape": baseline,
             "original_playback": original*gain, "reconstructed_playback": recon*level["ac_rms"]*gain}
    for name, wave in waves.items():
        if not np.isfinite(wave).all():
            raise ValueError("Nonfinite saved waveform")
        sf.write(out/(name+".wav"), wave, 16000, subtype="FLOAT")
    np.savez(out/"components.npz", **{k: v.detach().numpy() for k, v in components.items()}, check_waveforms=raw.numpy())
    diagnostics = diagnose(target.numpy(), recon, result, cfg)
    plot(out, target.numpy(), recon, baseline, result, cfg)
    save(out/"fit.json", {**result, "config_sha256": digest(cfg), "ancestry": ancestry, "original_level": level,
          "common_playback_gain": gain, "diagnostics": diagnostics,
          "per_resolution_check": {n: {k: v[0].tolist() for k, v in fields.items()} for n, fields in terms.items()},
          "physical_metadata": {k: "unknown" for k in ("RPM", "load", "throttle", "physical_vehicle_id", "weather", "distance")}})
    save(out/"failure.json", {"numerical_failure": False, "scientific_flags": diagnostics["flags"], "boundary_hits": diagnostics["boundary_hits"]})
    save(out/"hashes.json", file_hashes(out))
    return diagnostics


def synthetic_task(args):
    out_str, name, cfg, truth = args
    setup_cpu()
    out = Path(out_str)
    input_id = "synthetic_"+name
    original = shape(CapacityRenderer(cfg, input_id+"_truth").render(tensors(truth), "check")[0, 0]).detach().numpy()
    target, _ = normalize(original, cfg)
    result = fit(target, cfg, input_id, known_harmonic_weights=truth["harmonic_weights"] if name == "single_tone" else None)
    assessment = assess(name, truth, result, cfg)
    artifact(out/name, original, result, cfg, {"origin": "synthetic_fixture", "truth": truth})
    record = {"fixture": name, **assessment, "fit_seconds": result["fit_seconds"]}
    save(out/(name+".acceptance.json"), record)
    return record


def synthetic(out, workers=4):
    cfg, rows, lock = frozen(out)
    setup_cpu()
    target = out/"synthetic"
    target.mkdir(exist_ok=False)
    started = time.perf_counter()
    command = [sys.executable, "-m", "pytest", "CAST/improvement/tests", "CAST/generalization/tests", "CAST/tests", "tests/test_source_simulation.py", "-q", "-p", "no:cacheprovider", "--junitxml", str(target/"tests.xml")]
    tests = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
    (target/"tests.log").write_text(tests.stdout+tests.stderr)
    save(target/"tests.json", {"command": command, "passed": tests.returncode == 0})
    if tests.returncode:
        raise RuntimeError("Numerical/provenance test gate failed before real fitting")
    save(target/"acceptance_preregistered.json", {"criteria": cfg["acceptance"], "fixtures": {n: lift_controls(p, cfg) for n, p in fixtures().items()}, "config_sha256": digest(cfg), "workers": workers})
    records = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(synthetic_task, (str(target), name, cfg, lift_controls(truth, cfg))) for name, truth in fixtures().items()]
        for f in as_completed(futures):
            result = f.result(); records.append(result)
            print(json.dumps(result), flush=True)
    p = lift_controls(fixtures()["mixture"], cfg)
    r = CapacityRenderer(cfg, "ambiguity_gain")
    q = {**p, "envelope_knots": [x*1.2 for x in p["envelope_knots"]]}
    error = float((shape(r.render(tensors(p)))-shape(r.render(tensors(q))*7)).abs().max())
    ambiguity = {"flag": "gain_envelope_scale_unidentifiable", "max_difference": error, "passed": error < 3e-6}
    summary = {"passed": all(r["passed"] for r in records) and ambiguity["passed"], "records": sorted(records, key=lambda r: r["fixture"]),
               "gain_ambiguity": ambiguity, "seconds": time.perf_counter()-started, "workers": workers,
               "config_sha256": digest(cfg), "run_lock_sha256": sha(out/"lock.json")}
    save(target/"summary.json", summary)
    if not summary["passed"]:
        raise RuntimeError("Synthetic acceptance failed; real fitting remains blocked")


def training_task(args):
    out_str, row, cfg = args
    setup_cpu()
    out = Path(out_str)
    base, _, _ = read_frozen(PILOT)
    allowed = readl(out/"training_manifest.jsonl")
    path = checked_path(row, base, allowed)
    raw, sr = sf.read(path, dtype="float64", always_2d=True)
    if sr != row["native_sample_rate_hz"] or raw.shape != (row["decoded_frames"], 2):
        raise ValueError("Unexpected source shape/rate")
    original = observation(raw, sr, base)
    target, _ = normalize(original, base)
    result = fit(target, cfg, row["file_id"])
    diag = artifact(out/"fits"/row["file_id"], original, result, cfg,
         {"origin": "real_calibrated_observation_reconstruction", "role": "outer_fold0_training_only", "parent": row,
          "run_lock_sha256": sha(out/"lock.json"), "source_manifest_sha256": base["data"]["manifest_sha256"]})
    return {"file_id": row["file_id"], "class": row["canonical_class"], "group": row["provenance_group_id"],
            "check_loss": result["check_loss"], "baseline_check_loss": result["baseline_check_loss"],
            "fit_seconds": result["fit_seconds"], "diagnostics": diag,
            "worker_peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == "darwin" else 1024)}


def fit_bank(out, scope="pilot", workers=4):
    cfg, rows, lock = frozen(out)
    accepted = read(out/"synthetic/summary.json")
    if not accepted["passed"] or accepted["config_sha256"] != digest(cfg) or accepted["run_lock_sha256"] != sha(out/"lock.json"):
        raise ValueError("No valid synthetic acceptance")
    ids = set(read(out/"pilot_ids.json")) if scope == "pilot" else {r["file_id"] for r in rows}
    selected = [r for r in rows if r["file_id"] in ids]
    stage = out/("fit_"+scope+"_summary.json")
    if stage.exists():
        raise ValueError("Stage already completed; use its frozen artifacts")
    (out/"fits").mkdir(exist_ok=True)
    todo, records = [], []
    for row in selected:
        folder = out/"fits"/row["file_id"]
        if folder.exists():
            for name, expected in read(folder/"hashes.json").items():
                if sha(folder/name) != expected:
                    raise ValueError("Completed fit artifact changed")
            r = read(folder/"fit.json")
            if r["config_sha256"] != digest(cfg) or r["ancestry"]["parent"] != row:
                raise ValueError("Cannot resume changed fit")
            records.append({"file_id": row["file_id"], "class": row["canonical_class"], "group": row["provenance_group_id"],
                            "check_loss": r["check_loss"], "baseline_check_loss": r["baseline_check_loss"], "fit_seconds": r["fit_seconds"],
                            "diagnostics": r["diagnostics"], "reused_completed": True})
        else:
            todo.append(row)
    started = time.perf_counter()
    failures = []
    print(f"Fitting {len(todo)} new / {len(selected)} selected {scope} parents with {workers} CPU workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(training_task, (str(out), row, cfg)): row for row in todo}
        for f in as_completed(futures):
            row = futures[f]
            try:
                result = f.result(); records.append(result)
                print(json.dumps({"completed": len(records), "total": len(selected), "id": row["file_id"][:12], "check_loss": result["check_loss"], "fit_seconds": result["fit_seconds"]}), flush=True)
            except Exception as exc:
                fail = {"file_id": row["file_id"], "error": f"{type(exc).__name__}: {exc}"}; failures.append(fail)
                save(out/("failure_"+row["file_id"]+".json"), fail)
                print(json.dumps(fail), flush=True)
    summary = {"passed": len(records) == len(selected) and not failures, "scope": scope, "selected": len(selected),
               "records": sorted(records, key=lambda r: r["file_id"]), "failures": failures,
               "wall_seconds": time.perf_counter()-started, "workers": workers, "reused_fits": len(selected)-len(todo),
               "config_sha256": digest(cfg), "run_lock_sha256": sha(out/"lock.json")}
    save(stage, summary)
    if failures:
        raise RuntimeError("Real fitting had failures; all records preserved")

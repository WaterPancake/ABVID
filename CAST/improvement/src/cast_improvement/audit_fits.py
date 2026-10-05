"""Replay every completed training fit without rerunning its optimizer."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import time

import numpy as np
import soundfile as sf
import torch

from cast.audio import load_allowed, normalize, shape
from cast.artifacts import diagnose
from cast.config import ROOT, digest, save, sha
from cast.fit import Objective
from cast.provenance import read_frozen, snapshot
from cast.renderer import tensors, validate_controls
from cast_generalization.pipeline import file_hashes, read
from .data import HERE, PILOT, frozen, setup_cpu
from .engine import SmoothRenderer


def require_wave(path, expected):
    stored, rate = sf.read(path, dtype="float32")
    if rate != 16000 or not np.array_equal(stored, expected):
        raise ValueError(f"Waveform replay mismatch: {path.name}")


def verify_fit(folder, row, allowed, base, cfg, run_lock_hash):
    hashes = read(folder/"hashes.json")
    actual = file_hashes(folder)
    actual.pop("hashes.json", None)
    if actual != hashes:
        raise ValueError("Changed or incomplete fitted artifact inventory")
    fit = read(folder/"fit.json")
    expected_ancestry = {
        "origin": "real_calibrated_observation_reconstruction",
        "role": "outer_fold0_training_only", "parent": row,
        "run_lock_sha256": run_lock_hash,
        "source_manifest_sha256": base["data"]["manifest_sha256"],
    }
    if fit["ancestry"] != expected_ancestry or fit["config_sha256"] != digest(cfg):
        raise ValueError("Fit ancestry/configuration mismatch")
    if fit["input_id"] != row["file_id"]:
        raise ValueError("Fit input ID mismatch")
    original = load_allowed(row, base, allowed)
    target, level = normalize(original, cfg)
    require_wave(folder/"original.wav", original)
    require_wave(folder/"original_shape.wav", target.numpy())
    if level != fit["original_level"]:
        raise ValueError("Original level mismatch")
    renderer = SmoothRenderer(cfg, fit["input_id"])
    if renderer.streams() != fit["streams"]:
        raise ValueError("Fit/check random stream mismatch")
    objective = Objective(target, cfg)
    starts = fit["starts"]
    if [r["seed"] for r in starts] != cfg["seeds"]["starts"]:
        raise ValueError("Changed fit starts")
    history = np.array([r["fit_losses"] for r in fit["history"]])
    if history.shape != (cfg["optimizer"]["steps"]+1, len(starts)) or not np.isfinite(history).all():
        raise ValueError("Incomplete/nonfinite optimizer history")
    if [r["step"] for r in fit["history"]] != list(range(len(history))):
        raise ValueError("Optimizer steps missing or reordered")
    if fit["winner"] != int(np.argmin([s["fit_loss"] for s in starts])):
        raise ValueError("Winning start not chosen by fitting stream")
    if fit["baseline_check_winner"] != int(np.argmin([s["initial_check_loss"] for s in starts])):
        raise ValueError("Baseline not the best unoptimized checking start")
    if fit["parameters"] != starts[fit["winner"]]["parameters"]:
        raise ValueError("Saved controls differ from winning start")
    with torch.no_grad():
        for s in starts:
            for key, loss_key in (("parameters", "check_loss"), ("initial_parameters", "initial_check_loss")):
                validate_controls(s[key], cfg)
                checked = float(objective(renderer.render(tensors(s[key]), "check")).mean())
                if abs(checked-s[loss_key]) > 1e-5:
                    raise ValueError("Alternative-start checking loss replay failed")
        raw, components = renderer.render(tensors(fit["parameters"]), "check", components=True)
        recon = shape(raw)[0, 0].numpy()
        initial = starts[fit["baseline_check_winner"]]["initial_parameters"]
        baseline = shape(renderer.render(tensors(initial), "check"))[0, 0].numpy()
        loss, terms = objective(raw, detail=True)
    if abs(float(loss.mean())-fit["check_loss"]) > 1e-5:
        raise ValueError("Selected checking loss replay failed")
    if fit["baseline_check_loss"] != starts[fit["baseline_check_winner"]]["initial_check_loss"]:
        raise ValueError("Incorrect baseline score")
    require_wave(folder/"reconstructed_raw.wav", raw[0, 0].numpy())
    require_wave(folder/"reconstructed_shape.wav", recon)
    require_wave(folder/"baseline_shape.wav", baseline)
    gain = min(1., .95/max(float(np.abs(original).max()), float(np.abs(recon).max())*level["ac_rms"], 1e-12))
    if gain != fit["common_playback_gain"]:
        raise ValueError("Playback gain mismatch")
    require_wave(folder/"original_playback.wav", original*gain)
    require_wave(folder/"reconstructed_playback.wav", recon*level["ac_rms"]*gain)
    with np.load(folder/"components.npz", allow_pickle=False) as saved:
        expected = {k: v.numpy() for k, v in components.items()}
        expected["check_waveforms"] = raw.numpy()
        if set(saved.files) != set(expected) or any(not np.array_equal(saved[k], x) for k, x in expected.items()):
            raise ValueError("Renderer component replay failed")
    resolutions = {n: {k: v[0].tolist() for k, v in fields.items()} for n, fields in terms.items()}
    if resolutions != fit["per_resolution_check"]:
        raise ValueError("Per-resolution checking losses changed")
    diagnostics = diagnose(target.numpy(), recon, fit, cfg)
    if diagnostics != fit["diagnostics"]:
        raise ValueError("Scientific diagnostics changed")
    if read(folder/"failure.json") != {"numerical_failure": False, "scientific_flags": diagnostics["flags"], "boundary_hits": diagnostics["boundary_hits"]}:
        raise ValueError("Failure record changed")
    return {"file_id": row["file_id"], "fit_sha256": sha(folder/"fit.json"),
            "hashes_sha256": sha(folder/"hashes.json"), "class": row["canonical_class"],
            "group": row["provenance_group_id"], "checking_starts_replayed": len(starts)}


def audit(out, scope):
    setup_cpu()
    started = time.perf_counter()
    cfg, allowed, lock = frozen(out)
    base, _, _ = read_frozen(PILOT)
    stage = out/("fit_"+scope+"_summary.json")
    summary = read(stage)
    if not summary["passed"] or summary["failures"] or summary["config_sha256"] != digest(cfg) or summary["run_lock_sha256"] != sha(out/"lock.json"):
        raise ValueError("No completed successful fit stage")
    ids = set(read(out/"pilot_ids.json")) if scope == "pilot" else {r["file_id"] for r in allowed}
    if {r["file_id"] for r in summary["records"]} != ids or len(summary["records"]) != len(ids):
        raise ValueError("Stage IDs incomplete/duplicated")
    records = []
    for row in allowed:
        if row["file_id"] not in ids:
            continue
        records.append(verify_fit(out/"fits"/row["file_id"], row, allowed, base, cfg, sha(out/"lock.json")))
        if len(records) % 25 == 0:
            print(f"Raw/fit/alternative-start replay: {len(records)}/{len(ids)}", flush=True)
    if snapshot()["tracked_diff_sha256"] != read(out/"snapshot.json")["tracked_diff_sha256"]:
        raise ValueError("Tracked ABVID work changed")
    return {"passed": True, "scope": scope, "fit_count": len(records),
            "class_counts": dict(Counter(r["class"] for r in records)), "records": records,
            "fit_summary_sha256": sha(stage), "run_lock_sha256": sha(out/"lock.json"),
            "audit_source_sha256": sha(Path(__file__)), "all_waveforms_and_components_bit_exact": True,
            "all_checking_starts_recomputed": True, "loss_tolerance_unchanged": 1e-5,
            "raw_training_preprocessing_bit_exact": True, "held_audio_access": False,
            "tracked_ABVID_diff_unchanged": True, "wall_seconds": time.perf_counter()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scope", choices=["pilot", "full"], required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", args.run_id):
        parser.error("Use a simple run ID")
    out = HERE/"runs"/args.run_id
    destination = out/("fit_"+args.scope+"_verification.json")
    if destination.exists():
        parser.error("Verification already exists; preserve the original audit")
    result = audit(out, args.scope)
    save(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()

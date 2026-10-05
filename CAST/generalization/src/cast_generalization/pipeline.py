"""Freeze → train-only generation → permitted held observations → verification."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch

from cast.audio import normalize, observation
from cast.config import ROOT, canonical, digest, save, sha
from cast.provenance import environment, read_frozen, snapshot
from cast.renderer import validate_controls
from .method import compare, describe, render, sample, scales, stack

EXT = ROOT / "CAST/generalization"
ANCESTRY_FIELDS = ("file_id", "source_file_sha256", "decoded_audio_sha256", "paired_event_id", "duplicate_group_id", "candidate_recording_id")


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def jsonl(path, rows):
    with Path(path).open("xb") as out:
        for row in rows:
            out.write(canonical(row)+b"\n")


def readl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def file_hashes(base):
    # Finder can create/update this presentation metadata while a run is open.
    # Every experiment file, including unexpected non-Finder files, is hashed.
    return {str(p.relative_to(base)): sha(p) for p in sorted(base.rglob("*")) if p.is_file() and p.name != ".DS_Store"}


def implementation_hashes():
    files = sorted((EXT/"src").rglob("*.py")) + sorted((EXT/"tests").glob("*.py"))
    files += [EXT/n for n in ("config.json", "PROTOCOL.md", "DECISIONS.md", "README.md", "run.sh")]
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def disjoint(train, held):
    for key in ANCESTRY_FIELDS:
        a = {r.get(key) for r in train if r.get(key)}
        b = {r.get(key) for r in held if r.get(key)}
        if a & b:
            raise ValueError(f"Cross-split ancestry: {key}")


def validate_held(row, allowed_rows, base_cfg, cfg, root=ROOT):
    """Metadata/path checks only. This does not hash or decode an audio file."""
    allowed = {r["file_id"]: r for r in allowed_rows}
    if row.get("file_id") not in allowed or row != allowed[row["file_id"]]:
        raise ValueError("Held metadata differs from exact frozen manifest")
    checks = [row.get("dataset_id") == "IDMT", row.get("release_id") == "IDMT_V1",
              row.get("provenance_group_id") == cfg["held_group"],
              row.get("canonical_class") in base_cfg["data"]["classes"],
              row.get("class_id") == base_cfg["data"]["classes"].get(row.get("canonical_class")),
              row.get("split_role") == "source_logo_pool", row.get("admitted_for_training") is True,
              row.get("admitted_for_h1") is True, row.get("integrity_pass") is True,
              row.get("device_id") == "SE", row.get("provider_channel_pair") == "CH34",
              row.get("original_channel_ids") == [3, 4], row.get("native_channels") == 2,
              row.get("raw_audio_modified") is False]
    if not all(checks):
        raise ValueError("Disallowed held data")
    raw = Path(row["source_path"])
    path = root/raw
    base = root/base_cfg["data"]["allowed_audio_root"]
    if raw.is_absolute() or ".." in raw.parts or not raw.name.endswith("_SE_CH34.wav") or path.is_symlink() or base.resolve() != base.absolute() or path.resolve().parent != base.resolve():
        raise ValueError("Held path outside exact allowlisted root or symlink")
    for key in ("file_id", "source_file_sha256", "decoded_audio_sha256"):
        value = row.get(key, "")
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Invalid held hash")
    return path


def metadata(cfg, base_cfg, train):
    # read_frozen has already pinned all three parent metadata hashes.
    parent = read(ROOT/base_cfg["data"]["parent_lock"])
    fold = next(f for f in parent["folds"] if f["fold"] == 0)
    all_rows = readl(ROOT/base_cfg["data"]["manifest"])
    by_id = {r["file_id"]: r for r in all_rows}
    ids = fold["test_ids"]
    if len(ids) != len(set(ids)) or not set(ids) <= by_id.keys() or fold["held_out_group"] != cfg["held_group"]:
        raise ValueError("Invalid held selection")
    held = sorted([by_id[i] for i in ids], key=lambda r: r["file_id"])
    if set(ids) != {r["file_id"] for r in all_rows if r["provenance_group_id"] == cfg["held_group"]}:
        raise ValueError("Incomplete held-group selection")
    if dict(Counter(r["canonical_class"] for r in held)) != cfg["held_counts"]:
        raise ValueError("Held class count drift")
    disjoint(train, held)
    for row in held:
        validate_held(row, held, base_cfg, cfg)
    return held


def load_bank(cfg):
    parent = ROOT/cfg["parent_run"]
    if sha(parent/"lock.json") != cfg["parent_lock_sha256"] or sha(parent/"verification.json") != cfg["parent_verification_sha256"]:
        raise ValueError("Changed pilot lock or verification")
    base_cfg, train, _ = read_frozen(parent)
    verified = read(parent/"verification.json")
    if not verified["passed"] or {r["id"] for r in verified["records"]} != {r["file_id"] for r in train}:
        raise ValueError("Pilot not completely verified")
    pinned = read(parent/"snapshot.json")["environment"]
    env = environment()
    if env["versions"] != pinned["versions"] or env["python"].split()[0] != pinned["python"].split()[0]:
        raise ValueError("Pinned numerical environment changed")
    bank, train_desc = [], []
    for row in train:
        folder = parent/"pilot"/row["file_id"]
        for name, expected in read(folder/"hashes.json").items():
            if sha(folder/name) != expected:
                raise ValueError(f"Changed pilot artifact {row['file_id']}/{name}")
        p, flags = read(folder/"parameters.json"), read(folder/"failure.json")
        if p["ancestry"]["parent"] != row or p["config_sha256"] != digest(base_cfg) or flags["numerical_failure"]:
            raise ValueError("Invalid calibration ancestry")
        validate_controls(p["effective_controls"], base_cfg)
        x, sr = sf.read(folder/"original_shape.wav", dtype="float32")
        if sr != 16000 or x.shape != (32000,) or not np.isfinite(x).all():
            raise ValueError("Invalid training observation artifact")
        bank.append({"file_id": row["file_id"], "class": row["canonical_class"], "group": row["provenance_group_id"],
                     "parameters": p["effective_controls"], "flags": flags, "parent": row,
                     "parameters_sha256": sha(folder/"parameters.json"), "parent_config_sha256": digest(base_cfg)})
        train_desc.append({"file_id": row["file_id"], "class": row["canonical_class"],
                           **{k: v.tolist() for k, v in describe(x, base_cfg, cfg).items()}})
    if len(bank) != 50 or len({r["file_id"] for r in bank}) != 50 or dict(Counter(r["class"] for r in bank)) != {"car": 25, "truck": 25}:
        raise ValueError("Incorrect bank counts")
    if any(r["group"] == cfg["held_group"] for r in bank):
        raise ValueError("Held parent in bank")
    return base_cfg, train, bank, train_desc


def freeze(out):
    cfg = read(EXT/"config.json")
    base_cfg, train, bank, train_desc = load_bank(cfg)
    held = metadata(cfg, base_cfg, train)
    out.mkdir(parents=True, exist_ok=False)
    save(out/"config.resolved.json", cfg)
    save(out/"renderer.config.json", base_cfg)
    jsonl(out/"held_manifest.jsonl", held)
    jsonl(out/"training_manifest.jsonl", train)
    jsonl(out/"parameter_bank.jsonl", bank)
    jsonl(out/"training_descriptors.jsonl", train_desc)
    scale = {c: scales(stack([r for r in train_desc if r["class"] == c], cfg), cfg) for c in cfg["class_order"]}
    save(out/"scales.json", scale)
    schedule = []
    for arm in cfg["arms"]:
        for seed in cfg["seeds"]:
            for label in cfg["class_order"]:
                for i in range(cfg["samples_per_class_per_seed_per_arm"]):
                    schedule.append(sample(bank, cfg, label, seed, i, arm, base_cfg))
    jsonl(out/"sampling_schedule.jsonl", schedule)
    own_hashes = implementation_hashes()
    state = snapshot()
    save(out/"snapshot.json", {**state, "extension_source_hashes": own_hashes})
    for path in {**state["source_hashes"], **own_hashes}:
        target = out/"source_snapshot"/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/path, target)
    shutil.copyfile(EXT/"PROTOCOL.md", out/"PROTOCOL.md")
    # Full parent run and final pilot report are protected against later mutation.
    save(out/"parent_artifacts.json", file_hashes(ROOT/cfg["parent_run"]))
    save(out/"protected_files.json", {str(p.relative_to(ROOT)): sha(p) for p in [ROOT/"reports/CAST_pilot.md", ROOT/"CAST/README.md", ROOT/"CAST/DELIVERY_CHECKS.json", ROOT/"CAST/RESEARCH_LOG.md"]})
    names = [p.name for p in out.iterdir() if p.is_file()]
    lock = {"created_utc": now(), "held_audio_accessed": False, "config_sha256": digest(cfg),
            "bank_sha256": sha(out/"parameter_bank.jsonl"), "extension_source_hashes": own_hashes,
            "files": {name: sha(out/name) for name in sorted(names)},
            "held_count": len(held), "generated_count": len(schedule), "train_count": len(train),
            "preselected_held_ids": [next(r["file_id"] for r in held if r["canonical_class"] == c) for c in cfg["class_order"]],
            "historical_exposure": "H1 development; first CAST-held group; same site as one training group"}
    save(out/"lock.json", lock)
    return cfg, base_cfg


def frozen(out):
    lock = read(out/"lock.json")
    if implementation_hashes() != lock["extension_source_hashes"]:
        raise ValueError("Extension changed after freeze; use a new run")
    for name, expected in lock["files"].items():
        if sha(out/name) != expected:
            raise ValueError(f"Changed frozen artifact {name}")
    cfg, base_cfg = read(out/"config.resolved.json"), read(out/"renderer.config.json")
    if digest(cfg) != lock["config_sha256"]:
        raise ValueError("Changed configuration")
    read_frozen(ROOT/cfg["parent_run"])
    held = readl(out/"held_manifest.jsonl")
    train = readl(out/"training_manifest.jsonl")
    if metadata(cfg, base_cfg, train) != held:
        raise ValueError("Held selection drift")
    return cfg, base_cfg, lock


def tests(out):
    command = [sys.executable, "-m", "pytest", "CAST/generalization/tests", "CAST/tests", "tests/test_source_simulation.py", "-q", "-p", "no:cacheprovider", "--junitxml", str(out/"tests.xml")]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (out/"tests.log").write_text(result.stdout+result.stderr)
    save(out/"tests.json", {"command": command, "passed": result.returncode == 0, "returncode": result.returncode, "seconds": time.perf_counter()-started})
    print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError("Test gate failed before held audio access")


def generate(out):
    cfg, base_cfg, lock = frozen(out)
    if not read(out/"tests.json")["passed"]:
        raise ValueError("Tests must pass before generation")
    folder = out/"generated"
    folder.mkdir()
    records = []
    for i, chosen in enumerate(readl(out/"sampling_schedule.jsonl")):
        stem = f'{chosen["arm"]}/{chosen["seed"]}/{chosen["class"]}/{chosen["index"]:04d}'
        dest = folder/stem
        dest.mkdir(parents=True)
        wave, streams = render(chosen, base_cfg)
        gain = .95/max(float(np.abs(wave).max()), 1e-12)
        sf.write(dest/"shape.wav", wave, 16000, subtype="FLOAT")
        sf.write(dest/"playback.wav", wave*gain, 16000, subtype="FLOAT")
        rec = {**chosen, "bank_sha256": lock["bank_sha256"], "config_sha256": digest(cfg),
               "renderer_config_sha256": digest(base_cfg), "run_lock_sha256": sha(out/"lock.json"),
               "streams": streams, "playback_gain": gain, "path": str(dest.relative_to(out)),
               "waveform_sha256": sha(dest/"shape.wav"), "playback_sha256": sha(dest/"playback.wav"),
               "descriptors": {k: v.tolist() for k, v in describe(wave, base_cfg, cfg).items()}}
        save(dest/"sample.json", rec)
        records.append(rec)
        if (i+1) % 100 == 0:
            print(f"Generated {i+1}/{lock['generated_count']} without held audio", flush=True)
    jsonl(out/"generated_manifest.jsonl", records)
    save(out/"generation.complete.json", {"completed_utc": now(), "count": len(records),
         "generated_manifest_sha256": sha(out/"generated_manifest.jsonl"),
         "sampling_schedule_sha256": sha(out/"sampling_schedule.jsonl"),
         "run_lock_sha256": sha(out/"lock.json"), "artifacts": file_hashes(folder)})


def access_gate(out):
    cfg, base_cfg, lock = frozen(out)
    done = read(out/"generation.complete.json")
    if not read(out/"tests.json")["passed"] or done["count"] != lock["generated_count"] or done["run_lock_sha256"] != sha(out/"lock.json") or done["generated_manifest_sha256"] != sha(out/"generated_manifest.jsonl") or done["sampling_schedule_sha256"] != sha(out/"sampling_schedule.jsonl"):
        raise ValueError("Train-only generation must be complete before held access")
    if file_hashes(out/"generated") != done["artifacts"]:
        raise ValueError("Generated artifacts changed before held access")
    return cfg, base_cfg, lock


def decode_held(row, held, base_cfg, cfg):
    path = validate_held(row, held, base_cfg, cfg)
    if sha(path) != row["source_file_sha256"] or path.stat().st_size != row["source_file_bytes"]:
        raise ValueError("Held original byte hash or size mismatch")
    native, sr = sf.read(path, dtype="float64", always_2d=True)
    if native.shape != (row["decoded_frames"], 2) or sr != row["native_sample_rate_hz"]:
        raise ValueError("Held original native shape/rate mismatch")
    x = observation(native, sr, base_cfg)
    normalized, level = normalize(x, base_cfg)
    return x, normalized.numpy(), level


def evaluate(out):
    cfg, base_cfg, lock = access_gate(out)
    save(out/"held_access_receipt.json", {"started_utc": now(), "generation_complete_sha256": sha(out/"generation.complete.json"),
         "tests_sha256": sha(out/"tests.json"), "run_lock_sha256": sha(out/"lock.json"),
         "allowed_group": cfg["held_group"], "expected_count": sum(cfg["held_counts"].values()), "conditional_fitting": False})
    held = readl(out/"held_manifest.jsonl")
    folder = out/"held"
    folder.mkdir()
    records, failures = [], []
    for i, row in enumerate(held):
        dest = folder/row["file_id"]
        dest.mkdir()
        try:
            original, shape_wave, level = decode_held(row, held, base_cfg, cfg)
            gain = min(1.0, .95/max(float(np.abs(original).max()), 1e-12))
            sf.write(dest/"original.wav", original, 16000, subtype="FLOAT")
            sf.write(dest/"playback.wav", original*gain, 16000, subtype="FLOAT")
            record = {"file_id": row["file_id"], "class": row["canonical_class"], "group": row["provenance_group_id"],
                      "source_file_sha256": row["source_file_sha256"], "origin": "held_real_observation", "level": level,
                      "playback_gain": gain, "path": str(dest.relative_to(out)),
                      "descriptors": {k: v.tolist() for k, v in describe(shape_wave, base_cfg, cfg).items()},
                      "original_sha256": sha(dest/"original.wav"), "playback_sha256": sha(dest/"playback.wav")}
            save(dest/"observation.json", record)
            records.append(record)
        except Exception as exc:
            failure = {"file_id": row["file_id"], "class": row["canonical_class"], "stage": "held_observation", "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            save(dest/"failure.json", failure)
        if (i+1) % 100 == 0 or i+1 == len(held):
            print(f"Held descriptors {i+1}/{len(held)}; failures {len(failures)}", flush=True)
    jsonl(out/"held_descriptors.jsonl", records)
    save(out/"failures.json", {"selected": len(held), "completed": len(records), "failures": failures, "dropped_or_replaced": 0})
    save(out/"held.complete.json", {"completed_utc": now(), "count": len(records), "descriptor_sha256": sha(out/"held_descriptors.jsonl"), "artifacts": file_hashes(folder)})
    if failures:
        raise RuntimeError("Held input failures: incomplete scientific comparison; all failures retained")


def calculate_scores(out):
    cfg, _, _ = frozen(out)
    scale = read(out/"scales.json")
    gen, held = readl(out/"generated_manifest.jsonl"), readl(out/"held_descriptors.jsonl")
    train = readl(out/"training_descriptors.jsonl")
    result = {"group": cfg["held_group"], "held_groups": 1, "classes": {}, "domain": "real-calibrated synthesized observations versus held-group real IDMT observations", "session_confidence_interval": None}
    for c in cfg["class_order"]:
        h = stack([r["descriptors"] for r in held if r["class"] == c], cfg)
        if len(h["bands"]) != cfg["held_counts"][c]:
            raise ValueError("Missing held records")
        arms = {}
        for arm in cfg["arms"]:
            scores = []
            for seed in cfg["seeds"]:
                rows = [r for r in gen if (r["class"], r["arm"], r["seed"]) == (c, arm, seed)]
                if len(rows) != cfg["samples_per_class_per_seed_per_arm"]:
                    raise ValueError("Missing generated records")
                one = compare(stack([r["descriptors"] for r in rows], cfg), h, scale[c], cfg)
                one.update(seed=seed, unique_vectors=len({digest(r["parameters"]) for r in rows}),
                           unique_parents=len({p for r in rows for p in r["parent_ids"]}),
                           unique_groups=len({p for r in rows for p in r["parent_groups"]}))
                scores.append(one)
            arms[arm] = {"seeds": scores, "W1_mean": float(np.mean([s["W1"] for s in scores])),
                         "W1_min": min(s["W1"] for s in scores), "W1_max": max(s["W1"] for s in scores),
                         "coverage_mean": float(np.mean([s["coverage"] for s in scores])),
                         "family_spread_mean": {f: float(np.mean([s["families"][f]["spread_ratio"] for s in scores])) if all(s["families"][f]["spread_ratio"] is not None for s in scores) else None for f in cfg["primary_families"]}}
        reference = compare(stack([r for r in train if r["class"] == c], cfg), h, scale[c], cfg)
        gain = {a: 1-arms["joint"]["W1_mean"]/arms[a]["W1_mean"] for a in ("prototype", "marginals")}
        lo, hi = cfg["criteria"]["family_spread_ratio_bounds_each_class"]
        gates = {"distance_gain": all(g >= cfg["criteria"]["relative_W1_gain_over_each_control_each_class"] for g in gain.values()),
                 "marginal_coverage": arms["joint"]["coverage_mean"] >= cfg["criteria"]["mean_marginal_coverage_each_class"],
                 "spread": all(x is not None and lo <= x <= hi for x in arms["joint"]["family_spread_mean"].values())}
        result["classes"][c] = {"held_count": len(h["bands"]), "calibration_count": 25, "arms": arms, "training_real_reference": reference,
                                "joint_relative_gains": gain, "criteria": gates, "adequacy_passed": all(gates.values())}
    result["macro_W1"] = {a: float(np.mean([r["arms"][a]["W1_mean"] for r in result["classes"].values()])) for a in cfg["arms"]}
    result["adequacy_passed"] = all(r["adequacy_passed"] for r in result["classes"].values())
    result["worst_group"] = {"group": cfg["held_group"], "joint_macro_W1": result["macro_W1"]["joint"], "note": "Only one evaluated group; identical to group result"}
    return result


def verify(out):
    cfg, base_cfg, lock = access_gate(out)
    bank = readl(out/"parameter_bank.jsonl")
    schedule = readl(out/"sampling_schedule.jsonl")
    generated = readl(out/"generated_manifest.jsonl")
    if len(schedule) != len(generated):
        raise ValueError("Incomplete generated manifest")
    for chosen, record in zip(schedule, generated):
        repeated = sample(bank, cfg, chosen["class"], chosen["seed"], chosen["index"], chosen["arm"], base_cfg)
        if canonical(repeated) != canonical(chosen) or any(canonical(record[k]) != canonical(v) for k, v in chosen.items()):
            raise ValueError("Sampling replay or ancestry mismatch")
        wave, streams = render(record, base_cfg)
        dest = out/record["path"]
        saved, sr = sf.read(dest/"shape.wav", dtype="float32")
        played, playback_sr = sf.read(dest/"playback.wav", dtype="float32")
        if sr != 16000 or playback_sr != 16000 or not np.array_equal(saved, wave) or not np.array_equal(played, wave*record["playback_gain"]) or streams != record["streams"]:
            raise ValueError("Waveform or random-stream replay failed")
        actual = {k: v.tolist() for k, v in describe(saved, base_cfg, cfg).items()}
        if actual != record["descriptors"] or sha(dest/"shape.wav") != record["waveform_sha256"] or sha(dest/"playback.wav") != record["playback_sha256"]:
            raise ValueError("Generated descriptors or waveform hash changed")
        if read(dest/"sample.json") != record or record["bank_sha256"] != lock["bank_sha256"] or record["config_sha256"] != digest(cfg) or record["run_lock_sha256"] != sha(out/"lock.json"):
            raise ValueError("Generated sidecar provenance changed")
    print(f"Bit-exact generated replays: {len(generated)}", flush=True)
    done = read(out/"held.complete.json")
    if done["artifacts"] != file_hashes(out/"held") or done["descriptor_sha256"] != sha(out/"held_descriptors.jsonl"):
        raise ValueError("Held artifacts changed")
    held = readl(out/"held_manifest.jsonl")
    records = readl(out/"held_descriptors.jsonl")
    if [r["file_id"] for r in records] != [r["file_id"] for r in held]:
        raise ValueError("Held IDs incomplete or reordered")
    for row, record in zip(held, records):
        original, shape_wave, level = decode_held(row, held, base_cfg, cfg)
        dest = out/record["path"]
        saved, sr = sf.read(dest/"original.wav", dtype="float32")
        played, playback_sr = sf.read(dest/"playback.wav", dtype="float32")
        desc = {k: v.tolist() for k, v in describe(shape_wave, base_cfg, cfg).items()}
        if sr != 16000 or playback_sr != 16000 or not np.array_equal(original, saved) or not np.array_equal(original*record["playback_gain"], played) or desc != record["descriptors"] or level != record["level"]:
            raise ValueError("Held preprocessing replay failed")
        if record != read(dest/"observation.json") or record["source_file_sha256"] != row["source_file_sha256"] or sha(dest/"original.wav") != record["original_sha256"] or sha(dest/"playback.wav") != record["playback_sha256"]:
            raise ValueError("Held sidecar provenance failed")
    if calculate_scores(out) != read(out/"scores.json"):
        raise ValueError("Scores not reproducible")
    if read(out/"parent_artifacts.json") != file_hashes(ROOT/cfg["parent_run"]):
        raise ValueError("Parent pilot mutated")
    for name, expected in read(out/"protected_files.json").items():
        if sha(ROOT/name) != expected:
            raise ValueError("Protected prior deliverable mutated")
    before, current = read(out/"snapshot.json"), snapshot()
    if before["tracked_diff_sha256"] != current["tracked_diff_sha256"]:
        raise ValueError("Tracked ABVID diff changed")
    receipt = read(out/"held_access_receipt.json")
    generation = read(out/"generation.complete.json")
    if not lock["created_utc"] <= generation["completed_utc"] <= receipt["started_utc"] or receipt["generation_complete_sha256"] != sha(out/"generation.complete.json"):
        raise ValueError("Generation/access sequence not proven")
    return {"passed": True, "generated_bit_exact_replays": len(generated), "held_preprocessing_bit_exact_replays": len(held),
            "source_hashes_and_scores_reverified": True, "train_only_sampling_reverified": True,
            "parent_artifacts_unchanged": True, "tracked_ABVID_diff_unchanged": True,
            "generation_completed_before_held_audio": True, "failures": 0, "verified_utc": now()}


def workflow(out):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    timings = {}
    try:
        for name, fn in (("freeze", freeze), ("tests", tests), ("generation", generate), ("held_evaluation", evaluate)):
            t = time.perf_counter()
            print(f"Starting {name}", flush=True)
            fn(out)
            timings[name] = time.perf_counter()-t
        t = time.perf_counter()
        save(out/"scores.json", calculate_scores(out))
        timings["scoring"] = time.perf_counter()-t
        print("Verifying all generated and held artifacts", flush=True)
        t = time.perf_counter()
        save(out/"verification.json", verify(out))
        timings["verification"] = time.perf_counter()-t
        save(out/"timing.json", {"stages_seconds": timings, "through_verification_seconds": time.perf_counter()-started,
             "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
             "hardware": environment(), "torch_threads": 1, "generated_audio_seconds": 3000, "held_audio_seconds": 1140})
        from .report import report
        report(out)
        save(out/"deliverable_hashes.json", {name: sha(out/name) for name in ("scores.json", "verification.json", "timing.json", "CAST_generalization.md", "index.html", "coverage.png", "descriptors.png")})
        print(f"Complete: {out/'CAST_generalization.md'}", flush=True)
    except Exception as exc:
        if out.exists() and not (out/"run_failure.json").exists():
            save(out/"run_failure.json", {"error": f"{type(exc).__name__}: {exc}", "time_utc": now(),
                 "held_access_receipt_exists": (out/"held_access_receipt.json").exists(), "elapsed_seconds": time.perf_counter()-started})
        raise

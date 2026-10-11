"""Read-only, same-parent content audit for the archived smooth8-to-v11 chain.

This is an integrity/recipe audit, not another numerical optimization replay.
It never imports the archived runners, fits, generates, or scores a population.
Eligibility still requires the caller's current-fold training-role/parity gate.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from abvid.paths import resolve, roots
from abvid.provenance.hashing import canonical, sha256


STAGES = ("smooth8", "v7", "v8", "v9", "v10", "v11")
BANKS = {"v7": "parent_calibrated_bank.jsonl", "v8": "mixture_envelope_bank.jsonl",
         "v9": "calibrated_bank.jsonl", "v10": "temporal41_bank.jsonl",
         "v11": "calibrated_bank.jsonl"}
CALIBRATIONS = {"v7": "mixture_calibration", "v8": "envelope_calibration",
                "v9": "spectral_calibration", "v10": "temporal_calibration",
                "v11": "resolution_calibration"}
SAVED_FILES = {
    "smooth8": ["baseline_shape.wav", "components.npz", "diagnostics.png", "failure.json",
                "fit.json", "original.wav", "original_playback.wav", "original_shape.wav",
                "reconstructed_playback.wav", "reconstructed_raw.wav", "reconstructed_shape.wav"],
    "v9": ["calibration_reconstruction_0.wav", "calibration_reconstruction_1.wav", "derived_row.json", "runtime.json"],
    "v11": ["calibration_reconstruction_0.wav", "calibration_reconstruction_1.wav", "derived_row.json", "runtime.json"],
}
NUMERICAL_KEYS = ("audio", "normalization", "renderer", "loss", "optimizer",
                  "initialization", "seeds", "hardware", "diagnostics", "acceptance")
PROVENANCE_KEYS = ("file_id", "dataset_id", "release_id", "canonical_class", "class_id",
                   "provenance_group_id", "recording_session", "original_media_id",
                   "physical_vehicle_id", "candidate_recording_id", "source_time_axis",
                   "paired_event_id", "duplicate_group_id", "conservative_parent_proxy",
                   "source_file_sha256", "decoded_audio_sha256", "source_path", "relative_path",
                   "source_start_s", "source_end_s", "site_id", "native_sample_rate_hz",
                   "native_channels", "decoded_frames", "provider_channel_pair",
                   "original_channel_ids", "intended_channel_operation")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def streams(file_id, run=42):
    def seed(component, purpose, index=0):
        return int.from_bytes(hashlib.sha256(canonical(
            [run, file_id, component, purpose, index])).digest()[:8], "big") % 2**63
    return {"phase": seed("phase", "shared"),
            **{p: [seed("noise", p, i) for i in range(2)] for p in ("fit", "check")}}


def validate_controls(values, noise=8, envelope=5):
    sizes = {"spacing_hz": 3, "harmonic_weights": 8, "noise_weights": noise,
             "envelope_knots": envelope}
    require(set(values) == set(sizes) | {"harmonic_fraction"}, "control schema")
    for key, count in sizes.items():
        require(isinstance(values[key], list) and len(values[key]) == count, f"{key} dimensions")
    for key, value in values.items():
        vals = value if isinstance(value, list) else [value]
        require(all(isinstance(v, (float, int)) and math.isfinite(v) for v in vals), f"{key} nonfinite")
    for key in ("harmonic_weights", "noise_weights"):
        require(min(values[key]) >= 0 and abs(sum(values[key])-1) <= 1e-6, f"{key} simplex")
    for key, lo, hi in (("spacing_hz", 10, 400), ("envelope_knots", .1, 3),
                        ("harmonic_fraction", 0, 1)):
        vals = values[key] if isinstance(values[key], list) else [values[key]]
        require(all(lo-1e-6 <= v <= hi+1e-6 for v in vals), f"{key} bounds")


def validate_fit(fit, file_id, cfg):
    """Check all saved optimization states and strict first-minimum selection."""
    require(fit["input_id"] == file_id and fit["known_harmonic_weights"] is None, "fit input")
    require(fit["streams"] == streams(file_id, cfg["seeds"]["run"]), "fit streams")
    require(fit["initialization"]["rule"] == cfg["initialization"], "initialization recipe")
    starts, history = fit["starts"], fit["history"]
    require([s["seed"] for s in starts] == cfg["seeds"]["starts"], "all four starts required")
    require([h["step"] for h in history] == list(range(cfg["optimizer"]["steps"]+1)), "complete 301-state history required")
    require(all(len(h["fit_losses"]) == len(starts) and
                all(math.isfinite(v) for v in h["fit_losses"]) for h in history), "fit loss history")
    for i, start in enumerate(starts):
        validate_controls(start["initial_parameters"])
        validate_controls(start["parameters"])
        require(start["initial_fit_loss"] == history[0]["fit_losses"][i], "initial fit loss")
        losses = [h["fit_losses"][i] for h in history]
        require(start["fit_loss"] == min(losses) and start["best_step"] == losses.index(min(losses)), "strict best-state selection")
        require(all(math.isfinite(start[k]) for k in ("check_loss", "initial_check_loss")), "checking loss")
    winner = min(range(len(starts)), key=lambda i: starts[i]["fit_loss"])
    require(fit["winner"] == winner and fit["parameters"] == starts[winner]["parameters"], "fit winner")
    require(fit["fit_loss"] == starts[winner]["fit_loss"] and fit["check_loss"] == starts[winner]["check_loss"], "winner losses")
    require(fit["baseline_fit_loss"] == min(s["initial_fit_loss"] for s in starts), "baseline fit loss")
    check_winner = min(range(len(starts)), key=lambda i: starts[i]["initial_check_loss"])
    require(fit["baseline_check_winner"] == check_winner and
            fit["baseline_check_loss"] == starts[check_winner]["initial_check_loss"], "baseline checking winner")
    require(fit["selection"] == "fitted_winner_by_fit_stream_only_baseline_minimum_on_common_check_stream", "selection rule")


def validate_transition(stage, before, after, observation, fit_hash, cfg, settings):
    """Validate same-parent links, inherited records and unchanged controls."""
    calkey = CALIBRATIONS[stage]
    cal = after[calkey]
    for key, value in before.items():
        if key != "parameters":
            require(after.get(key) == value, f"{stage} altered ancestor {key}")
    require(digest(cal) == after[calkey+"_sha256"], f"{stage} calibration hash")
    for a, b in (("file_id", "file_id"), ("group", "group"),
                 (("class_label" if stage in ("v9", "v11") else "class"), "class")):
        require(cal[a] == after[b], f"{stage} calibration identity {a}")
    changes = {"v7": {"harmonic_fraction"}, "v8": {"envelope_knots"},
               "v9": {"noise_weights", "harmonic_fraction"}, "v10": {"envelope_knots"},
               "v11": {"noise_weights", "harmonic_fraction"}}[stage]
    for key in before["parameters"].keys()-changes:
        require(before["parameters"][key] == after["parameters"][key], f"{stage} changed fixed {key}")
    validate_controls(after["parameters"], noise=16 if stage == "v11" else 8,
                      envelope=41 if stage in ("v10", "v11") else 5)
    desc = observation["descriptors"]
    if stage == "v7":
        require(cal["original_harmonic_fraction"] == before["parameters"]["harmonic_fraction"], "v7 initial mixture")
        require(cal["harmonic_fraction"] == after["parameters"]["harmonic_fraction"], "v7 result mixture")
        require(cal["observed_bands"] == desc["bands"], "v7 same-parent bands")
    if stage in ("v8", "v10"):
        require(cal["observed_envelope"] == desc["envelope"], f"{stage} same-parent envelope")
        require(cal["envelope_knots"] == after["parameters"]["envelope_knots"], f"{stage} envelope result")
        require(abs(cal["preserved_mean_amplitude"]-cal["result_mean_amplitude"]) <= 1e-8, f"{stage} gauge")
        require(cal["solver"]["success"] or any("failure" in x for x in cal["flags"]), f"{stage} unrecorded solver failure")
    if stage == "v8":
        require(cal["original_fit_sha256"] == fit_hash and cal["observed_descriptor_sha256"] == digest(observation), "v8 ancestry")
        require(cal["original_envelope_knots"] == before["parameters"]["envelope_knots"], "v8 initial envelope")
    if stage in ("v9", "v10", "v11"):
        require(cal["initial_bank_row_sha256"] == digest(before), f"{stage} initial bank hash")
        require(cal["observation_sha256"] == digest(observation), f"{stage} observation hash")
    if stage == "v10":
        require(cal["initial_parameters"] == before["parameters"] and cal["smoothness"] == .01, "v10 initialization/recipe")
    if stage in ("v9", "v11"):
        require(cal["original_fit_sha256"] == fit_hash and cal["input_id"] == after["file_id"], f"{stage} original fit")
        require(cal["target_sha256"] == digest({k: desc[k] for k in ("log_spectrum", "bands")}), f"{stage} target")
        require(cal["streams"] == streams(after["file_id"]) and cal["settings"] == settings, f"{stage} streams/settings")
        require(cal["parameters"] == after["parameters"], f"{stage} result parameters")
        validate_controls(cal["initial_parameters"], noise=16 if stage == "v11" else 8,
                          envelope=41 if stage == "v11" else 5)
        if stage == "v9":
            require(cal["initial_parameters"] == before["parameters"], "v9 initialization")
        else:
            # Pure interpolation validates the documented lift; no optimization.
            from abvid.cast.population.resolution_model import lift_controls
            require(cal["initial_parameters"] == lift_controls(before["parameters"], cfg), "v11 spectral lift")
        history = cal["history"]
        require(len(history) == 151 and all(math.isfinite(v) for v in history), f"{stage} complete history")
        require(cal["initial_objective"] == history[0] and cal["best_objective"] == min(history)
                and cal["best_step"] == history.index(min(history)), f"{stage} strict best-state selection")


class Checks:
    def __init__(self):
        self.hashes = {}
        self.failures = []

    def check(self, label, action):
        try:
            return action()
        except (ValueError, KeyError, TypeError, IndexError, OSError, RuntimeError) as exc:
            self.failures.append(f"{label}: {type(exc).__name__}: {exc}")
            return None

    def hashed(self, ref, expected=None):
        path = resolve(ref, must_exist=True)
        found = sha256(path)
        self.hashes[ref] = found
        require(expected is None or found == expected, f"SHA256 mismatch {ref}")
        return path

    def json(self, ref):
        return json.loads(self.hashed(ref).read_text())

    def jsonl(self, ref):
        values = [json.loads(line) for line in self.hashed(ref).read_text().splitlines()]
        ids = [r["file_id"] for r in values]
        require(len(ids) == len(set(ids)), f"duplicate IDs {ref}")
        return {r["file_id"]: r for r in values}

    def closure(self, root, declared):
        for name, expected in declared.items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe closure path")
            self.check(name, lambda n=name, h=expected: self.hashed(root+"/"+n, h))


def _same_parent(parent, source):
    for key in PROVENANCE_KEYS:
        require(parent.get(key) == source.get(key), f"source provenance {key}")
    require(parent["dataset_id"] == "IDMT" and parent["release_id"] == "IDMT_V1"
            and parent["canonical_class"] in ("car", "truck"), "source dataset/class")
    require(parent["provider_channel_pair"] == "CH34" and parent["original_channel_ids"] == [3, 4], "source channels")
    require(parent == source, "source metadata differs from frozen source row")


def raw_reference(source, root):
    prefix = "dataset/IDMT_Traffic/audio/"
    require(source["source_path"].startswith(prefix), "raw source outside IDMT")
    name = source["source_path"][len(prefix):]
    require(Path(name).name == name and name.endswith(".wav"), "unsafe raw source filename")
    return root+"/"+name


def _recipe(cfg, numerical, name):
    for key in NUMERICAL_KEYS:
        require(cfg[key] == numerical["renderer_configs"][name][key], f"recipe {name}:{key}")


def _environment(snapshot, expected):
    env = snapshot["environment"]
    for key in ("machine", "python", "versions"):
        require(env[key] == expected[key], f"saved numerical environment {key}")


def _saved_closure(checks, folder, stage):
    """Read only the exact stage payload after a complete nonsymlink preflight."""
    resolved = resolve(folder, must_exist=True)
    scheme, relative = folder.split(":", 1)
    require(scheme == "archive", "saved parent artifacts must be archived")
    location = roots()[scheme]
    # resolve() deliberately permits some storage-root symlinks. Saved parent
    # closures have a narrower contract: no link at any component below root.
    lexical = location
    for part in Path(relative).parts:
        lexical = lexical/part
        require(not lexical.is_symlink(), "symlink in saved parent directory")
    require(lexical.resolve() == resolved and resolved.is_dir(), "saved parent directory escape")
    def direct_file(name):
        require(isinstance(name, str) and Path(name).name == name, "closure requires direct basenames")
        path = lexical/name
        require(not path.is_symlink(), f"symlink in saved parent payload: {name}")
        require(path.is_file() and path.resolve().parent == resolved, f"saved payload outside parent: {name}")
    direct_file("hashes.json")
    hashes = checks.json(folder+"/hashes.json")
    require(isinstance(hashes, dict), "saved closure must be a hash mapping")
    require(set(hashes) == set(SAVED_FILES[stage]), f"{stage} exact saved closure inventory required")
    require(all(isinstance(h, str) and len(h) == 64 and set(h) <= set("0123456789abcdef")
                for h in hashes.values()), "saved closure requires SHA256 values")
    # Do this for all entries before opening any payload, including audio/NPZ.
    for name in hashes:
        direct_file(name)
    checks.closure(folder, hashes)
    return hashes


def audit_saved_parents(contract):
    """Return every requested candidate, retaining failures without replacement.

    ``eligible`` is chain integrity only; it never grants a parent a training role.
    The contract is a pinned JSON mapping, with storage-qualified references.
    """
    shared = Checks()
    ids = contract["candidate_ids"]
    require(ids == sorted(set(ids)) and ids, "candidate IDs must be unique and sorted")
    require(set(contract["stages"]) == set(STAGES), "complete six-stage chain required")
    require(contract["per_parent_closure_files"] == SAVED_FILES, "undeclared saved closure filenames")
    for ref in contract["references"]:
        shared.check("reference", lambda r=ref: shared.hashed(r["path"], r["sha256"]))
    sources = shared.check("source manifest", lambda: shared.jsonl(contract["source_manifest"]["path"]))
    shared.check("source manifest hash", lambda: shared.hashed(**{
        "ref": contract["source_manifest"]["path"], "expected": contract["source_manifest"]["sha256"]}))
    numerical = shared.check("numerical contract", lambda: shared.json(contract["numerical_contract"]["path"]))
    shared.check("numerical contract hash", lambda: shared.hashed(contract["numerical_contract"]["path"], contract["numerical_contract"]["sha256"]))
    data = {}
    for stage in STAGES:
        root = contract["stages"][stage]
        def load_stage(s=stage, r=root):
            lock = shared.json(r+"/lock.json")
            shared.closure(r, lock["files"])
            for path, expected in lock["source_hashes"].items():
                shared.check(f"{s} snapshot", lambda p=path, h=expected: shared.hashed(r+"/source_snapshot/"+p, h))
            proto = shared.json(r+("/renderer.config.json" if s == "smooth8" else "/protocol.json"))
            data[s] = {"lock": lock, "protocol": proto}
            if s == "smooth8":
                data[s]["training"] = shared.jsonl(r+"/training_manifest.jsonl")
                data[s]["verification"] = shared.json(r+"/fit_full_verification.json")
                data[s]["summary"] = shared.json(r+"/fit_full_summary.json")
                _recipe(proto, numerical, "smooth8_v1")
                require(digest(proto) == lock["config_sha256"], "smooth8 config digest")
                _environment(shared.json(r+"/snapshot.json"), numerical["historical_non_executable_metadata"]["recorded_environment"])
            else:
                bank = shared.jsonl(r+"/"+BANKS[s])
                require(set(bank) == set(ids), f"{s} candidate population mismatch")
                data[s]["bank"] = bank
                _environment(proto["snapshot"], numerical["historical_non_executable_metadata"]["recorded_environment"])
                if s in ("v9", "v11"):
                    require(proto["input_ids"] == {i: i for i in ids}, f"{s} input namespaces")
                    data[s]["initial"] = shared.jsonl(r+"/initial_bank.jsonl")
                    data[s]["observations"] = shared.jsonl(r+"/observations.jsonl")
                    cfg = proto["renderer_config"]
                    _recipe(cfg, numerical, "smooth8_v1" if s == "v9" else "spectrum16_temporal41_v11")
                    summary = shared.json(r+"/calibration_summary.json")
                    require(summary["passed"] and summary["failures"] == [] and summary["count"] == len(ids), f"{s} calibration completion")
                    shared.hashed(r+"/calibrated_bank.jsonl", summary["bank_sha256"])
                    shared.hashed(r+"/lock.json", summary["lock_sha256"])
                    verification = shared.json(r+"/calibration_verification.json")
                    require(verification["passed"] and verification["calibrations_replayed"] == len(ids)
                            and verification["waveforms_replayed"] == 2*len(ids), f"{s} historical replay")
                    shared.hashed(r+"/calibration_summary.json", verification["summary_sha256"])
                else:
                    data[s]["observations"] = shared.jsonl(r+"/real_training_descriptors.jsonl")
                    for variant, cfg in proto["renderer_configs"].items():
                        _recipe(cfg, numerical, "smooth8_temporal41_v10" if variant == "temporal41" else "smooth8_v1")
                require(set(data[s]["observations"]) == set(ids), f"{s} observation IDs")
        shared.check(stage, load_stage)
    def shared_chain():
        require(set(data["smooth8"]["training"]) == set(ids), "smooth8 training IDs")
        verification = data["smooth8"]["verification"]
        require(verification["passed"] and verification["fit_count"] == len(ids)
                and verification["raw_training_preprocessing_bit_exact"]
                and verification["all_waveforms_and_components_bit_exact"]
                and verification["all_checking_starts_recomputed"], "smooth8 recorded replay")
        shared.hashed(contract["stages"]["smooth8"]+"/fit_full_summary.json", verification["fit_summary_sha256"])
        shared.hashed("archive:CAST/improvement/src/cast_improvement/audit_fits.py", verification["audit_source_sha256"])
        cfg = data["smooth8"]["protocol"]["data"]
        for field in ("manifest", "parent_lock", "source_lock"):
            shared.hashed("archive:"+cfg[field], cfg[field+"_sha256"])
        for s in ("v7", "v8"):
            proto = data[s]["protocol"]
            shared.hashed(contract["stages"]["smooth8"]+"/lock.json", proto["fit_lock_sha256"])
            shared.hashed(contract["stages"]["smooth8"]+"/fit_full_verification.json", proto["fit_audit_sha256"])
        for s, upstream in (("v9", "v8"), ("v11", "v10")):
            proto = data[s]["protocol"]
            require("archive:"+proto["upstream"] == contract["stages"][upstream], f"{s} upstream root")
            shared.hashed(contract["stages"][upstream]+"/audit_verification.json", proto["upstream_audit_sha256"])
            shared.hashed(contract["stages"][upstream]+"/summary.json", proto["upstream_summary_sha256"])
            cert = shared.json(contract["stages"][s]+"/calibration_verification.json")
            implementation = "spectrum_bank.py" if s == "v9" else "resolution_bank.py"
            shared.hashed(contract["stages"][s]+"/source_snapshot/CAST/improvement/src/cast_improvement/"+implementation,
                          cert["audit_code_sha256"])
        shared.hashed(contract["stages"]["v9"]+"/lock.json", data["v10"]["protocol"]["fit_lock_sha256"])
        shared.hashed(contract["stages"]["v9"]+"/calibration_verification.json", data["v10"]["protocol"]["fit_audit_sha256"])
        for s in ("v8", "v9", "v10", "v11"):
            require(data[s]["observations"] == data["v7"]["observations"], f"{s} observations changed")
        require(data["v9"]["initial"] == data["v8"]["bank"], "v9 initial bank not v8")
        require(data["v11"]["initial"] == data["v10"]["bank"], "v11 initial bank not v10")
    shared.check("transitive shared chain", shared_chain)
    records = []
    for file_id in ids:
        c = Checks()
        flags = {}
        def parent():
            source = sources[file_id]
            root = contract["stages"]["smooth8"]
            folder = root+"/fits/"+file_id
            hashes = _saved_closure(c, folder, "smooth8")
            fit = c.json(folder+"/fit.json")
            validate_fit(fit, file_id, data["smooth8"]["protocol"])
            require(fit["config_sha256"] == data["smooth8"]["lock"]["config_sha256"], "fit configuration hash")
            require(fit["ancestry"]["run_lock_sha256"] == shared.hashes[root+"/lock.json"], "fit lock ancestry")
            require(fit["ancestry"]["source_manifest_sha256"] == data["smooth8"]["protocol"]["data"]["manifest_sha256"], "fit source manifest ancestry")
            require(fit["ancestry"]["parent"] == data["smooth8"]["training"][file_id], "fit training manifest")
            _same_parent(fit["ancestry"]["parent"], source)
            old = next(r for r in data["smooth8"]["verification"]["records"] if r["file_id"] == file_id)
            require(old["fit_sha256"] == hashes["fit.json"] and old["checking_starts_replayed"] == 4,
                    "fit historical replay link")
            c.hashed(folder+"/hashes.json", old["hashes_sha256"])
            fail = c.json(folder+"/failure.json")
            require(fail["numerical_failure"] is False, "saved numerical failure")
            flags["smooth8"] = fail
            raw = c.hashed(raw_reference(source, contract["raw_source_root"]), source["source_file_sha256"])
            require(raw.stat().st_size == source["source_file_bytes"], "raw source byte count")
            import soundfile as sf
            info = sf.info(raw)
            require((info.samplerate, info.channels, info.frames) ==
                    (source["native_sample_rate_hz"], source["native_channels"], source["decoded_frames"]), "raw header shape/rate")
            before = {"file_id": file_id, "class": source["canonical_class"],
                      "group": source["provenance_group_id"], "parent": fit["ancestry"]["parent"],
                      "fit_sha256": hashes["fit.json"], "flags": fail, "parameters": fit["parameters"]}
            for stage in STAGES[1:]:
                row, obs = data[stage]["bank"][file_id], data[stage]["observations"][file_id]
                _same_parent(row["parent"], source)
                require(all(row[k] == obs[k] for k in ("file_id", "class", "group", "parent")), f"{stage} observation identity")
                settings = {k: numerical["expected_spectrum_calibration"][k]
                            for k in ("betas", "epsilon", "gradient_clip", "lr", "steps")}
                cfg = numerical["renderer_configs"]["spectrum16_temporal41_v11" if stage == "v11" else "smooth8_v1"]
                validate_transition(stage, before, row, obs, hashes["fit.json"], cfg, settings)
                calkey = CALIBRATIONS[stage]
                flags[stage] = row[calkey]["flags"]
                if stage == "v7":
                    require(row[calkey]["input_hashes"] == {k: hashes[k] for k in
                            ("fit.json", "components.npz", "original_shape.wav")}, "v7 component closure")
                if stage in ("v9", "v11"):
                    f = contract["stages"][stage]+"/parents/"+file_id
                    _saved_closure(c, f, stage)
                    require(c.json(f+"/derived_row.json") == row, f"{stage} saved row")
                c.hashes[f"canonical:{stage}:{file_id}"] = digest(row)
                before = row
        c.check("parent chain", parent)
        failures = [f"shared: {f}" for f in shared.failures] + c.failures
        records.append({"file_id": file_id, "eligible": not failures, "failures": failures,
                        "checked_hashes": dict(sorted(c.hashes.items())), "flags": flags,
                        "eligible_artifacts": (["v11_calibrated_parameter_row_and_retained_calibration_history",
                                                "smooth8_original_fit_all_starts_components_and_observation"]
                                               if not failures else []),
                        "reconstruction_reuse_allowed": False,
                        "unanchored_reconstruction_closure": {
                            "stages": ["v9", "v11"],
                            "status": "current_hash_closure_only_not_independently_anchored_to_historical_replay",
                            "rule": "exclude_saved_calibration_reconstruction_waveforms_from_reuse"}})
    summary = {"schema_version": 1, "scope": "same_parent_content_integrity_and_recipe_compatibility",
               "candidate_count": len(ids), "eligible_count": sum(r["eligible"] for r in records),
               "ineligible_count": sum(not r["eligible"] for r in records),
               "shared_failures": shared.failures, "shared_checked_hashes": dict(sorted(shared.hashes.items())),
               "per_parent_checked_hash_count": sum(len(r["checked_hashes"]) for r in records),
               "per_parent_file_hash_count": sum(sum(not k.startswith("canonical:") for k in r["checked_hashes"])
                                                 for r in records),
               "eligible_payload": "anchored_final_parameter_records_and_original_smooth8_same_parent_artifacts_only",
               "calibration_reconstruction_reuse_allowed": False,
               "failure_counts": dict(sorted(Counter(f for r in records for f in r["failures"]).items())),
               "new_optimization_replays": 0, "new_waveforms_generated": 0,
               "new_descriptors_computed": 0, "held_evaluation_descriptor_assets_accessed": False,
               "candidate_parent_descriptor_records_read": len(data.get("v7", {}).get("observations", {})),
               "descriptor_access_scope": "Original-fold0 saved candidate-parent observations only. Candidates can be held in other grouped folds; current-fold training-role filtering is a separate prerequisite. No source evaluation or scoring performed.",
               "raw_access": "candidate_training_files_only_SHA256_bytes_and_header_no_decode",
               "numerical_evidence": "fresh_hash_closure_and_state_checks_plus_pinned_historical_numerical_replay_records",
               "limitations": ["No fresh optimization or waveform/descriptor replay performed.",
                               "V9/v11 saved reconstructed waveforms have only present self-consistent hashes; historical certificates do not anchor those bytes. Their reuse is prohibited.",
                               "Eligibility requires separate current-fold training-role and packaged parity gates.",
                               "Unknown original sessions and physical vehicle identities remain unknown."]}
    return records, summary

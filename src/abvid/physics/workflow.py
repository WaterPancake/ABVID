"""Audited, additive development comparison; no classifier or target data access."""
from __future__ import annotations

import hashlib
import html
import importlib.metadata
import json
import platform
import resource
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import soundfile as sf
import torch

from abvid.cast.audio import shape
from abvid.cast.config import configuration as cast_config
from abvid.cast.fit import Objective, fit as cast_fit
from abvid.cast.population.engine import SmoothRenderer, model_config
from abvid.cast.renderer import Renderer, tensors
from abvid.cast.smooth_fit import fit as smooth_fit
from abvid.paths import repository_root, resolve
from . import data
from .config import load
from .engine import Engine, free_field, render
from .fit import default_engine, fit, prepare, rendered

DEFAULT_CONFIG = "repo:configs/physics/engine_v0.json"


def deterministic():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def baseline_check():
    lock = json.loads(resolve("repo:experiments/physics_engine/baseline_lock.json").read_text())
    for row in lock["references"]:
        if data.sha(resolve(row["path"])) != row["sha256"]:
            raise ValueError(f"CAST baseline changed: {row['path']}")
    return lock


def snapshot(out, cfg, reference):
    repo = repository_root()
    paths = sorted((repo/"src/abvid/physics").glob("*.py"))
    paths += [resolve(reference), resolve(cfg["input_lock"]),
              resolve("repo:experiments/physics_engine/baseline_lock.json")]
    paths += [p for p in (repo/"experiments/physics_engine").glob("*.md")]
    paths += [repo/"tests/test_physics_engine.py"]
    hashes = {}
    for path in paths:
        relative = path.relative_to(repo)
        target = out/"source_snapshot"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes["repo:"+str(relative)] = data.sha(path)
    return {"configuration": cfg, "config_path": reference, "source_hashes": hashes,
            "baseline": baseline_check(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "machine": platform.machine(), "processor": platform.processor(),
                            "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "torch", "soundfile", "matplotlib")}},
            "checkpoint": None, "classification": None, "dataset_version": data.REVISION,
            "role": cfg["scope"]["role"], "independent_groups": "unknown"}


def source_inventory(root, lock, download):
    path = root/"inventory.json"
    if not path.is_file():
        if not download:
            raise FileNotFoundError("Missing pinned inventory; use --download")
        files = data.inventory()
        if data.make_lock(files) != lock:
            raise ValueError("Remote inventory differs from frozen selection")
        data.save(path, files)
    files = json.loads(path.read_text())
    if data.make_lock(files) != lock:
        raise ValueError("Stored inventory differs from frozen selection")
    return files


def array_sha(x):
    return hashlib.sha256(np.asarray(x, dtype="<f4").tobytes()).hexdigest()


def metrics(waves, target):
    with torch.no_grad():
        objective = Objective(target, cast_config())
        total, terms = objective(waves, detail=True)
        normalized = shape(waves).numpy()
        truth = shape(target).numpy()
    # 50 ms non-overlapping RMS envelopes on the same normalized profile.
    a = np.sqrt(np.mean(normalized.reshape(len(waves), -1, 800)**2, axis=-1))
    b = np.sqrt(np.mean(truth.reshape(-1, 800)**2, axis=-1))
    return {"check_loss": float(total.mean()), "realization_losses": total.tolist(),
            "per_resolution": {n: {k: float(v.mean()) for k, v in term.items()} for n, term in terms.items()},
            "envelope_rmse": float(np.sqrt(np.mean((a-b)**2))),
            "raw_waveform_sha256_float32": [array_sha(x) for x in waves.numpy()]}


def diagnostics(out, raw, controls, waves, records):
    sr = 16000
    t = np.arange(len(raw))/sr
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), constrained_layout=True)
    axes[0].plot(t, controls["rpm"], label="RPM")
    axes[0].set(ylabel="RPM", title="Prescribed telemetry and observation reconstruction")
    ax = axes[0].twinx(); ax.plot(t, controls["torque_nm"], color="tab:orange"); ax.set_ylabel("Torque (Nm)")
    all_waves = {"reference synthetic": torch.tensor(raw), **{k: v[0] for k, v in waves.items()}}
    for name, y in all_waves.items():
        y = shape(y).numpy()
        f, p = welch(y, fs=sr, nperseg=2048)
        axes[1].plot(f, 10*np.log10(p+1e-12), label=name)
        env = np.sqrt(np.mean(y.reshape(-1, 800)**2, axis=1))
        axes[2].plot((np.arange(len(env))+.5)*.05, env, label=name)
    axes[1].set(xlim=(0, 4000), xlabel="Hz", ylabel="PSD (dB, unit RMS)")
    axes[1].legend(fontsize=8, ncol=2)
    axes[2].set(xlabel="Seconds", ylabel="50 ms RMS envelope")
    names = list(records)
    axes[3].bar(names, [records[k]["metrics"]["check_loss"] for k in names])
    axes[3].set(ylabel="MRSTFT checking loss", title="Lower is better; per-clip reconstruction, synthetic development data")
    fig.savefig(out/"diagnostics.png", dpi=135)
    plt.close(fig)


def demo(out, cfg):
    d = cfg["demo"]; sr = 16000; n = d["seconds"]*sr
    controls = {"rpm": np.linspace(d["rpm_start"], d["rpm_end"], n),
                "torque_nm": np.linspace(d["torque_start_nm"], d["torque_end_nm"], n)}
    started = time.perf_counter()
    source, components = render(**controls, engine=default_engine(d["cylinders"], cfg), seed=d["seed"])
    x = np.linspace(d["path_x_start_m"], d["path_x_end_m"], n)
    distance = np.hypot(x, d["path_y_m"])
    mic = free_field(source, distance, sound_speed_m_s=d["sound_speed_m_s"])
    elapsed = time.perf_counter()-started
    out.mkdir()
    np.savez(out/"states.npz", **controls, distance_m=distance, x_m=x)
    gain = min(1., .95/max(np.max(abs(source)), np.max(abs(mic))))
    for name, wave in {"source_relative": source, "microphone_relative": mic,
                       **components}.items():
        sf.write(out/(name+".wav"), wave, sr, subtype="FLOAT")
    for name, wave in (("source", source), ("passby", mic)):
        sf.write(out/(name+"_playback.wav"), wave*gain, sr, subtype="FLOAT")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    time_axis = np.arange(n)/sr
    axes[0].plot(time_axis, controls["rpm"]); axes[0].set(ylabel="RPM", title="Eight-second RPM sweep and prescribed pass-by")
    axes[1].plot(time_axis, distance); axes[1].set(ylabel="Distance (m)")
    axes[2].specgram(mic, NFFT=1024, Fs=sr, noverlap=768, cmap="magma", vmin=-85, vmax=-15)
    axes[2].set(ylim=(0, 2500), xlabel="Seconds", ylabel="Hz")
    fig.savefig(out/"demo.png", dpi=135); plt.close(fig)
    record = {"seconds": elapsed, "audio_seconds": d["seconds"], "real_time_factor": elapsed/d["seconds"],
              "configuration": d, "engine": default_engine(d["cylinders"], cfg).record(),
              "common_playback_gain": gain, "source_sha256_float32": array_sha(source),
              "microphone_sha256_float32": array_sha(mic), "units": "relative_uncalibrated_pressure",
              "endpoint": "zero_outside_emitted_support_tail_not_extended"}
    data.save(out/"parameters.json", record)
    return record


def clip(out, row, root, cfg):
    audio, rpm, torque, sr = data.decode(data.safe_path(root, row["path"]), row)
    raw, target, controls, levels = prepare(audio, rpm, torque, sr)
    identifier = row["capture_configuration"]+"_"+row["filename_trace_candidate"]
    out.mkdir()
    np.savez(out/"controls.npz", **controls)
    sf.write(out/"original.wav", raw, 16000, subtype="FLOAT")
    ccfg, scfg = cast_config(), model_config()
    records = {
        "physics_default": {"parameters": default_engine(row["cylinders_assumed"], cfg).record()},
        "physics_fitted": fit(target, controls, row["cylinders_assumed"], identifier, cfg),
        "cast_v0": cast_fit(target, ccfg, identifier),
        "cast_smooth8": smooth_fit(target, scfg, identifier),
    }
    with torch.no_grad():
        waves = {
            "physics_default": rendered(Engine(**records["physics_default"]["parameters"]), controls, identifier, "check", cfg),
            "physics_fitted": rendered(Engine(**records["physics_fitted"]["parameters"]), controls, identifier, "check", cfg),
            "cast_v0": Renderer(ccfg, identifier).render(tensors(records["cast_v0"]["parameters"]), "check")[0],
            "cast_smooth8": SmoothRenderer(scfg, identifier).render(tensors(records["cast_smooth8"]["parameters"]), "check")[0],
        }
        listen = {"original": np.array(raw)-levels["dc"],
                  **{k: shape(v[0]).numpy()*levels["ac_rms"] for k, v in waves.items()}}
    common_gain = min(1., 0.95/max(np.max(abs(x)) for x in listen.values()))
    scientific_failures = []
    for name, wave in waves.items():
        records[name]["metrics"] = metrics(wave, target)
        for i, w in enumerate(wave.numpy()):
            sf.write(out/f"{name}_raw_{i}.wav", w, 16000, subtype="FLOAT")
        if "baseline_check_loss" in records[name]:
            gain = 1-records[name]["metrics"]["check_loss"]/records[name]["baseline_check_loss"]
            records[name]["check_improvement_fraction"] = gain
            if gain < cfg["fit"]["minimum_check_improvement_fraction"]:
                scientific_failures.append({"arm": name, "reason": "less_than_5_percent_checking_improvement", "fraction": gain})
    for name, wave in listen.items():
        sf.write(out/f"{name}_playback.wav", wave*common_gain, 16000, subtype="FLOAT")
    best = records["physics_fitted"]
    keys = ("pulse_width_ms", "resonance_hz", "resonance_q", "noise_gain")
    best["grid_boundary_parameters"] = [k for k in keys if best["parameters"][k] in (cfg["fit"][k][0], cfg["fit"][k][-1])]
    if best["ambiguous"]:
        scientific_failures.append({"arm": "physics_fitted", "reason": "near_equivalent_grid_parameters", "count": len(best["near_equivalent_candidates"])})
    record = {"input_id": identifier, "parent": row, "preprocessing": levels,
              "target_sha256_float32": array_sha(target.numpy()), "common_playback_gain": common_gain,
              "records": records, "scientific_failures": scientific_failures,
              "comparison": "same_crop_bandlimit_and_objective_different_information_and_optimization_budgets"}
    data.save(out/"result.json", record)
    diagnostics(out, raw, controls, waves, records)
    return record


def index(out, results, summary):
    text = ["<!doctype html><html><head><meta charset='utf-8'><title>Physics engine and CAST comparison</title>",
            "<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;background:#f6f7fa;color:#182332} img{max-width:100%}section{background:white;padding:24px;margin:24px 0;border-radius:12px}td,th{padding:10px;text-align:left}audio{width:280px}p{line-height:1.5}</style></head><body>",
            "<h1>Physics engine and CAST comparison</h1><p>Sixteen synthetic development observations. RPM and torque are supplied to the physical model; CAST fits the observation without telemetry. No independent generalization or classifier claim.</p>",
            "<section><h2>Prescribed pass-by</h2><p>The same source approaches and recedes with distance attenuation and retarded-time Doppler. Relative pressure units; both players share one playback gain.</p>",
            "<p>Source <audio controls src='demo/source_playback.wav'></audio> Microphone <audio controls src='demo/passby_playback.wav'></audio></p><img src='demo/demo.png' alt='RPM, distance and microphone spectrogram'></section>"]
    text.append("<section><h2>Mean checking loss</h2><table><tr><th>Arm</th><th>MRSTFT</th><th>Envelope RMSE</th></tr>")
    for name, values in summary["arms"].items():
        text.append(f"<tr><td>{name}</td><td>{values['mean_check_loss']:.4f}</td><td>{values['mean_envelope_rmse']:.4f}</td></tr>")
    text.append("</table><p>Lower is better. These are correlated, in-sample reconstruction diagnostics.</p></section>")
    for row in results:
        key = row["input_id"]
        text.append(f"<section><h2>{html.escape(key)}</h2><p>Parent: {html.escape(row['parent']['path'])}</p>")
        for arm in ["original", *row["records"]]:
            text.append(f"<p>{arm.replace('_',' ')} <audio controls preload='none' src='clips/{key}/{arm}_playback.wav'></audio></p>")
        text.append(f"<img loading='lazy' src='clips/{key}/diagnostics.png' alt='Spectral, envelope and loss comparison'><p><a href='clips/{key}/result.json'>Parameters, losses, alternatives and failures</a></p></section>")
    text.append("<p><a href='audit.json'>Dataset audit</a> · <a href='summary.json'>Full results</a> · <a href='snapshot.json'>Provenance</a></p></body></html>")
    (out/"index.html").write_text("\n".join(text))


def run(reference, output, download=False):
    cfg = load(reference)
    if not output.startswith("artifact:physics_engine/"):
        raise ValueError("Run output must be an additive artifact:physics_engine/ path")
    out = resolve(output, must_exist=False)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deterministic()
    try:
        data.save(out/"snapshot.json", snapshot(out, cfg, reference))
        root = resolve(cfg["data_root"], must_exist=False)
        lock = json.loads(resolve(cfg["input_lock"]).read_text())
        files = source_inventory(root, lock, download)
        audit = data.audit(root, lock, files, download=download)
        data.save(out/"audit.json", audit)
        if audit["status"] != "pass":
            raise ValueError("Dataset admission failed; inspect audit.json")
        demo_record = demo(out/"demo", cfg)
        results, failures = [], []
        (out/"clips").mkdir()
        for number, row in enumerate(lock["selected"], 1):
            key = row["capture_configuration"]+"_"+row["filename_trace_candidate"]
            print(f"[{number}/{len(lock['selected'])}] {key}", flush=True)
            try:
                result = clip(out/"clips"/key, row, root, cfg)
                results.append(result)
                print(json.dumps({k: round(v["metrics"]["check_loss"], 5) for k, v in result["records"].items()}), flush=True)
            except Exception as exc:
                failures.append({"input_id": key, "type": type(exc).__name__, "error": str(exc)})
                data.save(out/"clips"/(key+"_failure.json"), failures[-1])
        arms = {}
        for name in ("physics_default", "physics_fitted", "cast_v0", "cast_smooth8"):
            values = [r["records"][name] for r in results]
            if values:
                losses = np.array([v["metrics"]["check_loss"] for v in values])
                arms[name] = {"mean_check_loss": float(losses.mean()), "median_check_loss": float(np.median(losses)),
                              "min_check_loss": float(losses.min()), "max_check_loss": float(losses.max()),
                              "mean_envelope_rmse": float(np.mean([v["metrics"]["envelope_rmse"] for v in values])),
                              "fit_seconds": sum(v.get("fit_seconds", 0) for v in values)}
        def group_results(field):
            return {g: {arm: float(np.mean([r["records"][arm]["metrics"]["check_loss"] for r in results if r["parent"][field] == g])) for arm in arms}
                    for g in sorted({r["parent"][field] for r in results})}
        summary = {"status": "pass" if len(results) == len(lock["selected"]) and not failures else "fail",
                   "successful_clips": len(results), "planned_clips": len(lock["selected"]),
                   "arms": arms, "by_timbre_engine_descriptive_only": group_results("engine"),
                   "by_filename_trace_candidate_descriptive_only": group_results("filename_trace_candidate"),
                   "uncertainty": "not_estimable_as_independent_group_CI_one_conservative_development_group",
                   "implementation_failures": failures,
                   "scientific_failures": [{"input_id": r["input_id"], **f} for r in results for f in r["scientific_failures"]],
                   "demo": demo_record, "audit_seconds": audit["seconds"],
                   "workflow_seconds": time.perf_counter()-started,
                   "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
                   "benchmark_preservation": baseline_check(), "output": output,
                   "data_domain": "synthetic_only", "target_or_protected_audio_opened": False,
                   "optimization": "physics_54_grid_candidates_two_fit_streams_CAST_4_starts_300_Adam_steps_two_fit_streams",
                   "no_performance_acceptance": "negative_reconstruction_results_are_valid_scientific_outcomes"}
        data.save(out/"summary.json", summary)
        index(out, results, summary)
        artifacts = {str(p.relative_to(out)): data.sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
        data.save(out/"artifacts.json", artifacts)
        print(json.dumps({"output": str(out), "status": summary["status"], "seconds": summary["workflow_seconds"]}), flush=True)
        return summary["status"] == "pass"
    except Exception as exc:
        data.save(out/"workflow_failure.json", {"type": type(exc).__name__, "error": str(exc), "elapsed_seconds": time.perf_counter()-started})
        raise


def replay(output, *, refit_first=False):
    started = time.perf_counter()
    out = resolve(output)
    saved = json.loads((out/"snapshot.json").read_text())
    cfg = load(saved["config_path"])
    if cfg != saved["configuration"]:
        raise ValueError("Replay configuration mismatch")
    baseline_check()
    for ref, expected in saved["source_hashes"].items():
        if data.sha(resolve(ref)) != expected:
            raise ValueError(f"Replay source changed: {ref}")
    for relative, expected in json.loads((out/"artifacts.json").read_text()).items():
        if data.sha(data.safe_path(out, relative)) != expected:
            raise ValueError(f"Artifact integrity failed: {relative}")
    deterministic()
    results = sorted((out/"clips").glob("*/result.json"))
    counts = {"clips": 0, "exact_waveforms": 0, "exact_refits": 0}
    for i, path in enumerate(results):
        result = json.loads(path.read_text()); identifier = result["input_id"]
        row = result["parent"]
        audio, rpm, torque, sr = data.decode(data.safe_path(resolve(cfg["data_root"]), row["path"]), row)
        raw, target, controls, _ = prepare(audio, rpm, torque, sr)
        if array_sha(target.numpy()) != result["target_sha256_float32"]:
            raise ValueError("Target reconstruction mismatch")
        for name, record in result["records"].items():
            if name.startswith("physics"):
                wave = rendered(Engine(**record["parameters"]), controls, identifier, "check", cfg)
            else:
                ccfg = cast_config() if name == "cast_v0" else model_config()
                renderer = Renderer if name == "cast_v0" else SmoothRenderer
                with torch.no_grad():
                    wave = renderer(ccfg, identifier).render(tensors(record["parameters"]), "check")[0]
            if metrics(wave, target) != record["metrics"]:
                raise ValueError(f"Waveform/metric replay mismatch: {identifier}/{name}")
            counts["exact_waveforms"] += len(wave)
        if refit_first and i == 0:
            for name, fitter, arguments in (
                ("physics_fitted", fit, (target, controls, row["cylinders_assumed"], identifier, cfg)),
                ("cast_v0", cast_fit, (target, cast_config(), identifier)),
                ("cast_smooth8", smooth_fit, (target, model_config(), identifier))):
                repeated = fitter(*arguments)
                original = result["records"][name]
                if any(value != original[key] for key, value in repeated.items() if key != "fit_seconds"):
                    raise ValueError(f"Exact refit mismatch: {name}")
                counts["exact_refits"] += 1
        counts["clips"] += 1
    if counts["clips"] != 16:
        raise ValueError("Incomplete replay population")
    states = np.load(out/"demo/states.npz")
    d = json.loads((out/"demo/parameters.json").read_text())
    source, _ = render(states["rpm"], states["torque_nm"], Engine(**d["engine"]), seed=cfg["demo"]["seed"])
    mic = free_field(source, states["distance_m"], sound_speed_m_s=cfg["demo"]["sound_speed_m_s"])
    if array_sha(source) != d["source_sha256_float32"] or array_sha(mic) != d["microphone_sha256_float32"]:
        raise ValueError("Demo replay mismatch")
    record = {"status": "pass", **counts, "demo_exact": True, "seconds": time.perf_counter()-started}
    print(json.dumps(record, sort_keys=True), flush=True)
    return record

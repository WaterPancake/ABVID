"""Descriptive in-sample results, with no classification or coverage claims."""
from collections import Counter
import html
import json
from pathlib import Path
import numpy as np

from .config import ROOT, HERE, save, sha


def grouped(records, key):
    groups = {}
    for value in sorted({r[key] for r in records}):
        all_rows = [r for r in records if r[key] == value]
        rows = [r for r in all_rows if r["status"] == "completed"]
        groups[value] = {"selected": len(all_rows), "completed": len(rows), "numerical_failures": len(all_rows)-len(rows)}
        if rows:
            groups[value].update({"median_initial_check": float(np.median([r["baseline_check_loss"] for r in rows])),
                "median_fitted_check": float(np.median([r["check_loss"] for r in rows])),
                "median_improvement_pct": float(np.median([r["check_improvement_fraction"] for r in rows])*100),
                "p90_fitted_check": float(np.quantile([r["check_loss"] for r in rows],.9)),
                "max_fitted_check": max(r["check_loss"] for r in rows),
                "insufficient_improvement": sum("insufficient_check_loss_improvement" in r["flags"] for r in rows),
                "median_spectral_error": float(np.median([r["spectral_relative_magnitude_error"] for r in rows])),
                "median_temporal_error": float(np.median([r["envelope_rmse"] for r in rows]))})
    return groups


def write_report(out):
    cfg = json.loads((out/"config.resolved.json").read_text())
    lock = json.loads((out/"lock.json").read_text())
    pilot = json.loads((out/"pilot/summary.json").read_text())
    syn = json.loads((out/"synthetic/summary.json").read_text())
    inventory = json.loads((out/"inventory.json").read_text())
    environment = json.loads((out/"snapshot.json").read_text())
    verification = json.loads((out/"verification.json").read_text())
    if not verification["passed"]:raise ValueError("Verification required before report")
    records = pilot["records"]
    rows = [r for r in records if r["status"] == "completed"]
    flags = Counter(flag for r in rows for flag in r["flags"])
    boundaries = Counter(flag for r in rows for flag in r["boundary_hits"])
    ranked = sorted(rows,key=lambda r:(r["check_loss"],r["input_id"]))
    examples = [("Preselected "+r["class"],r) for r in rows if r["input_id"] in inventory["preselected_example_ids"]]
    if ranked:
        examples.extend([(name,ranked[index]) for name,index in (("Best check loss",0),("Median check loss",(len(ranked)-1)//2),("Worst check loss",len(ranked)-1))])
    aggregate = {"by_class": grouped(records,"class"), "by_group": grouped(records,"group"),
                 "flags": dict(flags), "boundary_hits": dict(boundaries),
                 "examples": [{"rule":name,"input_id":r["input_id"]} for name,r in examples],
                 "config_sha256":lock["config_sha256"]}
    save(out/"aggregate.json",aggregate)
    base = "../"+str(out.relative_to(ROOT))
    lines = ["# CAST-0 through CAST-2 pilot", "",
        "CAST is runnable and the bounded pilot has completed. **The numerical prototype is complete; stable, identifiable acoustic coverage is not established. Stop at CAST-2.**", "",
        "The experiment fits an effective observation model. Domain: source-real IDMT observations → fitted synthetic observations, compared in sample with the same real observations. Synthetic acceptance is synthetic → synthetic reconstruction. There is no classifier, held-out validation score, transfer experiment, or claim of vehicle/RPM recovery.", "",
        "## Reproduce", "", "From the ABVID root, using the existing project environment without modifying it:", "",
        "```sh", "bash CAST/run.sh workflow --run-id cast_pilot_reproduction_01", "```", "",
        "Use a fresh run ID: existing runs are never overwritten. This performs the metadata freeze, numerical/provenance tests, seven synthetic fits and the gain-ambiguity check, all 50 IDMT fits, deterministic artifact replay, and reporting. The IDMT files and historical locks must already exist at the recorded paths. See [CAST/README.md](../CAST/README.md) for isolated installation and individual stages.", "",
        f"This run: `{out.name}`. [Resolved configuration]({base}/config.resolved.json), [immutable lock]({base}/lock.json), [source/environment snapshot]({base}/snapshot.json), [selected manifest]({base}/selected_manifest.jsonl), [verification]({base}/verification.json), and [audio gallery]({base}/index.html).", "",
        "## Frozen access and implementation decisions", "",
        "- Exactly 50 recordings (25 car / 25 truck), from five conservative groups at three sites. These are the lexicographically first five IDs per group/class within the existing H1 outer-fold-0, seed-42 training selection. No resampling of the training selection and no shortages.",
        "- Held-out group `connected_4001f06f57cfeee7` is excluded before audio access. No MELAUDIS audio/features, mixed-domain feature cache, supplied procedural/AudioLDM bank, military audio, or reserved audio is opened. File bytes are hashed only after exact ID/path/metadata/ancestry checks; symlinks and path traversal are rejected.",
        "- A single class-agnostic renderer has three 10–400 Hz effective-spacing knots, eight harmonic amplitude weights, eight noise-band energy weights, one harmonic-energy fraction, and five envelope knots. Source, background, path and sensor contributions are not identified separately.",
        "- Internal 8 kHz; disjoint rectangular FFT noise masks with DC/Nyquist excluded, each normalized to unit RMS. Harmonics are suppressed at/above Nyquist. A differentiable 41-tap Kaiser-5 FIR matches SciPy's fixed 8→16 kHz resampler. Linear endpoint interpolation and full component waveforms are recorded.",
        "- Input is the arithmetic mean of the two stored CH34 channels, float64 native→8→16 kHz polyphase resampling, centered two seconds, then float32. No padding. CAST fitting subtracts DC and divides by AC RMS; RMS below 1e-7 is rejected. H1 files/preprocessing are unchanged.",
        "- Four starts `[42,123,456,789]`, 300 Adam steps each; control LR 0.03 and spacing-coordinate LR 0.001 (100 Hz coordinate scale). Fixed coarse harmonic salience plus local log-parabolic refinement initializes trajectories. All classes have the same bounds, search and budget. The minimum fit-loss state is retained, including the initialization.",
        "- Two fixed fit-noise realizations are averaged. Two independently derived check realizations score the fitted winner and every initialization. The fitted winner is selected using fit loss only. The baseline is the lowest check loss among all unoptimized starts, a conservative comparison on identical check streams. No seed creates independent ancestry.",
        "- Loss averages relative-magnitude plus natural-log-magnitude error at FFT sizes 256/512/1024/2048; periodic Hann, N/4 hop, center=false, unnormalized one-sided FFT, epsilon 1e-5, no regularization or padded frames. Per-resolution terms are saved. Additional spectral/envelope/modulation diagnostics do not affect fitting.",
        "- Playback restores the original AC RMS to the reconstruction and uses one shared pair gain, capped at 0.95 peak without amplification. DC remains removed from the reconstruction. Original observations, shape-normalized waves, raw reconstruction, components and playback pairs are float WAV; independent peak normalization and clipping are not used.", "",
        "## Synthetic acceptance and repairs", "",
        "All criteria were fixed before real decoding. Early engineering runs exposed poor frequency updates and harmonic/subharmonic initialization; their records are retained under `CAST/development/`. Frequency step size and deterministic initialization were repaired without increasing the 300-step budget or relaxing the recovery/improvement tolerances. An unrestricted tone cannot uniquely identify harmonic order. The unambiguous single-tone fixture therefore fixes the known fundamental-only weights while optimizing frequency, noise fraction and envelope. A separate unrestricted-tone fit must recover the observed tone frequency to the same bin tolerance and explicitly retains the harmonic-order ambiguity. This is a synthetic identifiability constraint only; every real fit has all controls free.", "",
        "| Fixture | Fit-loss improvement | Max spacing error (Hz) | Acceptance |", "|---|---:|---:|---|"]
    for r in syn["records"]:
        m=r["measurements"]
        lines.append(f"| {r['fixture']} | {100*m['fit_improvement_fraction']:.2f}% | {m['spacing_max_error_hz']:.3f} | {'pass' if r['passed'] else 'FAIL'} |")
    lines.extend(["", "Single-tone recovery must be within 7.8125 Hz, the longest-STFT bin; harmonic recovery uses the same tolerance, changing trajectories 15.625 Hz, and noise-band L1 error ≤0.35. Every declared nontrivial fixture must improve over its best unoptimized fit-stream start by ≥5%. Missing-fundamental alternatives and the gain/envelope-scale ambiguity are explicitly preserved; unidentifiable physical parameters remain `unknown`.", "",
        f"[Numerical and provenance test log]({base}/tests.log), [synthetic results]({base}/synthetic/summary.json), and [preregistered fixtures/tolerances]({base}/synthetic/acceptance_preregistered.json).", "",
        "## Real in-sample results", "",
        f"{pilot['completed']}/{pilot['count']} fits completed; {pilot['failed']} numerical/input/artifact failures. {flags.get('insufficient_check_loss_improvement',0)}/{len(rows)} completed fits failed the predeclared ≥5% checking-loss improvement diagnostic. These engineering diagnostics are not perceptual or classification acceptance thresholds.", "",
        "| Class | n | Median initial check | Median fitted check | Median paired gain | p90 fitted | Worst fitted | <5% gain |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for label,g in aggregate["by_class"].items():
        if g["completed"]:
            lines.append(f"| {label} | {g['completed']} | {g['median_initial_check']:.4f} | {g['median_fitted_check']:.4f} | {g['median_improvement_pct']:.2f}% | {g['p90_fitted_check']:.4f} | {g['max_fitted_check']:.4f} | {g['insufficient_improvement']} |")
    lines.extend(["", "| Source group | n | Median fitted check | p90 | Worst | Median spectral residual | Median envelope RMSE | <5% gain |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for group,g in aggregate["by_group"].items():
        if g["completed"]:
            lines.append(f"| `{group}` | {g['completed']} | {g['median_fitted_check']:.4f} | {g['p90_fitted_check']:.4f} | {g['max_fitted_check']:.4f} | {g['median_spectral_error']:.4f} | {g['median_temporal_error']:.4f} | {g['insufficient_improvement']} |")
    lines.extend(["", "Spectral residual is relative error between long-term Welch magnitude envelopes; temporal residual is unit-RMS 50 ms envelope RMSE. The four-resolution fitting loss and these descriptors measure different properties. The first check realization is used consistently for plots/descriptors/playback; both checking realizations contribute to checking losses.", "", "Flags (overlapping, all completed fits retained): "+", ".join(f"`{k}`: {v}" for k,v in sorted(flags.items()))+".", "",
        "Boundary hits (within 1% of bounded controls, or weights below 0.001): "+", ".join(f"`{k}`: {v}" for k,v in sorted(boundaries.items()))+".", "",
        "Near-equal starts are within 5% of the best fit objective. Spacing differences above 7.8125 Hz or harmonic-fraction differences above 0.2 are flagged. Noise-dominant fits cannot establish meaningful spacing. Boundary and ambiguity flags are evidence to inspect, not reasons to discard inconvenient examples.", "",
        "## Examples and complete artifacts", "",
        "Preselected examples were chosen before decoding. Best/median/worst examples are ranked by checking loss with ID tie-breaking, including poor cases. Every selected record has a failure/diagnostic record.", "",
        "| Selection | Class | Check loss | Inspection |", "|---|---|---:|---|"])
    for name,r in examples:
        prefix=f"{base}/pilot/{r['input_id']}"
        lines.append(f"| {name} | {r['class']} | {r['check_loss']:.4f} | [original]({prefix}/original_playback.wav) · [reconstruction]({prefix}/reconstructed_playback.wav) · [plot]({prefix}/diagnostics.png) · [parameters]({prefix}/parameters.json) · [flags]({prefix}/failure.json) |")
    fit_times=[r["fit_seconds"] for r in rows]
    lines.extend(["", "Each clip also contains optimization history, all alternative starts and scores, per-resolution checking losses, original RMS/DC, source and parent-lock ancestry, component WAVs/NPZ, timing, and hashes. [Complete pilot summary]"+f"({base}/pilot/summary.json) and [aggregate JSON]({base}/aggregate.json).", "",
        "## Measured runtime and verification", "",
        f"CPU-only `{environment['environment']['machine']}` on `{environment['environment']['platform']}`; one Torch thread. Synthetic stage: {syn['elapsed_seconds']:.2f} s. Real pilot end-to-end: {pilot['elapsed_seconds']:.2f} s for 100 fitted audio seconds, or {pilot['seconds_per_fitted_second']:.2f} s per audio second. Median numerical-fit time: {np.median(fit_times) if fit_times else 0:.2f} s/clip. Process peak RSS: {pilot['process_peak_rss_bytes']/1024**3:.3f} GiB (lifetime high-water mark, not a per-clip incremental measurement).", "",
        f"The first allowed clip was timed before projecting the remaining pilot cost: [projection record]({base}/pilot/runtime_projection.json). Actual timings supersede that projection. Software: "+", ".join(f"{k} {v}" for k,v in environment["environment"]["versions"].items())+".", "",
        f"Verification rehashed all 50 permitted originals, checked per-clip artifact hashes/ancestry, replayed all completed fitted waveforms bit-for-bit in the pinned environment, recomputed checking losses, and checked common-gain playback. Existing tracked ABVID changes remained byte-identical to the starting diff. Commit `{environment['git_commit']}` was dirty; the saved source snapshot, hashes and dependency lock identify this implementation, not the commit alone. Config SHA256 `{lock['config_sha256']}`.", "",
        "## Adequacy verdict and limitations", "",
        "**Adequate as an inspectable bounded reconstruction prototype. Not yet evidence for a stable parameter distribution or real-world transfer.** Synthetic tests establish that numerical fitting works on declared renderer-generated cases. Real checking-loss improvements establish only better in-sample descriptor matching than initialization. Spectral and temporal residuals, parameter boundaries, alternative fits and noise-dominant solutions limit the interpretation of the effective controls.", "",
        "The smallest justified next revision is to inspect the saved spectral/temporal failures and repeated-start disagreements, then propose a separately versioned source-only change aimed at the dominant residual. Preserve these results as the control. Do not create a parameter distribution, add a classifier, evaluate held-out groups or targets, or enlarge the pilot without a new authorized stage. CAST-3 through CAST-6 were not implemented.", "",
        "Two-second observations give little evidence about slow dynamics. A low spectral loss is not a listening-test result, known vehicle identity, recovered RPM, dry source, or causal source/propagation separation. Five conservative groups at three sites do not identify independent physical vehicles or support session-population confidence claims. All 50 observations are engineering development exposures. Input recordings and generated derivatives retain their ancestry and local-only evaluation status; IDMT's recorded licence is CC BY-NC-ND 4.0. This task makes no redistribution/publication authorization claim.", ""])
    report="\n".join(lines)
    (out/"CAST_pilot.md").write_text(report.replace(base,".").replace("../CAST/README.md","../../README.md"))
    # Canonical user-requested additive report; refuse to replace an existing report.
    report_path=ROOT/"reports/CAST_pilot.md"
    if not report_path.exists():report_path.write_text(report)
    else:print(f"Existing {report_path} preserved; new report is in {out}",flush=True)
    cards=[]
    for r in records:
        rid=r["input_id"]; prefix=f"pilot/{rid}"
        if r["status"]!="completed":
            cards.append(f"<article><h2>{html.escape(rid)}</h2><p>Failed: {html.escape(r.get('error','unknown'))}</p></article>")
            continue
        flags_text=html.escape(", ".join(r["flags"]) or "No diagnostic flags")
        cards.append(f'<article><h2>{html.escape(r["class"])} · {rid[:12]}</h2><p>{r["group"]} · check loss {r["check_loss"]:.4f} · gain {100*r["check_improvement_fraction"]:.1f}%</p><p>{flags_text}</p><label>Original <audio controls preload="none" src="{prefix}/original_playback.wav"></audio></label><label>Reconstruction <audio controls preload="none" src="{prefix}/reconstructed_playback.wav"></audio></label><p><a href="{prefix}/parameters.json">Parameters & ancestry</a> · <a href="{prefix}/optimization.json">Optimization</a> · <a href="{prefix}/losses.json">Losses</a></p><img loading="lazy" src="{prefix}/diagnostics.png" alt="Spectral and temporal diagnostics"></article>')
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>CAST in-sample pilot</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;background:#f7f8fa;color:#172332}article{background:white;padding:24px;margin:24px 0;border:1px solid #ccd3dd;border-radius:10px}img{max-width:100%}label{display:inline-block;margin-right:20px}audio{display:block}h1,h2{line-height:1.3}</style><h1>CAST: all 50 in-sample observation fits</h1><p>Source-only IDMT engineering pilot. Common playback gain within each pair. Effective controls are not physical measurements. No classification or generalization result.</p>'+"\n".join(cards))
    save(out/"deliverable_hashes.json",{str(p.relative_to(out)):sha(p) for p in (out/"aggregate.json",out/"CAST_pilot.md",out/"index.html",out/"verification.json")})

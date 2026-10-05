"""Numeric components extracted from CAST/src/cast/artifacts.py."""
from pathlib import Path


import numpy as np


import soundfile as sf


import torch


from scipy.signal import stft, welch


from abvid.cast.audio import normalize, shape


from abvid.cast.config import digest, save, sha


from abvid.cast.fit import Objective


from abvid.cast.renderer import Renderer, tensors, validate_controls


def descriptors(x, cfg):
    n = cfg["diagnostics"]["envelope_frame_samples"]
    x = np.asarray(x, dtype=np.float64)
    x = (x-x.mean())/max(np.sqrt(np.mean((x-x.mean())**2)), 1e-12)
    env = np.sqrt(np.mean(x.reshape(-1, n)**2, axis=1))
    f, power = welch(x, cfg["audio"]["sample_rate_hz"], window="hann", nperseg=2048, noverlap=1024)
    bands = np.array([power[(f >= lo) & (f < hi)].sum() for lo, hi in zip(cfg["renderer"]["noise_edges_hz"][:-1], cfg["renderer"]["noise_edges_hz"][1:])])
    bands /= max(bands.sum(), 1e-12)
    mod = np.abs(np.fft.rfft(env-env.mean()))
    mod_freq = np.fft.rfftfreq(len(env), n/cfg["audio"]["sample_rate_hz"])
    keep = (mod_freq > 0) & (mod_freq <= cfg["diagnostics"]["modulation_max_hz"])
    sel = (f >= 10) & (f < 3200)
    salience = float(power[sel].max()/max(np.median(power[sel]), 1e-12))
    return {"envelope": env, "frequencies": f, "power": power, "band_proportions": bands,
            "modulation": mod[keep], "modulation_hz": mod_freq[keep], "peak_salience_ratio": salience,
            "dominant_peak_hz": float(f[sel][np.argmax(power[sel])])}


def diagnose(target, reconstruction, result, cfg):
    a, b = descriptors(target, cfg), descriptors(reconstruction, cfg)
    d = cfg["diagnostics"]
    p = result["parameters"]
    flags, boundary = [], []
    improvement = 1-result["check_loss"]/result["baseline_check_loss"]
    if improvement < d["scientific_failure_check_improvement_fraction"]:
        flags.append("insufficient_check_loss_improvement")
    for key, bounds in (("spacing_hz", cfg["renderer"]["spacing_hz_bounds"]), ("envelope_knots", cfg["renderer"]["envelope_bounds"]), ("harmonic_fraction", [0, 1])):
        lo, hi = bounds
        x = np.asarray(p[key]); tol = (hi-lo)*d["boundary_fraction"]
        if np.any((x <= lo+tol) | (x >= hi-tol)):
            boundary.append(key)
    for key in ("harmonic_weights", "noise_weights"):
        if min(p[key]) < d["weight_boundary_threshold"]:
            boundary.append(key)
    if p["harmonic_fraction"] < d["noise_dominant_mix_threshold"]:
        flags.append("spacing_unidentified_noise_dominant")
    if a["peak_salience_ratio"] < d["salience_resolvable_ratio"]:
        flags.append("no_resolvable_harmonic_salience")
    near = [s for s in result["starts"] if s["fit_loss"] <= result["fit_loss"]*(1+d["equally_good_relative_loss"])]
    spacing_range = float(np.ptp([s["parameters"]["spacing_hz"] for s in near], axis=0).max())
    mix_range = float(np.ptp([s["parameters"]["harmonic_fraction"] for s in near]))
    if spacing_range > d["spacing_disagreement_hz"] or mix_range > d["mix_disagreement"]:
        flags.append("equally_good_starts_disagree")
    if max(p["harmonic_weights"]) > .9 or p["harmonic_weights"][0] < .05:
        flags.append("harmonic_order_ambiguity")
    spectral_error = float(np.abs(np.sqrt(a["power"])-np.sqrt(b["power"])).mean()/(np.sqrt(a["power"]).mean()+1e-12))
    temporal_error = float(np.sqrt(np.mean((a["envelope"]-b["envelope"])**2)))
    if spectral_error > d["poor_spectral_relative_threshold"]:
        flags.append("large_spectral_residual")
    if temporal_error > d["poor_envelope_rmse_threshold"]:
        flags.append("large_temporal_residual")
    return {"flags": flags, "boundary_hits": boundary, "check_improvement_fraction": improvement,
            "spectral_relative_magnitude_error": spectral_error, "envelope_rmse": temporal_error,
            "band_energy_L1_error": float(np.abs(a["band_proportions"]-b["band_proportions"]).sum()),
            "modulation_L1_error": float(np.abs(a["modulation"]-b["modulation"]).mean()),
            "target_band_proportions": a["band_proportions"].tolist(), "render_band_proportions": b["band_proportions"].tolist(),
            "target_peak_salience_ratio": a["peak_salience_ratio"], "render_peak_salience_ratio": b["peak_salience_ratio"],
            "near_equal_start_count": len(near), "near_equal_spacing_spread_hz": spacing_range,
            "near_equal_mix_spread": mix_range, "physical_identifiability": "not_established"}


def plot(out, target, reconstructed, baseline, result, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sr = cfg["audio"]["sample_rate_hz"]
    a, b, c = [descriptors(x, cfg) for x in (target, reconstructed, baseline)]
    fig, ax = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    t = np.arange(len(target))/sr
    ax[0, 0].plot(t[::8], target[::8], label="Original", alpha=.65, lw=.5)
    ax[0, 0].plot(t[::8], reconstructed[::8], label="Reconstruction", alpha=.65, lw=.5)
    ax[0, 0].set(xlabel="Time (s)", ylabel="Unit RMS", title="Shape-normalized observations")
    ax[0, 0].legend()
    for data, name in ((a, "Original"), (b, "Fitted"), (c, "Best initial")):
        ax[0, 1].plot(data["frequencies"], 10*np.log10(data["power"]+1e-12), label=name)
        ax[1, 0].plot((np.arange(len(data["envelope"]))+.5)*.05, data["envelope"], label=name)
    ax[0, 1].set(xlim=(0, 4000), xlabel="Frequency (Hz)", ylabel="PSD (dB)", title="Spectral residual")
    ax[0, 1].legend()
    ax[1, 0].set(xlabel="Time (s)", ylabel="50 ms RMS", title="Temporal residual")
    hist = np.array([v["fit_losses"] for v in result["history"]])
    ax[1, 1].plot(hist)
    ax[1, 1].set(xlabel="Adam step", ylabel="Fit objective", title="All four starts")
    specs = []
    for x in (target, reconstructed):
        f, t, z = stft(x, sr, nperseg=512, noverlap=384, boundary=None, padded=False)
        specs.append(20*np.log10(np.abs(z)+1e-5))
    hi = max(s.max() for s in specs)
    for axis, spec, name in zip(ax[2], specs, ("Original spectrogram", "Reconstructed spectrogram")):
        axis.pcolormesh(t, f, spec, shading="auto", cmap="magma", vmin=hi-70, vmax=hi)
        axis.set(ylim=(0, 4000), xlabel="Time (s)", ylabel="Hz", title=name)
    fig.suptitle(result["input_id"][:30]+" — CAST observation fit")
    fig.savefig(out/"diagnostics.png", dpi=130)
    plt.close(fig)


def save_fit(out, original, result, cfg, ancestry):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    target, original_stats = normalize(original, cfg)
    p = validate_controls(result["parameters"], cfg)
    renderer = Renderer(cfg, result["input_id"])
    with torch.no_grad():
        raw, components = renderer.render(tensors(p), "check", components=True)
        recon = shape(raw)[0, 0].numpy()
        initial = result["starts"][result["baseline_check_winner"]]["initial_parameters"]
        baseline = shape(renderer.render(tensors(initial), "check"))[0, 0].numpy()
        total, terms = Objective(target, cfg)(raw, detail=True)
    # Raw renderer components are saved before overall shape normalization so
    # their sum exactly reconstructs the stored raw observation.
    waves = {"original": original, "original_shape": target.numpy(), "reconstructed_shape": recon,
             "reconstructed_raw": raw[0, 0].numpy(), "baseline_shape": baseline,
             **{k+"_component": components[k][0, 0].numpy() for k in ("harmonic", "noise")}}
    paired_recon = recon*original_stats["ac_rms"]
    common_gain = min(1.0, cfg["normalization"]["playback_peak_limit"]/max(float(np.abs(original).max()), float(np.abs(paired_recon).max()), 1e-12))
    waves["original_playback"] = original*common_gain
    waves["reconstructed_playback"] = paired_recon*common_gain
    sr = cfg["audio"]["sample_rate_hz"]
    for name, x in waves.items():
        sf.write(out/(name+".wav"), x, sr, subtype="FLOAT")
    # Losses average two independently generated check realizations; listening
    # and plots consistently show the first, without cherry-picking noise.
    np.savez(out/"components.npz", **{k: v.detach().numpy() for k, v in components.items()}, check_waveforms=raw.numpy())
    diagnostics = diagnose(target.numpy(), recon, result, cfg)
    plot(out, target.numpy(), recon, baseline, result, cfg)
    save(out/"parameters.json", {"config_sha256": digest(cfg), "renderer_version": cfg["renderer"]["version"],
        "effective_controls": p, "physical_metadata": {k: "unknown" for k in ("RPM", "throttle", "distance", "physical_vehicle_id", "weather", "load")},
        "streams": result["streams"], "input_id": result["input_id"], "ancestry": ancestry,
        "original_level": original_stats, "common_playback_gain": common_gain,
        "reconstruction_playback_dc": 0.0, "fit_normalization": cfg["normalization"]["fit"]})
    save(out/"losses.json", {"config_sha256": digest(cfg), **{k: result[k] for k in ("fit_loss", "check_loss", "baseline_fit_loss", "baseline_check_loss", "winner", "baseline_check_winner", "selection", "fit_seconds")},
                            "per_resolution_check": {n: {k: v[0].tolist() for k, v in terms.items()} for n, terms in terms.items()},
                            "diagnostics": diagnostics})
    save(out/"optimization.json", {k: result[k] for k in ("initialization", "starts", "history", "maximum_gradient_norm_before_clip", "known_harmonic_weights")})
    save(out/"failure.json", {"numerical_failure": False, "scientific_flags": diagnostics["flags"],
                             "boundary_hits": diagnostics["boundary_hits"], "ambiguity_not_physical_estimate": True})
    save(out/"hashes.json", {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    return diagnostics

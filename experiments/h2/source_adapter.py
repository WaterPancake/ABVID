"""Explicit frozen S0/S1 source parameters; no implicit path or target-data access.

Component definitions are ported from the P02 MIT-labelled dry-source functions;
see evidence/p02/_2_genSyntheticData.py and its release README. Source draws are
fully externalized in the frozen template manifest; no wrapper/fallback is used.
"""
import hashlib
import numpy as np
from scipy.signal import lfilter
from deterministic_filters import butterworth


def array_sha(x):
    return hashlib.sha256(np.asarray(x, dtype="<f8", order="C").tobytes()).hexdigest()


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64)**2)))


def peak_normalize(x):
    peak = float(np.max(np.abs(x)))
    if not np.isfinite(x).all() or peak <= 0:
        raise ValueError("Nonfinite or silent component")
    return x/peak


def gaussian(seed, n):
    return np.random.Generator(np.random.PCG64(seed)).normal(0, 1, n)


def synthesize(template, level, cfg, return_components=False):
    if level not in ("S0", "S1"):
        raise ValueError("Unknown source factor")
    s = cfg["source"]; fs = s["rate_hz"]; n = s["samples"]
    b = template["base"]; enrichment = template["S1"]
    t = np.arange(n, dtype=np.float64)/fs
    f0 = b["firing_hz"]
    instantaneous = np.full(n, f0)
    if level == "S1":
        instantaneous *= 1 + enrichment["frequency_fractional_amplitude"]*np.sin(
            2*np.pi*enrichment["frequency_modulation_hz"]*t + enrichment["phase_rad"])
        phase_increment = np.r_[0., np.cumsum(instantaneous[:-1])]*(2*np.pi/fs)
    else:
        # Exact released constant-frequency expression (not repeated addition).
        phase_increment = 2*np.pi*f0*t
    rolloff = b["engine_rolloff_db_per_order"] + (
        enrichment["engine_rolloff_offset_db_per_order"] if level == "S1" else 0)
    engine = np.zeros(n)
    for h, phase in enumerate(b["engine_phases_rad"], 1):
        if f0*h > fs/2: break
        engine += 10**(-rolloff*(h-1)/20)*np.sin(h*phase_increment+phase)
    engine *= 1 + b["engine_am_depth"]*np.sin(2*np.pi*(f0/2)*t)
    engine = peak_normalize(engine)
    low = max(20., b["tire_center_hz"]-b["tire_width_hz"]/2)
    high = min(fs/2-1., b["tire_center_hz"]+b["tire_width_hz"]/2)
    coeff = butterworth(4, [low/(fs/2), high/(fs/2)], band=True)
    tire = peak_normalize(lfilter(*coeff, gaussian(b["tire_noise_seed"], n)))
    components = {"engine": engine, "tire": tire}
    diagnostic = {"instantaneous_firing_hz": instantaneous, "phase_increment_rad": phase_increment}
    if "exhaust" in b["weights"]:
        coeff = butterworth(3, min(2000., fs/2-1)/(fs/2))
        broadband = peak_normalize(lfilter(*coeff, gaussian(b["exhaust_noise_seed"], n)))
        harmonics = np.zeros(n)
        for h, phase in enumerate(b["exhaust_phases_rad"], 1):
            if f0*h > fs/2: break
            harmonics += np.sin(h*phase_increment+phase)/h
        harmonics = peak_normalize(harmonics)
        components["exhaust"] = peak_normalize(.6*broadband+.4*harmonics)
        diagnostic["exhaust_broadband"] = broadband
        diagnostic["exhaust_harmonics"] = harmonics
    gains = {k: (10**(enrichment["component_gain_db"][k]/20) if level == "S1" else 1.) for k in components}
    mix = np.zeros(n)
    for k in ["engine", "tire", "exhaust"]:
        if k in components: mix += b["weights"][k]*gains[k]*components[k]
    mix = peak_normalize(mix)
    lo, hi = [round(v*fs) for v in s["rms_interval_s"]]
    before = rms(mix[lo:hi])
    if before <= s["gain_floor_rms"]: raise ValueError("Silent dry source")
    gain = s["rms_target"]/before
    waveform = np.asarray(mix*gain, dtype=np.float64)
    meta = dict(template_id=template["template_id"], source_level=level,
        base_parameters_sha256=template["base_parameters_sha256"], sample_rate_hz=fs, samples=n,
        waveform_sha256=array_sha(waveform), component_sha256={k:array_sha(v) for k,v in components.items()},
        component_gain_linear=gains, pre_rms=before, normalization_gain=gain,
        rms=rms(waveform[lo:hi]), peak=float(np.max(np.abs(waveform))), dtype="float64",
        firing_min_hz=float(instantaneous.min()), firing_max_hz=float(instantaneous.max()),
        engine_rolloff_db_per_order=rolloff)
    if return_components: return waveform, meta, components, diagnostic
    return waveform, meta

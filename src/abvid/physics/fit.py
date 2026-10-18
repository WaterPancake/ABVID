"""Bounded conditional reconstruction using the unchanged CAST spectral loss."""
from dataclasses import replace
from itertools import product
import hashlib
import time

import numpy as np
from scipy.signal import resample_poly
import torch

from abvid.cast.audio import observation, normalize
from abvid.cast.config import configuration as cast_configuration
from abvid.cast.fit import Objective
from .data import canonical
from .engine import Engine, render


def seeds(input_id, purpose, count=2):
    return [int.from_bytes(hashlib.sha256(canonical([42, input_id, "physics", purpose, i])).digest()[:8], "big") % 2**63 for i in range(count)]


def prepare(audio, rpm, torque, sr=48000):
    """Use CAST's exact two-audio-channel route; keep controls as separate data."""
    if sr != 48000 or audio.shape != (len(rpm), 2) or len(torque) != len(rpm):
        raise ValueError("Expected aligned 48 kHz audio and controls")
    cfg = cast_configuration()
    raw = observation(audio, sr, cfg)
    target, levels = normalize(raw, cfg)
    start = (2*((len(audio)+5)//6) - cfg["audio"]["samples"]) // 2
    times = (start + np.arange(len(raw))) / 16000
    native_times = np.arange(len(audio)) / sr
    controls = {"rpm": np.interp(times, native_times, rpm),
                "torque_nm": np.interp(times, native_times, torque)}
    return raw, target, controls, {**levels, "crop_start_seconds": start/16000,
                                 "native_samples": len(audio), "sample_rate_hz": 16000}


def rendered(engine, controls, input_id, purpose, cfg):
    waves = []
    for seed in seeds(input_id, purpose, cfg["fit"][purpose+"_realizations"]):
        source, _ = render(controls["rpm"], controls["torque_nm"], engine,
                           seed=seed, oversample=cfg["source"]["oversample"],
                           preroll_seconds=cfg["source"]["preroll_seconds"])
        matched = resample_poly(resample_poly(source, 1, 2, window=("kaiser", 5)),
                                2, 1, window=("kaiser", 5))
        waves.append(matched)
    return torch.tensor(np.stack(waves), dtype=torch.float32)


def default_engine(cylinders, cfg):
    return Engine(cylinders=cylinders, **{key: cfg["source"][key] for key in
                  ("pulse_width_ms", "resonance_hz", "resonance_q", "noise_gain", "torque_reference_nm", "idle_gain")})


def fit(target, controls, cylinders, input_id, cfg):
    started = time.perf_counter()
    objective = Objective(target, cast_configuration())
    keys = ("pulse_width_ms", "resonance_hz", "resonance_q", "noise_gain")
    base = default_engine(cylinders, cfg)
    history = []
    with torch.no_grad():
        initial_fit = float(objective(rendered(base, controls, input_id, "fit", cfg)).mean())
        initial_check = float(objective(rendered(base, controls, input_id, "check", cfg)).mean())
        for values in product(*(cfg["fit"][key] for key in keys)):
            parameters = replace(base, **dict(zip(keys, values)))
            loss = float(objective(rendered(parameters, controls, input_id, "fit", cfg)).mean())
            if not np.isfinite(loss):
                raise FloatingPointError("Nonfinite physics fitting loss")
            history.append({"parameters": parameters.record(), "fit_loss": loss})
        winner = int(np.argmin([r["fit_loss"] for r in history]))
        best = history[winner]
        check_loss = float(objective(rendered(Engine(**best["parameters"]), controls, input_id, "check", cfg)).mean())
    alternatives = [i for i, row in enumerate(history) if row["fit_loss"] <= best["fit_loss"]*(1+cfg["fit"]["equivalent_relative_loss"])]
    return {"input_id": input_id, "parameters": best["parameters"], "winner": winner,
            "fit_loss": best["fit_loss"], "check_loss": check_loss,
            "baseline_fit_loss": initial_fit, "baseline_check_loss": initial_check,
            "history": history, "near_equivalent_candidates": alternatives,
            "ambiguous": len(alternatives) > 1,
            "fit_seconds": time.perf_counter()-started,
            "streams": {p: seeds(input_id, p) for p in ("fit", "check")},
            "parameter_interpretation": "observation_conditioned_effective_pulse_and_resonator_not_identified_engine_geometry"}

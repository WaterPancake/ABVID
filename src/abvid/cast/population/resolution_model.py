"""Numeric components extracted from CAST/improvement/src/cast_improvement/resolution_model.py."""
from copy import deepcopy


import numpy as np


from abvid.cast.population.capacity_engine import CapacityRenderer, model_config as capacity_config


def model_config():
    cfg = capacity_config()
    cfg["experiment_id"] = "cast_spectrum16_temporal41_v11"
    cfg["renderer"].update(version="cast_spectrum16_temporal41_v11", envelope_control_count=41)
    return cfg


class ResolutionRenderer(CapacityRenderer):
    """Same continuous spectral interpolation and original waveform physics."""


def validate_controls(values, cfg):
    expected = {"spacing_hz": (3,), "harmonic_weights": (8,), "noise_weights": (16,),
                "harmonic_fraction": (), "envelope_knots": (41,)}
    if set(values) != set(expected) or cfg["renderer"].get("envelope_control_count") != 41:
        raise ValueError("Spectral resolution parameter schema mismatch")
    for key, dims in expected.items():
        a = np.asarray(values[key])
        if a.shape != dims or not np.isfinite(a).all():
            raise ValueError("Invalid spectral resolution control tensor")
    for key, bounds in (("spacing_hz", cfg["renderer"]["spacing_hz_bounds"]),
                        ("envelope_knots", cfg["renderer"]["envelope_bounds"]), ("harmonic_fraction", [0, 1])):
        a = np.asarray(values[key])
        if ((a < bounds[0]-1e-6) | (a > bounds[1]+1e-6)).any():
            raise ValueError("Spectral resolution bounds violated")
    for key in ("harmonic_weights", "noise_weights"):
        a = np.asarray(values[key])
        if (a < 0).any() or abs(a.sum()-1) > 1e-6:
            raise ValueError("Invalid spectral resolution simplex")
    return values


def lift_controls(values, cfg):
    p = deepcopy(values)
    if np.asarray(p["noise_weights"]).shape != (8,) or np.asarray(p["envelope_knots"]).shape != (41,):
        raise ValueError("Expected temporal41 controls with eight noise weights")
    edges = np.asarray(cfg["renderer"]["noise_edges_hz"], dtype=float)
    centers = (edges[:-1]+edges[1:])/2
    density = np.maximum(p["noise_weights"], 1e-12)/np.diff(edges)
    new_centers = cfg["renderer"]["noise_control_centers_hz"]
    power = np.exp(np.interp(new_centers, centers, np.log(density))) * np.asarray(cfg["renderer"]["noise_control_reference_widths_hz"])
    p["noise_weights"] = (power/power.sum()).tolist()
    return validate_controls(p, cfg)

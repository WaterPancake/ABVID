"""Numeric components extracted from CAST/improvement/src/cast_improvement/capacity_engine.py."""
from copy import deepcopy


import numpy as np


import torch


from abvid.cast.population.engine import SmoothRenderer, linear_basis, model_config as smooth_config


def model_config():
    cfg = deepcopy(smooth_config())
    edges = np.array(cfg["renderer"]["noise_edges_hz"], dtype=float)
    old = (edges[:-1]+edges[1:])/2
    centers = np.sort(np.concatenate([old, (old[:-1]+old[1:])/2, [4000.]]))
    limits = np.r_[0., (centers[:-1]+centers[1:])/2, 4000.]
    cfg["experiment_id"] = "cast_capacity_noise16_env9_v3"
    cfg["renderer"].update(version="cast_noise16_env9_v3", noise_control_centers_hz=centers.tolist(),
        noise_control_reference_widths_hz=np.diff(limits).tolist(), envelope_control_count=9,
        noise_filter="log_power_linear_at_sixteen_fixed_points", noise_weights="simplex_point_power_times_Voronoi_reference_width")
    return cfg


def validate_controls(values, cfg):
    expected = {"spacing_hz": (3,), "harmonic_weights": (8,), "noise_weights": (16,),
                "harmonic_fraction": (), "envelope_knots": (9,)}
    if set(values) != set(expected):
        raise ValueError("Capacity parameter schema mismatch")
    for key, dims in expected.items():
        a = np.asarray(values[key])
        if a.shape != dims or not np.isfinite(a).all():
            raise ValueError("Invalid capacity control tensor")
    for key, bounds in (("spacing_hz", cfg["renderer"]["spacing_hz_bounds"]),
                        ("envelope_knots", cfg["renderer"]["envelope_bounds"]), ("harmonic_fraction", [0, 1])):
        a = np.asarray(values[key])
        if np.any(a < bounds[0]-1e-6) or np.any(a > bounds[1]+1e-6):
            raise ValueError("Capacity control outside unchanged bounds")
    for key in ("harmonic_weights", "noise_weights"):
        a = np.asarray(values[key])
        if (a < 0).any() or abs(a.sum()-1) > 1e-6:
            raise ValueError("Invalid capacity simplex")
    return values


def lift_controls(values, cfg):
    p = deepcopy(values)
    old_edges = np.array(cfg["renderer"]["noise_edges_hz"], dtype=float)
    old_centers = (old_edges[:-1]+old_edges[1:])/2
    density = np.maximum(p["noise_weights"], 1e-12)/np.diff(old_edges)
    target = cfg["renderer"]["noise_control_centers_hz"]
    powers = np.exp(np.interp(target, old_centers, np.log(density)))*np.array(cfg["renderer"]["noise_control_reference_widths_hz"])
    p["noise_weights"] = (powers/powers.sum()).tolist()
    p["envelope_knots"] = np.interp(np.linspace(0, 1, 9), np.linspace(0, 1, 5), p["envelope_knots"]).tolist()
    return validate_controls(p, cfg)


class CapacityRenderer(SmoothRenderer):
    def __init__(self, cfg, input_id, sampling=False):
        super().__init__(cfg, input_id, sampling=sampling)
        centers = np.array(cfg["renderer"]["noise_control_centers_hz"], dtype=float)
        if centers.shape != (16,) or not np.all(np.diff(centers) > 0):
            raise ValueError("Invalid fixed spectral centers")
        freq = np.fft.rfftfreq(self.n, 1/self.sr)
        self.basis = linear_basis(freq, centers)
        upper = np.searchsorted(centers, freq, side="right").clip(1, len(centers)-1)
        self.lower, self.upper = torch.tensor(upper-1), torch.tensor(upper)
        self.fraction = torch.tensor(((freq-centers[upper-1])/(centers[upper]-centers[upper-1])).clip(0, 1), dtype=torch.float32)
        self.widths = torch.tensor(cfg["renderer"]["noise_control_reference_widths_hz"], dtype=torch.float32)
        if self.widths.shape != (16,) or not torch.all(self.widths > 0):
            raise ValueError("Invalid spectral reference widths")

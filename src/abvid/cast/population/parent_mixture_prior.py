"""Numeric components extracted from CAST/improvement/src/cast_improvement/parent_mixture_prior.py."""
import numpy as np


def calibrate(original, harmonic_power, noise_power, target):
    h, n, y = [np.asarray(v, dtype=np.float64) for v in (harmonic_power, noise_power, target)]
    if not np.isfinite(original) or not 0 <= original <= 1:
        raise ValueError("Invalid original harmonic fraction")
    if any(v.shape != (8,) or not np.isfinite(v).all() or (v < 0).any() for v in (h,n,y)) or abs(y.sum()-1) > 1e-6:
        raise ValueError("Invalid component powers or observed band proportions")
    base = {"original_harmonic_fraction": original, "harmonic_power": h.tolist(), "noise_power": n.tolist(),
            "observed_bands": y.tolist(), "harmonic_fraction": original, "flags": []}
    if original in (0.,1.) or h.sum() <= 1e-12 or n.sum() <= 1e-12:
        base["flags"].append("component_calibration_undefined_at_boundary_or_zero_power")
        return base
    hsum, nsum = float(h.sum()), float(n.sum())
    hs, ns = h/hsum, n/nsum
    d = hs-ns; identifiability = float(d@d)
    base["squared_component_shape_separation"] = identifiability
    if identifiability <= 1e-10:
        base["flags"].append("mixture_calibration_not_identifiable")
        return base
    beta = float(np.clip((y-ns)@d/identifiability, 0., 1.))
    fraction = float(beta*nsum/((1-beta)*hsum+beta*nsum))
    before = (original*h+(1-original)*n)/(original*hsum+(1-original)*nsum)
    after = beta*hs+(1-beta)*ns
    base.update(harmonic_fraction=fraction, harmonic_band_power_fraction=beta,
                original_expected_band_squared_error=float(np.square(before-y).sum()),
                calibrated_expected_band_squared_error=float(np.square(after-y).sum()))
    if base["calibrated_expected_band_squared_error"] > base["original_expected_band_squared_error"]+1e-12:
        raise ValueError("Bounded projection worsened its declared objective")
    if fraction in (0.,1.): base["flags"].append("calibrated_mixture_boundary")
    if fraction < .05: base["flags"].append("spacing_unidentified_after_noise_dominant_calibration")
    return base

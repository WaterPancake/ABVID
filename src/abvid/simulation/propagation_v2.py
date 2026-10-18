"""Renderer v2: H2 propagation with air-filter latency declared, not implicit.

The v1 renderer (``propagation.render_sources``) applies each symmetric 11-tap
air-absorption FIR causally. That adds five samples per application: five on the
direct path and ten on the two reflected legs. The relative direct/reflected
delay therefore exceeds the geometric value by 0.625 ms at 8 kHz, which the H2
attenuation/latency control identified as the source of the H2 path gain.

v2 applies the same coefficients centred (zero added delay) by default, so
relative arrival times follow geometry only. Any extra latency must be passed
explicitly and is returned in the render record. ``air="causal"`` reproduces v1
exactly, with its implicit latency reported. v1 itself is unchanged.
"""
import numpy as np
from numba import njit

from .propagation import fractional_delay, reference_manager, time_varying_fir

AIR_MODES = ("off", "centered", "causal")


@njit(cache=False, fastmath=False)
def centered_fir(x, coefficients):
    """sum_k h[n,k] x[n+c-k] with c the centre tap; zero outside the signal."""
    out = np.zeros_like(x)
    center = coefficients.shape[1] // 2
    for n in range(len(x)):
        for k in range(coefficients.shape[1]):
            j = n + center - k
            if 0 <= j < len(x):
                for column in range(x.shape[1]):
                    out[n, column] += coefficients[n, k] * x[j, column]
    return out


def shift(x, samples):
    """Delay by a declared non-negative integer sample count, zero-filled."""
    if int(samples) != samples or samples < 0:
        raise ValueError("Declared latency must be a non-negative integer sample count")
    samples = int(samples)
    if samples == 0:
        return x
    out = np.zeros_like(x)
    out[samples:] = x[:-samples]
    return out


def render(sources, plan, *, air="centered", ground=True, direct_latency=0, reflected_latency=0):
    """Render the direct path and, if ``ground``, the reflected path.

    Returns components, their sum and a latency record. Latency values are in
    samples at the plan's rate and are added after the physical rendering.
    """
    if air not in AIR_MODES:
        raise ValueError(f"air must be one of {AIR_MODES}")
    if air != "off" and plan["air"] is None:
        raise ValueError("Plan was built without air coefficients")
    x = np.asarray(sources, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or len(x) != len(plan["d"]) or not np.isfinite(x).all() or not np.any(x):
        raise ValueError("Finite explicit length-matched nonzero source required")
    obj = reference_manager()
    dl = obj.primaryDelLine
    filt = {"off": None, "centered": centered_fir, "causal": time_varying_fir}[air]
    taps = plan["air"][0].shape[1] if plan["air"] is not None else 0
    implicit = (taps // 2) if air == "causal" else 0

    direct = fractional_delay(x, plan["delays"][:, 0], dl._sinc_table, dl._delta)
    if filt is not None:
        direct = filt(direct, plan["air"][0])
    direct = shift(direct / plan["d"][:, None], direct_latency)
    reflected = np.zeros_like(direct)
    if ground:
        first = fractional_delay(x, plan["delays"][:, 1], dl._sinc_table, dl._delta)
        if filt is not None:
            first = filt(first, plan["air"][1])
        second = time_varying_fir(first, plan["ground"])
        second = fractional_delay(second, plan["delays"][:, 2], dl._sinc_table, dl._delta)
        if filt is not None:
            second = filt(second, plan["air"][2])
        reflected = shift(second / (plan["a"] + plan["b"])[:, None], reflected_latency)
    total = direct + reflected
    if not np.isfinite(total).all():
        raise ValueError("Nonfinite renderer output")
    latency = {"air_mode": air, "ground": bool(ground),
               "implicit_direct": implicit, "implicit_reflected": 2 * implicit if ground else 0,
               "declared_direct": int(direct_latency), "declared_reflected": int(reflected_latency) if ground else 0}
    latency["relative_reflected_minus_direct"] = (
        latency["implicit_reflected"] + latency["declared_reflected"]
        - latency["implicit_direct"] - latency["declared_direct"]) if ground else None
    return {"direct": direct, "reflected": reflected, "total": total, "latency": latency}

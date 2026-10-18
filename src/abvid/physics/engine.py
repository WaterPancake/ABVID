"""Four-stroke firing clock, causal pulse response, and one exhaust resonance.

This is a reduced-order physical *analogy*, not a combustion/CFD solver. Pressure
is relative, cylinder timing is evenly spaced, and the torque-to-pulse map is a
declared heuristic. Unlike CAST, the controls are prescribed physical states.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import bilinear, butter, lfilter, resample_poly, sosfilt


@dataclass(frozen=True)
class Engine:
    cylinders: int = 4
    pulse_width_ms: float = 0.8
    resonance_hz: float = 400.0
    resonance_q: float = 2.0
    noise_gain: float = 0.03
    torque_reference_nm: float = 400.0
    idle_gain: float = 0.15

    def validate(self):
        if type(self.cylinders) is not int or self.cylinders not in (1, 2, 3, 4, 6, 8, 12):
            raise ValueError("Unsupported cylinder count")
        bounds = {"pulse_width_ms": (0.1, 5), "resonance_hz": (50, 2000),
                  "resonance_q": (0.5, 5), "noise_gain": (0, 0.5),
                  "torque_reference_nm": (1, 1000), "idle_gain": (0, 1)}
        for key, (lo, hi) in bounds.items():
            value = getattr(self, key)
            if not np.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"Invalid engine parameter: {key}")
        return self

    def record(self):
        return asdict(self)


def firing_events(rpm, sample_rate_hz, cylinders):
    """Fractional sample positions of uniform four-stroke firing events.

    RPM is piecewise constant on [i/fs, (i+1)/fs). Starting crank phase is zero.
    A stopped interval has no firings, including at its left boundary.
    """
    rpm = np.asarray(rpm, dtype=np.float64)
    if rpm.ndim != 1 or not len(rpm) or not np.isfinite(rpm).all() or (rpm < 0).any() or (rpm > 10000).any():
        raise ValueError("Expected finite RPM in [0, 10000]")
    if sample_rate_hz <= 0 or type(cylinders) is not int or cylinders <= 0:
        raise ValueError("Invalid clock settings")
    increments = rpm * cylinders / (120 * sample_rate_hz)
    phase = np.concatenate(([0.], np.cumsum(increments)))
    cycles = np.arange(np.ceil(phase[-1]))
    if not len(cycles):
        return np.empty(0)
    indexes = np.searchsorted(phase, cycles, side="right") - 1
    valid = indexes < len(rpm)
    indexes, cycles = indexes[valid], cycles[valid]
    return indexes + (cycles - phase[indexes]) / increments[indexes]


def render(rpm, torque_nm, engine=Engine(), *, sample_rate_hz=16000,
           seed=42, oversample=4, preroll_seconds=0.25):
    """Return relative source pressure and separate tonal/noise components.

    A unit-area event is split across adjacent oversampled samples. Cascaded
    exponential responses approximate h(t)=t exp(-t/tau)/tau^2. The response
    then passes through H(s)=w0^2/(s^2+w0*s/Q+w0^2), discretized by bilinear
    transformation. A 20 Hz highpass removes DC. Anti-aliased downsampling is
    scipy resample_poly (Kaiser 5). No component or per-clip RMS normalization.
    """
    engine.validate()
    rpm, torque = np.asarray(rpm, dtype=np.float64), np.asarray(torque_nm, dtype=np.float64)
    if rpm.ndim != 1 or rpm.shape != torque.shape or not len(rpm) or not np.isfinite(rpm).all() or not np.isfinite(torque).all():
        raise ValueError("Controls must be finite, equally sized 1-D arrays")
    if (rpm < 0).any() or (rpm > 10000).any() or (torque < -200).any() or (torque > 1000).any():
        raise ValueError("Controls outside prototype domain")
    if sample_rate_hz != 16000 or type(oversample) is not int or oversample not in (2, 4, 8):
        raise ValueError("Supported output: 16 kHz; oversampling: 2, 4 or 8")
    if not np.isfinite(preroll_seconds) or not 0.05 <= preroll_seconds <= 1:
        raise ValueError("Invalid preroll")
    sr = sample_rate_hz * oversample
    prefix = round(preroll_seconds * sample_rate_hz)
    rpm = np.pad(rpm, (prefix, 0), mode="edge")
    torque = np.pad(torque, (prefix, 0), mode="edge")
    time = np.arange(len(rpm) * oversample) / oversample
    r = np.interp(time, np.arange(len(rpm)), rpm)
    t = np.interp(time, np.arange(len(torque)), torque)
    amplitude = engine.idle_gain + np.maximum(t, 0) / engine.torque_reference_nm
    events = firing_events(r, sr, engine.cylinders)
    indexes = np.floor(events).astype(int)
    fraction = events - indexes
    weights = np.interp(events, np.arange(len(amplitude)), amplitude) * sr / 100
    impulses = np.zeros(len(r))
    np.add.at(impulses, indexes, weights * (1 - fraction))
    valid = indexes + 1 < len(impulses)
    np.add.at(impulses, indexes[valid] + 1, weights[valid] * fraction[valid])
    pole = np.exp(-1 / (sr * engine.pulse_width_ms * 0.001))
    pulse = lfilter([(1-pole)**2], [1, -2*pole, pole*pole], impulses)
    # sqrt(fs) scaling preserves continuous white-noise spectral density when
    # changing oversampling for convergence checks.
    noise = np.random.default_rng(seed).normal(size=len(r)) * np.sqrt(sr/16000)
    noise *= engine.noise_gain * amplitude * (r > 0)
    w0 = 2*np.pi*engine.resonance_hz
    b, a = bilinear([w0*w0], [1, w0/engine.resonance_q, w0*w0], fs=sr)
    highpass = butter(2, 20, fs=sr, btype="highpass", output="sos")
    def process(x):
        y = sosfilt(highpass, lfilter(b, a, x))
        return resample_poly(y, 1, oversample, window=("kaiser", 5))[prefix:]
    components = {"firing": process(pulse), "broadband": process(noise)}
    wave = components["firing"] + components["broadband"]
    if not np.isfinite(wave).all():
        raise FloatingPointError("Nonfinite physical source")
    return wave, components


def free_field(source, distance_m, *, sample_rate_hz=16000, sound_speed_m_s=343.0):
    """Retarded-time point-source pressure: x(te) / r(te), tr=te+r(te)/c.

    Unit reference distance is 1 m. Linear interpolation; zero outside emitted
    support. Radial motion changes arrival times (kinematic Doppler), with no
    added Doppler formula. No ground, air absorption, cabin, wind or directivity.
    This omits the moving-monopole convective amplitude correction.
    """
    source = np.asarray(source, dtype=np.float64)
    if source.ndim != 1 or len(source) < 2 or not np.isfinite(source).all():
        raise ValueError("Expected finite 1-D source")
    distance = np.broadcast_to(np.asarray(distance_m, dtype=np.float64), source.shape)
    if not np.isfinite(distance).all() or (distance < 1).any():
        raise ValueError("Distance must be finite and at least 1 m")
    if not np.isfinite(sound_speed_m_s) or sound_speed_m_s <= 0 or sample_rate_hz <= 0:
        raise ValueError("Invalid propagation constants")
    if (np.abs(np.diff(distance)) * sample_rate_hz >= sound_speed_m_s).any():
        raise ValueError("Only subsonic radial trajectories are supported")
    emission = np.arange(len(source)) / sample_rate_hz
    arrival = emission + distance / sound_speed_m_s
    return np.interp(emission, arrival, source/distance, left=0, right=0)

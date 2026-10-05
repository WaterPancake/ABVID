"""Numeric components extracted from CAST/src/cast/renderer.py."""
import hashlib


import math


import numpy as np


from scipy.signal import firwin


import torch


import torch.nn.functional as F


from abvid.cast.audio import shape


from abvid.cast.config import canonical


def seed_for(cfg, input_id, component, purpose, index=0):
    value = [cfg["seeds"]["run"], input_id, component, purpose, index]
    return int.from_bytes(hashlib.sha256(canonical(value)).digest()[:8], "big") % 2**63


def interpolate(knots, n):
    return F.interpolate(knots.unsqueeze(1), size=n, mode="linear", align_corners=True).squeeze(1)


def oscillator(spacing, weights, phases, sr):
    """Reject instantaneous components >= Nyquist, including exactly Nyquist."""
    orders = torch.arange(1, weights.shape[-1]+1, dtype=torch.float64)
    phase = 2*math.pi*torch.cumsum(spacing.double(), -1)/sr
    wave = torch.sin(phase[:, None, :]*orders[None, :, None] + phases[None, :, None])
    mask = spacing[:, None, :]*orders[None, :, None] < sr/2
    return (wave.to(weights.dtype)*mask*weights[:, :, None]).sum(1)


class Parameters(torch.nn.Module):
    def __init__(self, values, cfg):
        super().__init__()
        self.cfg = cfg
        self.spacing_raw = torch.nn.Parameter(torch.tensor(values["spacing_hz"], dtype=torch.float32)/cfg["optimizer"]["spacing_coordinate_scale_hz"])
        self.harmonic_logits = torch.nn.Parameter(torch.tensor(values["harmonic_weights"], dtype=torch.float32).clamp_min(1e-12).log())
        self.noise_logits = torch.nn.Parameter(torch.tensor(values["noise_weights"], dtype=torch.float32).clamp_min(1e-12).log())
        self.mix_raw = torch.nn.Parameter(torch.tensor(values["harmonic_fraction"], dtype=torch.float32))
        lo, hi = cfg["renderer"]["envelope_bounds"]
        env = torch.tensor(values["envelope_knots"], dtype=torch.float32)
        self.envelope_raw = torch.nn.Parameter(torch.logit(((env-lo)/(hi-lo)).clamp(1e-6, 1-1e-6)))

    def controls(self):
        r = self.cfg["renderer"]
        lo, hi = r["envelope_bounds"]
        return {"spacing_hz": (self.spacing_raw*self.cfg["optimizer"]["spacing_coordinate_scale_hz"]).clamp(*r["spacing_hz_bounds"]),
                "harmonic_weights": self.harmonic_logits.softmax(-1),
                "noise_weights": self.noise_logits.softmax(-1),
                "harmonic_fraction": self.mix_raw.clamp(0, 1),
                "envelope_knots": lo+(hi-lo)*self.envelope_raw.sigmoid()}

    def project(self):
        # Project raw coordinates too, so a clamped endpoint can re-enter the range.
        with torch.no_grad():
            lo, hi = self.cfg["renderer"]["spacing_hz_bounds"]
            scale = self.cfg["optimizer"]["spacing_coordinate_scale_hz"]
            self.spacing_raw.clamp_(lo/scale, hi/scale)
            self.mix_raw.clamp_(0, 1)


class Renderer:
    def __init__(self, cfg, input_id):
        self.cfg, self.input_id = cfg, input_id
        r = cfg["renderer"]
        self.n, self.sr = r["samples"], r["internal_sample_rate_hz"]
        g = torch.Generator().manual_seed(seed_for(cfg, input_id, "phase", "shared"))
        self.phases = torch.rand(r["harmonics"], generator=g, dtype=torch.float64)*2*math.pi
        taps = firwin(r["upsampler_taps"], 0.5, window=("kaiser", cfg["audio"]["kaiser_beta"])) * 2
        self.taps = torch.tensor(taps, dtype=torch.float32).reshape(1, 1, -1)
        self.banks = {}

    def upsample(self, x):
        dims = x.shape
        flat = x.reshape(-1, dims[-1])
        up = torch.stack((flat, torch.zeros_like(flat)), -1).flatten(1)
        out = F.conv1d(up[:, None, :], self.taps, padding=self.taps.shape[-1]//2)[:, 0]
        return out.reshape(*dims[:-1], self.n*2)

    def noise_bank(self, purpose):
        if purpose not in self.banks:
            freq = torch.fft.rfftfreq(self.n, 1/self.sr)
            edges = self.cfg["renderer"]["noise_edges_hz"]
            count = self.cfg["seeds"][purpose+"_realizations"]
            banks = []
            for i in range(count):
                g = torch.Generator().manual_seed(seed_for(self.cfg, self.input_id, "noise", purpose, i))
                noise = torch.randn(self.n, generator=g)
                spec = torch.fft.rfft(noise)
                bands = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    mask = (freq >= lo) & (freq < hi) & (freq > 0) & (freq < self.sr/2)
                    bands.append(shape(torch.fft.irfft(spec*mask, n=self.n)))
                banks.append(torch.stack(bands))
            self.banks[purpose] = torch.stack(banks)
        return self.banks[purpose]

    def render(self, controls, purpose="fit", components=False):
        f0 = interpolate(controls["spacing_hz"], self.n)
        harmonic = shape(oscillator(f0, controls["harmonic_weights"], self.phases, self.sr))
        # Disjoint FFT bands with unit energy make the softmax noise weights
        # energy proportions before the envelope. No class-specific branch exists.
        noise = torch.einsum("bk,rkn->brn", controls["noise_weights"].sqrt(), self.noise_bank(purpose))
        noise = shape(noise)
        envelope = interpolate(controls["envelope_knots"], self.n)
        envelope = envelope/envelope.square().mean(-1, keepdim=True).sqrt()
        mix = controls["harmonic_fraction"][:, None, None]
        # The tiny clamp gives finite endpoint derivatives; exact zero gain is
        # restored with where, allowing a truly harmonic-only/noise-only fit.
        hg = torch.where(mix > 0, mix.clamp_min(1e-12).sqrt(), 0)
        ng = torch.where(mix < 1, (1-mix).clamp_min(1e-12).sqrt(), 0)
        h = harmonic[:, None, :]*hg*envelope[:, None, :]
        n = noise*ng*envelope[:, None, :]
        wave = self.upsample(h+n)
        if components:
            return wave, {"harmonic": self.upsample(h.expand_as(n)), "noise": self.upsample(n),
                          "envelope_internal": envelope, "spacing_internal_hz": f0,
                          "noise_bands_internal": self.noise_bank(purpose)}
        return wave

    def streams(self):
        return {"phase": seed_for(self.cfg, self.input_id, "phase", "shared"),
                **{p: [seed_for(self.cfg, self.input_id, "noise", p, i) for i in range(self.cfg["seeds"][p+"_realizations"])] for p in ("fit", "check")}}


def tensors(values):
    return {k: torch.tensor([v], dtype=torch.float32) for k, v in values.items()}


def serialize(controls, index=0):
    return {k: v[index].detach().cpu().tolist() for k, v in controls.items()}


def validate_controls(values, cfg):
    expected = {"spacing_hz": (3,), "harmonic_weights": (8,), "noise_weights": (8,),
                "harmonic_fraction": (), "envelope_knots": (5,)}
    if set(values) != set(expected):
        raise ValueError("Parameter schema mismatch")
    for key, dims in expected.items():
        x = np.asarray(values[key])
        if x.shape != dims or not np.isfinite(x).all():
            raise ValueError("Invalid parameter tensor")
    for key, bounds in (("spacing_hz", cfg["renderer"]["spacing_hz_bounds"]),
                        ("envelope_knots", cfg["renderer"]["envelope_bounds"]), ("harmonic_fraction", [0, 1])):
        x = np.asarray(values[key])
        if np.any(x < bounds[0]-1e-6) or np.any(x > bounds[1]+1e-6):
            raise ValueError("Parameter bounds violated")
    for key in ("harmonic_weights", "noise_weights"):
        x = np.asarray(values[key])
        if (x < 0).any() or abs(x.sum()-1) > 1e-6:
            raise ValueError("Invalid simplex weights")
    return values

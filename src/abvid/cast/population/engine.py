"""Numeric components extracted from CAST/improvement/src/cast_improvement/engine.py."""
import numpy as np


import torch


from abvid.cast.audio import shape


from abvid.cast.config import configuration


from abvid.cast.renderer import Renderer, interpolate, oscillator, seed_for


def model_config(variant="smooth8"):
    if variant not in ("smooth8",):
        raise ValueError("Unfrozen renderer variant")
    cfg = configuration()
    cfg["renderer"]["version"] = "cast_smooth_noise_v1_8"
    cfg["renderer"]["noise_filter"] = "log_power_linear_interpolation_at_original_band_centers"
    cfg["renderer"]["noise_weights"] = "simplex_point_power_times_original_band_width"
    cfg["seeds"]["sampling_realizations"] = 1
    return cfg


def linear_basis(grid, centers):
    """Constant endpoint extrapolation, linear interpolation, partition of one."""
    grid, centers = np.asarray(grid), np.asarray(centers)
    upper = np.searchsorted(centers, grid, side="right").clip(1, len(centers)-1)
    lower = upper-1
    frac = ((grid-centers[lower])/(centers[upper]-centers[lower])).clip(0, 1)
    result = np.zeros((len(grid), len(centers)), dtype=np.float32)
    result[np.arange(len(grid)), lower] = 1-frac
    result[np.arange(len(grid)), upper] += frac
    return torch.tensor(result)


class SmoothRenderer(Renderer):
    def __init__(self, cfg, input_id, sampling=False):
        super().__init__(cfg, input_id)
        edges = np.array(cfg["renderer"]["noise_edges_hz"], dtype=np.float64)
        freq = np.fft.rfftfreq(self.n, 1/self.sr)
        centers = (edges[:-1]+edges[1:])/2
        self.basis = linear_basis(freq, centers)
        upper = np.searchsorted(centers, freq, side="right").clip(1, len(centers)-1)
        self.lower, self.upper = torch.tensor(upper-1), torch.tensor(upper)
        self.fraction = torch.tensor(((freq-centers[upper-1])/(centers[upper]-centers[upper-1])).clip(0, 1), dtype=torch.float32)
        self.widths = torch.tensor(np.diff(edges), dtype=torch.float32)
        self.spectra = {}
        if sampling:
            g = torch.Generator().manual_seed(seed_for(cfg, input_id, "phase", "sampling"))
            self.phases = torch.rand(cfg["renderer"]["harmonics"], generator=g, dtype=torch.float64)*2*np.pi

    def spectral_gain(self, weights):
        log_density = (weights.clamp_min(1e-12)/self.widths).log()
        # Two-point interpolation avoids batch-size-dependent GEMM reductions.
        log_gain = log_density[:, self.lower]*(1-self.fraction)+log_density[:, self.upper]*self.fraction
        gain = torch.exp(.5*log_gain)
        # Exclude DC and exact Nyquist, preserving the parent's alias boundary.
        mask = torch.ones(gain.shape[-1]); mask[0] = 0; mask[-1] = 0
        return gain*mask

    def white_spectra(self, purpose):
        if purpose not in self.spectra:
            values = []
            for i in range(self.cfg["seeds"][purpose+"_realizations"]):
                g = torch.Generator().manual_seed(seed_for(self.cfg, self.input_id, "noise", purpose, i))
                values.append(torch.fft.rfft(torch.randn(self.n, generator=g)))
            self.spectra[purpose] = torch.stack(values)
        return self.spectra[purpose]

    def render(self, controls, purpose="fit", components=False):
        f0 = interpolate(controls["spacing_hz"], self.n)
        harmonic = shape(oscillator(f0, controls["harmonic_weights"], self.phases, self.sr))
        gain = self.spectral_gain(controls["noise_weights"])
        noise = shape(torch.fft.irfft(self.white_spectra(purpose)[None]*gain[:, None], n=self.n))
        envelope = interpolate(controls["envelope_knots"], self.n)
        envelope = envelope/envelope.square().mean(-1, keepdim=True).sqrt()
        mix = controls["harmonic_fraction"][:, None, None]
        hg = torch.where(mix > 0, mix.clamp_min(1e-12).sqrt(), 0)
        ng = torch.where(mix < 1, (1-mix).clamp_min(1e-12).sqrt(), 0)
        h = harmonic[:, None]*hg*envelope[:, None]
        n = noise*ng*envelope[:, None]
        wave = self.upsample(h+n)
        if components:
            return wave, {"harmonic": self.upsample(h.expand_as(n)), "noise": self.upsample(n),
                          "envelope_internal": envelope, "spacing_internal_hz": f0,
                          "noise_spectral_gain": gain}
        return wave

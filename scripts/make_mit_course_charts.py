"""Generate real-data / physics charts for the MIT course notes (6.003, 6.551J).

Uses only numpy/scipy/matplotlib; writes PNGs into each course's assets/ dir.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_6003 = ROOT / "docs" / "notes" / "mit-6.003-signals-systems" / "assets"
OUT_6551 = ROOT / "docs" / "notes" / "mit-6.551j-acoustics" / "assets"
for d in (OUT_6003, OUT_6551):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# 6.003 charts
# ---------------------------------------------------------------------------
def chart_aliasing():
    """A 5500 Hz tone sampled at 10 kHz aliases to 4500 Hz (Think DSP example)."""
    fs = 10_000.0
    t = np.linspace(0, 0.002, 8000)
    real = np.cos(2 * np.pi * 5500 * t)
    ts = np.arange(0, 0.002, 1 / fs)
    samples = np.cos(2 * np.pi * 5500 * ts)
    alias_t = np.linspace(0, 0.002, 8000)
    alias = np.cos(2 * np.pi * 4500 * alias_t)

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(t * 1000, real, "b-", lw=1, alpha=0.55, label="5500 Hz (real)")
    ax.plot(ts * 1000, samples, "ro", ms=5, label="samples @ 10 kHz")
    ax.plot(alias_t * 1000, alias, "g--", lw=1.5, alpha=0.9, label="4500 Hz (alias)")
    style_ax(ax, "Aliasing: 5500 Hz sampled at 10 kHz looks like 4500 Hz",
             "Time (ms)", "Amplitude")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_6003 / "aliasing.png")
    plt.close(fig)


def chart_db_scale():
    ratio = np.logspace(-3, 3, 400)
    amp_db = 20 * np.log10(ratio)
    pow_db = 10 * np.log10(ratio)

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(ratio, amp_db, label="amplitude: 20·log₁₀(A₂/A₁)")
    ax.plot(ratio, pow_db, label="power: 10·log₁₀(P₂/P₁)")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(1, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_ylim(-80, 80)
    style_ax(ax, "Decibel scale: amplitude vs power ratios",
             "Ratio", "dB")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_6003 / "db_scale.png")
    plt.close(fig)


def chart_filters():
    """First-order low/high pass and a band-pass magnitude response."""
    f = np.logspace(1, 5, 800)
    fc = 1000.0
    lp = 1 / np.sqrt(1 + (f / fc) ** 2)            # RC low pass
    hp = (f / fc) / np.sqrt(1 + (f / fc) ** 2)      # RC high pass
    # band-pass as combination of two poles
    bp = 1 / np.sqrt((1 + (f / 500) ** 2) * (1 + (f / 5000) ** 2))

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.semilogx(f, 20 * np.log10(lp), label="low-pass (fc=1 kHz)")
    ax.semilogx(f, 20 * np.log10(hp), label="high-pass (fc=1 kHz)")
    ax.semilogx(f, 20 * np.log10(bp), label="band-pass (0.5–5 kHz)")
    ax.axvline(fc, color="k", ls=":", lw=0.8)
    ax.set_ylim(-30, 5)
    style_ax(ax, "Filter magnitude responses (first-order)",
             "Frequency (Hz)", "Magnitude (dB)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_6003 / "filter_shapes.png")
    plt.close(fig)


def chart_spectrogram():
    """STFT of a frequency sweep — how a time-varying signal looks in TF."""
    fs = 16000.0
    t = np.linspace(0, 1.0, int(fs))
    # chirp + a steady harmonic + noise
    f0 = np.linspace(200, 4000, len(t))
    x = 0.8 * np.sin(2 * np.pi * np.cumsum(f0) / fs)
    x += 0.4 * np.sin(2 * np.pi * 3000 * t)
    x += 0.15 * np.random.default_rng(0).normal(size=len(t))

    n_fft = 512
    hop = 128
    win = np.hanning(n_fft)
    frames = []
    for start in range(0, len(x) - n_fft, hop):
        frames.append(np.fft.rfft(x[start:start + n_fft] * win))
    S = np.abs(np.array(frames)).T
    freqs = np.fft.rfftfreq(n_fft, 1 / fs)
    times = np.arange(S.shape[1]) * hop / fs

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    im = ax.imshow(20 * np.log10(S + 1e-6), aspect="auto", origin="lower",
                   extent=[times[0], times[-1], freqs[0], freqs[-1]],
                   cmap="magma", vmin=-40, vmax=0)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("dB")
    style_ax(ax, "STFT of a rising sweep + steady harmonic + noise",
             "Time (s)", "Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(OUT_6003 / "spectrogram_stft.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6.551J charts
# ---------------------------------------------------------------------------
def chart_inverse_square():
    """Spherical spreading: pressure ~ 1/r, intensity ~ 1/r^2."""
    r = np.linspace(1, 20, 400)
    p = 1 / r
    i = (1 / r) ** 2
    fig, ax1 = plt.subplots(figsize=(5.6, 3.4))
    ax1.plot(r, p, label="pressure p ∝ 1/r")
    ax1.plot(r, i, label="intensity I ∝ 1/r²")
    ax1.set_xlabel("Distance r (arbitrary units)")
    ax1.set_ylabel("Amplitude (linear)")
    ax1.grid(True, alpha=0.3, ls="--")
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.plot(r, 20 * np.log10(p), "k--", lw=0.8, alpha=0.6, label="p in dB (20·log₁₀)")
    ax2.set_ylabel("Pressure (dB re r=1)")
    ax2.set_ylim(-40, 5)
    ax1.set_title("Spherical spreading: inverse distance vs inverse square")
    fig.tight_layout()
    fig.savefig(OUT_6551 / "inverse_square.png")
    plt.close(fig)


def chart_wavelength():
    """Wavelength vs frequency in air (c=343 m/s)."""
    f = np.logspace(1.3, 5, 400)
    lam = 343 / f
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.loglog(f, lam)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Wavelength (m)")
    for fx, label in [(100, "100 Hz: 3.4 m"), (1000, "1 kHz: 0.34 m"),
                      (10000, "10 kHz: 3.4 cm")]:
        ax.plot(fx, 343 / fx, "ro", ms=5)
        ax.annotate(label, (fx, 343 / fx), textcoords="offset points",
                    xytext=(6, -10), fontsize=8)
    ax.set_title("Wavelength vs frequency in air (c = 343 m/s)")
    ax.grid(True, which="both", alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(OUT_6551 / "wavelength_frequency.png")
    plt.close(fig)


def chart_array_directivity():
    """Two in-phase simple sources: |g(θ)| = |cos((kd/2) cosθ)| for d=λ/8..2λ."""
    th = np.linspace(0, 2 * np.pi, 720)
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.6), subplot_kw={"polar": True})
    for ax, d_lam in zip(axes, [1 / 8, 1 / 4, 1 / 2, 1.0]):
        g = np.abs(np.cos((2 * np.pi * d_lam / 2) * np.cos(th)))
        ax.plot(th, g, lw=1.5)
        ax.set_title(f"d = λ/{1 / d_lam:.0f}" if d_lam < 1 else f"d = {d_lam:.0f}λ",
                     fontsize=9, pad=8)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(np.pi / 180 * np.array([0, 90, 180, 270]))
        ax.set_yticks([])
    fig.suptitle("Directivity of two in-phase equal sources (L3): spacing d",
                 fontsize=10, y=1.05)
    fig.tight_layout()
    fig.savefig(OUT_6551 / "array_directivity.png")
    plt.close(fig)


def chart_standing_wave():
    """Rigid-wall reflection: P(x) = 2P+ cos(kx) at λ = 1 m."""
    lam = 1.0
    x = np.linspace(0, -2 * lam, 800)
    k = 2 * np.pi / lam
    p = 2 * np.cos(k * x)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(x, p)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="r", ls="--", lw=1, label="rigid wall (x=0)")
    for n in [1, 3, 5]:
        ax.axvline(-n * lam / 4, color="g", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlim(-2, 0)
    style_ax(ax, "Standing wave from rigid-wall reflection: P(x)=2P⁺cos(kx)",
             "Distance from wall (m)", "Pressure")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_6551 / "standing_wave.png")
    plt.close(fig)


def chart_itd():
    """ITD = (a/c)(θ + sinθ) for a spherical head of radius a."""
    a = 0.0875  # ~human head radius, m
    c = 343.0
    th = np.linspace(-90, 90, 181)
    th_rad = np.radians(th)
    itd = (a / c) * (th_rad + np.sin(th_rad))
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.plot(th, itd * 1e6, lw=2)
    ax.axhline(0, color="k", lw=0.8)
    style_ax(ax, "Interaural time difference vs azimuth (L4, sphere model)",
             "Azimuth (deg)", "ITD (µs)")
    fig.tight_layout()
    fig.savefig(OUT_6551 / "itd_azimuth.png")
    plt.close(fig)


def main():
    chart_aliasing()
    chart_db_scale()
    chart_filters()
    chart_spectrogram()
    chart_inverse_square()
    chart_wavelength()
    chart_array_directivity()
    chart_standing_wave()
    chart_itd()
    print("6.003 assets:", len(list(OUT_6003.glob("*.png"))))
    print("6.551J assets:", len(list(OUT_6551.glob("*.png"))))
    for d in (OUT_6003, OUT_6551):
        for p in sorted(d.glob("*.png")):
            print(" -", p.relative_to(ROOT), f"{p.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()

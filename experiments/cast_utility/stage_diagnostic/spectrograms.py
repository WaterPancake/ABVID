"""Mel spectrograms through BEATs's own front end: original vs CAST-fitted clips.

Groups the training clips by the saved real-only BEATs head's verdict on their
fitted re-render (smooth8, realization 0): still right (p(correct class) >= 0.9)
and very wrong (< 0.1). Descriptive, training data only; no fitting. Run after
listen.py --originals and verdicts.py, in the baseline environment:

    .venv-baseline/bin/python -B experiments/cast_utility/stage_diagnostic/spectrograms.py
"""
from io import BytesIO
import json

import numpy as np

from abvid.cast.grouped_storage import read_bundle
from abvid.cast.stage_diagnostic import BASE_ENV, design, destination, environment, feature_path, wave_hash
from abvid.paths import resolve

RUN = "artifact:cast/diagnostics/cast_stage_diagnosis_20261025_v1"
OUT = "artifact:cast/listening/stage_diagnosis_20261025_v1"
STAGE, RIGHT, WRONG = "smooth8_check_0", 0.9, 0.1
DB, HOP_S = 10/np.log(10), 0.010
TICKS = [100, 250, 500, 1000, 2000, 4000, 8000]
GROUPS = [("very_wrong", f"Very wrong (fitted p(correct) < {WRONG})", "#c0362c"),
          ("still_right", f"Still right (fitted p(correct) ≥ {RIGHT})", "#2f6fb0")]


def band_centres(n=128, low=20.0, high=8000.0):
    """Kaldi fbank bin centres: equally spaced on 1127 ln(1 + f/700)."""
    lo, hi = (1127*np.log1p(f/700) for f in (low, high))
    return 700*np.expm1((lo + (hi-lo)/(n+1)*np.arange(1, n+1))/1127)


def inputs(d):
    out = resolve(OUT)
    if json.loads((out / "originals_ids.json").read_text()) != d["train_ids"]:
        raise ValueError("Saved originals are in another order")
    original = np.load(out / "originals.npy", allow_pickle=False)
    with np.load(feature_path("real"), allow_pickle=False) as cache:
        expected = dict(zip(cache["ids"].astype(str), cache["wave_sha256"].astype(str)))
    if any(wave_hash(w) != expected[i] for i, w in zip(d["train_ids"], original)):
        raise ValueError("Saved original differs from the classifier input")
    _, payload = read_bundle(destination(RUN) / "render" / STAGE)
    if json.loads(payload["ids.json"]) != d["train_ids"]:
        raise ValueError("Fitted bundle is in another order")
    fitted = np.load(BytesIO(payload["waves.npy"]), allow_pickle=False)
    text = (out / "verdicts.js").read_text()
    verdicts = json.loads(text[text.index("=")+1:].strip().rstrip(";"))["p_truck"]["BEATs768"]
    return original, fitted, verdicts


def logmel(encoder, waves):
    import torch
    with torch.inference_mode():  # fbank_mean=0, std=0.5 undoes BEATs's affine normalization
        fbank = encoder.preprocess(torch.from_numpy(waves), fbank_mean=0.0, fbank_std=0.5).numpy()
    return DB*fbank.astype(np.float64)  # float32 means over clips x frames drift by ~0.1 dB


def axes_freq(ax, centres):
    pos = np.interp(np.log(TICKS), np.log(centres), np.arange(len(centres)))
    ax.set_yticks(pos, [f"{t//1000}k" if t >= 1000 else str(t) for t in TICKS])
    ax.axhline(pos[TICKS.index(4000)], color="white", lw=.6, ls="--", alpha=.7)


def clip_figure(path, o, f, title, centres, vmin, vmax, plt):
    fig, ax = plt.subplots(1, 3, figsize=(11, 2.5), constrained_layout=True)
    extent = [0, len(o)*HOP_S, -.5, len(centres)-.5]
    for a, x, name in ((ax[0], o, "Original (classifier input)"), (ax[1], f, "CAST fitted (smooth8)")):
        im = a.imshow(x.T, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap="magma")
        a.set_title(name, fontsize=9)
        axes_freq(a, centres)
    fig.colorbar(im, ax=ax[:2], label="log-mel (dB)", shrink=.9, pad=.01)
    im = ax[2].imshow((f-o).T, origin="lower", aspect="auto", extent=extent, vmin=-30, vmax=30, cmap="RdBu_r")
    ax[2].set_title("Fitted − original (dB)", fontsize=9)
    axes_freq(ax[2], centres)
    fig.colorbar(im, ax=ax[2], shrink=.9, pad=.01)
    for a in ax:
        a.set_xlabel("s", fontsize=8)
        a.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def summary_figure(path, cls, groups, centres, dead, plt):
    """`dead` bands hold the filterbank floor in every clip (a 25 ms frame cannot resolve them)."""
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    live = lambda x: np.where(dead, np.nan, x)
    for key, label, colour in GROUPS:
        o, f = groups[key]
        n = len(o)
        ax[0].plot(centres, live(o.mean((0, 1))), color=colour, label=f"{label}, original (n={n})")
        ax[0].plot(centres, live(f.mean((0, 1))), color=colour, ls="--", label=f"{label}, fitted")
        diff = (f-o).mean(1)
        q1, med, q3 = (live(q) for q in np.percentile(diff, [25, 50, 75], axis=0))
        ax[1].plot(centres, med, color=colour, label=label)
        ax[1].fill_between(centres, q1, q3, color=colour, alpha=.18, lw=0)
        ratio = np.median(f.std(1)/o.std(1), axis=0)
        ax[2].plot(centres, live(ratio), color=colour, label=label)
    titles = ["Mean spectrum over time", "Fitted − original per band (median, IQR)",
              "Frame-to-frame variation kept (fitted / original)"]
    ylabels = ["log-mel (dB)", "dB", "ratio of temporal std"]
    for a, t, y in zip(ax, titles, ylabels):
        a.set_xscale("log")
        a.set_xticks(TICKS, [f"{t//1000}k" if t >= 1000 else str(t) for t in TICKS])
        a.axvspan(4000, centres[-1], color="grey", alpha=.15, lw=0)
        a.set_title(t, fontsize=10)
        a.set_xlabel("Hz")
        a.set_ylabel(y)
    ax[1].axhline(0, color="grey", lw=.8)
    ax[2].axhline(1, color="grey", lw=.8)
    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=7)
    fig.suptitle(f"{cls.capitalize()}s: what the CAST fit changes, by whether BEATs still recognizes the clip "
                 f"(grey: above the 4 kHz route ceiling, filter leakage only)", fontsize=11)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from abvid.representations.beats import load_encoder
    environment(BASE_ENV)
    d, parents, _ = design(RUN)
    original, fitted, verdicts = inputs(d)
    encoder, _ = load_encoder(dict(d["h1_config"]["encoder"], checkpoint=str(resolve(d["checkpoint"]))))
    o_mel, f_mel = logmel(encoder, original), logmel(encoder, fitted)
    centres = band_centres(o_mel.shape[-1])
    vmax = float(np.percentile(np.concatenate([o_mel, f_mel]), 99.9))
    dead = np.concatenate([o_mel, f_mel]).std(axis=(0, 1)) < 0.1
    out = resolve(OUT) / "spectrograms"
    out.mkdir(exist_ok=False)
    cls_of = {p["file_id"]: p["source"]["canonical_class"] for p in parents}
    report = []
    for cls in ("truck", "car"):
        groups, cards = {}, []
        for key, label, _ in GROUPS:
            picks, clips = [], []
            for k, fid in enumerate(d["train_ids"]):
                if cls_of[fid] != cls:
                    continue
                po, pf = verdicts[f"{fid[:12]}|original|0"], verdicts[f"{fid[:12]}|smooth8|0"]
                correct = (lambda p: p) if cls == "truck" else (lambda p: 1-p)
                if correct(po) <= .5 or not (correct(pf) < WRONG if key == "very_wrong" else correct(pf) >= RIGHT):
                    continue
                png = f"spectrograms/{cls}_{key}_{fid[:12]}.png"
                clip_figure(resolve(OUT) / png, o_mel[k], f_mel[k], f"{cls} {fid[:12]} · P(truck) original {po:.0%} → fitted {pf:.0%}",
                            centres, vmax-70, vmax, plt)
                picks.append(k)
                clips.append({"id": fid[:12], "po": po, "pf": pf, "png": png})
            groups[key] = (o_mel[picks], f_mel[picks])
            cards.append({"key": key, "label": label, "clips": clips})
        summary_figure(out / f"summary_{cls}.png", cls, groups, centres, dead, plt)
        report.append({"class": cls, "summary": f"spectrograms/summary_{cls}.png", "groups": cards})
    html = resolve("repo:experiments/cast_utility/stage_diagnostic/spectrograms.html").read_text()
    with (resolve(OUT) / "spectrograms.html").open("x") as handle:
        handle.write(html.replace("__REPORT__", json.dumps(report)))
    print("; ".join(f"{r['class']}: " + ", ".join(f"{g['key']} {len(g['clips'])}" for g in r["groups"]) for r in report))


if __name__ == "__main__":
    main()

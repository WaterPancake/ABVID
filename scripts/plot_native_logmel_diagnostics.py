"""Post-hoc log-Mel examples tied to frozen held-out predictions, not model tuning."""

import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import shutil

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio

from vehicle_audio.audio import resample_audio
from vehicle_audio.benchmark_followup import (
    FrozenBenchmark,
    MANIFEST,
    sha256,
    write_json,
)
from vehicle_audio.pretrained_evaluation import validate_panns_checkpoint


def relative_db(power, reference):
    return 10 * np.log10(np.maximum(power, 1e-10) / max(float(reference), 1e-10))


def window_scores(b, folds):
    values = defaultdict(list)
    for fold in folds:
        for j, i in enumerate(fold["indices"]):
            y = b.labels[i]
            values[int(i)].append(
                {
                    "fold_id": fold["fold_id"],
                    "threshold": fold["threshold"],
                    **{
                        f"{m}_p_wheeled": float(fold[m][j])
                        for m in ("classical", "semantic", "fusion")
                    },
                    **{
                        f"{m}_correct": bool(
                            (
                                fold[m][j]
                                >= (fold["threshold"] if m == "fusion" else 0.5)
                            )
                            == y
                        )
                        for m in ("classical", "semantic", "fusion")
                    },
                }
            )
    return {
        i: {
            "folds": rows,
            **{
                key: float(np.mean([r[key] for r in rows]))
                for key in rows[0]
                if key != "fold_id"
            },
        }
        for i, rows in values.items()
    }


def select_examples(b, scores, profiles):
    def choose(source_part, objective):
        indices = [i for i, r in enumerate(b.records) if source_part in r["source_id"]]
        return max(
            indices,
            key=lambda i: (
                *objective(scores[i]),
                -b.records[i]["window_start_seconds"],
            ),
        )

    good_t = choose(
        "abrams-bright-star",
        lambda s: (
            min(s["classical_correct"], s["semantic_correct"], s["fusion_correct"]),
            -s["fusion_p_wheeled"],
        ),
    )
    good_w = choose(
        "maserati",
        lambda s: (s["fusion_correct"], s["semantic_correct"], s["fusion_p_wheeled"]),
    )
    bad_w = choose(
        "stryker-convoy", lambda s: (1 - s["fusion_correct"], -s["fusion_p_wheeled"])
    )
    condition = b.records[bad_w]["operating_condition"]
    candidates = [
        i
        for i, r in enumerate(b.records)
        if r["vehicle_class"] == "tracked" and r["operating_condition"] == condition
    ]
    if not candidates:
        candidates = list(np.flatnonzero(b.labels == 0))
    z = profiles - profiles.mean(axis=1, keepdims=True)
    z /= np.linalg.norm(z, axis=1, keepdims=True).clip(1e-12)
    similar_t = max(candidates, key=lambda i: float(z[i] @ z[bad_w]))
    hmmwv = choose(
        "hmmwv-m1151", lambda s: (1 - s["fusion_correct"], -s["fusion_p_wheeled"])
    )
    ford = choose(
        "ford-model-t", lambda s: (1 - s["fusion_correct"], -s["fusion_p_wheeled"])
    )
    disagreement_t = choose(
        "t72-bmp3",
        lambda s: (
            s["semantic_correct"] - s["classical_correct"],
            1 - s["fusion_correct"],
            s["classical_p_wheeled"] - s["semantic_p_wheeled"],
        ),
    )
    disagreement_w = choose(
        "abarth",
        lambda s: (
            s["classical_correct"] - s["semantic_correct"],
            1 - s["fusion_correct"],
            s["classical_p_wheeled"] - s["semantic_p_wheeled"],
        ),
    )
    return [
        (
            "correct_examples",
            "Better-classified examples (best available in these sessions)",
            [good_t, good_w],
        ),
        (
            "spectral_overlap",
            "Cross-class spectral resemblance (deliberately selected)",
            [similar_t, bad_w],
        ),
        ("wheeled_failures", "Wheeled failures: HMMWV and Ford Model T", [hmmwv, ford]),
        (
            "branch_disagreement",
            "Branch disagreement: fusion can overturn correct decisions",
            [disagreement_t, disagreement_w],
        ),
    ], {
        "similarity_metric": "cosine of mean-centered time-averaged log-Mel profiles (64 bins)",
        "similarity": float(z[similar_t] @ z[bad_w]),
        "same_operating_condition": b.records[similar_t]["operating_condition"]
        == condition,
    }


def draw_spectrogram(ax, db, centers, title, upper_frequency):
    image = ax.imshow(
        db,
        origin="lower",
        aspect="auto",
        extent=(0, 2, -0.5, 63.5),
        vmin=-70,
        vmax=0,
        cmap="magma",
        interpolation="nearest",
    )
    frequencies = [100, 500, 1000, 2000, 4000, 8000] + (
        [14000] if upper_frequency > 8000 else []
    )
    positions = np.interp(frequencies, centers, np.arange(64))
    ax.set_yticks(
        positions, [f"{f / 1000:g}k" if f >= 1000 else str(f) for f in frequencies]
    )
    ax.set(
        xlabel="Time within selected window (s)",
        ylabel="Frequency (Hz; Mel spacing)",
        title=title,
    )
    ax.title.set_fontsize(9)
    if upper_frequency > 8000:
        ax.axhline(
            np.interp(8000, centers, np.arange(64)), color="cyan", ls="--", lw=0.9
        )
    return image


def build(output):
    if output.exists():
        raise FileExistsError(
            "Use a new output directory for a fresh diagnostic gallery"
        )
    b = FrozenBenchmark()
    predictions = b.predict(b.semantic, b.classical)
    b.verify_native(predictions)
    scores = window_scores(b, predictions)
    audio = []
    for r in b.records:
        x, sr = sf.read(
            MANIFEST.parent / r["audio_path"], dtype="float32", always_2d=True
        )
        if sr != 16000 or x.shape != (32000, 1):
            raise ValueError("Unexpected model input shape")
        audio.append(torch.from_numpy(x[:, 0].copy()))
    audio = torch.stack(audio)
    diagnostic_mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        f_min=20,
        f_max=8000,
        power=2.0,
    )
    with torch.inference_mode():
        powers = diagnostic_mel(audio).numpy()
    log_profiles = (10 * np.log10(np.maximum(powers, 1e-10))).mean(axis=2)
    groups, overlap = select_examples(b, scores, log_profiles)
    selected = sorted({i for _, _, indices in groups for i in indices})
    checkpoint = validate_panns_checkpoint(b.metrics["pretrained_encoder"]["path"])
    from panns_inference import AudioTagging

    tagging = AudioTagging(checkpoint_path=checkpoint["path"], device="cpu")
    network = tagging.model.eval()
    with torch.inference_mode():
        x = resample_audio(audio[selected], 16000, 32000)
        actual_logmel = (
            network.logmel_extractor(network.spectrogram_extractor(x))
            .squeeze(1)
            .transpose(1, 2)
            .numpy()
        )
    actual_by_index = dict(zip(selected, actual_logmel, strict=True))
    common_ref = float(powers[selected].max())
    panns_ref_db = float(actual_logmel.max())
    # HTK centers, exactly matching the diagnostic transform's default scale.
    mel_edges = np.linspace(
        2595 * np.log10(1 + 20 / 700), 2595 * np.log10(1 + 8000 / 700), 66
    )
    diagnostic_centers = 700 * (10 ** (mel_edges[1:-1] / 2595) - 1)
    panns_centers = librosa.mel_frequencies(n_mels=66, fmin=50, fmax=14000)[1:-1]
    output.mkdir(parents=True)
    (output / "audio").mkdir()
    examples = []
    cards = []
    short_names = {
        "candidate-target-tracked-abrams-bright-star-2017": "Abrams",
        "target-wheeled-maserati-granturismo-exhaust": "Maserati",
        "candidate-target-wheeled-m1126-stryker-convoy-pinon-2024": "Stryker",
        "candidate-target-wheeled-hmmwv-m1151-training-2014": "HMMWV",
        "target-wheeled-ford-model-t-start": "Ford Model T",
        "candidate-target-tracked-t72-bmp3-102nd-march-2022": "T-72 / BMP-3 source",
        "target-wheeled-abarth-205-goodwood": "Abarth",
    }
    for group_id, title, indices in groups:
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        fig.subplots_adjust(
            left=0.065, right=0.90, top=0.85, bottom=0.08, hspace=0.57, wspace=0.31
        )
        fig.suptitle(
            title
            + "\nPost-hoc held-out development examples; contrast is not proof of class separability",
            fontsize=15,
        )
        group_cards = []
        for row, i in enumerate(indices):
            r, score = b.records[i], scores[i]
            name = short_names.get(r["source_id"], r["vehicle_model"])
            stamp = f"{r['window_start_seconds']:g}–{r['window_end_seconds']:g}s"
            title_prefix = f"{name} | TRUE {r['vehicle_class']} | source {stamp}"
            common_db = relative_db(powers[i], common_ref)
            local_db = relative_db(powers[i], powers[i].max())
            draw_spectrogram(
                axes[row, 0],
                common_db,
                diagnostic_centers,
                title_prefix + "\n16 kHz • shared power reference",
                8000,
            )
            draw_spectrogram(
                axes[row, 1],
                local_db,
                diagnostic_centers,
                f"Per-clip contrast only • {r['operating_condition']}\nCorrect folds: classical {score['classical_correct']:.0%}, semantic {score['semantic_correct']:.0%}, fusion {score['fusion_correct']:.0%}",
                8000,
            )
            image = draw_spectrogram(
                axes[row, 2],
                actual_by_index[i] - panns_ref_db,
                panns_centers,
                f"Actual PANNs log-Mel frontend, before BN\nMean P(wheeled): C {score['classical_p_wheeled']:.2f}, S {score['semantic_p_wheeled']:.2f}, F {score['fusion_p_wheeled']:.2f}",
                14000,
            )
            axes[row, 2].text(
                0.04,
                0.94,
                "Above 8 kHz was discarded upstream",
                transform=axes[row, 2].transAxes,
                color="cyan",
                fontsize=8,
                va="top",
            )
            source = b.catalog[r["source_id"]]
            segment = next(
                s
                for s in source["condition_segments"]
                if s["start_seconds"] <= r["window_start_seconds"]
                and s["end_seconds"] >= r["window_end_seconds"]
                and s["operating_condition"] == r["operating_condition"]
            )
            local_audio = Path("audio") / f"{r['sample_id']}.wav"
            shutil.copy2(MANIFEST.parent / r["audio_path"], output / local_audio)
            waveform = audio[i].numpy()
            spectrum = abs(np.fft.rfft(waveform * np.hanning(len(waveform)))) ** 2
            bins = np.fft.rfftfreq(len(waveform), 1 / 16000)
            low_fraction = float(
                spectrum[bins < 100].sum() / max(spectrum.sum(), 1e-12)
            )
            example = {
                "group": group_id,
                "manifest_index": i,
                "record": r,
                "scores": score,
                "audio_sha256": sha256(output / local_audio),
                "audio_path": str(local_audio),
                "segment_notes": segment.get("notes"),
                "below_100hz_power_fraction": low_fraction,
                "rms_dbfs": float(20 * np.log10(np.sqrt(np.mean(waveform**2)))),
            }
            examples.append(example)
            group_cards.append(
                f"<article><h3>{html.escape(title_prefix)}</h3><p>{html.escape(r['operating_condition'])}; fusion correct in {score['fusion_correct']:.0%} of {len(score['folds'])} held-out contexts.</p><audio controls src='{local_audio}'></audio><p>{html.escape(segment.get('notes', ''))}</p><small>{html.escape(r['license'])} — {html.escape(r['attribution'])}</small></article>"
            )
        color_axis = fig.add_axes([0.93, 0.14, 0.012, 0.64])
        fig.colorbar(
            image,
            cax=color_axis,
            label="Power (dB relative to indicated reference); display floor −70 dB",
        )
        fig.savefig(output / f"{group_id}.png", dpi=150)
        plt.close(fig)
        cards.append(
            f"<section><h2>{html.escape(title)}</h2><img src='{group_id}.png'><div class='pair'>{''.join(group_cards)}</div></section>"
        )
    # Magnitude profiles expose similarity without pretending the time structure is identical.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, (_, title, indices) in zip(axes, groups[:2], strict=True):
        for i in indices:
            r = b.records[i]
            profile = log_profiles[i] - log_profiles[i].max()
            ax.semilogx(
                diagnostic_centers,
                profile,
                label=short_names.get(r["source_id"], r["vehicle_model"]),
            )
        ax.set(
            title=title.split(" (")[0],
            xlabel="Frequency (Hz)",
            ylabel="Time-mean log-Mel profile (dB, peak aligned)",
            ylim=(-75, 2),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(output / "spectral_profiles.png", dpi=150)
    plt.close(fig)
    parameter_count = sum(
        p.numel()
        for name, p in network.named_parameters()
        if not name.startswith(("spectrogram_extractor.", "logmel_extractor."))
    )
    metadata = {
        **b.metadata(
            {
                "purpose": "post-hoc visual diagnostics, no model changes",
                "diagnostic_fft": 1024,
                "diagnostic_hop": 256,
                "display_dynamic_range_db": 70,
                "diagnostic_mels": 64,
            }
        ),
        "script_sha256": sha256(__file__),
        "examples": examples,
        "spectral_overlap_selection": overlap,
        "network_parameter_count_excluding_fixed_frontend": parameter_count,
        "panns_frontend": {
            "sample_rate": 32000,
            "fft": 1024,
            "hop": 320,
            "mels": 64,
            "fmin": 50,
            "fmax": 14000,
        },
        "display_reference": {
            "shared_diagnostic_power": common_ref,
            "shared_panns_db": panns_ref_db,
        },
        "selection": "Best/worst examples ranked by held-out correctness then probability; tracked resemblance matched on mean-centered log-Mel profile, same operating state when available; not representative or a new test.",
        "probability_warning": "Uncalibrated mean across separate models that each held out this entire session. Correctness uses each model's original threshold, not thresholding the mean score.",
    }
    write_json(output / "examples.json", metadata)
    np.savez_compressed(
        output / "plot_arrays.npz",
        selected_indices=selected,
        mel_power=powers[selected],
        panns_logmel=actual_logmel,
        diagnostic_centers=diagnostic_centers,
        panns_centers=panns_centers,
    )
    page = """<!doctype html><meta charset='utf-8'><title>ABVID log-Mel diagnostics</title>
<style>body{font:17px system-ui;background:#101827;color:#e8edf5;max-width:1500px;margin:24px auto;padding:0 20px}img{width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}article{background:#202d40;padding:15px}small{color:#bdcbe0}audio{width:100%}section{margin:40px 0}</style>
<h1>ABVID · Log-Mel diagnostic gallery</h1><p>Exact 2-second benchmark inputs, channel 0. Listen without video. These are deliberately selected development examples—not evidence of overall separability.</p>
<p>Left: shared energy reference. Middle: each clip's peak set to 0 dB for visual comparison only; the model was not normalized this way. Right: actual PANNs log-Mel before batch normalization; the cyan line marks 8 kHz. Upper frequencies cannot be recovered by upsampling.</p>
<p>Correct-fold percentages use all five or seven models that excluded that entire session. Scores are uncalibrated; averaging them does not define a new decision rule. No reserved recordings, retraining, denoising or benchmark changes.</p>
"""
    (output / "index.html").write_text(
        page
        + "".join(cards)
        + "<h2>Time-averaged spectral shapes</h2><img src='spectral_profiles.png'><p>Similarity in an averaged spectrum discards temporal information and is not proof that a listener or another model cannot distinguish the clips.</p>"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "cnn_parameters": parameter_count,
                "examples": [
                    {
                        "group": e["group"],
                        "source": e["record"]["source_id"],
                        "start": e["record"]["window_start_seconds"],
                        "correct_fraction": e["scores"]["fusion_correct"],
                        "low_frequency_power": e["below_100hz_power_fraction"],
                    }
                    for e in examples
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(".artifacts/logmel_diagnostics_7t5w_v1")
    )
    torch.set_num_threads(4)
    build(parser.parse_args().output)

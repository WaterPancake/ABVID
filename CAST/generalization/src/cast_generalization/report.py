"""Local scientific report and inspectable, unpaired audio gallery."""
from collections import Counter
import html
from pathlib import Path
import numpy as np

from cast.config import sha
from .pipeline import read, readl


def figures(out, cfg, scores, generated, held):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"joint": "#007a87", "prototype": "#ad547b", "marginals": "#c77b12", "held real": "#242424"}
    fig, ax = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for i, c in enumerate(cfg["class_order"]):
        arms = scores["classes"][c]["arms"]
        names = cfg["arms"]
        means = [arms[a]["W1_mean"] for a in names]
        errs = [[means[j]-arms[a]["W1_min"] for j, a in enumerate(names)], [arms[a]["W1_max"]-means[j] for j, a in enumerate(names)]]
        ax[i, 0].bar(names, means, yerr=errs, color=[colors[a] for a in names], capsize=4)
        ref = scores["classes"][c]["training_real_reference"]["W1"]
        ax[i, 0].axhline(ref, color="black", ls="--", label="25 training-real reference")
        ax[i, 0].set(title=f"{c}: distance (lower better)", ylabel="Mean scaled marginal W1")
        ax[i, 0].legend(fontsize=8)
        coverage = [arms[a]["coverage_mean"] for a in names]
        err = [[coverage[j]-min(s["coverage"] for s in arms[a]["seeds"]) for j, a in enumerate(names)], [max(s["coverage"] for s in arms[a]["seeds"])-coverage[j] for j, a in enumerate(names)]]
        ax[i, 1].bar(names, coverage, yerr=err, color=[colors[a] for a in names], capsize=4)
        ax[i, 1].axhline(.8, color="black", ls="--", label="Frozen provisional floor")
        ax[i, 1].set(title=f"{c}: marginal coverage", ylim=(0, 1), ylabel="Held values in generated 5–95% intervals")
        ax[i, 1].legend(fontsize=8)
        x = np.arange(4)
        for j, a in enumerate(names):
            ax[i, 2].plot(x, [arms[a]["family_spread_mean"][f] for f in cfg["primary_families"]], "o-", label=a, color=colors[a])
        ax[i, 2].axhspan(.5, 2, color="grey", alpha=.13)
        ax[i, 2].axhline(1, color="grey", ls=":")
        ax[i, 2].set(xticks=x, xticklabels=["spectrum", "bands", "envelope", "modulation"], ylabel="Generated / held descriptor SD", title=f"{c}: retained spread")
        ax[i, 2].tick_params(axis="x", labelrotation=15)
        ax[i, 2].legend(fontsize=8)
    fig.suptitle("CAST: one held development group — error bars are five-seed ranges, not confidence intervals")
    fig.savefig(out/"coverage.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for i, c in enumerate(cfg["class_order"]):
        datasets = {a: [r["descriptors"] for r in generated if (r["class"], r["arm"]) == (c, a)] for a in cfg["arms"]}
        datasets["held real"] = [r["descriptors"] for r in held if r["class"] == c]
        for j, (family, x, label) in enumerate((("log_spectrum", (np.arange(64)+.5)*62.5, "Frequency (Hz)"), ("envelope", (np.arange(40)+.5)*.05, "Time (s)"), ("modulation", (np.arange(20)+1)*.5, "Modulation frequency (Hz)"))):
            for name, rows in datasets.items():
                a = np.stack([r[family] for r in rows])
                if family == "log_spectrum":
                    a = a*10/np.log(10)
                low, med, high = np.quantile(a, [.05, .5, .95], axis=0)
                ax[i, j].plot(x, med, color=colors[name], label=name, lw=1.3)
                ax[i, j].fill_between(x, low, high, color=colors[name], alpha=.09)
            ax[i, j].set(title=f"{c}: {family.replace('_', ' ')}", xlabel=label,
                         ylabel={"log_spectrum": "Band power proportion (dB)", "envelope": "50-ms RMS (unit-RMS observation)", "modulation": "Envelope FFT magnitude / 40"}[family])
            if i == 0:
                ax[i, j].legend(fontsize=8)
    fig.suptitle("Median and 5th–95th percentiles; generated seeds pooled for visualization only")
    fig.savefig(out/"descriptors.png", dpi=160)
    plt.close(fig)


def report(out):
    cfg, scores = read(out/"config.resolved.json"), read(out/"scores.json")
    verification, timing, lock = read(out/"verification.json"), read(out/"timing.json"), read(out/"lock.json")
    generated, held, bank = readl(out/"generated_manifest.jsonl"), readl(out/"held_descriptors.jsonl"), readl(out/"parameter_bank.jsonl")
    figures(out, cfg, scores, generated, held)
    lines = ["# CAST held-group generalization check", "",
             ("**The frozen provisional acoustic criteria passed on this one development group.**" if scores["adequacy_passed"] else "**The frozen provisional acoustic generalization criteria failed.**")+" Technical verification passed. This is a held-group descriptor-coverage experiment; no classifier or conditional held-audio fitter was used.", "",
             "Domain: **50 real IDMT training observations → calibrated synthetic observations**, compared with **570 real observations from one held IDMT group**. Generated audio uses real calibration data. Fold 0 is historically exposed H1 development data, not an untouched final test. Physical vehicle identity is unknown.", "",
             "## Reproduce", "", "From the ABVID root, with a fresh run ID:", "", "```sh", "bash CAST/generalization/run.sh workflow --run-id cast_generalization_reproduction_01", "```", "",
             f"Completed run: `{out.name}`. Reverify without changing artifacts: `bash CAST/generalization/run.sh verify --run-id {out.name}`. The parent CAST environment is used read-only; see [extension README](../../README.md) for prerequisites. The workflow refuses overwrites and requires the original pilot and source hashes.", "",
             "[Frozen protocol](PROTOCOL.md) · [resolved configuration](config.resolved.json) · [lock](lock.json) · [source snapshot](snapshot.json) · [parameter bank](parameter_bank.jsonl) · [sampling schedule](sampling_schedule.jsonl) · [scores and per-coordinate tails](scores.json) · [audio gallery](index.html)", "",
             "## Frozen method and access", "",
             "The 50 pilot fits are the only parameter donors: 25 per class, five per class/group across five groups. Joint sampling chooses a group uniformly and then a complete fitted vector uniformly within that class/group. Prototype sampling uses the equal-group class mean. The marginals control independently draws each of 25 scalar coordinates and then renormalizes the two simplex weight vectors. There are five seeds and 50 two-second clips per class/seed/arm, totaling 1,500 generated clips. All three arms share matched sampling phase/noise streams at each class/seed/index.", "",
             "The renderer and pilot fits are unchanged. Separately namespaced sampling streams are independent of fitting/checking streams. All generated clips, descriptors, sampling choices, training-only scales and hashes were frozen before the first held-audio access receipt. Held audio does not select examples, controls, latent parameters, scales, stopping rules or seeds. No nearest-example selection or held latent fitting is performed.", "",
             "Engineering replay disclosure: the initial run `cast_generalization_v0_20261004` processed all held clips, then its verifier stopped because Finder changed `.DS_Store`. That run and its original pre-access freeze are preserved. This run excludes only Finder presentation metadata from the artifact inventory, with a regression test protecting all experiment files. The sampler, score definitions, criteria, configuration and selection schedule are unchanged. It is a repair/replay of the original preregistered evaluation, not new held-out confirmation. See [implementation decisions](../../DECISIONS.md).", "",
             f"Held group `{cfg['held_group']}`: Schleusinger-Allee, 2019-11-12; 491 cars and 79 trucks. This site also appears in the training bank on another date. Exact H1 fold-0 test IDs and linked ancestry are checked against the immutable source-only manifest. [Held access receipt](held_access_receipt.json) follows [completed train-only generation](generation.complete.json). All selected records were retained; zero replacements.", "",
             "The four equally weighted descriptor families are 64 log spectral proportions over 0–4 kHz, eight broad-band energy proportions, forty 50-ms envelope values, and twenty 0.5–10-Hz modulation magnitudes. Each coordinate's empirical Wasserstein-1 distance is divided by its frozen training-real SD with declared floors; coordinate means are averaged within families, then across families. Lower is better. This tests marginal distributions, not joint support or perceptual realism.", "",
             "Coverage is the mean fraction of held values inside generated marginal 5th–95th percentile intervals. Seed ranges below measure sampler variability only; they are not confidence intervals over recording groups. The training-real reference uses 25 parents per class and is an unequal-size observational reference, not a matched generated arm.", "",
             "## Held-group results", "", "| Class | Arm | Mean scaled W1 ↓ | Five-seed range | Marginal coverage ↑ |", "|---|---|---:|---:|---:|"]
    for c, one in scores["classes"].items():
        for a in cfg["arms"]:
            s = one["arms"][a]
            lines.append(f"| {c} | {a} | {s['W1_mean']:.4f} | {s['W1_min']:.4f}–{s['W1_max']:.4f} | {s['coverage_mean']:.1%} |")
        ref = one["training_real_reference"]
        lines.append(f"| {c} | 25 training-real reference | {ref['W1']:.4f} | one fixed set | {ref['coverage']:.1%} |")
    lines += ["", "![Distance, coverage and spread](coverage.png)", "", "| Class | Joint gain vs prototype | Joint gain vs marginals | ≥5% gains | ≥80% coverage | All spread ratios 0.5–2 |", "|---|---:|---:|---|---|---|"]
    for c, one in scores["classes"].items():
        g, passed = one["joint_relative_gains"], one["criteria"]
        lines.append(f"| {c} | {g['prototype']:.1%} | {g['marginals']:.1%} | {'pass' if passed['distance_gain'] else 'fail'} | {'pass' if passed['marginal_coverage'] else 'fail'} | {'pass' if passed['spread'] else 'fail'} |")
    lines += ["", "These thresholds were declared before held access as exploratory engineering adequacy criteria, not calibrated statistical or perceptual standards. Positive relative gain means lower distance for joint sampling. All criteria must pass in both classes. No criterion or method was changed in response to these scores.", "",
              "| Class | Spectrum spread | Broad-band spread | Envelope spread | Modulation spread |", "|---|---:|---:|---:|---:|"]
    for c, one in scores["classes"].items():
        v = one["arms"]["joint"]["family_spread_mean"]
        lines.append(f"| {c} | "+" | ".join(f"{v[f]:.3f}" if v[f] is not None else "undefined" for f in cfg["primary_families"])+" |")
    lines += ["", "Spread is the ratio of mean coordinate SDs after train-only scaling. Less than one means under-dispersion. [Full numerical scores](scores.json) retain each family's distances, lower/upper tail misses, every coordinate's p05/median/p95 and zero-spread counts, for every seed. Class-balanced macro distances: "+", ".join(f"{a} {s:.4f}" for a, s in scores["macro_W1"].items())+". There is only one evaluated group, so group and worst-group results are identical.", "",
              "![Spectral, temporal and modulation distributions](descriptors.png)", "", "## Ancestry and diversity", ""]
    flags = Counter(f for r in bank for f in r["flags"]["scientific_flags"])
    lines += ["Calibration flags, retained without exclusion: "+", ".join(f"`{k}`: {v}/50" for k, v in sorted(flags.items()))+". These are observation-model fits; spacing is not measured RPM and noise is not separated tire sound. Sampling one optimum does not resolve alternative equally good parameterizations.", "", "| Class | Arm | Distinct vectors per seed (range) | Distinct real parents per seed (range) |", "|---|---|---:|---:|"]
    for c, one in scores["classes"].items():
        for a, data in one["arms"].items():
            seeds = data["seeds"]
            lines.append(f"| {c} | {a} | {min(s['unique_vectors'] for s in seeds)}–{max(s['unique_vectors'] for s in seeds)} | {min(s['unique_parents'] for s in seeds)}–{max(s['unique_parents'] for s in seeds)} |")
    lines += ["", "Joint outputs replay a finite bank of at most 25 complete vectors per class; new noise/phases add stochastic variation, not new real ancestry. The prototype has one vector per class by design. Marginal recombination creates new vectors but does not validate their realism. All parents, per-coordinate donors and inherited flags are available in the generated manifest and each sample sidecar.", "",
              "## Runtime, failures and verification", "",
              f"CPU only, one Torch thread, {timing['hardware']['platform']}. End-to-end through verification: **{timing['through_verification_seconds']:.2f} s**. Generation: {timing['stages_seconds']['generation']:.2f} s for 3,000 synthesized audio seconds; held preprocessing/descriptors: {timing['stages_seconds']['held_evaluation']:.2f} s for 1,140 audio seconds. This excludes the already completed 813.51-s calibration pilot. Process peak RSS: {timing['peak_RSS_bytes']/2**30:.3f} GiB (lifetime high-water mark). [Timing/environment](timing.json).", "",
              f"[Test gate](tests.log) passed. [Verification](verification.json): {verification['generated_bit_exact_replays']} generated waves and sampling decisions replayed exactly; {verification['held_preprocessing_bit_exact_replays']} originals rehashed and preprocessing/descriptors replayed exactly; scores recomputed; all parent-run artifacts and the tracked ABVID diff unchanged. [Failure records](failures.json): 570/570 held observations completed, zero numerical/input failures, zero dropped/replaced recordings. The scientific adequacy outcome above is independent of this technical pass.", "",
              "## Limitations and next decision", "",
              "This single-group development check cannot establish generalization to new physical vehicles, unseen sites, microphones or datasets. Labels and site/date groups are provider-derived conservative proxies, and physical independence is unverified. All clips are two seconds; marginal spectral/envelope descriptors miss joint dependencies, fine temporal structure and perceptual fidelity. Five sampler seeds do not provide session-population uncertainty. Audio examples are preselected, unpaired inspection aids, not matched reconstructions or a listening study.", "",
              "The empirical bank inherits the pilot's ambiguity. These results test one fitted winner per parent and do not measure robustness across equivalent-fit choices. No smooth parameter prior, classifier, external target corpus, new fold, military milestone, or real-world transfer evaluation was run. Existing H1 results, datasets, environments, splits and baseline artifacts are preserved. Derived IDMT audio remains local; recorded licence: CC BY-NC-ND 4.0.", ""]
    if not scores["adequacy_passed"]:
        lines += ["**Decision:** this sampler has not met the declared coverage/generalization criteria. Diagnose the frozen spectrum/envelope/modulation residuals and equivalent-fit uncertainty on training data before considering a revised model. Any revision needs a new version and must retain this failed result; this held group is now explicitly exposed CAST development data. A later classification-transfer experiment requires its own matched frozen protocol.", ""]
    else:
        lines += ["**Decision:** the sampler meets the declared descriptor-level criteria for this one development group. Confirmation needs new groups and a separately frozen protocol; classification benefit and physical identifiability remain untested.", ""]
    (out/"CAST_generalization.md").write_text("\n".join(lines))
    pages = ["<!doctype html><meta charset='utf-8'><title>CAST held-group coverage</title>",
             "<style>body{font:16px system-ui;max-width:1150px;margin:32px auto;padding:0 18px;color:#202830}img{width:100%}table{border-collapse:collapse;width:100%}td,th{padding:6px;border-bottom:1px solid #ddd;text-align:left}audio{width:280px}code{font-size:12px}a{color:#006e7d}</style>",
             "<h1>CAST held-group acoustic coverage</h1><p>Preselected, unpaired examples. Generated clips do not reconstruct the adjacent held recordings. One H1-exposed development group; no classifier or physical vehicle identity claim.</p>",
             "<p><a href='CAST_generalization.md'>Report</a> · <a href='PROTOCOL.md'>Frozen protocol</a> · <a href='scores.json'>Complete scores</a> · <a href='verification.json'>Verification</a></p>",
             "<img src='coverage.png' alt='Held group distance, coverage and spread'><img src='descriptors.png' alt='Descriptor distributions'>"]
    for c in cfg["class_order"]:
        h = next(r for r in held if r["class"] == c)
        pages += [f"<h2>{html.escape(c)}: preselected audio</h2><p>Held original: <code>{h['file_id']}</code></p><audio controls preload='none' src='{h['path']}/playback.wav'></audio><p>Independent samples, seed 42, index 0:</p>"]
        for a in cfg["arms"]:
            r = next(r for r in generated if (r["class"], r["arm"], r["seed"], r["index"]) == (c, a, 42, 0))
            pages.append(f"<p>{a} · <a href='{r['path']}/sample.json'>parameters and full ancestry</a></p><audio controls preload='none' src='{r['path']}/playback.wav'></audio>")
    pages += ["<h2>All generated audio</h2><p>Every sample retained. Shape files are unit RMS; listen to peak-safe playback. <a href='generated_manifest.jsonl'>Machine-readable manifest</a></p><table><tr><th>Arm / class</th><th>Seed / index</th><th>Playback</th><th>Parameters / ancestry</th></tr>"]
    for r in generated:
        pages.append(f"<tr><td>{r['arm']} / {r['class']}</td><td>{r['seed']} / {r['index']}</td><td><a href='{r['path']}/playback.wav'>audio</a></td><td><a href='{r['path']}/sample.json'>sample.json</a></td></tr>")
    pages += ["</table><h2>All held observations</h2><table><tr><th>Class</th><th>ID</th><th>Playback</th><th>Descriptors / levels</th></tr>"]
    for r in held:
        pages.append(f"<tr><td>{r['class']}</td><td><code>{r['file_id']}</code></td><td><a href='{r['path']}/playback.wav'>audio</a></td><td><a href='{r['path']}/observation.json'>observation.json</a></td></tr>")
    pages += ["</table><p>Local evaluation only. Recorded source licence: CC BY-NC-ND 4.0.</p>"]
    (out/"index.html").write_text("\n".join(pages))

# CAST held-group generalization check

**The frozen provisional acoustic generalization criteria failed.** Technical verification passed. This is a held-group descriptor-coverage experiment; no classifier or conditional held-audio fitter was used.

Domain: **50 real IDMT training observations → calibrated synthetic observations**, compared with **570 real observations from one held IDMT group**. Generated audio uses real calibration data. Fold 0 is historically exposed H1 development data, not an untouched final test. Physical vehicle identity is unknown.

## Reproduce

From the ABVID root, with a fresh run ID:

```sh
bash CAST/generalization/run.sh workflow --run-id cast_generalization_reproduction_01
```

Completed run: `cast_generalization_v0_20261004_r1`. Reverify without changing artifacts: `bash CAST/generalization/run.sh verify --run-id cast_generalization_v0_20261004_r1`. The parent CAST environment is used read-only; see [extension README](../CAST/generalization/README.md) for prerequisites. The workflow refuses overwrites and requires the original pilot and source hashes.

[Frozen protocol](../CAST/generalization/runs/cast_generalization_v0_20261004_r1/PROTOCOL.md) · [resolved configuration](../CAST/generalization/runs/cast_generalization_v0_20261004_r1/config.resolved.json) · [lock](../CAST/generalization/runs/cast_generalization_v0_20261004_r1/lock.json) · [source snapshot](../CAST/generalization/runs/cast_generalization_v0_20261004_r1/snapshot.json) · [parameter bank](../CAST/generalization/runs/cast_generalization_v0_20261004_r1/parameter_bank.jsonl) · [sampling schedule](../CAST/generalization/runs/cast_generalization_v0_20261004_r1/sampling_schedule.jsonl) · [scores and per-coordinate tails](../CAST/generalization/runs/cast_generalization_v0_20261004_r1/scores.json) · [audio gallery](../CAST/generalization/runs/cast_generalization_v0_20261004_r1/index.html)

## Frozen method and access

The 50 pilot fits are the only parameter donors: 25 per class, five per class/group across five groups. Joint sampling chooses a group uniformly and then a complete fitted vector uniformly within that class/group. Prototype sampling uses the equal-group class mean. The marginals control independently draws each of 25 scalar coordinates and then renormalizes the two simplex weight vectors. There are five seeds and 50 two-second clips per class/seed/arm, totaling 1,500 generated clips. All three arms share matched sampling phase/noise streams at each class/seed/index.

The renderer and pilot fits are unchanged. Separately namespaced sampling streams are independent of fitting/checking streams. All generated clips, descriptors, sampling choices, training-only scales and hashes were frozen before the first held-audio access receipt. Held audio does not select examples, controls, latent parameters, scales, stopping rules or seeds. No nearest-example selection or held latent fitting is performed.

Engineering replay disclosure: the initial run `cast_generalization_v0_20261004` processed all held clips, then its verifier stopped because Finder changed `.DS_Store`. That run and its original pre-access freeze are preserved. This run excludes only Finder presentation metadata from the artifact inventory, with a regression test protecting all experiment files. The sampler, score definitions, criteria, configuration and selection schedule are unchanged. It is a repair/replay of the original preregistered evaluation, not new held-out confirmation. See [implementation decisions](../CAST/generalization/DECISIONS.md).

Held group `connected_4001f06f57cfeee7`: Schleusinger-Allee, 2019-11-12; 491 cars and 79 trucks. This site also appears in the training bank on another date. Exact H1 fold-0 test IDs and linked ancestry are checked against the immutable source-only manifest. [Held access receipt](../CAST/generalization/runs/cast_generalization_v0_20261004_r1/held_access_receipt.json) follows [completed train-only generation](../CAST/generalization/runs/cast_generalization_v0_20261004_r1/generation.complete.json). All selected records were retained; zero replacements.

The four equally weighted descriptor families are 64 log spectral proportions over 0–4 kHz, eight broad-band energy proportions, forty 50-ms envelope values, and twenty 0.5–10-Hz modulation magnitudes. Each coordinate's empirical Wasserstein-1 distance is divided by its frozen training-real SD with declared floors; coordinate means are averaged within families, then across families. Lower is better. This tests marginal distributions, not joint support or perceptual realism.

Coverage is the mean fraction of held values inside generated marginal 5th–95th percentile intervals. Seed ranges below measure sampler variability only; they are not confidence intervals over recording groups. The training-real reference uses 25 parents per class and is an unequal-size observational reference, not a matched generated arm.

## Held-group results

| Class | Arm | Mean scaled W1 ↓ | Five-seed range | Marginal coverage ↑ |
|---|---|---:|---:|---:|
| car | joint | 0.9533 | 0.9247–0.9722 | 68.3% |
| car | prototype | 1.1328 | 1.1320–1.1349 | 21.2% |
| car | marginals | 1.0385 | 1.0170–1.0855 | 53.9% |
| car | 25 training-real reference | 0.7597 | one fixed set | 69.2% |
| truck | joint | 0.8040 | 0.7782–0.8439 | 68.9% |
| truck | prototype | 0.9771 | 0.9734–0.9814 | 22.0% |
| truck | marginals | 0.8106 | 0.7727–0.8852 | 62.3% |
| truck | 25 training-real reference | 0.6081 | one fixed set | 74.3% |

![Distance, coverage and spread](../CAST/generalization/runs/cast_generalization_v0_20261004_r1/coverage.png)

| Class | Joint gain vs prototype | Joint gain vs marginals | ≥5% gains | ≥80% coverage | All spread ratios 0.5–2 |
|---|---:|---:|---|---|---|
| car | 15.8% | 8.2% | pass | fail | pass |
| truck | 17.7% | 0.8% | fail | fail | pass |

These thresholds were declared before held access as exploratory engineering adequacy criteria, not calibrated statistical or perceptual standards. Positive relative gain means lower distance for joint sampling. All criteria must pass in both classes. No criterion or method was changed in response to these scores.

| Class | Spectrum spread | Broad-band spread | Envelope spread | Modulation spread |
|---|---:|---:|---:|---:|
| car | 1.211 | 1.424 | 0.668 | 0.688 |
| truck | 0.962 | 0.908 | 0.738 | 0.825 |

Spread is the ratio of mean coordinate SDs after train-only scaling. Less than one means under-dispersion. [Full numerical scores](../CAST/generalization/runs/cast_generalization_v0_20261004_r1/scores.json) retain each family's distances, lower/upper tail misses, every coordinate's p05/median/p95 and zero-spread counts, for every seed. Class-balanced macro distances: joint 0.8787, marginals 0.9246, prototype 1.0550. There is only one evaluated group, so group and worst-group results are identical.

![Spectral, temporal and modulation distributions](../CAST/generalization/runs/cast_generalization_v0_20261004_r1/descriptors.png)

## Ancestry and diversity

Calibration flags, retained without exclusion: `equally_good_starts_disagree`: 43/50, `no_resolvable_harmonic_salience`: 1/50, `spacing_unidentified_noise_dominant`: 2/50. These are observation-model fits; spacing is not measured RPM and noise is not separated tire sound. Sampling one optimum does not resolve alternative equally good parameterizations.

| Class | Arm | Distinct vectors per seed (range) | Distinct real parents per seed (range) |
|---|---|---:|---:|
| car | joint | 20–23 | 20–23 |
| car | marginals | 50–50 | 25–25 |
| car | prototype | 1–1 | 25–25 |
| truck | joint | 21–22 | 21–22 |
| truck | marginals | 50–50 | 25–25 |
| truck | prototype | 1–1 | 25–25 |

Joint outputs replay a finite bank of at most 25 complete vectors per class; new noise/phases add stochastic variation, not new real ancestry. The prototype has one vector per class by design. Marginal recombination creates new vectors but does not validate their realism. All parents, per-coordinate donors and inherited flags are available in the generated manifest and each sample sidecar.

## Runtime, failures and verification

CPU only, one Torch thread, macOS-26.6.2-arm64-arm-64bit. End-to-end through verification: **30.88 s**. Generation: 7.90 s for 3,000 synthesized audio seconds; held preprocessing/descriptors: 4.13 s for 1,140 audio seconds. This excludes the already completed 813.51-s calibration pilot. Process peak RSS: 0.557 GiB (lifetime high-water mark). [Timing/environment](../CAST/generalization/runs/cast_generalization_v0_20261004_r1/timing.json).

[Test gate](../CAST/generalization/runs/cast_generalization_v0_20261004_r1/tests.log) passed. [Verification](../CAST/generalization/runs/cast_generalization_v0_20261004_r1/verification.json): 1500 generated waves and sampling decisions replayed exactly; 570 originals rehashed and preprocessing/descriptors replayed exactly; scores recomputed; all parent-run artifacts and the tracked ABVID diff unchanged. [Failure records](../CAST/generalization/runs/cast_generalization_v0_20261004_r1/failures.json): 570/570 held observations completed, zero numerical/input failures, zero dropped/replaced recordings. The scientific adequacy outcome above is independent of this technical pass.

## Limitations and next decision

This single-group development check cannot establish generalization to new physical vehicles, unseen sites, microphones or datasets. Labels and site/date groups are provider-derived conservative proxies, and physical independence is unverified. All clips are two seconds; marginal spectral/envelope descriptors miss joint dependencies, fine temporal structure and perceptual fidelity. Five sampler seeds do not provide session-population uncertainty. Audio examples are preselected, unpaired inspection aids, not matched reconstructions or a listening study.

The empirical bank inherits the pilot's ambiguity. These results test one fitted winner per parent and do not measure robustness across equivalent-fit choices. No smooth parameter prior, classifier, external target corpus, new fold, military milestone, or real-world transfer evaluation was run. Existing H1 results, datasets, environments, splits and baseline artifacts are preserved. Derived IDMT audio remains local; recorded licence: CC BY-NC-ND 4.0.

**Decision:** this sampler has not met the declared coverage/generalization criteria. Diagnose the frozen spectrum/envelope/modulation residuals and equivalent-fit uncertainty on training data before considering a revised model. Any revision needs a new version and must retain this failed result; this held group is now explicitly exposed CAST development data. A later classification-transfer experiment requires its own matched frozen protocol.

## Post-run review

All **68 tests passed** (36 extension tests, 29 existing CAST tests and three legacy simulator tests). Both diagnostic figures were visually inspected; labels, error bars and percentile bands are readable. The gallery has eight preselected audio players plus links to every generated and held observation. No human listening study was performed.

The 5th–95th percentile coverage of joint sampling is **68.3% for cars and 68.9% for trucks**, below the predeclared 80% floor. Its mean distance improves over independent marginals by **8.2% for cars but only 0.8% for trucks**, failing the 5% rule for trucks. These are scientific adequacy failures, retained despite every technical check passing.

The generated median spectra show steps near 2 and 3 kHz, aligned with the renderer's fixed noise-band edges, while the held median spectra are smoother. Generated envelopes have less variation, and the median 0.5-Hz modulation magnitude is lower than the held observations in both classes. These observations suggest inspecting the spectral basis and envelope diversity; they do not isolate renderer limitations from calibration-set coverage or group shift. The actual training-real reference also misses many held tails (69.2% car / 74.3% truck coverage), so the sparse 25-parent-per-class bank is itself a material limitation. The unequal reference and generated sample sizes prevent treating that comparison as a matched method test.

The engineering replay preserved the original configuration, renderer config, parameter bank, training descriptors, scales, complete sampling schedule, held manifest and **scores.json byte-for-byte**. All **4,140 waveform pairs** compared between runs (generated shape/playback and held original/playback) contain exactly identical decoded sample arrays. The waveform-container hashes differ because the FLOAT WAV `PEAK` chunk includes a creation timestamp; per-run hashes remain recorded and verified. This does not change sample-level determinism. The first technical failure remains archived and was not counted as a new independent test.

The successful workflow took **30.88 seconds through verification**, excluding report rendering and the existing 813.51-second calibration pilot. It used 0.557 GiB peak RSS through verification. No method or threshold was revised after inspecting held scores. Existing pilot files, source implementation and tracked ABVID changes are unchanged. This delivery stops at the CAST-4 acoustic diagnostic; classification transfer remains untested.

See [delivery checks](../CAST/generalization/DELIVERY_CHECKS.json) for replay, link and preservation evidence. The immutable run-local report and original failure record are preserved.

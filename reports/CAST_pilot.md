# CAST-0 through CAST-2 pilot

CAST is runnable and the bounded pilot has completed. **The numerical prototype is complete; stable, identifiable acoustic coverage is not established. Stop at CAST-2.**

The experiment fits an effective observation model. Domain: source-real IDMT observations → fitted synthetic observations, compared in sample with the same real observations. Synthetic acceptance is synthetic → synthetic reconstruction. There is no classifier, held-out validation score, transfer experiment, or claim of vehicle/RPM recovery.

## Reproduce

From the ABVID root, using the existing project environment without modifying it:

```sh
bash CAST/run.sh workflow --run-id cast_pilot_reproduction_01
```

Use a fresh run ID: existing runs are never overwritten. This performs the metadata freeze, numerical/provenance tests, seven synthetic fits and the gain-ambiguity check, all 50 IDMT fits, deterministic artifact replay, and reporting. The IDMT files and historical locks must already exist at the recorded paths. See [CAST/README.md](https://github.com/WaterPancake/ABVID/blob/77daa1b339906d35b6fa0a59a166ecf2eac01174/CAST/README.md) for isolated installation and individual stages.

This run: `cast_pilot_v0_20261004`. [Resolved configuration](ARTIFACT_INDEX.md#artifact-21f55c937240), [immutable lock](ARTIFACT_INDEX.md#artifact-9b383e5ccf58), [source/environment snapshot](ARTIFACT_INDEX.md#artifact-12b86b93068d), [selected manifest](ARTIFACT_INDEX.md#artifact-214218cbf6c2), [verification](ARTIFACT_INDEX.md#artifact-9c6abaac6da0), and [audio gallery](ARTIFACT_INDEX.md#artifact-6480fca2972d).

## Frozen access and implementation decisions

- Exactly 50 recordings (25 car / 25 truck), from five conservative groups at three sites. These are the lexicographically first five IDs per group/class within the existing H1 outer-fold-0, seed-42 training selection. No resampling of the training selection and no shortages.
- Held-out group `connected_4001f06f57cfeee7` is excluded before audio access. No MELAUDIS audio/features, mixed-domain feature cache, supplied procedural/AudioLDM bank, military audio, or reserved audio is opened. File bytes are hashed only after exact ID/path/metadata/ancestry checks; symlinks and path traversal are rejected.
- A single class-agnostic renderer has three 10–400 Hz effective-spacing knots, eight harmonic amplitude weights, eight noise-band energy weights, one harmonic-energy fraction, and five envelope knots. Source, background, path and sensor contributions are not identified separately.
- Internal 8 kHz; disjoint rectangular FFT noise masks with DC/Nyquist excluded, each normalized to unit RMS. Harmonics are suppressed at/above Nyquist. A differentiable 41-tap Kaiser-5 FIR matches SciPy's fixed 8→16 kHz resampler. Linear endpoint interpolation and full component waveforms are recorded.
- Input is the arithmetic mean of the two stored CH34 channels, float64 native→8→16 kHz polyphase resampling, centered two seconds, then float32. No padding. CAST fitting subtracts DC and divides by AC RMS; RMS below 1e-7 is rejected. H1 files/preprocessing are unchanged.
- Four starts `[42,123,456,789]`, 300 Adam steps each; control LR 0.03 and spacing-coordinate LR 0.001 (100 Hz coordinate scale). Fixed coarse harmonic salience plus local log-parabolic refinement initializes trajectories. All classes have the same bounds, search and budget. The minimum fit-loss state is retained, including the initialization.
- Two fixed fit-noise realizations are averaged. Two independently derived check realizations score the fitted winner and every initialization. The fitted winner is selected using fit loss only. The baseline is the lowest check loss among all unoptimized starts, a conservative comparison on identical check streams. No seed creates independent ancestry.
- Loss averages relative-magnitude plus natural-log-magnitude error at FFT sizes 256/512/1024/2048; periodic Hann, N/4 hop, center=false, unnormalized one-sided FFT, epsilon 1e-5, no regularization or padded frames. Per-resolution terms are saved. Additional spectral/envelope/modulation diagnostics do not affect fitting.
- Playback restores the original AC RMS to the reconstruction and uses one shared pair gain, capped at 0.95 peak without amplification. DC remains removed from the reconstruction. Original observations, shape-normalized waves, raw reconstruction, components and playback pairs are float WAV; independent peak normalization and clipping are not used.

## Synthetic acceptance and repairs

All criteria were fixed before real decoding. Early engineering runs exposed poor frequency updates and harmonic/subharmonic initialization; their records are retained under `CAST/development/`. Frequency step size and deterministic initialization were repaired without increasing the 300-step budget or relaxing the recovery/improvement tolerances. An unrestricted tone cannot uniquely identify harmonic order. The unambiguous single-tone fixture therefore fixes the known fundamental-only weights while optimizing frequency, noise fraction and envelope. A separate unrestricted-tone fit must recover the observed tone frequency to the same bin tolerance and explicitly retains the harmonic-order ambiguity. This is a synthetic identifiability constraint only; every real fit has all controls free.

| Fixture | Fit-loss improvement | Max spacing error (Hz) | Acceptance |
|---|---:|---:|---|
| single_tone | 99.39% | 0.187 | pass |
| unrestricted_tone | 99.18% | 155.515 | pass |
| harmonic | 97.13% | 0.424 | pass |
| noise | 10.06% | 45.792 | pass |
| changing | 98.34% | 0.361 | pass |
| mixture | 19.94% | 55.210 | pass |
| missing_fundamental | 98.88% | 95.190 | pass |

Single-tone recovery must be within 7.8125 Hz, the longest-STFT bin; harmonic recovery uses the same tolerance, changing trajectories 15.625 Hz, and noise-band L1 error ≤0.35. Every declared nontrivial fixture must improve over its best unoptimized fit-stream start by ≥5%. Missing-fundamental alternatives and the gain/envelope-scale ambiguity are explicitly preserved; unidentifiable physical parameters remain `unknown`.

[Numerical and provenance test log](ARTIFACT_INDEX.md#artifact-b7fe1376bc65), [synthetic results](ARTIFACT_INDEX.md#artifact-647741c53979), and [preregistered fixtures/tolerances](ARTIFACT_INDEX.md#artifact-d0d66140c9fd).

## Real in-sample results

50/50 fits completed; 0 numerical/input/artifact failures. 0/50 completed fits failed the predeclared ≥5% checking-loss improvement diagnostic. These engineering diagnostics are not perceptual or classification acceptance thresholds.

| Class | n | Median initial check | Median fitted check | Median paired gain | p90 fitted | Worst fitted | <5% gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| car | 25 | 1.7229 | 1.4021 | 19.58% | 1.4563 | 1.4742 | 0 |
| truck | 25 | 1.7562 | 1.3664 | 22.08% | 1.4178 | 1.4255 | 0 |

| Source group | n | Median fitted check | p90 | Worst | Median spectral residual | Median envelope RMSE | <5% gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| `connected_654b5df57ec64387` | 10 | 1.4120 | 1.4572 | 1.4598 | 0.2506 | 0.1172 | 0 |
| `connected_667018ac7661cccb` | 10 | 1.3636 | 1.3979 | 1.4067 | 0.2465 | 0.1089 | 0 |
| `connected_72766641e1daf436` | 10 | 1.3912 | 1.4192 | 1.4252 | 0.2360 | 0.0997 | 0 |
| `connected_b00d69ddbb66d0fe` | 10 | 1.3416 | 1.3982 | 1.4021 | 0.2408 | 0.1086 | 0 |
| `connected_c6148cf0cafb5753` | 10 | 1.4141 | 1.4303 | 1.4742 | 0.2475 | 0.1059 | 0 |

Spectral residual is relative error between long-term Welch magnitude envelopes; temporal residual is unit-RMS 50 ms envelope RMSE. The four-resolution fitting loss and these descriptors measure different properties. The first check realization is used consistently for plots/descriptors/playback; both checking realizations contribute to checking losses.

Flags (overlapping, all completed fits retained): `equally_good_starts_disagree`: 43, `no_resolvable_harmonic_salience`: 1, `spacing_unidentified_noise_dominant`: 2.

Boundary hits (within 1% of bounded controls, or weights below 0.001): `spacing_hz`: 2.

Near-equal starts are within 5% of the best fit objective. Spacing differences above 7.8125 Hz or harmonic-fraction differences above 0.2 are flagged. Noise-dominant fits cannot establish meaningful spacing. Boundary and ambiguity flags are evidence to inspect, not reasons to discard inconvenient examples.

## Examples and complete artifacts

Preselected examples were chosen before decoding. Best/median/worst examples are ranked by checking loss with ID tie-breaking, including poor cases. Every selected record has a failure/diagnostic record.

| Selection | Class | Check loss | Inspection |
|---|---|---:|---|
| Preselected car | car | 1.4569 | [original](ARTIFACT_INDEX.md#artifact-e1d8cc0e7137) · [reconstruction](ARTIFACT_INDEX.md#artifact-aeae42fe468b) · [plot](ARTIFACT_INDEX.md#artifact-56ff4ac00e7c) · [parameters](ARTIFACT_INDEX.md#artifact-cb3047e96031) · [flags](ARTIFACT_INDEX.md#artifact-e0d492319fde) |
| Preselected truck | truck | 1.3967 | [original](ARTIFACT_INDEX.md#artifact-65892fd135b1) · [reconstruction](ARTIFACT_INDEX.md#artifact-7fda9dd0d9b7) · [plot](ARTIFACT_INDEX.md#artifact-2df2c320aa41) · [parameters](ARTIFACT_INDEX.md#artifact-3985bd681665) · [flags](ARTIFACT_INDEX.md#artifact-b10780420d60) |
| Best check loss | car | 1.2647 | [original](ARTIFACT_INDEX.md#artifact-64563b79bf30) · [reconstruction](ARTIFACT_INDEX.md#artifact-e83676a5d1d4) · [plot](ARTIFACT_INDEX.md#artifact-4e2c5b7b3705) · [parameters](ARTIFACT_INDEX.md#artifact-e8fc55cc386d) · [flags](ARTIFACT_INDEX.md#artifact-2530e5b01f0d) |
| Median check loss | truck | 1.3913 | [original](ARTIFACT_INDEX.md#artifact-1d944f6cda8c) · [reconstruction](ARTIFACT_INDEX.md#artifact-61f0571dbcfd) · [plot](ARTIFACT_INDEX.md#artifact-7b0e9249c757) · [parameters](ARTIFACT_INDEX.md#artifact-37d16a216f7d) · [flags](ARTIFACT_INDEX.md#artifact-681af6072f05) |
| Worst check loss | car | 1.4742 | [original](ARTIFACT_INDEX.md#artifact-4b785e8d6406) · [reconstruction](ARTIFACT_INDEX.md#artifact-93e455b0a5cf) · [plot](ARTIFACT_INDEX.md#artifact-09a310ca7e5c) · [parameters](ARTIFACT_INDEX.md#artifact-4aa98c9140e0) · [flags](ARTIFACT_INDEX.md#artifact-cd2ec3ff708d) |

Each clip also contains optimization history, all alternative starts and scores, per-resolution checking losses, original RMS/DC, source and parent-lock ancestry, component WAVs/NPZ, timing, and hashes. [Complete pilot summary](ARTIFACT_INDEX.md#artifact-c7f14db42889) and [aggregate JSON](ARTIFACT_INDEX.md#artifact-155ba4b50b9c).

## Measured runtime and verification

CPU-only `arm64` on `macOS-26.6.2-arm64-arm-64bit`; one Torch thread. Synthetic stage: 113.41 s. Real pilot end-to-end: 813.51 s for 100 fitted audio seconds, or 8.14 s per audio second. Median numerical-fit time: 15.85 s/clip. Process peak RSS: 0.689 GiB (lifetime high-water mark, not a per-clip incremental measurement).

The first allowed clip was timed before projecting the remaining pilot cost: [projection record](ARTIFACT_INDEX.md#artifact-b4ce33a9c5bf). Actual timings supersede that projection. Software: matplotlib 3.11.1, numpy 2.4.6, pytest 9.1.1, scipy 1.17.1, soundfile 0.14.0, torch 2.13.0.

Verification rehashed all 50 permitted originals, checked per-clip artifact hashes/ancestry, replayed all completed fitted waveforms bit-for-bit in the pinned environment, recomputed checking losses, and checked common-gain playback. Existing tracked ABVID changes remained byte-identical to the starting diff. Commit `0d242c6d76ff036747f89576ee15d5f7399f5891` was dirty; the saved source snapshot, hashes and dependency lock identify this implementation, not the commit alone. Config SHA256 `88f0f975b9c39c7510c0576f815be1b2cd685070faf9fd045950e496b5b6b5bb`.

## Adequacy verdict and limitations

**Adequate as an inspectable bounded reconstruction prototype. Not yet evidence for a stable parameter distribution or real-world transfer.** Synthetic tests establish that numerical fitting works on declared renderer-generated cases. Real checking-loss improvements establish only better in-sample descriptor matching than initialization. Spectral and temporal residuals, parameter boundaries, alternative fits and noise-dominant solutions limit the interpretation of the effective controls.

The smallest justified next revision is to inspect the saved spectral/temporal failures and repeated-start disagreements, then propose a separately versioned source-only change aimed at the dominant residual. Preserve these results as the control. Do not create a parameter distribution, add a classifier, evaluate held-out groups or targets, or enlarge the pilot without a new authorized stage. CAST-3 through CAST-6 were not implemented.

Two-second observations give little evidence about slow dynamics. A low spectral loss is not a listening-test result, known vehicle identity, recovered RPM, dry source, or causal source/propagation separation. Five conservative groups at three sites do not identify independent physical vehicles or support session-population confidence claims. All 50 observations are engineering development exposures. Input recordings and generated derivatives retain their ancestry and local-only evaluation status; IDMT's recorded licence is CC BY-NC-ND 4.0. This task makes no redistribution/publication authorization claim.

## Post-run review

The complete workflow took **933.75 s (15.56 minutes)** including the tests, synthetic stage, pilot, verification and automatic report. The recorded peak after verification was **0.692 GiB**; the pilot-stage high-water mark above was 0.689 GiB. All 29 CAST tests passed, as did three separately executed legacy source-simulator regression tests. The gallery contains 100 playback controls and 50 plots; all report/gallery artifact links were checked. The preselected case and the systematically ranked best/worst plots were visually inspected. No human listening study was performed.

The unrestricted-tone fixture recovered its **observed dominant frequency within 0.378 Hz**, while selecting a different harmonic-order explanation. Its large raw-spacing difference in the table is therefore explicitly *not* unique parameter recovery. Noise-only, mixed and missing-fundamental fixtures likewise do not establish unique spacing. The known-fundamental single-tone fixture's constraint is used only for the numerical recovery test and is recorded in its optimization artifact; real fits have no such prior.

The v0 `peak_salience_ratio` is a peak-to-median Welch-power heuristic. Its `no_resolvable_harmonic_salience` flag should be read as a conservative inspection cue, not a validated harmonic-series detector: broadband spectral coloration can also create large peak ratios. The initialization's separate harmonic-comb search is recorded, but its candidate ranking is not a calibrated physical confidence estimate.

The clearest next revision to propose is **identifiability-aware reporting of equivalent fits**, before treating one latent vector as a distribution target: retain the near-equal alternatives, mark noise-dominant spacing as unresolved, and distinguish observed harmonic peaks from their nonunique fundamental/order explanations. The synthetic tone and missing-fundamental cases already justify that revision; they do not justify adding an architecture or consuming another dataset. It was not implemented as a later CAST stage here. The best-loss real example still has an overestimated late envelope, and the worst-loss example has a shifted/smoothed amplitude peak and spectral detail left unmatched. These residuals support a limited reconstruction verdict, even when aggregate loss improves.

The immutable run-local report and numerical artifacts remain as generated. This final review adds interpretation without altering the frozen method, fits, acceptance criteria, or their hashes. See [engineering decisions](ARTIFACT_INDEX.md#artifact-2ee5e66db70e), [CAST research log](https://github.com/WaterPancake/ABVID/blob/77daa1b339906d35b6fa0a59a166ecf2eac01174/CAST/RESEARCH_LOG.md), and [delivery checks](https://github.com/WaterPancake/ABVID/blob/77daa1b339906d35b6fa0a59a166ecf2eac01174/CAST/DELIVERY_CHECKS.json).

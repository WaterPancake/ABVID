---
title: Vehicle Acoustic Classification Study Guide
aliases:
  - ABVID Study Plan
  - Vehicle Acoustics Learning Roadmap
tags:
  - abvid
  - acoustics
  - digital-signal-processing
  - machine-learning
  - study-plan
status: active
created: 2026-08-19
---

# Vehicle Acoustic Classification Study Guide

This guide builds the technical and mathematical background needed to understand,
implement, and discuss the ABVID passive vehicle-acoustics project. It follows the
project's milestone order:

1. deterministic audio augmentation;
2. tracked-versus-wheeled classification;
3. controlled synthetic vehicle sources;
4. multichannel simulation, localization, and beamforming;
5. real-world transfer protocols and their negative results;
6. noise-invariant representation learning, pretrained encoders, and leakage-safe
   session evaluation.

Milestones 1-6 are implemented and documented in this repository. The open question,
guarded by the Milestone 7 gate, is whether a development run clears the criteria
required before hierarchical and open-set classification can begin. Later milestones
remain architectural guidance only.

The default pace is **15 weeks at 5–7 hours per week**. If time is limited, complete
the items marked **Core** and postpone the optional material.

## How to use this guide

For each week:

- [ ] Read or watch the assigned core source.
- [ ] Write the important equations by hand.
- [ ] Complete the repository exercise.
- [ ] Explain the week's concepts aloud without looking at notes.
- [ ] Record questions and misconceptions under a dated heading in your own notes.

Do not measure progress by how many pages you read. Measure it by whether you can:

- predict what a parameter change will do;
- connect an equation to code in the repository;
- recognize an invalid experiment;
- explain the project's assumptions and limitations clearly.

## Codebase map

Every milestone follows the same pattern. Use this table to navigate from a concept
to the file that implements or evaluates it. Entry points are runnable commands or
`scripts/*.py`; the evaluation logic lives in `src/vehicle_audio/*_evaluation.py`.

| Concept | Entry point | Config | Source | Tests | Report |
|---|---|---|---|---|---|
| Augmentation engine | `python -m vehicle_audio.cli generate` | `configs/default.yaml` | `src/vehicle_audio/{augment,mixing,audio,dataset,manifest}.py` | `tests/test_{augment,mixing,audio,dataset,manifest}.py` | `docs/milestones_1_3_report.md` |
| Classical/CNN baselines | `scripts/train_baseline.py` | — | `src/vehicle_audio/baseline.py` | `tests/test_baseline.py` | `docs/milestones_1_3_report.md` |
| Procedural sources | `python -m vehicle_audio.cli synthesize-sources` | `configs/procedural_vehicles.yaml` | `src/vehicle_audio/source_simulation.py` | `tests/test_source_simulation.py` | `docs/milestones_1_3_report.md` |
| Microphone array | `python -m vehicle_audio.cli generate-array` | `configs/multichannel.yaml` | `src/vehicle_audio/{multichannel,array_dataset}.py` | `tests/test_multichannel.py` | `docs/milestone4_report.md` |
| Localization/beamforming | `scripts/evaluate_multichannel.py` | — | `src/vehicle_audio/multichannel_evaluation.py` | — | `docs/milestone4_report.md` |
| Real corpus preparation | `python -m vehicle_audio.cli prepare-real` | `configs/real_corpus.yaml` | `src/vehicle_audio/real_corpus.py` | `tests/test_real_corpus.py` | `docs/milestone5_status.md` |
| Transfer evaluation | `scripts/evaluate_transfer.py` | — | `src/vehicle_audio/transfer_evaluation.py` | `tests/test_transfer_evaluation.py` | `docs/milestone5_status.md` |
| Invariance training | `scripts/evaluate_invariance.py` | — | `src/vehicle_audio/invariance_evaluation.py` | `tests/test_invariance_evaluation.py` | `docs/milestone6_report.md` |
| Seed aggregation | `scripts/aggregate_invariance.py` | — | `src/vehicle_audio/invariance_aggregation.py` | `tests/test_invariance_aggregation.py` | `docs/milestone6_improvement_checkpoint.md` |
| Pretrained PANNs | `scripts/evaluate_pretrained.py` | — | `src/vehicle_audio/pretrained_evaluation.py` | `tests/test_pretrained_evaluation.py` | `docs/milestone6_pretrained_transfer_report.md` |
| Semantic session eval | `scripts/evaluate_semantic_sessions.py` | — | `src/vehicle_audio/semantic_session_evaluation.py` | `tests/test_semantic_session_evaluation.py` | `docs/milestone6_semantic_session_report.md` |
| Fusion session eval | `scripts/evaluate_fusion_sessions.py` | — | `src/vehicle_audio/fusion_session_evaluation.py` | `tests/test_fusion_session_evaluation.py` | `docs/milestone6_fusion_session_report.md` |

Repository conventions to notice as you work:

- configurations live in `configs/`, entry points in `scripts/`, and evaluation logic
  in `src/vehicle_audio/*_evaluation.py`;
- every experiment writes `experiment.json`, `splits.json`, model checkpoints,
  `metrics.json`, and CSV/plot summaries under `runs/<milestone>_<name>_seed<seed>/`;
- `data/generated/` holds paired `clean.wav` + `corrupted.wav` events with
  `manifest.jsonl`; `data/real_eval_v2/` holds the reviewed native-real windows;
- machine-readable milestone reports live in `docs/`.

## Target level of understanding

At the end of this plan, you should be able to explain:

- how sample rate, filtering, convolution, and controlled-SNR mixing work;
- why a log-Mel spectrogram is useful for vehicle classification;
- what MFCC, spectral centroid, bandwidth, and rolloff measure;
- how a CNN learns from time-frequency structure;
- why recording-session leakage invalidates an evaluation;
- why accuracy must be reported against SNR and test domain;
- how engine RPM produces harmonic orders;
- why an unseen-vehicle test is different from a random test split;
- what domain randomization can and cannot establish;
- how a microphone array localizes a source and why more channels are not always
  better for classification;
- why invariance training can help in-distribution yet fail to transfer to real audio;
- what a leakage-safe nested session evaluation is and why thresholds are selected
  on inner folds only;
- why an external pretrained encoder is not a strict synthetic-only baseline;
- what the Milestone 7 gate requires before hierarchical or open-set work;
- which ABVID recordings are real and which observations or sources are synthetic.

# Prerequisite check

You can begin immediately if you are comfortable with basic Python. Review the
mathematics as it appears rather than delaying all DSP work until the math is
complete.

## Minimum mathematics

- [x] Rearrange algebraic equations.
- [x] Work with powers, roots, exponentials, and base-10 logarithms.
- [x] Recognize sine and cosine waves.
- [x] Understand vectors, matrices, means, and variance.
- [x] Understand derivatives as local rates of change.
- [x] Interpret probability as a number between 0 and 1.

If several items are unfamiliar, use the free companion site for
[Mathematics for Machine Learning](https://mml-book.github.io/). Prioritize:

1. linear algebra;
2. probability and distributions;
3. vector calculus;
4. continuous optimization.

Do not attempt to finish the whole book before beginning Week 1.

# Week 1 — Waveforms, sampling, decibels, and SNR

## Outcomes

You should understand:

- discrete audio samples and sample rate;
- amplitude versus power;
- frequency, phase, and periodicity;
- the Nyquist frequency and aliasing;
- why amplitude ratios use 20 in a decibel equation while power ratios use 10;
- how controlled-SNR noise mixing is calculated.

## Core sources

1. [Think DSP — Digital Signal Processing in Python](https://greenteapress.com/thinkdsp/thinkdsp.pdf): read the introductory waveform and spectrum chapters.
2. [MIT 6.003 Signals and Systems](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/): use the lectures or notes on discrete signals and sampling.
3. [MIT Acoustics of Speech and Hearing lecture notes](https://ocw.mit.edu/courses/6-551j-acoustics-of-speech-and-hearing-fall-2004/pages/lecture-notes/): read Lecture 1 for amplitude, frequency, phase, spectra, and decibels.

## Equations

Amplitude gain:

$$
G_{dB}=20\log_{10}\left(\frac{A_2}{A_1}\right)
$$

Power SNR:

$$
\operatorname{SNR}_{dB}=10\log_{10}\left(\frac{P_s}{P_n}\right)
$$

Nyquist frequency:

$$
f_{\text{Nyquist}}=\frac{f_s}{2}
$$

## Repository exercise

Read:

- [`src/vehicle_audio/mixing.py`](../src/vehicle_audio/mixing.py)
- [`tests/test_mixing.py`](../tests/test_mixing.py)

Then:

- [ ] Derive the noise scaling factor used by `mix_at_snr`.
- [ ] Predict how much larger the noise amplitude becomes when SNR changes from 20 dB to 0 dB.
- [ ] Run the SNR tests and connect each assertion to the equation above.

```bash
uv run pytest tests/test_mixing.py -q
```

## Explain aloud

> What does a 0 dB SNR mean, and why does it not mean that the waveform is silent?

# Week 2 — Fourier transforms, STFT, and spectrograms

## Outcomes

You should understand:

- how sinusoids form a frequency representation;
- the difference between FFT and STFT;
- window length, hop length, and overlap;
- time resolution versus frequency resolution;
- magnitude versus power spectra;
- spectral leakage and window functions.

## Core sources

1. [MIT RES.6-007 Signals and Systems](https://ocw.mit.edu/courses/res-6-007-signals-and-systems-spring-2011/): study the Fourier transform and sampling lectures.
2. [MIT 6.003 reading assignments and notes](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/pages/readings/): use the Fourier and sampling sections as a reference.
3. [Official torchaudio audio-feature tutorial](https://docs.pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html): work through waveform, spectrogram, and Mel-spectrogram examples.

## Repository exercise

Generate the toy inputs and visualization:

```bash
uv run python scripts/create_toy_audio.py --output .artifacts/study_toy_inputs
uv run python -m vehicle_audio.cli generate \
  --config configs/default.yaml \
  --targets .artifacts/study_toy_inputs/targets \
  --backgrounds .artifacts/study_toy_inputs/backgrounds \
  --impulse-responses .artifacts/study_toy_inputs/impulse_responses \
  --output .artifacts/study_toy_generated \
  --num-samples 6 \
  --seed 42
```

Then use [`scripts/inspect_sample.py`](../scripts/inspect_sample.py).

- [ ] Identify steady harmonic lines in the clean spectrogram.
- [ ] Identify broadband noise in the corrupted spectrogram.
- [ ] Change FFT size and describe the visible tradeoff.
- [ ] Change hop length and describe what changes and what does not.

## Explain aloud

> Why can a single FFT hide an acceleration event that an STFT makes visible?

# Week 3 — Filters, resampling, convolution, and nonlinear distortion

## Outcomes

You should understand:

- low-pass, high-pass, and band-pass filters;
- cutoff frequency, filter order, and frequency response;
- resampling and anti-alias filtering;
- impulse responses and convolution;
- hard clipping and harmonic distortion;
- dynamic-range compression.

## Core sources

1. [MIT 6.003 Signals and Systems](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/): focus on LTI systems, convolution, impulse response, and frequency response.
2. [PyTorch audio feature-extraction tutorial](https://docs.pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html): study the resampling comparison.
3. Review the relevant filtering and convolution chapters of [Think DSP](https://greenteapress.com/thinkdsp/thinkdsp.pdf).

## Key equation

Discrete convolution:

$$
y[n]=(x*h)[n]=\sum_k x[k]h[n-k]
$$

## Repository exercise

Read:

- [`src/vehicle_audio/augment.py`](../src/vehicle_audio/augment.py)
- [`tests/test_augment.py`](../tests/test_augment.py)

- [ ] Match every augmentation function to its physical or recording interpretation.
- [ ] Plot the same harmonic input before and after each filter.
- [ ] Explain why round-trip resampling returns to the configured output sample rate.
- [ ] Explain why clipping and compression are not interchangeable.

```bash
uv run pytest tests/test_augment.py -q
```

## Explain aloud

> What does convolution with a room impulse response represent physically?

# Week 4 — Practical acoustics and microphones

## Outcomes

You should understand:

- sound pressure, intensity, and sound-pressure level;
- inverse-distance and inverse-square relationships;
- free-field assumptions;
- direct sound, early reflections, and diffuse reverberation;
- room impulse response and RT60;
- microphone frequency response and directionality;
- why real environments violate simple propagation assumptions.

## Core sources

1. [MIT Acoustics of Speech and Hearing lecture notes](https://ocw.mit.edu/courses/6-551j-acoustics-of-speech-and-hearing-fall-2004/pages/lecture-notes/): Lectures 1–3.
2. [MIT Music and Technology — physics of sound](https://ocw.mit.edu/courses/21m-380-music-and-technology-recording-techniques-and-audio-production-fall-2016/ae4e07f3da3b04c46a93486b9ac9d6d6_MIT21M_380F16_ses02_note.pdf): inverse-distance, inverse-square, and dB relationships.
3. [MIT Music and Technology — room acoustics and reverberation](https://ocw.mit.edu/courses/21m-380-music-and-technology-recording-techniques-and-audio-production-fall-2016/d315c75efb01cb76ca897958570bab87_MIT21M_380F16_ses18_note.pdf).

## Important distinction

For an ideal spherical wave in a free field:

$$
p(r)\propto\frac{1}{r}
$$

but intensity follows:

$$
I(r)\propto\frac{1}{r^2}
$$

This distinction prevents the common error of saying that sound pressure itself
always follows an inverse-square law.

## Optional practical source

[Pyroomacoustics room-simulation documentation](https://pyroomacoustics.readthedocs.io/en/stable/pyroomacoustics.room.html) shows how sources, microphones, and room impulse responses are represented in a simulator. Treat it as preparation for the Week 11 microphone-array material, not as a replacement for understanding the equations.

## Repository exercise

- [ ] Locate the random gain, impulse-response, and microphone-response stages in the pipeline.
- [ ] Write down which effects are physically modeled and which are approximations.
- [ ] Explain why the final measured SNR may differ from the requested mixing-stage SNR.

## Explain aloud

> Why is “6 dB quieter per distance doubling” only an approximation in a real environment?

# Week 5 — Log-Mel, MFCC, and classical audio features

## Outcomes

You should understand:

- linear-frequency versus Mel-frequency spectrograms;
- logarithmic power scaling;
- how MFCCs are derived from log-Mel energies;
- spectral centroid, bandwidth, rolloff, and flatness;
- why time aggregation choices matter;
- what information handcrafted features may discard.

## Core sources

1. [Official torchaudio feature-extraction tutorial](https://docs.pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html).
2. [Official torchaudio MFCC documentation](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.MFCC.html).
3. Review the spectrum and spectrogram chapters of [Think DSP](https://greenteapress.com/thinkdsp/thinkdsp.pdf).

## Repository exercise

Read [`src/vehicle_audio/baseline.py`](../src/vehicle_audio/baseline.py).

- [ ] Trace waveform input into the log-Mel extractor.
- [ ] Record `sample_rate`, `n_fft`, `hop_length`, `n_mels`, `f_min`, and `f_max`.
- [ ] Calculate the approximate time represented by one hop.
- [ ] Explain why features must be calculated from one explicitly selected microphone channel.

## Explain aloud

> What does an MFCC summarize, and why might a CNN retain information that an MFCC baseline loses?

# Week 6 — Classical classification and trustworthy evaluation

## Outcomes

You should understand:

- logistic regression and decision boundaries;
- class probabilities and cross-entropy;
- train, validation, and test roles;
- accuracy, balanced accuracy, precision, recall, and F1;
- confusion matrices;
- group-based splitting and source leakage;
- majority-class and chance baselines.

## Core sources

1. [Scikit-learn model-evaluation guide](https://scikit-learn.org/stable/modules/model_evaluation.html).
2. [Scikit-learn GroupKFold documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).
3. [Scikit-learn F1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).
4. [PyTorch neural network from first principles](https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html): the first model is logistic regression.

## Equations

Precision and recall:

$$
\operatorname{precision}=\frac{TP}{TP+FP}
$$

$$
\operatorname{recall}=\frac{TP}{TP+FN}
$$

F1:

$$
F_1=2\frac{\operatorname{precision}\cdot\operatorname{recall}}
{\operatorname{precision}+\operatorname{recall}}
$$

## Repository exercise

- [ ] Inspect a generated manifest and count unique `recording_session` values.
- [ ] Verify that an original source never crosses train and test boundaries.
- [ ] Compare accuracy with balanced accuracy on an intentionally imbalanced toy prediction.
- [ ] Build a confusion matrix by hand from ten predictions.

## Non-negotiable project rule

All windows from one original recording belong to the same group:

```text
one source recording
→ one recording_session
→ one train/validation/test split
```

## Explain aloud

> Why can a model achieve excellent random-window accuracy while learning nothing that generalizes to a new vehicle recording?

# Week 7 — Neural networks, CNNs, and PyTorch

## Outcomes

You should understand:

- tensors, modules, parameters, and computation graphs;
- forward passes, loss, backpropagation, and optimization;
- convolution kernels and feature maps;
- nonlinear activations and pooling;
- batch normalization;
- softmax class probabilities;
- overfitting and checkpoint selection.

## Core sources

1. [Official PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/index.html): tensors, data loaders, model building, autograd, optimization, and saving models.
2. [PyTorch autograd tutorial](https://docs.pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html).
3. [Stanford CS231n syllabus and course materials](https://cs231n.stanford.edu/2021/schedule.html): linear classifiers, losses, backpropagation, CNNs, and training practice. The examples are visual, but the convolution and optimization principles transfer to spectrograms.

## Repository exercise

Read:

- [`src/vehicle_audio/baseline.py`](../src/vehicle_audio/baseline.py)
- [`tests/test_baseline.py`](../tests/test_baseline.py)

- [ ] Draw the CNN as boxes with tensor dimensions.
- [ ] Calculate its parameter count layer by layer.
- [ ] Explain temporal pooling and what timing information it removes.
- [ ] Identify the loss function, optimizer, and checkpoint-selection metric.
- [ ] Explain why training loss is not a test result.

## Explain aloud

> What does a convolutional filter learn when its input is a log-Mel spectrogram?

# Week 8 — Vehicle acoustics and order analysis

## Outcomes

You should understand:

- RPM, rotational frequency, and engine order;
- firing events and harmonics;
- throttle versus engine load;
- idle, acceleration, steady speed, and deceleration;
- transmission whine and gear changes;
- tire-road broadband noise;
- track-link impacts and periodic modulation;
- why tracked-versus-wheeled cues can be confounded by vehicle identity.

## Core sources

1. [Ansys introduction to order analysis](https://ansyshelp.ansys.com/public/Views/Secured/corp/v242/en/Sound_SAS_UG/Sound/UG_SAS/what_order_analysis__is_76022.html).
2. [Dewesoft order-tracking manual](https://downloads.dewesoft.com/manuals/dewesoft-order-tracking-manual-en.pdf): focus on RPM-dependent spectral lines and order tracking.
3. Revisit the STFT material from Week 2 and connect diagonal spectral ridges to changing RPM.

## Key equation

Rotational order frequency:

$$
f_{\text{order}}=\frac{\operatorname{RPM}}{60}\times\operatorname{order}
$$

## Repository exercise

- [ ] Compare spectrograms for idle, acceleration, steady speed, and deceleration recordings.
- [ ] Mark frequency components that rise with speed.
- [ ] Separate hypotheses about engine, transmission, tire, and track sounds.
- [ ] List at least five non-mobility cues a classifier might memorize.

Examples of confounders include microphone, codec, location, background, vehicle
model, recording gain, and session-specific reverberation.

## Explain aloud

> Why is “the model detects track clatter” a hypothesis rather than a conclusion from high classification accuracy?

# Week 9 — Synthetic data, domain shift, and generalization

## Outcomes

You should understand:

- source simulation versus microphone/environment augmentation;
- domain shift and nuisance variables;
- domain randomization;
- in-distribution versus out-of-distribution evaluation;
- unseen-state, unseen-environment, and unseen-vehicle tests;
- synthetic-to-synthetic versus synthetic-to-real claims.

## Core sources

1. [Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907). The application is visual robotics, but the experimental principle is relevant.
2. Review [Pyroomacoustics room simulation](https://pyroomacoustics.readthedocs.io/en/stable/pyroomacoustics.room.html) as an example of separating source, room, and microphone variables.
3. Read the Milestone 3 and cross-milestone rules in [`AGENTS.md`](../AGENTS.md).

## Conceptual model

Treat an observation as:

$$
x=g(v,s,e,m,n)
$$

where:

- $v$ is vehicle identity or mobility class;
- $s$ is operating state;
- $e$ is environment;
- $m$ is microphone response and geometry;
- $n$ is background noise.

The objective is to preserve information about mobility class while preventing the
model from depending on a single value of the nuisance variables.

## Repository exercise

- [ ] Design an unseen-state split without sharing source sessions.
- [ ] Design an unseen-environment split.
- [ ] Design an unseen-vehicle split with at least one held-out model per class.
- [ ] State which conclusions each split supports and does not support.

## Explain aloud

> Why does good synthetic-to-synthetic performance not demonstrate transfer to real recordings?

# Week 10 — Calibration, experiment reporting, and technical communication

## Outcomes

You should understand:

- predicted probability versus empirical correctness;
- reliability diagrams and expected calibration error;
- why discrimination and calibration are different;
- reproducible experiment metadata;
- how to present limitations without underselling the work.

## Core sources

1. [Scikit-learn probability-calibration guide](https://scikit-learn.org/stable/modules/calibration.html).
2. [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).
3. Review the cross-milestone experiment rules in [`AGENTS.md`](../AGENTS.md).

## Repository exercise

For one experiment, verify that its artifacts record:

- [ ] code or git revision;
- [ ] complete configuration;
- [ ] random seed;
- [ ] manifest hash or dataset version;
- [ ] train, validation, and test groups;
- [ ] model checkpoint;
- [ ] aggregate and per-SNR metrics;
- [ ] test domain;
- [ ] known limitations.

Prepare a five-minute explanation with this structure:

1. research question;
2. dataset and source domain;
3. corruption process;
4. leakage-safe split;
5. baselines;
6. result versus SNR;
7. limitations;
8. next discriminating experiment.

## Explain aloud

> If a model reports 90% confidence, what observation would show that this confidence is calibrated?

# Week 11 — Microphone arrays, direction of arrival, and beamforming (Milestone 4)

## Outcomes

You should understand:

- uniform linear array geometry and the broadside-referenced azimuth convention;
- the far-field plane-wave assumption and inter-microphone arrival delay;
- fractional delay and why sample-quantized delay is not sufficient;
- GCC-PHAT as a phase-only weighted cross-correlation and why that weighting is robust;
- SRP-PHAT as a search over candidate source directions;
- delay-and-sum beamforming and what it does to the signal-to-noise ratio;
- angular error, and why a linear array is ambiguous outside the front half-plane.

## Core sources

1. [Knapp and Carter, "The generalized correlation method for estimation of time delay" (1976)](https://ui.adsabs.harvard.edu/abs/1976ITASS..24..320K/abstract) — the origin of GCC-PHAT.
2. [DiBiase, "A High-Accuracy, Low-Latency Technique for Talker Localization in Reverberant Environments Using Microphone Arrays" (2000)](https://www.semanticscholar.org/paper/4e3c7d012c9312c177b740db3dcf3838e5d78e8a) — the SRP-PHAT method.
3. Revisit the [pyroomacoustics room-simulation documentation](https://pyroomacoustics.readthedocs.io/en/stable/pyroomacoustics.room.html) from Week 4 and notice how microphone positions are represented.
4. Read the results and limitation sections of [`docs/milestone4_report.md`](../docs/milestone4_report.md).

## Key equations

The repo measures azimuth from array broadside: `0` degrees is positive y and
positive angles rotate toward positive x, giving a direction unit vector
`u = (sin θ, cos θ, 0)`. For a plane wave, the arrival delay at microphone `i`
relative to microphone 0 is:

$$
\tau_i = -\frac{p_i \cdot u}{c}
$$

where `p_i` is the microphone position and `c` is the speed of sound (343 m/s in
the code). GCC-PHAT estimates the delay between two channels from the phase-only
cross-spectrum:

$$
\hat R(\tau)=\int \frac{X_1(\omega) X_2^*(\omega)}{|X_1(\omega) X_2^*(\omega)|} e^{j\omega\tau}\,d\omega
\qquad
\hat\tau = \arg\max_\tau \hat R(\tau)
$$

## Repository exercise

Read the array simulation and estimators in [`src/vehicle_audio/multichannel.py`](../src/vehicle_audio/multichannel.py):

- [ ] Verify that `azimuth_unit_vector` matches the broadside convention above.
- [ ] Derive the delay formula from `far_field_arrival_delays_seconds` and confirm the sign convention.
- [ ] Explain why `fractional_delay` interpolates rather than rounding to integer samples.
- [ ] Compare the weighting in `gcc_phat_delay_samples` with the plain cross-correlation case.
- [ ] State the assumption behind `delay_and_sum_beamform` and what it trades off.

Run the multichannel tests, which are self-contained and need no audio corpus:

```bash
uv run pytest tests/test_multichannel.py -q
```

Then open `runs/m4_multichannel_paired_sessions_seed42/metrics.json` and
[`docs/milestone4_report.md`](../docs/milestone4_report.md), and verify by hand:

- [ ] Recompute balanced accuracy from the single-microphone confusion matrix `[[84, 0], [13, 71]]` and confirm 92.26%.
- [ ] Confirm the aggregate ordering: 2-mic feature fusion beat 4-mic beamforming for classification.
- [ ] Confirm that 4 microphones localized better than 2 (mean angular error 9.00° vs 22.22° for GCC-PHAT).
- [ ] Reconcile the two: why can more microphones help localization but not strictly help classification?

## Explain aloud

> Why is a four-microphone uniform linear array restricted to the unambiguous front half-plane `(-90, 90)`?

> Why does GCC-PHAT generally beat plain cross-correlation in noise and reverberation?

# Week 12 — Real-world transfer and the sim-to-real protocol (Milestone 5)

## Outcomes

You should understand:

- why native-real windows come from one explicitly selected channel with provenance;
- the fails-closed corpus audit and why it refuses to proceed on incomplete data;
- the three transfer protocols: real-only, synthetic-only, and synthetic plus a
  fraction of real training data;
- why the held-out recording session is the test unit, not the window;
- learning curves and what would count as closing the sim-to-real gap;
- how to interpret a negative result: source/session overfitting versus category learning.

## Core sources

1. Re-read the cross-milestone experiment rules and the dataset leakage rules in [`AGENTS.md`](../AGENTS.md) with this week's questions in mind.
2. Read [`docs/milestone5_status.md`](../docs/milestone5_status.md) in full, including the split, hashes, and limitations.
3. Revisit the Week 9 domain-randomization paper and note what it does **not** establish about real-world transfer.

## Repository exercise

- [ ] Read `audit_real_manifest` in [`src/vehicle_audio/real_corpus.py`](../src/vehicle_audio/real_corpus.py) and list every condition that makes the audit fail closed.
- [ ] Read `validate_transfer_protocol` and `stratified_fraction_indices` in [`src/vehicle_audio/transfer_evaluation.py`](../src/vehicle_audio/transfer_evaluation.py) and state how the 1/5/10/25% real fractions are stratified.
- [ ] Run the self-contained transfer tests:

```bash
uv run pytest tests/test_real_corpus.py tests/test_transfer_evaluation.py -q
```

- [ ] Open `runs/m5_transfer_seed42/metrics.json` and the learning-curve plot, and confirm the headline numbers: real-only 10.2% and synthetic-only 32.8% balanced accuracy on the fixed real test set.
- [ ] Explain why every model misses the held-out wheeled session even though the synthetic-only model is the best of the three on real audio.

## Explain aloud

> If a model scores 95% on synthetic-to-synthetic but 30% on real-to-real, what exactly did it learn?

> What would a convincing sim-to-real transfer result look like, given the corpus constraints described in the report?

# Week 13 — Noise-invariant representation learning (Milestone 6, part 1)

## Outcomes

You should understand:

- the paired clean/corrupted examples produced by Milestone 1 and the invariance objective;
- the difference between augmentation-only training and explicit consistency training;
- positive, hard-positive (same vehicle, different state), and hard-negative (different vehicle) pairs;
- a combined objective of class-weighted cross-entropy, cosine consistency, and a cosine triplet margin;
- the projection head and why leaving the classifier encoder free matters;
- nuisance partitions (held-out noise category, held-out geometry, microphone response);
- why repeated training seeds must share a fixed split seed before they may be aggregated.

## Core sources

1. [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) — the triplet-margin idea used in the invariance objective.
2. [A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)](https://arxiv.org/abs/2002.05709) — broader context for learning invariant embeddings.
3. Read [`docs/milestone6_report.md`](../docs/milestone6_report.md) and [`docs/milestone6_improvement_checkpoint.md`](../docs/milestone6_improvement_checkpoint.md).

## Key equation

The triplet-margin objective in the code pushes an anchor `a` closer to a positive
`p` than to a negative `n` by at least a margin `m`:

$$
L_{\text{triplet}}=\max\left(0,\;d(a,p)-d(a,n)+m\right)
$$

where `d` is cosine distance. The total training objective combines this with
class-weighted cross-entropy and a clean/corrupted cosine consistency term; read
`evaluate_invariance.py` for the exact weights (`--consistency-weight`,
`--hard-positive-weight`, `--hard-negative-weight`, `--triplet-margin`).

## Repository exercise

- [ ] Read `hard_positive_partner_indices` and `hard_negative_partner_indices` in [`src/vehicle_audio/invariance_evaluation.py`](../src/vehicle_audio/invariance_evaluation.py) and state which metadata makes a partner a "hard" pair.
- [ ] Read `build_nuisance_partitions` and list which conditions are held out of training.
- [ ] Read `aggregate_invariance_runs` in [`src/vehicle_audio/invariance_aggregation.py`](../src/vehicle_audio/invariance_aggregation.py) and note which fields must match before runs are combined.
- [ ] Run the invariance tests:

```bash
uv run pytest tests/test_invariance_evaluation.py tests/test_invariance_aggregation.py -q
```

**Predict, then verify.** Before reading the results, write down your prediction:
will invariance training transfer to the fixed native-real sessions better or worse
than augmentation-only training, and why? Then open `docs/milestone6_report.md` and
reconcile your answer with the seed-42 numbers (invariance training raised synthetic
all-corruption balanced accuracy from 77.7% to 78.9% and embedding cosine similarity
from 0.845 to 0.916, but real balanced accuracy fell to 12.5% versus 29.0% for
clean-only and 48.4% for augmentation-only, with 0% recall on the held-out wheeled
session).

## Explain aloud

> Why can making `z_clean ≈ z_corrupted` help in-distribution accuracy yet hurt transfer to real audio?

> Why is a hard negative stronger evidence than an easy one, and what metadata does the repo use to build hard negatives?

# Week 14 — Pretrained encoders and leakage-safe session evaluation (Milestone 6, part 2)

## Outcomes

You should understand:

- external pretraining (PANNs on AudioSet) and why it is not a strict synthetic-only baseline;
- a frozen encoder with a linear probe versus fine-tuning;
- the 2,048-dimensional embedding versus the 527 AudioSet outputs as representations;
- nested leave-session-pair-out evaluation and why regularization must be selected only on inner folds;
- semantic regularization, equal-weight ensembles, and per-fold decision thresholds;
- the Milestone 7 gate and why a strong mean alone does not clear it.

## Core sources

1. [PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/abs/1912.10211).
2. Read [`docs/milestone6_pretrained_transfer_report.md`](../docs/milestone6_pretrained_transfer_report.md), [`docs/milestone6_semantic_session_report.md`](../docs/milestone6_semantic_session_report.md), and [`docs/milestone6_fusion_session_report.md`](../docs/milestone6_fusion_session_report.md).

## Repository exercise

- [ ] Read the five Milestone 7 gate criteria at the end of [`docs/milestone6_fusion_session_report.md`](../docs/milestone6_fusion_session_report.md) and state why 77.35% nested mean balanced accuracy passes gates 2 and 3 but not gates 1, 4, and 5.
- [ ] Read `nested_leave_session_pair_out_fusion` and `_fused_probabilities` in [`src/vehicle_audio/fusion_session_evaluation.py`](../src/vehicle_audio/fusion_session_evaluation.py) and describe what one outer fold excludes.
- [ ] Open `runs/m6_fusion_nested_sessions/metrics.json` and find the Ford Model T wheeled-recall collapse (0–14.71% in several contexts). State what this implies about the "wheeled" category at the session level.
- [ ] Run the self-contained session tests:

```bash
uv run pytest tests/test_semantic_session_evaluation.py tests/test_fusion_session_evaluation.py tests/test_pretrained_evaluation.py -q
```

## Explain aloud

> Why must the wheeled decision threshold be selected inside each outer fold rather than once on the whole corpus?

> Why does a 77% mean balanced accuracy not authorize Milestone 7 when one held-out session sits near 22% recall?

# Week 15 — Capstone: designing the Milestone 7 experiment

## Outcomes

You should be able to:

- state the hierarchical taxonomy and its reliability ordering
  (vehicle present → tracked/wheeled → family → model);
- define open-set evaluation: known-class accuracy, unknown detection rate,
  false-known rate, and confidence calibration;
- explain why UNKNOWN must be an allowed answer rather than a forced prediction;
- design a discriminating experiment rather than reusing the current corpus.

## Core sources

1. Re-read the Milestone 7 section and the research priority list in [`AGENTS.md`](../AGENTS.md).
2. Re-read the gate assessment in [`docs/milestone6_fusion_session_report.md`](../docs/milestone6_fusion_session_report.md).

## Capstone exercise

- [ ] Design an experiment that would clear all five Milestone 7 gate criteria. Specify the corpus changes, the split, and the metrics you would report.
- [ ] Identify which hierarchy levels the current fusion run supports and which it does not, using the session-level numbers from Week 14.
- [ ] Write a short protocol for measuring unknown detection and false-known rate using vehicle models excluded from training.
- [ ] State which conclusions each of your proposed splits would and would not support.

## Explain aloud

> What is the difference between an "unknown vehicle" and an "unseen operating state"?

> Design the smallest experiment that would distinguish "the model knows the wheeled category" from "the model memorized three wheeled recordings."

# Equation reference

## Decibels

Amplitude ratio:

$$
20\log_{10}\left(\frac{A_2}{A_1}\right)
$$

Power ratio:

$$
10\log_{10}\left(\frac{P_2}{P_1}\right)
$$

## Noise scaling for requested SNR

Given signal power $P_s$, original noise power $P_n$, and requested SNR $R$ in dB:

$$
\alpha=\sqrt{\frac{P_s}{P_n10^{R/10}}}
$$

The mixture is:

$$
y=x+\alpha n
$$

## Convolution

$$
y[n]=\sum_k x[k]h[n-k]
$$

## Cross-entropy

$$
L=-\log p(y_{\text{correct}}\mid x)
$$

## Balanced accuracy

For two classes:

$$
\operatorname{balanced\ accuracy}=\frac{1}{2}
\left(\operatorname{recall}_{tracked}+\operatorname{recall}_{wheeled}\right)
$$

# Vocabulary checklist

You should be able to define each term in one or two sentences.

## DSP and acoustics

- [ ] waveform
- [ ] sample rate
- [ ] Nyquist frequency
- [ ] aliasing
- [ ] amplitude
- [ ] power
- [ ] phase
- [ ] decibel
- [ ] SNR
- [ ] FFT
- [ ] STFT
- [ ] window function
- [ ] spectral leakage
- [ ] impulse response
- [ ] convolution
- [ ] frequency response
- [ ] reverberation
- [ ] RT60
- [ ] clipping
- [ ] compression
- [ ] uniform linear array
- [ ] broadside
- [ ] azimuth
- [ ] far field
- [ ] plane wave
- [ ] fractional delay
- [ ] time difference of arrival (TDOA)
- [ ] GCC-PHAT
- [ ] SRP-PHAT
- [ ] beamforming
- [ ] delay-and-sum
- [ ] angular error

## Audio features and ML

- [ ] log-Mel spectrogram
- [ ] MFCC
- [ ] spectral centroid
- [ ] spectral bandwidth
- [ ] spectral rolloff
- [ ] logistic regression
- [ ] CNN
- [ ] temporal pooling
- [ ] cross-entropy
- [ ] backpropagation
- [ ] optimizer
- [ ] checkpoint
- [ ] calibration
- [ ] embedding
- [ ] cosine distance
- [ ] linear probe
- [ ] frozen encoder
- [ ] external pretraining
- [ ] triplet margin
- [ ] hard positive
- [ ] hard negative
- [ ] projection head
- [ ] invariance

## Experimental design

- [ ] recording session
- [ ] source leakage
- [ ] group split
- [ ] confounder
- [ ] nuisance variable
- [ ] ablation
- [ ] domain shift
- [ ] domain randomization
- [ ] unseen state
- [ ] unseen environment
- [ ] unseen vehicle
- [ ] synthetic-to-synthetic
- [ ] real-to-real
- [ ] synthetic-to-real
- [ ] nested cross-validation
- [ ] leave-session-pair-out
- [ ] inner fold
- [ ] outer fold
- [ ] ensemble
- [ ] decision threshold
- [ ] learning curve
- [ ] open-set classification
- [ ] unknown detection
- [ ] false-known rate
- [ ] gate criteria

# Questions you should be ready to answer

- [ ] Which parts of the current data are real recordings?
- [ ] Which parts are synthetically generated or corrupted?
- [ ] At what stage is SNR controlled and measured?
- [ ] Why can later microphone effects change final measured SNR?
- [ ] Why are channels never silently averaged?
- [ ] What metadata makes a sample reproducible?
- [ ] Why is a recording-session split mandatory?
- [ ] What does balanced accuracy add beyond ordinary accuracy?
- [ ] What does performance versus SNR reveal?
- [ ] What acoustic cues might distinguish tracked and wheeled vehicles?
- [ ] What alternative confounders might explain the same result?
- [ ] What would constitute evidence of unseen-vehicle generalization?
- [ ] What would constitute evidence of synthetic-to-real transfer?
- [ ] Why is a linear array restricted to the front half-plane, and how is azimuth measured in this repo?
- [ ] Why did two-microphone feature fusion beat four-microphone beamforming for classification?
- [ ] What does the Milestone 5 negative result demonstrate about synthetic-to-real transfer?
- [ ] What is the difference between augmentation-only training and explicit consistency training?
- [ ] Why can invariance training help in-distribution yet hurt real transfer?
- [ ] Why is a PANNs-based model not a strict synthetic-only baseline?
- [ ] Why must regularization and decision thresholds be selected on inner folds only?
- [ ] What are the five Milestone 7 gate criteria, and why does 77% mean balanced accuracy not suffice?
- [ ] What is the difference between an unknown vehicle and an unseen operating state?

# Optional deeper resources

Use these after completing the core plan:

- [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html) for probability, classifiers, uncertainty, and deeper statistical treatment.
- [Stanford CS231n](https://cs231n.stanford.edu/2021/schedule.html) for optimization, CNNs, regularization, and practical training diagnostics.
- [MIT Acoustics of Speech and Hearing](https://ocw.mit.edu/courses/6-551j-acoustics-of-speech-and-hearing-fall-2004/) for more rigorous physical acoustics.
- [Pyroomacoustics documentation](https://pyroomacoustics.readthedocs.io/en/stable/) for later room and microphone-array simulation.

# Completion criterion

The plan is complete when you can give a 10-minute project explanation, answer the
questions above, derive the SNR scaling equation, inspect a split for leakage,
connect the principal DSP and classifier equations to their implementations in the
repository, and explain the Milestone 4-6 results (array localization and
classification, the sim-to-real negative result, and the invariance findings) with
their limitations. The capstone is finished when you can also design the experiment
that would clear the Milestone 7 gate.

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
4. later work in multichannel acoustics and synthetic-to-real transfer.

The default pace is **10 weeks at 5–7 hours per week**. If time is limited, complete
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
- which ABVID recordings are real and which observations or sources are synthetic.

# Prerequisite check

You can begin immediately if you are comfortable with basic Python. Review the
mathematics as it appears rather than delaying all DSP work until the math is
complete.

## Minimum mathematics

- [ ] Rearrange algebraic equations.
- [ ] Work with powers, roots, exponentials, and base-10 logarithms.
- [ ] Recognize sine and cosine waves.
- [ ] Understand vectors, matrices, means, and variance.
- [ ] Understand derivatives as local rates of change.
- [ ] Interpret probability as a number between 0 and 1.

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

[Pyroomacoustics room-simulation documentation](https://pyroomacoustics.readthedocs.io/en/stable/pyroomacoustics.room.html) shows how sources, microphones, and room impulse responses are represented in a simulator. Treat it as preparation for later milestones, not as a replacement for understanding the equations.

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

# Optional deeper resources

Use these after completing the core plan:

- [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html) for probability, classifiers, uncertainty, and deeper statistical treatment.
- [Stanford CS231n](https://cs231n.stanford.edu/2021/schedule.html) for optimization, CNNs, regularization, and practical training diagnostics.
- [MIT Acoustics of Speech and Hearing](https://ocw.mit.edu/courses/6-551j-acoustics-of-speech-and-hearing-fall-2004/) for more rigorous physical acoustics.
- [Pyroomacoustics documentation](https://pyroomacoustics.readthedocs.io/en/stable/) for later room and microphone-array simulation.

# Completion criterion

The plan is complete when you can give a 10-minute project explanation, answer the
questions above, derive the SNR scaling equation, inspect a split for leakage, and
connect the principal DSP and classifier equations to their implementations in the
repository.

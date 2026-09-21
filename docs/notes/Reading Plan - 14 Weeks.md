---
title: Reading Plan - 14 Weeks
tags:
  - abvid
  - study-plan
  - reading
status: active
created: 2026-08-23
---

# Reading Plan — 14 Weeks

One consolidated reading plan combining the **course/textbook sources**, the
**research papers**, and the **repo milestone reports**. This is the *"what to read,
in what order"* view; the [[README]] study guide is the *"what to do each week"* view.
Compressed from the guide's 15 weeks to 14 by folding calibration/reporting into the
evaluation week.

> Rule of thumb: **repo reports are primary sources; papers are citations.** Papers
> explain *why a method exists*; the milestone reports show *what happened on your data*.

## The two roles papers play

- **Method papers** (origin of a technique) — read abstract + the one technique
  section, then **verify against repo code + milestone report**. Do not read
  cover-to-cover.
- **Paradigm paper** (Domain Randomization) — read for the *experimental principle*,
  then **revisit later to note what it does NOT establish** (the M5 lesson).

## At a glance

| Wk | Focus | Primary reading | Paper(s) |
|---|---|---|---|
| 1 | Waveforms, sampling, dB, SNR | Think DSP Ch 1–4 · 6.003 L1–2 + sampling · 6.551J L1 | — |
| 2 | Fourier, STFT, spectrograms | 6.003 Fourier lectures · torchaudio tutorial · Think DSP spectrum | — |
| 3 | Filters, resampling, convolution | 6.003 L8/L9/L11 · Think DSP filters · torchaudio resampling | — |
| 4 | Practical acoustics & mics | 6.551J L1–3 · 21M.380 (2 PDFs) | — |
| 5 | Log-Mel, MFCC, features | torchaudio MFCC · Think DSP review | — |
| 6 | Classical classification + evaluation | scikit-learn: eval, GroupKFold, F1, calibration | — |
| 7 | Neural nets, CNNs, PyTorch | PyTorch basics + autograd · CS231n | — |
| 8 | Vehicle acoustics & order analysis | Ansys · Dewesoft manual · revisit STFT | — |
| 9 | Synthetic data & domain shift | Domain Randomization · pyroomacoustics · AGENTS.md | **Domain Randomization** |
| 10 | Arrays, DOA, beamforming (M4) | pyroomacoustics positions · M4 report | **Knapp & Carter · DiBiase** |
| 11 | Real-world transfer (M5) | AGENTS.md rules · M5 report | *revisit Domain Randomization* |
| 12 | Invariance learning (M6-1) | M6 report + checkpoint | **FaceNet · SimCLR** |
| 13 | Pretrained encoders + session eval (M6-2) | PANNs · M6 reports (3) | **PANNs** |
| 14 | Capstone: design M7 | AGENTS.md M7 · gate assessment | — |

## Weekly detail

### Week 1 — Waveforms, sampling, decibels, SNR
- **Reading:** [[Think DSP - Digital Signal Processing in Python]] Ch 1–4 (waveforms,
  spectrum); [[MIT 6.003 Signals and Systems]] L1–L2 + sampling (Oppenheim & Willsky Ch 7);
  [[MIT 6.551J Acoustics of Speech and Hearing]] L1 (measurement, dB).
- **Verify:** `mixing.py` + `tests/test_mixing.py` — derive the `mix_at_snr` noise scaling.
- **Papers:** —

### Week 2 — Fourier, STFT, spectrograms
- **Reading:** 6.003 Fourier lectures (L14–18) + readings (O&S Ch 3/4/5); torchaudio
  feature-extraction tutorial; Think DSP spectrum/spectrogram chapters.
- **Verify:** `scripts/create_toy_audio.py` + `scripts/inspect_sample.py` — FFT size / hop tradeoffs.
- **Papers:** —

### Week 3 — Filters, resampling, convolution, nonlinear distortion
- **Reading:** 6.003 L8 (convolution), L9/L11 (frequency response); Think DSP filter
  chapters; torchaudio resampling comparison.
- **Verify:** `augment.py` + `tests/test_augment.py` — every stage → physical interpretation.
- **Papers:** —

### Week 4 — Practical acoustics & microphones
- **Reading:** 6.551J L1–3; 21M.380 *physics of sound* + *room acoustics* PDFs
  (inverse-distance vs inverse-square, reflections).
- **Note:** this is the physics foundation for Week 10 arrays; not a code week.
- **Papers:** —

### Week 5 — Log-Mel, MFCC, classical audio features
- **Reading:** torchaudio feature tutorial + MFCC docs; Think DSP spectrum review.
- **Verify:** `baseline.py` log-Mel path; `inspect_sample.py`.
- **Papers:** —

### Week 6 — Classical classification + trustworthy evaluation (+ calibration)
- **Reading:** scikit-learn model-evaluation guide, GroupKFold, F1, calibration.
- **Verify:** classical baseline with a GroupKFold split; re-read leakage rules in `AGENTS.md`.
- **Papers:** —

### Week 7 — Neural networks, CNNs, PyTorch
- **Reading:** PyTorch Learn the Basics + autograd tutorial; CS231n (linear
  classifiers, losses, backprop, CNNs).
- **Verify:** the CNN baseline in `baseline.py`; run a training run.
- **Papers:** —

### Week 8 — Vehicle acoustics & order analysis
- **Reading:** Ansys order-analysis intro; Dewesoft order-tracking manual (RPM-dependent
  spectral lines); revisit Week 2 STFT.
- **Verify:** `source_simulation.py` harmonic-engine model — connect RPM → harmonic ridges.
- **Papers:** —

### Week 9 — Synthetic data, domain shift, generalization
- **Reading:** **Domain Randomization** (arXiv 1703.06907) — the *paradigm* paper;
  pyroomacoustics room simulation; AGENTS.md M3 + cross-milestone rules.
- **Verify:** procedural corpus generation (`procedural_vehicles.yaml`).
- **Papers:** **Domain Randomization** — read for the principle only.

### Week 10 — Microphone arrays, direction of arrival, beamforming (M4)
- **Reading:** **Knapp & Carter 1976** (GCC-PHAT origin); **DiBiase 2000** (SRP-PHAT
  origin); pyroomacoustics mic positions; `milestone4_report.md` results + limitations.
- **Verify:** `multichannel.py` `gcc_phat_azimuth` / `srp_phat_azimuth`; the angular
  error table.
- **Papers:** **Knapp & Carter**, **DiBiase** — method papers, read abstract + the
  correlation/steered-response section.

### Week 11 — Real-world transfer and the sim-to-real protocol (M5)
- **Reading:** re-read cross-milestone + leakage rules (`AGENTS.md`); `milestone5_status.md`
  in full (split, hashes, limitations); **revisit Domain Randomization** and note what it
  does **not** establish.
- **Verify:** the negative result — 10.2% real-only, 32.8% synthetic-only balanced acc;
  every model misses the held-out wheeled session.
- **Papers:** *revisit* Domain Randomization (paradigm) — the M5 counterexample.

### Week 12 — Noise-invariant representation learning (M6, part 1)
- **Reading:** **FaceNet** (arXiv 1503.03832) — triplet margin; **SimCLR** (arXiv
  2002.05709) — contrastive context; `milestone6_report.md` + `milestone6_improvement_checkpoint.md`.
- **Verify:** `invariance_evaluation.py`; predict-then-verify: invariance helps
  in-distribution (77.7→78.9%) but hurts real transfer (12.5% vs 48.4% augmentation).
- **Papers:** **FaceNet**, **SimCLR** — method papers; read the loss sections.

### Week 13 — Pretrained encoders + leakage-safe session evaluation (M6, part 2)
- **Reading:** **PANNs** (arXiv 1912.10211); `milestone6_pretrained_transfer_report.md`,
  `milestone6_semantic_session_report.md`, `milestone6_fusion_session_report.md`.
- **Verify:** `pretrained_evaluation.py`; nested leave-session-pair-out; the 77.35% fusion
  gate (passes gates 2–3, not all five).
- **Papers:** **PANNs** — method paper; read for the AudioSet-output embedding idea.

### Week 14 — Capstone: design the Milestone 7 experiment
- **Reading:** re-read M7 + research-priority list (`AGENTS.md`); gate assessment in
  `milestone6_fusion_session_report.md`.
- **Deliverable:** design the experiment that clears all five gate criteria; protocol
  for unknown detection / false-known rate using excluded vehicle models.
- **Papers:** —

## Course / textbook source map

| Source | Weeks |
|---|---|
| **Think DSP** | 1–3, 5 |
| **MIT 6.003** (Signals & Systems) | 1–3 — Fourier, sampling, convolution, filters |
| **MIT 6.551J** (Acoustics) | 1, 4 — + L4 (localization) & L10–11 (mics) for Week 10 depth |
| **21M.380** (2 PDFs: physics of sound, room acoustics) | 4 |
| **torchaudio** (feature tutorials) | 2, 5 |
| **scikit-learn** (eval, GroupKFold, F1, calibration) | 6 |
| **PyTorch** (basics + autograd) + **CS231n** | 7 |
| **Ansys / Dewesoft** (order analysis) | 8 |
| **pyroomacoustics** (room/mic simulation) | 9, 10 |

## Paper map

| Week | Paper | Role | Verify against |
|---|---|---|---|
| 9 | Domain Randomization (1703.06907) | paradigm | M5 transfer result (counterexample) |
| 10 | Knapp & Carter (1976) | method | `multichannel.py` GCC-PHAT |
| 10 | DiBiase (2000) | method | `multichannel.py` SRP-PHAT |
| 12 | FaceNet (1503.03832) | method | `invariance_evaluation.py` triplet |
| 12 | SimCLR (2002.05709) | method | same invariance objectives |
| 13 | PANNs (1912.10211) | method | `pretrained_evaluation.py` |

## Companion vault notes

- [[Think DSP - Digital Signal Processing in Python]] — condensed source note
- [[MIT 6.003 Signals and Systems]] — condensed source note
- [[MIT 6.551J Acoustics of Speech and Hearing]] — condensed source note
- [[README]] — the full condensed ABVID study guide (the "do" view)

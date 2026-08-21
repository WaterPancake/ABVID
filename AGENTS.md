# Project Roadmap

The long-term objective is to develop and evaluate robust passive acoustic vehicle detection and classification under adverse recording conditions.

The project should progress incrementally. **Do not skip milestones merely because later work appears more interesting.**

Each milestone should produce a reproducible experiment and documented results before moving forward.

---

# Milestone 1 — Synthetic Audio Augmentation Engine

## Goal

Build a deterministic offline pipeline that transforms relatively clean vehicle recordings into controlled synthetic microphone observations.

```text
clean target
     +
background
     ↓
propagation/environment effects
     ↓
microphone effects
     ↓
corrupted observation
```

## Required Features

Implement:

1. random gain;
2. background-noise mixing;
3. controlled SNR;
4. low/high/band-pass filtering;
5. random EQ;
6. resampling degradation;
7. clipping;
8. dynamic-range compression;
9. impulse-response convolution;
10. temporal cropping;
11. approximate microphone frequency response.

Support paired:

```text
clean.wav
corrupted.wav
metadata
```

All generated samples must be reproducible from metadata and random seed.

## Validation

Tests must verify:

- SNR;
- output duration;
- sample rate;
- tensor dimensions;
- deterministic generation;
- manifest correctness;
- numerical stability.

Generate a toy dataset using synthetic harmonic signals before using real recordings.

## Deliverable

A CLI similar to:

```bash
python -m vehicle_audio.cli generate \
    --config configs/default.yaml \
    --targets data/targets \
    --backgrounds data/backgrounds \
    --output data/generated \
    --num-samples 1000 \
    --seed 42
```

---

# Milestone 2 — Baseline Vehicle Classifier

## Goal

Establish whether vehicle-category information survives increasingly severe acoustic corruption.

Initial task:

```text
vehicle audio
      ↓
tracked / wheeled
```

Do not begin with exact vehicle identification.

## Models

Implement at least two baselines.

### Classical baseline

Investigate simple features such as:

- MFCC;
- spectral centroid;
- spectral rolloff;
- spectral bandwidth;
- harmonic structure.

Use a lightweight classifier such as logistic regression, SVM, or random forest.

### Neural baseline

Implement:

```text
waveform
   ↓
log-Mel spectrogram
   ↓
CNN or pretrained audio encoder
   ↓
temporal pooling
   ↓
classifier
```

Prefer a pretrained environmental-audio representation where practical.

Keep the encoder/classifier interface modular.

## Dataset Splitting

**Never randomly split windows from the same recording across train and test.**

Split by:

```text
recording_session
```

or an equivalent source identifier.

This requirement applies to all later milestones.

## Evaluation

Report:

- accuracy;
- balanced accuracy;
- precision/recall;
- F1;
- confusion matrix;
- calibration where appropriate.

Most importantly, evaluate performance versus SNR:

```text
SNR     Accuracy
30 dB   ...
20 dB   ...
10 dB   ...
 5 dB   ...
 0 dB   ...
-5 dB   ...
-10 dB  ...
```

Save experiment configuration and random seed with every result.

---

# Milestone 3 — Controlled Synthetic Vehicle Corpus

## Goal

Replace the limited collection of clean recordings with a controllable synthetic vehicle-audio source.

Possible sources include:

- game/simulation engines;
- procedural engine models;
- appropriately licensed recorded assets.

Keep this component separate from the augmentation engine.

The conceptual pipeline should remain:

```text
VEHICLE SIMULATOR
      ↓
clean-ish source
      ↓
AUGMENTATION ENGINE
      ↓
simulated microphone observation
```

## Vehicle Variables

Where supported, record:

- vehicle ID;
- vehicle class;
- tracked/wheeled;
- engine state;
- RPM;
- throttle;
- speed;
- acceleration;
- load;
- source/listener geometry.

Do not infer unavailable variables.

## Dataset Design

Begin with approximately:

```text
2+ tracked vehicles
2+ wheeled vehicles
```

and only expand once the pipeline works.

Generate multiple operating states for every vehicle.

Avoid allowing vehicle identity to become trivially correlated with:

- one environment;
- one speed;
- one recording position;
- one background;
- one recording session.

## Experiments

Evaluate generalization across:

### Unseen operating state

Train on some RPM/speed ranges and evaluate others.

### Unseen environment

Train using some corruption/environment distributions and test others.

### Unseen vehicle

Train category classification while holding out an entire vehicle model.

This experiment is particularly important:

```text
train:
tracked A
tracked B
wheeled A
wheeled B

test:
tracked C
wheeled C
```

The model should ideally recognize category-level acoustic characteristics rather than memorize individual sound assets.

---

# Milestone 4 — Multichannel Acoustic Simulation

## Goal

Investigate whether multiple synchronized microphones improve detection and classification under adverse conditions.

Do not build this before the single-channel system works.

## Simulator

Represent each microphone observation approximately as:

```text
x_i(t) =
    attenuation_i *
    convolution(source, impulse_response_i)
    + noise_i
```

Support configurable microphone geometry.

Each microphone may have different:

- propagation delay;
- attenuation;
- impulse response;
- background noise;
- frequency response.

Output tensors should use:

```python
[num_channels, num_samples]
```

## Direction of Arrival

Implement established methods before attempting learned localization.

Candidates include:

- GCC-PHAT;
- SRP-PHAT.

Validate localization using synthetic sources with known positions.

Report angular error.

## Beamforming

Implement a conventional beamforming baseline.

Compare:

```text
single microphone

vs.

multiple microphones
without beamforming

vs.

multiple microphones
with beamforming
```

Evaluate all three against increasing noise and interference.

## Deliverable

Produce plots showing:

```text
classification accuracy vs SNR

localization error vs SNR

classification accuracy vs microphone count
```

---

# Milestone 5 — Real-World Transfer

## Goal

Determine whether models trained using synthetic data learn features that transfer to real recordings.

This is one of the project's primary research questions.

## Core Experiment

Train using:

```text
synthetic data ONLY
```

and evaluate using:

```text
real recordings NEVER SEEN during training
```

Start with:

```text
tracked vs wheeled
```

before attempting vehicle-family or model identification.

## Experiments

Compare:

### A — Real only

```text
train: real
test: real
```

### B — Synthetic only

```text
train: synthetic
test: real
```

### C — Synthetic pretraining + limited real data

```text
train:
synthetic
+
1% / 5% / 10% / 25% real

test:
held-out real
```

Generate a learning curve showing how much real data is required to close the synthetic-to-real gap.

## Domain Analysis

Extract model embeddings and compare distributions for:

```text
synthetic tracked
real tracked

synthetic wheeled
real wheeled
```

Use dimensionality reduction only for visualization, not as proof of representation quality.

---

# Milestone 6 — Noise-Invariant Representation Learning

## Goal

Explicitly train representations to preserve vehicle information while discarding nuisance acoustic variation.

Use paired examples created by Milestone 1:

```text
clean event
     ↓
   encoder
     ↓
   z_clean


same event + corruption
     ↓
   encoder
     ↓
   z_corrupted
```

Train toward:

```text
z_clean ≈ z_corrupted
```

while maintaining separation between different vehicle classes.

## Contrastive Dataset

Support:

### Positive pairs

Same source event under different corruption.

### Hard positives

Same vehicle under different:

- RPM;
- speed;
- distance;
- environment;
- microphone.

### Easy negatives

Different coarse vehicle classes.

### Hard negatives

Different vehicles with similar acoustic characteristics.

## Objective

Investigate objectives of the form:

```text
classification loss
+
lambda * representation consistency loss
```

and/or an established contrastive objective.

Do not invent a novel loss unless existing methods have been properly baselined.

## Evaluation

Compare:

```text
standard supervised training

vs.

augmentation only

vs.

representation-invariance training
```

under:

- decreasing SNR;
- unseen noise;
- unseen microphone response;
- unseen environment;
- real-world recordings.

---

# Milestone 7 — Hierarchical and Open-Set Classification

## Goal

Move beyond binary tracked/wheeled classification while avoiding forced predictions for unknown vehicles.

Classification should become hierarchical:

```text
vehicle
├── tracked
│   ├── family A
│   ├── family B
│   └── unknown tracked
│
└── wheeled
    ├── family C
    ├── family D
    └── unknown wheeled
```

Possible hierarchy:

```text
vehicle present
      ↓
tracked / wheeled
      ↓
vehicle category
      ↓
vehicle family
      ↓
specific model
```

Do not proceed to finer classification unless the preceding level is reliable.

## Open-Set Evaluation

The system must be allowed to return:

```text
UNKNOWN
```

rather than forcing every observation into a known vehicle class.

Evaluate using vehicle models completely excluded from training.

Measure:

- known-class accuracy;
- unknown detection rate;
- false-known rate;
- confidence calibration.

---

# Milestone 8 — Integrated Passive Acoustic Demonstrator

## Goal

Combine the successful components into a passive sensing prototype for detection, classification, localization, and logging.

The system should perform:

```text
continuous multichannel audio
          ↓
     event detection
          ↓
      localization
          ↓
     classification
          ↓
 confidence aggregation
          ↓
        logging
```

## Temporal Aggregation

Do not classify an entire event from a single short window.

Support overlapping inference windows:

```text
t0 ───── t1
     t0.5 ───── t1.5
          t1 ───── t2
```

Aggregate predictions over time.

Output approximately:

```json
{
  "timestamp": "...",
  "vehicle_detected": 0.98,
  "bearing_deg": 74.2,
  "bearing_confidence": 0.81,
  "mobility_class": {
    "tracked": 0.91,
    "wheeled": 0.09
  },
  "vehicle_family": "unknown",
  "classification_confidence": 0.52
}
```

## Demonstration

Evaluate the complete system using controlled, non-weaponized vehicle pass-by or recorded-audio experiments.

Measure:

- detection probability;
- false-positive rate;
- classification accuracy;
- localization error;
- inference latency;
- compute utilization.

The demonstrator should remain a passive sensing and research system.

---

# Cross-Milestone Experiment Rules

Every experiment must record:

```text
git commit
configuration
random seed
dataset version
train/test split
model checkpoint
metrics
```

Prefer machine-readable experiment metadata.

Never report performance without clearly identifying the test domain.

Distinguish:

```text
synthetic → synthetic

real → real

synthetic → real

synthetic + real → real
```

These are fundamentally different results.

# Dataset Leakage Rules

Treat recordings from the same original source as belonging to the same group.

For example, if a ten-minute recording is divided into 100 clips:

```text
ALL 100 clips belong to one recording_session.
```

They must not be distributed across train and test.

Where possible, also track:

```text
source vehicle
recording location
recording date/session
original media source
simulation run
```

Use group-based splitting.

# Research Priority

Prioritize questions in this order:

1. Can vehicle presence be detected reliably?
2. Can tracked and wheeled vehicles be distinguished?
3. Does classification survive severe acoustic corruption?
4. Does the representation generalize to unseen vehicles?
5. Does multichannel audio improve robustness?
6. Does synthetic training transfer to real audio?
7. Can representation learning reduce the sim-to-real gap?
8. Can vehicle families/models be distinguished?
9. Can unknown vehicles be rejected reliably?

Do not optimize exact-model identification before demonstrating category-level generalization.

# Immediate Work

Unless explicitly instructed otherwise, Codex should currently work only on:

**Milestone 1 — Synthetic Audio Augmentation Engine.**

Later milestones are architectural guidance and should influence interfaces where doing so is inexpensive, but they are **not authorization to implement the entire roadmap at once**.

After completing Milestone 1:

1. run all tests;
2. generate a small example dataset;
3. document usage;
4. summarize implementation decisions;
5. identify technical debt relevant to Milestone 2;
6. stop and request review before proceeding.

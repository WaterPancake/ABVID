# Project Roadmap

The long-term objective is to develop and evaluate robust passive acoustic vehicle detection and classification under adverse recording conditions.

The immediate objective is **ABVID Benchmark v0.1 with an inspectable audio replay demo**:
a reproducible, portfolio-ready research artifact built from Milestones 1-6. Packaging
these existing capabilities does not require passing the Milestone 7 gate and does not
authorize hierarchical classification or the full Milestone 8 system.

The project should progress incrementally. **Do not skip milestones merely because later work appears more interesting.**

Each milestone should produce a reproducible experiment and documented results before moving forward.

---

# Current Status

Milestones 1-6 are implemented and evaluated: the augmentation engine, baseline classifiers,
the procedural synthetic corpus, multichannel simulation, real-world transfer, and
representation-invariance training. The project is at the **Milestone 7 entry gate**.

The critical data constraint is independent, provenance-clean, human-reviewed **recording
sessions per class**. As of 2026-09-20, the reviewed catalog and local target discovery
contain **7 tracked / 5 wheeled development sessions**, excluding protected evaluation
sources. The refreshed 7/5 nested development benchmark is complete: the primary
fusion scored 39.70% mean balanced accuracy, with 51.98% tracked recall and 27.42%
wheeled recall, and failed gates 2-4.

Historical real-to-real nested development balanced accuracy was 77.35%, but individual
sessions fell below the required recall floor. The frozen Sherman/PDSounds confirmation
was completed on 2026-08-24 and scored 49.69% balanced accuracy on that real session pair;
it failed confirmation and must not be repeated or treated as an unused test. The roughly
38% synthetic-to-real result is an observed result under the tested protocols, not a proven
performance ceiling.

See `docs/milestone6_fusion_session_report.md` for the historical development and completed
locked results, and `docs/low_snr_audibility_improvement.md` for later development work.
`docs/milestone6_results_rundown.md` predates the locked result and current admissions;
reconcile it against versioned artifacts before treating it as a live gate assessment.
The source catalog and collection manifest are `configs/audio_sources.yaml` and
`data/collection_manifest.jsonl`. T90M and JLTV remain reserved for a future locked pair;
JLTV still requires audiovisual segment review.

On 2026-09-20 the operator excluded startup intervals from active training/evaluation;
raw recordings and review history remain in `configs/archived_startup_intervals.yaml`.
Historical benchmark snapshots/reels are unchanged and can still contain startup.
The new uncapped non-startup corpus has 785 two-second windows (486 tracked / 299
wheeled) across the same 7/5 sessions. Its preprocessing-grid protocol is
`docs/preprocessing_grid_protocol.md`; compare preprocessing against the unchanged
control on this new version, not directly against the older capped corpus.
The completed 24-way grid scored 51.86% mean balanced accuracy for the primary
PANNs head (77.31% tracked / 26.41% wheeled recall), versus its matched unchanged
control at 53.73%. The classical comparator improved from 43.23% to 47.26%.
Both retained a zero-recall held-out case; gates 2-4 remain failed and confirmation
was not run. Results and artifacts: `docs/preprocessing_grid_results.md`.

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

## Entry Gate

Milestone 7 must not begin until all of the following pass on the same versioned
development corpus and frozen protocol, documented in an updated
`docs/milestone6_results_rundown.md` with links to the experiment artifacts:

1. at least five independent reviewed development recording sessions per class, excluding
   reserved evaluation sessions;
2. nested unseen-session mean balanced accuracy >= 75%;
3. mean recall >= 70% for both tracked and wheeled;
4. no held-out session below 50% window recall;
5. confirmation on a newly admitted locked pair, evaluated exactly once by the frozen rule.

Average performance alone (gates 2-3) is not sufficient: a model that can fail an entire
unseen recording session is not ready for hierarchical or open-set work.

Historical passes do not carry over to a changed dataset or model. Preregister the next
development protocol and confirmation acceptance criteria, pass the development gates,
then freeze the checkpoint, preprocessing, aggregation, and decision rule before running
the reserved pair exactly once. Do not inspect its predictions during development.

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

The project is at the **Milestone 7 entry gate**; Milestones 1-6 are complete. Do not begin
Milestone 7 until every gate in the Milestone 7 section passes.

## Priority

- **Make the benchmark the primary deliverable.** Release ABVID Benchmark v0.1 with three
  separate tracks: native unseen-session real audio, controlled corruption of held-out real
  audio, and simulated microphone arrays. Compare existing classical, pretrained-audio, and
  strongest development baselines before adding architectures. Version manifests, reviewed
  intervals, licenses, split roles, hashes, configurations, seeds, commits, and checkpoints;
  report per-class, per-session, worst-session results, and session-level uncertainty. Hold
  out original background recordings when claiming unseen-noise performance. Reconcile
  historical dataset/model roles and stale reports first; later adaptation used
  Sherman/PDSounds as development sources, so current exclusion flags cannot retroactively
  make those checkpoints independent of them. Preserve the original locked result unchanged.
- **Build an inspectable replay demo.** Let users select reviewed development/demo audio,
  listen to original and corrupted versions, adjust SNR/background/microphone response, and
  inspect waveform, spectrogram, window predictions, event aggregation, provenance, model
  version, annotations, and processing time. Include success and failure cases. A simple 2D
  interface and an optional explicitly simulated array/localization view are sufficient;
  defer game-engine integration. Never use reserved recordings for interactive exploration.
  Label the initial demo as offline replay: the current full-recording peak-relative
  audibility gate is neither a causal streaming implementation nor a vehicle-presence detector.
- **Close the independent-session data gate and control confounds.** Finish review of staged
  sources before seeking replacements. With 7 tracked / 5 wheeled development sessions,
  the minimum session-count gate is met without consuming the reserved T90M/JLTV pair;
  finish JLTV review for confirmation only. Continue favoring heavy wheeled
  vehicles and vary location, device, distance, and operating state so class does not simply
  encode military versus civilian recording context. Aim beyond the five-session minimum
  toward roughly ten development sessions per class plus several untouched evaluation
  sessions per class; this is a collection target, not a reliability guarantee. Rebuild the
  reviewed development corpus and improve the failed nested gate assessment before frozen
  confirmation. Collect vehicle-absent scenes before claiming presence detection or false-alarm
  performance; those capabilities are not prerequisites for packaging the replay benchmark.
- **Package a portfolio-ready release.** Provide one-command demo startup, a small
  redistribution-cleared example dataset, reproducible benchmark commands and report,
  documented model/data limitations, an architecture explanation, and a short demonstration
  video. Verify installation and the sample workflow, retain regression tests, and show
  measured runtime on the demonstrated hardware. Keep native-real, controlled-corruption,
  and simulated-array claims distinct; do not present the release as field-validated vehicle
  detection, real-array localization, or completed Milestone 7/8 functionality.

## Guardrails

- Later milestones remain architectural guidance, not authorization to implement the entire
  roadmap at once.
- Keep session-grouped splits and the fail-closed provenance audit on every admission.
- Always identify the test domain (synthetic-to-synthetic / real-to-real / synthetic-to-real /
  synthetic + real-to-real) before reporting performance.

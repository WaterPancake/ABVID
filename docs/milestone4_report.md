# Milestone 4 — Multichannel Acoustic Simulation

Date: 2026-08-19  
Status: implemented and evaluated  
Test domain: procedural synthetic vehicle targets plus recorded backgrounds through a
simulated microphone array; this is **synthetic-to-synthetic**, not real-world transfer

## What was implemented

The Milestone 4 pipeline is separate from the Milestone 1 single-microphone
augmentation engine. It provides:

- deterministic near-field propagation to configurable three-dimensional microphone
  positions;
- per-microphone propagation delay and inverse-distance attenuation;
- independent microphone impulse responses and approximate frequency responses;
- independent crops from one selected background recording plus small independent
  sensor noise;
- exact global controlled-SNR mixing;
- synchronized output tensors shaped `[num_channels, num_samples]`;
- GCC-PHAT and SRP-PHAT direction-of-arrival estimates;
- conventional delay-and-sum beamforming steered by each estimator;
- leakage-safe classifier evaluation grouped by `recording_session`;
- classification and localization plots versus SNR, and classification accuracy
  versus microphone count.

The relevant implementation is in `src/vehicle_audio/multichannel.py`,
`src/vehicle_audio/array_dataset.py`, and
`src/vehicle_audio/multichannel_evaluation.py`. The checked experiment configuration
is `configs/multichannel.yaml`.

## Reproducible experiment

### Dataset generation

```bash
uv run python -m vehicle_audio.cli generate-array \
  --config configs/multichannel.yaml \
  --source-manifest data/generated/m3_controlled_seed42/manifest.jsonl \
  --backgrounds data/backgrounds \
  --output data/generated/m4_array_paired_sessions_seed42 \
  --num-samples 896 --seed 42
```

Dataset version (SHA-256 of `array_manifest.jsonl`):

```text
cf467dc467ab57637121947e14096414fb7a4893f857b723b91e4246c35ba2c1
```

The corpus contains 128 base events. Every base event has the same source crop,
background crops, geometry, microphone responses, and impulse responses at 30, 20,
10, 5, 0, -5, and -10 dB. Only the controlled noise scale changes. This prevents the
SNR sweep from being confounded by different source events.

| Property | Count |
|---|---:|
| Observations | 896 |
| Base events | 128 |
| Observations per SNR | 128 |
| Tracked / wheeled | 448 / 448 |
| Idle / accelerating / steady / decelerating | 224 each |
| Distinct recording sessions represented | 64 |
| Channels per observation | 4 |
| Samples per channel | 32,000 at 16 kHz |

Each of the 64 procedural source sessions occurs in two base events and therefore 14
observations. Background use was 224 animals/insects, 434 rain, and 238 road-traffic
observations. These background counts are recorded for transparency; they are not
balanced experimental factors.

### Evaluation

```bash
uv run python scripts/evaluate_multichannel.py \
  --manifest data/generated/m4_array_paired_sessions_seed42/array_manifest.jsonl \
  --output runs/m4_multichannel_paired_sessions_seed42 \
  --seed 42 --epochs 30 --batch-size 64 --learning-rate 0.01 \
  --group-field recording_session \
  --test-domain 'procedural synthetic -> simulated multichannel synthetic (paired SNR, session-balanced)'
```

The grouped split contains 40/12/12 complete source recording sessions and
560/168/168 observations for train/validation/test. Group overlap is empty. Every
test SNR contains 24 observations. All classifier comparisons use the same classical
features, training-only normalization, linear classifier, seed, and split.

## Results

### Aggregate classification

| Representation | Accuracy | Balanced accuracy | Macro F1 | ECE |
|---|---:|---:|---:|---:|
| Single microphone | 92.26% | 92.26% | 92.22% | 0.0384 |
| 2-mic feature fusion | **98.21%** | **98.21%** | **98.21%** | 0.0507 |
| 4-mic feature fusion | 97.62% | 97.62% | 97.62% | 0.0538 |
| 2-mic GCC-PHAT beamforming | 97.02% | 97.02% | 97.02% | 0.0547 |
| 4-mic GCC-PHAT beamforming | 96.43% | 96.43% | 96.43% | 0.0489 |
| 2-mic SRP-PHAT beamforming | 97.02% | 97.02% | 97.02% | 0.0552 |
| 4-mic SRP-PHAT beamforming | 97.02% | 97.02% | 97.02% | 0.0470 |

The single-microphone confusion matrix was `[[84, 0], [13, 71]]`; the best aggregate
result, two-microphone feature fusion, was `[[82, 2], [1, 83]]`. Matrix rows are true
`tracked`, `wheeled` labels and columns are predicted labels.

### Classification accuracy versus SNR

| Representation | 30 | 20 | 10 | 5 | 0 | -5 | -10 dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single microphone | 95.8% | 95.8% | 91.7% | 91.7% | 91.7% | 91.7% | 87.5% |
| 2-mic feature fusion | 100% | 100% | 100% | 100% | 100% | 100% | 87.5% |
| 4-mic feature fusion | 100% | 100% | 100% | 100% | 100% | 95.8% | 87.5% |
| 4-mic GCC beamforming | 100% | 100% | 100% | 100% | 95.8% | 95.8% | 83.3% |
| 4-mic SRP beamforming | 100% | 100% | 100% | 100% | 100% | 91.7% | 87.5% |

### Localization

| Estimator | Mean error | Median error | 90th percentile | Within 10 degrees |
|---|---:|---:|---:|---:|
| 2-mic GCC-PHAT | 22.22° | 9.70° | 52.42° | 50.60% |
| 4-mic GCC-PHAT | **9.00°** | 2.88° | **21.99°** | 64.29% |
| 2-mic SRP-PHAT | 21.45° | 10.00° | 53.00° | 56.55% |
| 4-mic SRP-PHAT | 11.18° | **2.00°** | 45.00° | **70.24%** |

Four-microphone GCC mean error rose from 7.50° at 30 dB to 13.44° at -10 dB.
Four-microphone SRP mean error rose from 8.54° to 18.17° over the same sweep.

## Interpretation

On this checked synthetic test domain, multiple microphones materially improved
category classification over the one-microphone baseline. Simple two-microphone
feature fusion was the highest-scoring representation. Estimated-direction
beamforming also improved aggregate classification, but it did not beat feature
fusion and more microphones were not monotonically better for classification.

Four microphones substantially reduced localization error compared with two. GCC
had the lower mean and tail error; SRP had the lower median error and more estimates
within ten degrees. These are useful baselines, not evidence that either estimator is
universally preferable.

## Limitations and technical debt

- Vehicle targets remain procedural synthetic audio. The result may exploit
  simulator-specific class signatures and says nothing yet about synthetic-to-real
  transfer.
- The held-out test has 168 observations and only 24 per SNR. Accuracy differences of
  one observation are 4.17 percentage points within an SNR slice.
- The uniform linear array cannot resolve front/back ambiguity. Only the configured
  front half-plane is evaluated, with no elevation estimation.
- Background channels use independent crops from the same recording. This is a
  controlled adverse-noise baseline, not a physically complete spatial diffuse-noise
  or interferer simulation.
- Reflections are independent sparse per-microphone impulse responses rather than
  geometry-consistent room acoustics.
- Beamforming uses a conventional far-field steering model; the simulator uses exact
  point-to-microphone path lengths.
- Classification comparisons use handcrafted features and a linear model. A neural
  multichannel model is intentionally deferred until these controlled baselines are
  reviewed.

## Artifacts

The checked run directory `runs/m4_multichannel_paired_sessions_seed42` contains:

- `experiment.json`, `metrics.json`, and `splits.json`;
- `models.pt` and the versioned `array_features.pt` cache;
- `classification_by_snr.csv` and `localization_by_snr.csv`;
- `classification_accuracy_vs_snr.png`;
- `localization_error_vs_snr.png`;
- `classification_accuracy_vs_microphone_count.png`.

The workspace is not a Git checkout, so experiment metadata correctly records
`git_commit: unavailable_not_git_checkout`.

## Milestone 5 preparation boundary

The OpenClaw curator generated a proposal-only priority review at
`openclaw/vehicle-audio-curator/proposals/2026-08-19-milestone5-priority-review.yaml`.
It performed no downloads. Two tracked candidates are ready for explicit operator
approval, one tracked candidate needs further rights/content review, and the current
wheeled proposals are traffic backgrounds rather than isolated wheeled target
sessions. A credible synthetic-to-real experiment therefore remains blocked on a
balanced set of independent, reviewed real tracked and wheeled target recordings.

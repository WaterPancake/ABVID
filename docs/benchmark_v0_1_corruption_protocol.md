# Controlled-corruption 7/5 protocol v1

Specified before evaluating corrupted inputs, 2026-09-20.

Domain: **real-trained probes -> held-out real recordings with controlled added
real background audio**, plus a separate simulated microphone-response stress test.
This is not procedural synthetic-to-real transfer, presence detection, a locked
confirmation, or a new Milestone 7 gate attempt.

## Frozen inputs and decisions

- Use the exact 593 native windows, 12 original sessions, and 35 outer session-pair
  folds of `native_real_7t5w`; all variants of a source stay in that source's role.
- Reuse saved classical and semantic regularization ensembles and the original
  inner-selected fusion threshold. No fit, calibration, threshold search, gating,
  selection, or adaptation on corrupted inputs. Verify all native recalls first.
- Evaluate every window against each of four recorded backgrounds at 30, 20, 10,
  5, 0, -5, -10 dB. Select background channel 0, resample to 16 kHz. One seed (42);
  choose deterministic crop offsets per original window/background, shared across
  SNR levels. Do not imply robustness across independently resampled repetitions.
- Added-background SNR is the ratio of **the whole native window** (vehicle plus
  existing ambient sound) to the added background. It is not measured engine-only
  SNR. Save requested/measured ratios, offset, source hashes and waveform digest.
- If mixing exceeds 0.99 peak, apply a common attenuation to signal and noise to
  avoid clipping without changing SNR. Record that gain. There is no loudness
  normalization otherwise. This controlled test therefore includes gain effects.
- Analyze road traffic separately as competing-vehicle interference: at negative
  SNR the added traffic can dominate the intended target. Never interpret its
  labels as the class of every audible vehicle.
- Independently apply two specified frequency responses to the native inputs,
  without added noise; no microphone-by-noise factorial in v1. These curves are
  approximations, not measurements of named hardware.

## Background holdout and provenance

All four entire original background recordings are evaluation-only relative to the
current native-trained ABVID probes, whose train/validation inputs contain no added
background assets. Verify distinct original raw hashes and absence from target
source hashes. Do not randomly assign crops of one background to train and test.
These assets were used in historical experiments and are not newly sealed noise
tests. The external AudioSet encoder's exposure cannot be excluded; no claim of
background independence from external pretraining is made. Licenses, source pages,
sidecars and normalized hashes are snapshotted before evaluation.

## Reporting and reproducibility

Report each background/SNR and microphone condition separately for all three models:
mean pair-balanced accuracy, each class recall, each session's mean and worst
pairing-context recall, and the minimum session-context recall. Environmental-noise
curves equally average the three background sessions, not their duration; traffic
is separate. Include native reference, paired changes, confusion matrices, macro
precision/recall/F1, and descriptive session-bootstrap intervals. Do not treat the
35 overlapping folds or thousands of corrupted windows as independent recordings.
Intervals resample 7 tracked / 5 wheeled session means and omit refit/new-background
uncertainty. Scores are not calibrated operational probabilities.

Persist configuration, seed, code/commit, split/checkpoint hashes, input checksums,
per-window prediction tensors, transformation manifests, and runtime/environment.
Local generated audio/features are caches, not a redistribution release. Refuse
changed inputs on resume. Keep native artifacts and all protected recordings untouched.

Configuration: [`benchmark_v0_1_corruption_7t5w.yaml`](../configs/benchmark_v0_1_corruption_7t5w.yaml).

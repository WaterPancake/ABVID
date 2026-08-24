# Low-SNR adaptation and audibility-aware event inference

## Status

Complete on 2026-08-24 for development use. This work implements the first
four agreed improvements:

1. reject events with insufficient audible audio before classification;
2. aggregate overlapping windows across an event;
3. adapt the classifier using more low-SNR examples from real vehicle sources.
4. add and evaluate a modern tracked-vehicle recording.

The fifth improvement, separate vehicle-presence and mobility-classification
evaluation, remains pinned until labeled vehicle-absent audio is available.

## Data correction and development corpus

`data/real_eval_v2` was stale: it had been prepared before later source reviews
rejected the Romanian TR-85 recording and left the MPF recording pending. It
must not be used. The replacement `data/real_development_v3` contains only
currently admitted, human-reviewed sources:

| Class | Windows | Source sessions |
|---|---:|---:|
| tracked | 137 | 3 |
| wheeled | 124 | 4 |
| total | 261 | 7 |

All windows from one original recording retain one `recording_session`. The
manifest SHA-256 is
`c01cd498831267f745adc7a00b3258b088db55d93794c6cafd04cd8ff8adafcb`.

Rebuild it with:

```bash
.venv/bin/python -m vehicle_audio.cli prepare-real \
  --config configs/real_corpus.yaml \
  --targets data/targets \
  --output data/real_development_v3
```

## Controlled low-SNR training set

The deterministic Milestone 1 pipeline generated 1,400 two-second examples
from the seven admitted real source sessions. This is real recorded vehicle
audio subjected to synthetic corruption; it is not native low-SNR field audio.

| SNR | Samples |
|---:|---:|
| 10 dB | 278 |
| 5 dB | 269 |
| 0 dB | 289 |
| -5 dB | 283 |
| -10 dB | 281 |

The set has 703 tracked and 697 wheeled examples. The maximum absolute error
between requested and measured mix SNR is `5.178e-07 dB`. Configuration is
`configs/real_low_snr.yaml`, seed is 42, and the manifest SHA-256 is
`2ac4d01dd05ba26d162a8d04140bba38017014f6db2efcb1d812b8bc3352ed74`.

```bash
.venv/bin/python -m vehicle_audio.cli generate \
  --config configs/real_low_snr.yaml \
  --targets data/targets \
  --backgrounds data/backgrounds \
  --output data/generated/real_low_snr_v1_seed42 \
  --num-samples 1400 \
  --seed 42
```

## Source-held-out low-SNR adaptation

The experiment starts from the frozen Milestone 6 PANNs AudioSet paired linear
probe. The PANNs Cnn14 encoder remains frozen; only the tracked/wheeled linear
head is adapted. Twelve outer folds hold out one tracked and one wheeled source
session at a time. No windows from either held-out source appear in that fold's
training data. Each record's out-of-fold probabilities are averaged across all
folds in which its source was held out.

The 50 epochs, learning rate `1e-4`, and weight decay `1e-4` were fixed before
outer-fold scoring. Equal class/session/window weighting prevents a long source
from dominating. Configuration and seed are in
`configs/real_low_snr_adaptation.yaml`.

Test domain: **controlled low-SNR augmentation of held-out real recording
sessions**.

| Method | Accuracy | Balanced accuracy | Macro F1 | Tracked recall | Wheeled recall | Sessions correct |
|---|---:|---:|---:|---:|---:|---:|
| frozen probe | 48.79% | 48.80% | 48.73% | 45.23% | 52.37% | 3/7 |
| source-held-out adaptation | 65.50% | 65.48% | 65.45% | 68.99% | 61.98% | 6/7 |

Adapted balanced accuracy by SNR was 57.25% at -10 dB, 69.36% at -5 dB,
65.39% at 0 dB, 68.40% at 5 dB, and 66.50% at 10 dB. These values are not a
causal SNR curve because each SNR bucket contains different randomly sampled
events and backgrounds. A paired factorial sweep is required to attribute a
change specifically to SNR.

Sherman was the remaining failed source when every window was included. Its
mean out-of-fold prediction was 33.11% tracked / 66.89% wheeled. This is
consistent with the earlier diagnosis that the long, faint approach and fade
dominate its window count.

```bash
.venv/bin/python scripts/evaluate_low_snr_adaptation.py \
  --config configs/real_low_snr_adaptation.yaml \
  --manifest data/generated/real_low_snr_v1_seed42/manifest.jsonl \
  --base-probe-bundle runs/m6_panns_audioset_transfer/seed_42/probe_models.pt \
  --panns-checkpoint .artifacts/models/panns/Cnn14_mAP=0.431.pth \
  --output runs/real_low_snr_adaptation_v1_seed42
```

The final all-development-session state is saved as
`final_all_development_sessions`, but it has no independent score and is only a
candidate for a fresh recording.

## Audibility gate and temporal aggregation

`vehicle-audio infer-event` now:

- creates two-second windows at a one-second hop;
- scores every window with the frozen encoder and adapted probe;
- retains windows whose RMS is within a calibrated level of the event peak;
- abstains if fewer than three windows remain; and
- averages retained tracked/wheeled probabilities over time.

The threshold candidates were declared in
`configs/audibility_gate_calibration.yaml`. Calibration used only source-held-out
predictions. The selection rule first maximized session-balanced accuracy, then
active-window balanced accuracy, then coverage. It selected `-15 dB` relative
to the event's peak-RMS window.

| Calibration quantity | Result |
|---|---:|
| active-window coverage | 77.01% (201/261) |
| active-window balanced accuracy | 73.59% |
| tracked recall | 77.27% |
| wheeled recall | 69.91% |
| source-held-out session balanced accuracy | 87.50% |
| sessions correct | 6/7 |
| abstained sessions | 0/7 |

At the event level, the gate retained 15/64 Sherman windows and changed the
source-held-out event decision to tracked with mean probability 89.42%. The one
remaining failed development session was the Ford Model T recording, which was
confidently classified as tracked. All candidate thresholds had the same 87.5%
session-balanced accuracy; `-15 dB` won on active-window balanced accuracy.

```bash
.venv/bin/python scripts/calibrate_audibility_gate.py \
  --config configs/audibility_gate_calibration.yaml \
  --native-manifest data/real_development_v3/real_manifest.jsonl \
  --adaptation-metrics runs/real_low_snr_adaptation_v1_seed42/metrics.json \
  --adaptation-models runs/real_low_snr_adaptation_v1_seed42/models.pt \
  --panns-checkpoint .artifacts/models/panns/Cnn14_mAP=0.431.pth \
  --output runs/audibility_gate_calibration_v1
```

Example event inference:

```bash
.venv/bin/python -m vehicle_audio.cli infer-event \
  --audio path/to/event.wav \
  --probe-bundle runs/real_low_snr_adaptation_v1_seed42/models.pt \
  --state-key final_all_development_sessions \
  --panns-checkpoint .artifacts/models/panns/Cnn14_mAP=0.431.pth \
  --relative-rms-threshold-db -15 \
  --minimum-active-windows 3 \
  --output runs/event_result.json
```

The full Sherman and PDSounds files were used only as integration smoke tests.
The gate retained 15/101 Sherman windows and returned tracked at 99.12%; it
retained 23/43 PDSounds windows and returned wheeled at 96.35%. Both sources
trained the final head, so these numbers are not evaluation evidence.

## What the audibility gate is not

Relative energy cannot distinguish a vehicle from speech, gunfire, music,
construction, or another loud non-vehicle event. The result metadata therefore
records `is_vehicle_presence_detector: false` and
`engineering_audibility_gate_not_vehicle_detection`. A true first-stage
detector and separate detector/classifier metrics require labeled
vehicle-absent recordings. Adding metric plumbing without those negatives would
produce a misleading evaluation, so that work remains pinned.

## Modern MPF source and refreshed experiment

Human auditory review admitted the public-domain 2023 Mobile Protected
Firepower testbed recording as a fourth independent tracked session. The labels
preserve the review literally and do not invent speed, RPM, acceleration, or
deceleration values:

| Interval | Label | Review observation |
|---|---|---|
| 0:00-0:32 | idle | idle vehicle sound |
| 0:32-0:50 | mixed | moving; exact dynamic state unavailable |
| 0:51-1:10 | unknown | mostly engine sound; motion state unavailable |
| 1:11-1:30 | mixed | squeaky sound consistent with tracks |
| 1:30-end | excluded | silence |

The canonical catalog and local sidecar now mark these as human-reviewed
segments and admit the source. The one-second gaps at 0:50-0:51 and 1:10-1:11
are not silently assigned to either adjacent condition.

### Pre-admission frozen result

Before MPF entered training or gate calibration, the previously frozen v1
all-development-session probe and its fixed `-15 dB` gate were applied once to
the complete MPF file. This is a native-real, unseen-modern-source check—not a
synthetic corruption result. It retained 91/97 windows and correctly returned
tracked with probability 59.01%.

Fully contained overlapping windows show that the result is condition
dependent:

| Reviewed interval | Windows | Mean tracked probability | Tracked argmax windows |
|---|---:|---:|---:|
| idle | 31 | 71.73% | 26/31 |
| moving, exact state unknown | 17 | 18.21% | 3/17 |
| engine sound, motion unknown | 18 | 50.08% | 9/18 |
| squeaky tracks | 18 | 96.15% | 18/18 |

The model recognizes the overt tracked-motion cue much more reliably than the
generic moving/engine intervals. These overlapping windows are correlated and
come from one recording, so they diagnose this source rather than estimate a
population operating-state accuracy.

### Versioned corpora

`data/real_development_v4` now has 325 native windows: 201 tracked and 124
wheeled, with four independent recording sessions per class. Sixty-four windows
come from MPF: 24 idle, 27 mixed, and 13 unknown. The audit reports complete
provenance/content review and no blockers. Manifest SHA-256:
`3789bed5d3fbee511c66eba3e51347891b628a02dc9707f5cbb43e0dbb59cbd4`.

The refreshed low-SNR corpus contains 1,600 paired examples: 804 tracked and
796 wheeled. A new `class_session_condition_balanced` strategy samples class,
then recording session, then reviewed condition; sessions receive 180-222
examples rather than MPF being over-sampled merely because it has four segment
annotations. One exact-silence background crop was deterministically rejected
and replaced on the second attempt. The maximum absolute SNR error is
`5.057e-07 dB`. Manifest SHA-256:
`3fa4777c404133d9d01e6e0bef0c4a78d7f1801a0b9bb749e1ee00a9b76e0e3d`.

```bash
.venv/bin/python -m vehicle_audio.cli prepare-real \
  --config configs/real_corpus.yaml \
  --targets data/targets \
  --output data/real_development_v4

.venv/bin/python -m vehicle_audio.cli generate \
  --config configs/real_low_snr_session_balanced.yaml \
  --targets data/targets \
  --backgrounds data/backgrounds \
  --output data/generated/real_low_snr_v2_seed42 \
  --num-samples 1600 \
  --seed 42
```

### Eight-session source-held-out adaptation

The unchanged 50-epoch adaptation protocol now has 16 outer folds: every
Cartesian pair of one held-out tracked and one held-out wheeled source. MPF is
never present in training for any fold used to score MPF.

Test domain: **controlled low-SNR augmentation of held-out real recording
sessions**.

| Method | Accuracy | Balanced accuracy | Macro F1 | Tracked recall | Wheeled recall | Sessions correct |
|---|---:|---:|---:|---:|---:|---:|
| frozen probe | 46.00% | 46.04% | 45.71% | 38.56% | 53.52% | 3/8 |
| source-held-out adaptation | 63.94% | 63.91% | 63.80% | 69.78% | 58.04% | 6/8 |

MPF is correct while held out, with mean tracked probability 66.12%. Sherman
remains misclassified at 40.29% tracked, and the Model T remains misclassified
at 47.24% wheeled. Adapted balanced accuracy by SNR is 59.28% at -10 dB,
62.09% at -5 dB, 68.48% at 0 dB, 64.61% at 5 dB, and 65.05% at 10 dB.

The v2 adapted balanced accuracy is 1.58 percentage points below v1's 65.48%,
but this is not an apples-to-apples regression: v2 adds an unseen vehicle and
session, changes the sampling distribution to session-balanced, and evaluates
eight rather than seven sessions. The useful controlled conclusion is that v2
still improves its own frozen baseline by 17.87 points and raises tracked
recall by 31.22 points.

```bash
.venv/bin/python scripts/evaluate_low_snr_adaptation.py \
  --config configs/real_low_snr_adaptation.yaml \
  --manifest data/generated/real_low_snr_v2_seed42/manifest.jsonl \
  --base-probe-bundle runs/m6_panns_audioset_transfer/seed_42/probe_models.pt \
  --panns-checkpoint .artifacts/models/panns/Cnn14_mAP=0.431.pth \
  --output runs/real_low_snr_adaptation_v2_seed42
```

### Refreshed audibility calibration

The predeclared threshold search selected `-18 dB` on source-held-out native
development predictions. It retains 284/325 windows (87.38%), with 71.83%
active-window balanced accuracy, 74.23% tracked recall, and 69.42% wheeled
recall. Event aggregation gets 7/8 sessions correct (87.5% session-balanced
accuracy) with no abstentions. MPF is correct at 60.85% tracked; Sherman becomes
correct at 78.32% tracked; Model T remains the sole failed session.

```bash
.venv/bin/python scripts/calibrate_audibility_gate.py \
  --config configs/audibility_gate_calibration.yaml \
  --native-manifest data/real_development_v4/real_manifest.jsonl \
  --adaptation-metrics runs/real_low_snr_adaptation_v2_seed42/metrics.json \
  --adaptation-models runs/real_low_snr_adaptation_v2_seed42/models.pt \
  --panns-checkpoint .artifacts/models/panns/Cnn14_mAP=0.431.pth \
  --output runs/audibility_gate_calibration_v2
```

This is still development calibration, not a locked external result. The final
v2 all-session model contains MPF training data and therefore has no independent
MPF score.

## Artifacts and reproducibility

Implementation commits:

- `73dc31038512d2f1d4219f08c877b5e3965fb2c5`: event inference and low-SNR adaptation;
- `88ff0955a6e957ed7fc7ff9c3eb62cdbde434c0b`: source-held-out gate calibration;
- `38857ff8b389fe5f4c908a7233bb15c4b4a65cdf`: MPF admission,
  session-balanced sampling, and deterministic silent-background retry.

Important artifact SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| v1 adaptation metrics | `3c82543f7755563ce2a334e9ed455ab13ca4c1919c5133e89f5f988263917610` |
| v1 adaptation models | `584bde5e3ad35d70d7acadbaa8b6dc756c04f82cc02cdc2307ed44906ba5c550` |
| v1 gate calibration | `6902d1185763ce1157799c26a73a102e6ee523f7ffb0866a14953ccb66b90de5` |
| pre-admission MPF inference | `5a38c55df97dd762aa9c9bdbe6e5990a21b2a4a54e8d7191d3e512fbf848741a` |
| v2 adaptation metrics | `71504495d9e655d55118ea50faca20f930268a420b8624d2365d0643b4f88555` |
| v2 adaptation models | `0bb8ff91d43cd52382b6e2ea589aec560b36e449b1969e655e2f8b12103d5c61` |
| v2 adaptation splits | `841206ee05741d751fafe962f9b90f249be7348cdf4a830de18f0bbeb6cd5590` |
| v2 adaptation predictions | `6c934fd8f6f2ff8785259bdcac7177887c561be461767f8e6cd0336d52b6312e` |
| v2 gate calibration | `2ee669e6b8842b1da04c6f563885db8cad700543ef7008f3af9260c3960c369b` |
| v2 selected gate config | `4a34dd908b9a08060ede9c91efd4250c41d8c2c8c6c9b7ad150a88d7247aa3ab` |

Limitations remain substantial: there are only eight development source
sessions; MPF is the only modern tracked session; the low-SNR observations are
synthetic corruptions; threshold calibration and adaptation use development
sources; and vehicle, era, site, and recording method are not controlled.

Validation: all 90 repository tests pass. Full-file inference writes detailed
per-window JSON while printing only the compact event summary.

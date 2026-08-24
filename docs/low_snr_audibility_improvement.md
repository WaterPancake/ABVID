# Low-SNR adaptation and audibility-aware event inference

## Status

Complete on 2026-08-24 for development use. This work implements the first
three agreed improvements:

1. reject events with insufficient audible audio before classification;
2. aggregate overlapping windows across an event;
3. adapt the classifier using more low-SNR examples from real vehicle sources.

The fourth improvement, adding modern tracked recordings, is at the mandatory
human content-review gate. The fifth, separate vehicle-presence and mobility
classification evaluation, remains pinned until labeled vehicle-absent audio is
available.

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

## Modern tracked acquisition gate

The Mobile Protected Firepower testbed recording is already downloaded at
`data/targets/tracked/us_army_armor_cavalry_collection_tankodrome_2023/mpf_testbed.wav`.
It is modern, public domain, and provisionally segmented from 0 to 90 seconds,
but it is deliberately excluded from the corpus pending human auditory review.
The reviewer must identify vehicle-dominant intervals, speech/music/gunfire or
other confounds, and recognizable operating states. No new source was admitted
or downloaded by bypassing the curator's source-specific approval gate.

## Artifacts and reproducibility

Implementation commits:

- `73dc31038512d2f1d4219f08c877b5e3965fb2c5`: event inference and low-SNR adaptation;
- `88ff0955a6e957ed7fc7ff9c3eb62cdbde434c0b`: source-held-out gate calibration.

Important artifact SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| adaptation metrics | `3c82543f7755563ce2a334e9ed455ab13ca4c1919c5133e89f5f988263917610` |
| adaptation models | `584bde5e3ad35d70d7acadbaa8b6dc756c04f82cc02cdc2307ed44906ba5c550` |
| adaptation splits | `432582ac62daedc66ccd8070822185133dcbd1b76556bb1d45179146202d4b84` |
| adaptation predictions | `9a4eb377819656e5a5969acf714b10270ab8234d62a1deaf818211f450952575` |
| gate calibration | `6902d1185763ce1157799c26a73a102e6ee523f7ffb0866a14953ccb66b90de5` |
| selected gate config | `17d368f7a902ccdc1820b595837ce785adc7724ec1cccc4575b4e733a4326653` |

Limitations remain substantial: there are only seven development source
sessions; the low-SNR observations are synthetic corruptions; threshold
calibration and adaptation use development sources; source vehicle, era, site,
and recording method are not controlled; and no fresh modern tracked source has
yet been admitted for an independent test.

Validation: all 87 repository tests pass. Both full-file inference smoke tests
wrote per-window JSON artifacts while printing only the compact event summary.

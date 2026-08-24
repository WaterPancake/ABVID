# Historical checkpoint audit: Sherman / PDSounds v1

## Status

Complete on 2026-08-24.

This is a **post-hoc diagnostic**, not a second independent locked test. The
Sherman/PDSounds pair had already been opened by the frozen Milestone 6 fusion
evaluation. These results may be used to diagnose the existing systems, but
must not be used to tune a model or claim a new locked-test result.

## Protocol

- Audit code/config commit: `7914f3cb879257191740c0897a6436abd73a2156`
- Configuration: `configs/historical_checkpoint_audit.yaml`
- Configuration SHA-256:
  `c22756a0c1890ca109ff9c79099189c81464799ed0563de1286423ca48013f71`
- Dataset version / manifest SHA-256:
  `23229121344ff45c549a9641362f34f99a936d52cf193341a1b8ebcc85b0e5f6`
- Test domain: historical frozen checkpoint -> reviewed native real pair
- Input: channel 0, 16 kHz, two-second windows
- Support: 87 windows from two source-grouped sessions
  - tracked: 64 overlapping windows from one M4 Sherman pass-by
  - wheeled: 23 overlapping windows from one passenger-car start/drive
- Decision rule: each checkpoint's native two-class argmax rule. The already
  executed frozen fusion result retains its preselected 0.25 wheeled threshold.
- Training, fine-tuning, threshold selection, and checkpoint selection on this
  corpus: none

The roster was committed before scores were generated. It contains every saved
method in the selected final Milestone 2-6 checkpoint bundles. Milestone 1 has
no classifier. Only Milestone 4's one-microphone baseline is applicable;
localization, beamforming, and microphone-count comparisons require synchronized
multichannel audio and known geometry.

## Main result

The strongest retrospective result was the Milestone 6 frozen PANNs AudioSet
paired linear probe:

- accuracy: 68.97%
- balanced accuracy: 76.12%
- macro F1: 67.58%
- tracked recall: 60.94% (39/64)
- wheeled recall: 91.30% (21/23)
- confusion matrix: `[[39, 25], [2, 21]]`
- temporally averaged session decisions: 2/2 correct

It classified the one startup window, both idle windows, and 18/20 wheeled
mixed/drive windows correctly. It classified 39/64 tracked pass-by windows
correctly. These condition counts are descriptive only: startup and idle have
far too little support for condition-level claims.

The PANNs paired probe improved balanced accuracy by 5.30 percentage points over
the corresponding corruption-only PANNs probe (70.82% -> 76.12%). On this pair,
adding development-real adaptation did not beat the paired synthetic probe in
balanced accuracy. The 100% adaptation model had the highest raw accuracy
(78.16%) but only 65.66% balanced accuracy because wheeled recall fell to
39.13%; this is why raw accuracy is misleading for the 64/23 class imbalance.

## Results by milestone

| Milestone | Checkpoint/method | Accuracy | Balanced accuracy | Macro F1 | Tracked recall | Wheeled recall | Sessions correct |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | classical grouped | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0/2 |
| 2 | CNN grouped | 4.60% | 8.70% | 4.40% | 0.00% | 17.39% | 0/2 |
| 3 | classical unseen state | 1.15% | 2.17% | 1.14% | 0.00% | 4.35% | 0/2 |
| 3 | classical unseen environment | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0/2 |
| 3 | classical unseen vehicle | 2.30% | 4.35% | 2.25% | 0.00% | 8.70% | 0/2 |
| 3 | CNN unseen vehicle | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0/2 |
| 4 | classical single microphone | 9.20% | 17.39% | 8.42% | 0.00% | 34.78% | 0/2 |
| 5 | synthetic only | 11.49% | 16.17% | 11.45% | 6.25% | 26.09% | 0/2 |
| 5 | real only | 49.43% | 34.99% | 34.99% | 65.62% | 4.35% | 0/2 |
| 5 | synthetic + 1% real | 11.49% | 17.56% | 11.31% | 4.69% | 30.43% | 0/2 |
| 5 | synthetic + 5% real | 10.34% | 13.99% | 10.33% | 6.25% | 21.74% | 0/2 |
| 5 | synthetic + 10% real | 5.75% | 3.91% | 5.43% | 7.81% | 0.00% | 0/2 |
| 5 | synthetic + 25% real | 6.90% | 4.69% | 6.45% | 9.38% | 0.00% | 0/2 |
| 5 | synthetic + 100% real | 10.34% | 7.03% | 9.38% | 14.06% | 0.00% | 0/2 |
| 6 | standard supervised CNN | 11.49% | 20.35% | 10.74% | 1.56% | 39.13% | 0/2 |
| 6 | augmentation-only CNN | 1.15% | 0.78% | 1.14% | 1.56% | 0.00% | 0/2 |
| 6 | paired-supervised CNN | 2.30% | 2.96% | 2.30% | 1.56% | 4.35% | 0/2 |
| 6 | representation-invariance CNN | 2.30% | 2.96% | 2.30% | 1.56% | 4.35% | 0/2 |
| 6 | projected-invariance CNN | 1.15% | 0.78% | 1.14% | 1.56% | 0.00% | 0/2 |
| 6 | PANNs embedding, corruption only | 40.23% | 42.66% | 38.86% | 37.50% | 47.83% | 1/2 |
| 6 | PANNs embedding, paired | 36.78% | 47.28% | 36.78% | 25.00% | 69.57% | 1/2 |
| 6 | PANNs AudioSet, corruption only | 63.22% | 70.82% | 62.09% | 54.69% | 86.96% | 1/2 |
| 6 | **PANNs AudioSet, paired** | **68.97%** | **76.12%** | **67.58%** | **60.94%** | **91.30%** | **2/2** |
| 6 | PANNs + 1% real | 65.52% | 72.38% | 64.15% | 57.81% | 86.96% | 2/2 |
| 6 | PANNs + 5% real | 67.82% | 73.95% | 66.20% | 60.94% | 86.96% | 2/2 |
| 6 | PANNs + 10% real | 67.82% | 73.95% | 66.20% | 60.94% | 86.96% | 2/2 |
| 6 | PANNs + 25% real | 67.82% | 72.55% | 65.82% | 62.50% | 82.61% | 2/2 |
| 6 | PANNs + 100% real | 78.16% | 65.66% | 67.39% | 92.19% | 39.13% | 1/2 |
| 6 | previously frozen semantic/classical fusion | 50.57% | 49.69% | 47.20% | 51.56% | 47.83% | 1/2 |

## Interpretation

The early procedural/small-CNN systems do not transfer to this pair. Several
show confident class reversal rather than mere uncertainty. For example, the
Milestone 2 classical model predicts all 64 Sherman windows as wheeled and all
23 car windows as tracked. Its saved synthetic-domain test balanced accuracy
was 88.86%, so this is direct evidence that its synthetic success did not imply
real category transfer.

Generic AudioSet semantics are substantially more useful here than either the
handcrafted features or the compact learned embedding. The AudioSet output
probe also outperforms the already frozen semantic/classical fusion on this
pair. That does not authorize replacing the frozen model based on these opened
data; it nominates a model for a new, independently frozen comparison.

## Limitations

- There is one source recording per class. The session-level sample size is two.
- Adjacent windows overlap by one second and are not independent observations.
- Vehicle identity, era, source site, and class are perfectly confounded.
- The tracked source is a historical Sherman; no modern tracked vehicle is in
  this audit.
- The wheeled operating-state support is 1 startup, 2 idle, and 20 mixed
  windows. The tracked source has only the mixed/pass-by label.
- No SNR is available for native recordings, so accuracy-versus-SNR cannot be
  measured on this corpus.
- Calibration statistics are descriptive on 87 correlated windows and should
  not be generalized.

## Next decision

Freeze a small candidate comparison before opening the next recordings:

1. PANNs AudioSet paired linear probe (primary candidate);
2. PANNs AudioSet corruption-only probe (controlled pairing ablation);
3. previously frozen semantic/classical fusion (existing benchmark).

Then acquire and review at least one modern tracked recording and another
independent wheeled recording, with matched idle, acceleration, deceleration,
and pass-by states where possible. Keep entire sources grouped. Use that fresh
pair once, without threshold changes, as the next locked transfer check. Move
the current Sherman/PDSounds pair into a clearly marked development-diagnostic
role only after that protocol is frozen.

## Reproduction and artifacts

```bash
.venv/bin/python scripts/evaluate_historical_checkpoints.py \
  --config configs/historical_checkpoint_audit.yaml \
  --manifest .artifacts/real_locked_pair_sherman_pdsounds_v1/real_manifest.jsonl \
  --output runs/historical_checkpoint_audit_sherman_pdsounds_v1 \
  --repo-root . \
  --batch-size 16
```

The evaluator refuses to overwrite an existing audit directory. Machine-readable
artifacts are under `runs/historical_checkpoint_audit_sherman_pdsounds_v1/`:

- `metrics.json` SHA-256:
  `825c8c4471840d37233d7469c0b3969a72a546b199ec4e06a512a82e6e58b1dd`
- `comparison.csv` SHA-256:
  `ac28d11c09af8448d891d3e9a40f7ac9af88640d0db2c51fc9ee94da9d43c464`
- `window_predictions.csv` SHA-256:
  `966e71fff60af541dec9c7682484769aeea7f5f4f852a6792eb12e9513e198c5`
- `native_features.pt`
- `panns_features.pt`

Validation:

- 79/79 repository tests passed.
- Checkpoint class order was `tracked`, `wheeled` throughout.
- Confusion matrices were independently reconstructed from prediction rows.
- The PANNs AudioSet extraction matched the prior locked feature extraction to
  a maximum absolute difference of `1.2219e-06` (batch-level floating-point
  variation).

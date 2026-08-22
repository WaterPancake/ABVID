# Semantic/classical late-fusion evaluation

Date: 2026-08-22

Status: strongest leakage-safe real-session category diagnostic so far; Milestone 7
gate not met

Performance claim: **real-only development evaluation on the exact seven-session
corpus identified below; not an independent final test**

## Experiment identity

| Field | Value |
|---|---|
| Run | `runs/m6_fusion_nested_sessions` |
| Git commit | `54fa6dac6e0cfd32dcdcead1772d34cbb97a8c93` |
| Dataset version | `84fb1ef0e9f44df7f8d5c48400ea3f9b3bac3de2b3bdf5ae63004b073883c1dd` |
| Test domain | `real -> real nested unseen recording-session pairs` |
| Native-real corpus | 271 two-second windows; 4 tracked and 3 wheeled sessions |
| External encoder | Frozen PANNs Cnn14 pretrained on AudioSet |
| PANNs checkpoint SHA-256 | `0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31` |
| Semantic branch | Fixed 35-label vehicle/engine subset of AudioSet outputs |
| Classical branch | 53 MFCC and compact spectral/temporal features, channel 0 |
| Branch models | Session-balanced logistic probes at `C = 0.001, 0.01, 0.1, 1, 10` |
| Fusion | Equal mean of semantic and classical ensemble probabilities |
| Threshold candidates | Wheeled probability `0.20` through `0.60` in steps of `0.05` |
| Outer folds | Every tracked-session x wheeled-session pair: 4 x 3 = 12 |
| Inner folds | Every remaining tracked-session x wheeled-session pair: 6 per outer fold |
| Random seed | 42 |

The two feature families made complementary errors in the preceding development
audit: semantic AudioSet outputs recognized Goodwood and Maserati well but rejected
the Ford Model T, while classical MFCC/spectral features recognized the Ford session
but rejected Maserati. This experiment therefore averages the two probability
vectors with fixed `0.5/0.5` weights. It does not fit a fusion layer.

For each outer pair, every probe is fitted only on the remaining sessions. The
wheeled decision threshold is selected by mean balanced accuracy across six inner
held-out session pairs, then frozen for the outer pair. Training weights give equal
mass to classes, sessions within a class, and windows within a session. The saved
split artifact records every inner and outer sample ID. An audit found zero overlap
for outer train/test, inner train/validation, and inner validation/outer test.

The fusion and threshold grids were chosen after inspecting earlier results on this
same corpus. The nested mechanics are leakage-safe, but the performance remains a
post-hoc development result and requires a newly admitted locked pair.

## Results

| Metric across 12 outer pairs | Mean | Sample SD | Median | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| Balanced accuracy | **77.35%** | 12.08% | 76.79% | 60.28% | 97.66% |
| Macro F1 | 73.66% | 15.98% | 74.85% | 48.55% | 97.58% |
| Tracked recall | **77.42%** | 24.39% | 84.11% | 22.22% | 100.00% |
| Wheeled recall | **77.28%** | 28.32% | 95.00% | 29.41% | 100.00% |

Relative to the semantic-only regularization ensemble, fusion raises mean balanced
accuracy by 4.43 percentage points and wheeled recall by 21.25 points, while tracked
recall falls by 12.40 points. This is a much more balanced average boundary, not a
uniformly reliable one.

The inner-selected threshold ranges from `0.20` to `0.60`. Both held-out sessions
receive the correct recording-level decision in only 6 of 12 outer folds (50%). The
semantic-only ensemble achieved 8 of 12, despite its worse window-level average.
This disagreement and the large per-class standard deviations expose threshold and
session instability.

Specific failures remain:

- the Ford Model T wheeled recall falls to 29.41% in one outer context;
- the DVIDS Romanian tank tracked recall falls to 22.22% in one outer context;
- two other Ford contexts remain below 50% wheeled window recall;
- threshold selection can move in opposite directions when only two wheeled
  training sessions remain.

Probability calibration is not reported. The threshold is selected for balanced
accuracy, and treating the raw fused probability as calibrated confidence would be
misleading with seven sessions.

## Milestone 7 gate assessment

| Gate | Requirement | Result | Status |
|---|---|---:|---|
| 1 | At least 5 reviewed sessions per class | 4 tracked / 3 wheeled | Fail |
| 2 | Nested mean balanced accuracy at least 75% | 77.35% | **Pass** |
| 3 | Mean recall at least 70% for both classes | 77.42% / 77.28% | **Pass** |
| 4 | No held-out session below 50% window recall | Minimum 22.22% / 29.41% | Fail |
| 5 | Newly admitted locked test pair | None | Fail |

This is the first leakage-safe development run to pass gates 2 and 3. It does not
authorize Milestone 7 because a stronger mean cannot compensate for catastrophic
individual sessions or the missing corpus and confirmation requirements.

## Frozen confirmation protocol

A deployable rule is now frozen before any confirmation pair is admitted:

| Field | Value |
|---|---|
| Run | `runs/m6_fusion_frozen_development_v2` |
| Code commit | `ae42478a98bc0e2bfdc3b75da6314a87388af14b` |
| Frozen checkpoint | `frozen_fusion_model.pt` |
| Checkpoint SHA-256 | `8be2081987124dfce5abe59fb973a81c05188cd681a2924a27f67681814e9e6e` |
| Selected global wheeled threshold | `0.25` |
| Status | `frozen_awaiting_locked_evaluation` |

The global threshold was selected across the same 12 leave-session-pair-out
development folds, then both five-member probe ensembles were refitted on all 271
development windows. At the selected threshold, the development-selection summary
is 83.73% mean balanced accuracy, 74.81% tracked recall, and 92.66% wheeled recall.
These numbers are selection statistics and **not** an independent performance claim.
The minimum tracked recall remains 46.88%, so the development corpus still fails the
per-session floor as well.

The frozen checkpoint records the development manifest hash and protected values for
recording session, source ID, original media URL, normalized-source hash, and raw
hash. `scripts/evaluate_locked_fusion.py` requires exactly one provenance-complete
unseen session per class, rejects any protected overlap, validates the PANN checkpoint
and both feature definitions, writes the split and window predictions, and refuses to
overwrite a completed locked result. This closes the procedural gap for gate 5; it
does not satisfy the gate until real new data are evaluated once.

## Next action

The highest-value action remains new independent data, not a larger classifier:

1. admit at least one reviewed heavy/idle or ordinary pass-by wheeled session and
   one independently recorded moving tracked session;
2. keep one newly admitted tracked/wheeled pair locked while the fusion rule and
   threshold policy are frozen;
3. rerun the nested development protocol on the expanded remainder;
4. evaluate the frozen policy once on the locked pair;
5. proceed to hierarchical/open-set work only if all five gates pass.

Approved Wikimedia candidates remain unavailable through the compliant collector.
Earlier requests returned HTTP 429; a single Vanwall retry after the cooldown on
2026-08-22 failed at DNS resolution before download. The collector manifest records
the latest outcome for each source. No audio from these failed attempts was admitted,
and no alternate route or rate-limit bypass was used.

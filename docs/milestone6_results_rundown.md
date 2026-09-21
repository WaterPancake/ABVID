# Milestone 6 results rundown

Date: 2026-08-22; reconciled with later source/evaluation history 2026-09-20

Milestone: **Noise-Invariant Representation Learning**

Status: historical Milestone 6 record; Milestone 7 entry gate not met

This document preserves the experiment state assembled on 2026-08-22. For the live
Benchmark v0.1 source roles and gate, use [`benchmark_v0_1.md`](benchmark_v0_1.md)
and `configs/benchmark_v0_1.yaml`. The catalog now contains 7 tracked / 5 wheeled
development sessions. The refreshed primary fusion scored 39.70% mean balanced
accuracy and failed development gates 2-4; see
[`benchmark_v0_1_native_real_results.md`](benchmark_v0_1_native_real_results.md).
The later [failure diagnosis](benchmark_v0_1_failure_diagnostics.md) and
[30-condition corruption evaluation](benchmark_v0_1_corruption_results.md) reuse
that frozen 7/5 baseline; neither changes the failed native gate assessment.
The Sherman/PDSounds locked evaluation was completed on 2026-08-24 and failed; it is
documented in [`milestone6_fusion_session_report.md`](milestone6_fusion_session_report.md).

### Current non-startup preprocessing-grid gate assessment (2026-09-20)

The operator parked startup intervals while preserving recordings and history.
The new uncapped 785-window development corpus still has 7 tracked / 5 wheeled
sessions. Nested inner-only preprocessing/C selection scored **51.86% balanced
accuracy, 77.31% tracked recall, 26.41% wheeled recall**, and **0% worst-session
recall** for the primary PANNs head. Its matched unchanged-input control scored
53.73%. The classical comparator improved from 43.23% to 47.26% but also retained
zero-recall cases. This does not replace historical dataset results above.

| Gate | Current result |
|---|---|
| At least five reviewed sessions per class | Pass: 7 tracked / 5 wheeled |
| Mean balanced accuracy ≥75% | Fail: 51.86% |
| Both recalls ≥70% | Fail: wheeled 26.41% |
| Every held-out session recall ≥50% | Fail: minimum 0% |
| New frozen confirmation | Not attempted; reserved pair untouched |

See [full results/protocol](preprocessing_grid_results.md),
[versioned artifacts](../benchmarks/v0.1/preprocessing_grid_v1/README.md), and
[verification](../benchmarks/v0.1/preprocessing_grid_v1/verification.json).
Milestone 7 remains blocked; benchmark/demo packaging may continue.

## Executive summary

Milestone 6 successfully built and evaluated paired clean/corrupted representation
training, but it did **not** demonstrate that noise invariance alone solves
synthetic-to-real tracked-versus-wheeled classification.

The original representation-invariance model did what its objective requested on
synthetic data: clean and corrupted versions of an event moved closer in embedding
space, and aggregate synthetic balanced accuracy improved slightly. That improvement
did not transfer to the initial held-out real recordings. In the first controlled
run, the invariance model achieved 78.9% balanced accuracy on all held-out synthetic
corruptions but only 12.5% on the two held-out native-real sessions.

Subsequent controlled work established the following:

- five-seed ablations did not find a repeatable synthetic-to-real benefit from the
  paired or projected invariance objectives;
- a frozen PANNs AudioSet representation improved the fixed real result, especially
  after adapting its linear head with all available real-development data, but the
  result remained session-limited and variable;
- a nested real-only semantic probe exposed severe source dependence, especially a
  failure on the Ford Model T recording;
- late fusion of semantic PANN outputs and classical MFCC/spectral features produced
  the strongest leakage-safe real-session development average: **77.35% balanced
  accuracy**, **77.42% tracked recall**, and **77.28% wheeled recall**;
- individual held-out sessions still failed catastrophically, with minimum recalls
  of 22.22% for tracked and 29.41% for wheeled;
- the evaluated corpus had only four tracked and three wheeled recording sessions;
- the later Sherman/PDSounds confirmation scored 49.69% balanced accuracy and failed
  to confirm the development result.

The correct conclusion is therefore: **Milestone 6 improved the experimental method
and produced a credible binary development baseline, but it did not establish a
reliable category classifier suitable for Milestone 7.**

## Experimental scope and domains

Results in this report come from different domains and must not be combined as if
they measured the same problem.

| Domain | Meaning in these experiments |
|---|---|
| `synthetic -> synthetic` | Train on procedural/augmented vehicle events and test on held-out synthetic recording sessions or corruptions |
| `synthetic -> real` | Fit the ABVID classifier on synthetic examples and evaluate on native-real recordings |
| `external real pretraining + synthetic -> real` | Use frozen PANNs, pretrained on AudioSet, then fit an ABVID probe on synthetic examples |
| `synthetic + limited real -> real` | Adapt only the PANN classifier head with grouped real-development data, then test on fixed real sessions |
| `real -> real` | Fit on complete real recording sessions and evaluate on different, fully excluded real sessions |

The strongest late-fusion result is a `real -> real` **development** result. It is not
evidence that the original synthetic-only invariance objective transfers to real
recordings.

### Principal datasets

| Dataset | Size and grouping | Use |
|---|---|---|
| Initial controlled synthetic corpus | Grouped procedural sessions with paired clean/corrupted observations | First three-method invariance experiment |
| Factorial synthetic corpus | 2,688 fully crossed two-second paired observations | Five-seed objective ablation and nuisance diagnostics |
| Native-real corpus v2 | 271 two-second windows from 4 tracked and 3 wheeled recording sessions | Fixed transfer tests, all-session diagnostics, and nested unseen-session evaluation |

Every split kept all windows from an original recording in one
`recording_session`. No experiment randomly distributed windows from the same source
across train and test.

## How the milestone was achieved

### 1. Paired representation-learning pipeline

Milestone 1 supplied paired observations of the same source event:

```text
clean event --------------------> encoder -> z_clean
same event with corruption ----> encoder -> z_corrupted
```

The initial model was a small log-Mel CNN with a 64-dimensional pooled embedding.
Three controlled methods used the same architecture, initialization, grouped split,
class weighting, optimizer, and corrupted validation set:

| Method | Training behavior |
|---|---|
| Standard supervised | Weighted tracked/wheeled cross-entropy on clean inputs |
| Augmentation only | The same cross-entropy on corrupted inputs |
| Representation invariance | Clean and corrupted classification plus paired consistency, hard-positive consistency, and triplet-margin hard-negative loss |

The invariance objective used:

- clean/corrupted cosine consistency with weight `0.5`;
- same-vehicle hard-positive cosine consistency with weight `0.1`;
- an established cosine triplet-margin loss with weight `0.1` and margin `0.2`.

Hard positives used the same vehicle with maximally different available state/session
and nuisance metadata. Hard negatives used the opposite coarse class while matching
operating condition, background, geometry, microphone state, and SNR as closely as
possible. All 111 eligible initial training anchors had both a hard positive and a
hard negative.

### 2. Factorial corpus and five-seed controls

The first run was too small and used only one training seed. A 2,688-observation
factorial corpus was therefore generated so the same paired-event protocol could be
repeated across seeds `7`, `19`, `42`, `73`, and `101`, with split seed `42` fixed.

The comparison expanded to five methods:

- standard supervised;
- augmentation only;
- paired supervised classification without an explicit consistency loss;
- direct representation invariance;
- projected representation invariance using a separate projection head.

This separated the value of seeing paired inputs from the value of explicitly
forcing the classifier embedding to be invariant.

### 3. Frozen environmental-audio representation

Because the small CNN consistently failed on real data, a frozen PANNs Cnn14 encoder
pretrained on AudioSet was introduced as an established environmental-audio
baseline. Two representations were compared:

- the 2,048-dimensional PANN embedding;
- the 527 AudioSet class outputs.

Each was evaluated with corrupted-only and clean-plus-corrupted probe training. The
encoder remained frozen. A separate experiment then adapted only the best linear
head using nested fractions of the grouped real-development partition.

### 4. Leakage-safe real-session diagnostics

The repeatedly viewed fixed AMX-30/Maserati test pair could no longer answer whether
the category boundary generalized across all real sources. A stricter real-only
diagnostic was therefore added:

- every outer fold excluded one complete tracked and one complete wheeled session;
- all `4 x 3 = 12` tracked/wheeled held-out pairs were evaluated;
- logistic regularization was selected only through inner held-out session pairs;
- training gave equal mass to each class, each session within a class, and each
  window within a session.

The semantic representation used a fixed 35-label vehicle/engine subset of the 527
AudioSet outputs. This removed unrelated near-zero speech/music outputs that had
acquired unstable coefficients in the unrestricted probe.

### 5. Complementary late fusion

Error analysis found complementary failures:

- the semantic PANN branch recognized Goodwood and Maserati but nearly always
  rejected the Ford Model T;
- the 53-feature classical MFCC/spectral branch recognized Ford better but performed
  poorly on Maserati.

The final development model averaged the two branch probability vectors with fixed
`0.5/0.5` weights. Each branch was itself an equal-probability ensemble of
session-balanced logistic probes fitted at:

```text
C = 0.001, 0.01, 0.1, 1, 10
```

For every outer pair, the wheeled threshold was selected from `0.20` through `0.60`
using only the six inner session-pair folds. An audit found zero overlap for:

- outer train versus outer test;
- inner train versus inner validation;
- inner validation versus outer test.

### 6. Frozen confirmation protocol

One global model was frozen before admitting any new confirmation data. Global
threshold `0.25` was selected using leave-session-pair-out development predictions,
then both five-member probe ensembles were refitted on all 271 development windows.

The frozen checkpoint records protected development provenance, including session,
source ID, media URL, and source hashes. The locked evaluator requires exactly one
new provenance-complete session per class and refuses any overlap or completed-result
overwrite.

The frozen model is awaiting a genuinely new pair. Its development-selection
statistics are not an independent test result.

## Results by experimental stage

### Stage A: first controlled invariance experiment

Run: `runs/m6_invariance_seed42`

Commit: `88062f6c83eb028dbcada963b6c48834785499f0`

| Test domain | Standard supervised | Augmentation only | Representation invariance |
|---|---:|---:|---:|
| Seen synthetic corruption | 70.5% | 66.7% | **72.7%** |
| Unseen synthetic noise | 40.0% | **50.0%** | 35.0% |
| Unseen synthetic microphone | 70.3% | 65.4% | **76.9%** |
| Unseen synthetic environment proxy | **91.3%** | 73.9% | 89.1% |
| All held-out synthetic corruptions | 77.7% | 73.4% | **78.9%** |
| Native-real held-out sessions | 29.0% | **48.4%** | 12.5% |

The representation objective increased clean/corrupted similarity:

| Method | Mean paired cosine similarity | Mean normalized L2 distance |
|---|---:|---:|
| Standard supervised | 0.845 | 0.443 |
| Augmentation only | 0.890 | 0.356 |
| Representation invariance | **0.916** | **0.333** |

Interpretation: the objective achieved synthetic embedding consistency, but the
invariance model was the worst real-transfer model. The 48.4% augmentation-only real
score was also not useful; it came from predicting nearly everything as tracked.

### Stage B: five-seed factorial objective comparison

Runs: `runs/m6_factorial_2s_session_diagnostic/seed_{7,19,42,73,101}`

Commit: `6bfe7bcaefcdcb431c7b31df98cbabf245154535`

Values are mean balanced accuracy plus/minus sample standard deviation across five
training seeds.

| Method | Synthetic all-corruption BA | Fixed native-real BA |
|---|---:|---:|
| Standard supervised | 75.14% +/- 2.30% | 35.97% +/- 9.43% |
| Augmentation only | 76.24% +/- 0.76% | **38.79% +/- 9.54%** |
| Paired supervised | **76.65% +/- 1.39%** | 36.75% +/- 5.96% |
| Direct representation invariance | 75.83% +/- 1.72% | 35.66% +/- 4.76% |
| Projected representation invariance | 76.60% +/- 0.96% | 37.05% +/- 6.84% |

All five methods averaged only 0.7% to 1.3% wheeled recall on the fixed real test.
The paired and projected objectives therefore did not provide a repeatable
synthetic-to-real improvement.

The all-session diagnostic was highly source-dependent:

| Method | Session-balanced real accuracy |
|---|---:|
| Standard supervised | **62.58% +/- 11.44%** |
| Augmentation only | 46.96% +/- 5.16% |
| Paired supervised | 54.32% +/- 9.43% |
| Direct representation invariance | 47.52% +/- 2.08% |
| Projected representation invariance | 57.67% +/- 7.35% |

Every method nearly failed Maserati while performing well on Ford and Goodwood,
showing that recording source and vehicle identity dominated the average behavior.

### Stage C: frozen PANNs transfer and limited-real adaptation

Runs: `runs/m6_panns_audioset_transfer/seed_{7,19,42,73,101}`

Commit: `84195a105157ff7722e4804b65d35d9158dc867c`

| Representation and probe input | Synthetic BA | Fixed real BA | Tracked recall | Wheeled recall |
|---|---:|---:|---:|---:|
| 2,048 embedding, corrupted | 79.91% +/- 0.89% | 31.50% +/- 6.18% | 40.00% | 23.00% |
| 2,048 embedding, paired | 81.28% +/- 2.22% | 36.96% +/- 9.04% | 31.25% | 42.67% |
| 527 AudioSet outputs, corrupted | **86.40% +/- 1.24%** | 40.00% +/- 8.17% | 25.00% | 55.00% |
| 527 AudioSet outputs, paired | 83.57% +/- 3.25% | **47.90% +/- 8.76%** | 23.13% | 72.67% |

The paired AudioSet-output probe improved fixed real balanced accuracy by 9.11
percentage points over the small-CNN augmentation baseline, but it flipped the class
bias: wheeled recall improved while tracked recall fell to 23.13%.

Adapting only the linear head with all 95 real-development windows produced:

| Metric | Five-seed result |
|---|---:|
| Fixed real balanced accuracy | **65.50% +/- 13.85%** |
| Tracked recall | 65.00% |
| Wheeled recall | 66.00% |

The training data were still inadequate: 88 tracked windows came from two sessions,
while seven wheeled windows came from one Goodwood session. Fractions below 100%
produced a non-monotonic learning curve.

### Stage D: nested semantic real-session evaluation

Run: `runs/m6_semantic_nested_sessions_ensemble`

Commit: `4752f1cfe321d98708764e452ec9007d392d0c21`

| Metric across 12 outer pairs | Nested-selected `C` | Regularization ensemble |
|---|---:|---:|
| Mean balanced accuracy | 70.72% | **72.92%** |
| Mean macro F1 | 65.56% | **68.32%** |
| Mean tracked recall | 89.37% | **89.81%** |
| Mean wheeled recall | 52.07% | **56.03%** |
| Both sessions correct | 7/12 | **8/12** |

The ensemble improved the mean but remained strongly tracked-biased. Ford Model T
wheeled recall was only 0% to 14.71% across its outer contexts.

### Stage E: nested semantic/classical late fusion

Run: `runs/m6_fusion_nested_sessions`

Commit: `54fa6dac6e0cfd32dcdcead1772d34cbb97a8c93`

| Metric across 12 outer pairs | Mean | Sample SD | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Balanced accuracy | **77.35%** | 12.08% | 60.28% | 97.66% |
| Macro F1 | 73.66% | 15.98% | 48.55% | 97.58% |
| Tracked recall | **77.42%** | 24.39% | 22.22% | 100.00% |
| Wheeled recall | **77.28%** | 28.32% | 29.41% | 100.00% |

Fusion increased mean balanced accuracy by 4.43 points and wheeled recall by 21.25
points relative to the semantic-only ensemble, at the cost of 12.40 points of tracked
recall. Only 6 of 12 outer pairs received two correct recording-level decisions,
versus 8 of 12 for the semantic-only ensemble.

This was the first leakage-safe development run to pass the mean balanced-accuracy
and mean per-class-recall gates. It still failed the individual-session requirement.

### Stage F: frozen development rule

Run: `runs/m6_fusion_frozen_development_v2`

Code commit: `ae42478a98bc0e2bfdc3b75da6314a87388af14b`

Checkpoint SHA-256:
`8be2081987124dfce5abe59fb973a81c05188cd681a2924a27f67681814e9e6e`

| Selection statistic | Value |
|---|---:|
| Selected global wheeled threshold | 0.25 |
| Development-selection balanced accuracy | 83.73% |
| Development-selection tracked recall | 74.81% |
| Development-selection wheeled recall | 92.66% |
| Minimum tracked recall | 46.88% |

These are model-selection statistics on the reused development corpus. They are
included to identify the frozen rule, not to make a performance claim. The artifact
was originally written as `frozen_awaiting_locked_evaluation`; its one-shot locked
evaluation was later completed and must not be repeated.

## Deviations from the original Milestone 6 plan

The following deviations were deliberate and are part of the interpretation of the
results.

| Deviation | Why it was used | Consequence and reporting boundary |
|---|---|---|
| Far source geometry was used as the unseen-environment partition | The generated corpus did not contain impulse-response observations spanning distinct rooms/environments | This is a propagation/geometry proxy, not a true unseen-room result |
| SNR slices contained different events rather than the same base events at every SNR | The first corpus was not a paired causal SNR sweep | Non-monotonic SNR accuracy cannot be interpreted as a clean causal degradation curve |
| A factorial two-second corpus replaced the smaller initial training subset | The one-seed run had only 111 eligible training anchors and weak nuisance support | Improved control and repeatability, but did not by itself improve real transfer |
| Paired-supervised and projected-invariance ablations were added | Needed to distinguish paired exposure from direct pressure on the classifier embedding | Both were treated as established controls; neither earned promotion from the real result |
| Frozen PANNs was added | The small CNN lacked real environmental-audio knowledge | PANNs was pretrained on real AudioSet data, so this is not a strict `synthetic only -> real` representation claim |
| Only the PANN linear head was adapted with real data | Full encoder fine-tuning was unjustified with seven sessions | Reduced overfitting risk, but the real learning curve remained data-limited and non-monotonic |
| The semantic probe used 35 of 527 AudioSet outputs | The unrestricted probe assigned unstable coefficients to unrelated near-zero labels | The fixed semantic subset improved auditability, but it was introduced after shortcut analysis and remains development work |
| An equal ensemble over all regularization values was added | Inner selection of one `C` was unstable across the small number of sessions | The ensemble was added after observing the original nested result; its gain is post-hoc development evidence |
| Classical features and late probability fusion were added | Semantic and classical branches made complementary Ford/Maserati errors | Fusion is outside a pure representation-invariance objective and was selected after examining development errors |
| The wheeled threshold was selected for balanced accuracy | The fused probabilities were strongly biased and `0.5` was not a useful operating point | Thresholded probabilities were not reported as calibrated confidence |
| The fixed AMX-30/Maserati pair was reused during development | It was the only established fixed pair when Milestone 6 began | It is no longer an acceptable final confirmation set |
| A global threshold was selected across all development folds and the model was refit on all seven sessions | Needed a fully frozen rule before admitting new data | The resulting 83.73% is a selection statistic, not a test result |
| The first locked pair failed confirmation | Sherman/PDSounds scored 49.69% balanced accuracy and the Sherman session was misclassified | Preserve the result; preregister and freeze a refreshed protocol before evaluating the separate T90M/JLTV pair once |

## Rejected or non-promoted approaches

Several alternatives were examined but not promoted to the formal result:

- direct and projected invariance losses did not produce a repeatable fixed-real gain;
- the 2,048-dimensional PANN embedding transferred worse than the semantic AudioSet
  output vector;
- the full 527-output real-session probe was less stable than the fixed 35-feature
  semantic subset;
- nonlinear real-session probes and unconstrained threshold tuning were unstable on
  seven sessions;
- richer exploratory encoders were not added to the dependency surface when they did
  not outperform the audited PANN/classical baseline.

These negative results helped prevent architecture growth from being mistaken for
category-level generalization.

## Milestone 7 gate assessment

The gate was fixed before beginning hierarchical or open-set work:

| Gate | Requirement | Current evidence boundary | Status |
|---|---|---:|---|
| 1 | At least five reviewed development sessions per class | Current catalog/local discovery: 7 tracked / 5 wheeled | **Pass** |
| 2 | Nested unseen-session mean balanced accuracy at least 75% | Current 7/5 primary fusion: 39.70% | Fail |
| 3 | Mean recall at least 70% for both classes | Current 7/5 primary fusion: 51.98% / 27.42% | Fail |
| 4 | No held-out session below 50% window recall | Current minimum: 0.00% | Fail |
| 5 | Passing confirmation on a newly admitted locked pair | Sherman/PDSounds: 49.69%, Sherman misclassified | Fail |

Historical passes do not carry over to a changed corpus or model. Milestone 7 should
not begin while the current corpus fails the performance gates, the model can fail an entire
unseen recording session, and confirmation has not passed.

## Reproducibility and audit artifacts

Every formal experiment records its commit, configuration, seed, dataset hash,
grouped split, checkpoint, and metrics. The principal artifacts are:

- [`milestone6_report.md`](milestone6_report.md): first controlled run;
- [`milestone6_improvement_checkpoint.md`](milestone6_improvement_checkpoint.md):
  five-seed factorial diagnostic;
- [`milestone6_pretrained_transfer_report.md`](milestone6_pretrained_transfer_report.md):
  frozen PANNs and limited-real head adaptation;
- [`milestone6_semantic_session_report.md`](milestone6_semantic_session_report.md):
  nested semantic real-session evaluation;
- [`milestone6_fusion_session_report.md`](milestone6_fusion_session_report.md): late
  fusion, gate assessment, and frozen confirmation protocol;
- `runs/m6_fusion_frozen_development_v2/frozen_fusion_model.pt`: frozen checkpoint;
- `scripts/evaluate_locked_fusion.py`: one-shot locked-pair evaluator.

The first CPU invariance run was repeated from its manifest-validated feature cache;
excluding only the creation timestamp, metrics were identical and every checkpoint
tensor matched exactly. Later aggregate tools reject mismatched commits, datasets,
splits, configurations, supports, or adaptation sample IDs.

At the time this rundown was assembled, the complete repository test suite passed
with 70 tests.

## Final assessment

Milestone 6 answered three questions:

1. **Can paired training make synthetic representations less sensitive to modeled
   corruption?** Yes, on the controlled synthetic corpus.
2. **Did that invariance objective close the synthetic-to-real gap?** No.
3. **Can a carefully audited real-session model distinguish tracked and wheeled
   vehicles on average?** Provisionally yes on the seven-session development corpus,
   but not reliably for every unseen session.

The next defensible step is to add independent reviewed real sessions, reserve one
new tracked/wheeled pair untouched, and evaluate the already frozen rule exactly
once. No hierarchy, family classifier, or open-set threshold should be developed
until all five gate conditions pass.

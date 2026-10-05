# H1 regularization transfer sensitivity

2026-10-04. Protocol `h1_regularization_transfer_v1`, run `regularization_transfer_20261004_v1`. **Complete: IDMT real → MELAUDIS real, car/truck, exposed development evaluation.**

**BEATs regularization improves balanced accuracy and truck recall, but does not establish the primary macro-F1 improvement.** MFCC has no clear overall benefit. The source-domain gains therefore transfer only partly, and the domain gap remains. This is a classifier-control result; no simulator component was changed.

## Frozen comparison and evidence

- [Frozen protocol](../experiments/h1_transfer_sensitivity/frozen/regularization_transfer_20261004_v1/PROTOCOL.md), [configuration](../experiments/h1_transfer_sensitivity/frozen/regularization_transfer_20261004_v1/config.json), [lock and hashes](../experiments/h1_transfer_sensitivity/frozen/regularization_transfer_20261004_v1/lock.json).
- [Exact saved models and training IDs](../experiments/h1_transfer_sensitivity/frozen/regularization_transfer_20261004_v1/model_references.json), [all per-model/group scores](../experiments/h1_transfer_sensitivity/results/regularization_transfer_20261004_v1/evaluations.json), [aggregates and paired intervals](../experiments/h1_transfer_sensitivity/results/regularization_transfer_20261004_v1/summary.json), [independent verification](../experiments/h1_transfer_sensitivity/results/regularization_transfer_20261004_v1/verification.json).
- [Commands](../experiments/h1_transfer_sensitivity/README.md), [previous source-only diagnostics](H1_source_diagnostics.md), [unchanged H1](H1_baseline_results.md).

Reuse the 190/class IDMT selections, six outer groups and five seeds, two-second mean-channel native→8 kHz→16 kHz observations, frozen BEATs768/MFCC26, and threshold .5. The reference uses C=1; the candidate reuses each saved outer head with C selected previously by five inner IDMT groups. BEATs selected .01 in 26/30 cases and .1 in four; MFCC selected .01/.1/1 in 3/14/13 cases. No new choices, fitting, extraction, augmentation, threshold adjustment or bandwidth changes occurred.

All 8,066 admitted MELAUDIS excerpts are evaluated: 7,810 cars and 256 trucks. The target was already exposed in R0/H1; this freeze preceded new candidate inference, not all prior knowledge of the target. Candidate validation uses additional IDMT labels compared with the original untuned baseline. All target features enter only prediction/scoring, never scaler fitting or model selection.

## Primary and class-level results

Values are percentages. Source F1 averages six held-out group scores; target F1 is computed on the full target for each model and averaged over 30 source fold/seed combinations. These are averages of individual models, not ensembles. Brackets are conditional 95% paired-bootstrap intervals.

| Representation / arm | Source F1 | Target F1 [95% CI] | Target BA | Car recall | Truck recall | Truck precision |
| --- | --- | --- | --- | --- | --- | --- |
| BEATs768 / baseline | 53.08 | 33.96 [27.05, 40.77] | 57.38 | 45.51 | 69.24 | 4.38 |
| BEATs768 / nested_C | 54.89 | 34.59 [26.36, 44.34] | 62.42 | 45.57 | 79.27 | 5.06 |
| MFCC26 / baseline | 56.23 | 31.34 [22.88, 35.50] | 58.47 | 39.35 | 77.59 | 4.13 |
| MFCC26 / nested_C | 55.56 | 30.37 [22.73, 34.82] | 58.70 | 37.48 | 79.92 | 4.13 |

Paired candidate-minus-reference effects, in percentage points:

| Representation | Macro-F1 gain | BA gain | Car-recall gain | Truck-recall gain |
| --- | --- | --- | --- | --- |
| BEATs768 | +0.63 [-2.21, 5.13] | +5.04 [2.73, 7.99] | +0.06 [-4.91, 7.63] | +10.03 [4.14, 14.80] |
| MFCC26 | -0.98 [-2.00, 0.72] | +0.23 [-0.51, 1.11] | -1.87 [-3.82, 0.98] | +2.33 [0.14, 4.64] |

**BEATs:** +0.63 F1 points [−2.21,5.13] crosses both zero and the preregistered three-point practical reference. Neither a positive nor a practically important F1 benefit is established. BA improves +5.04 points [2.73,7.99] and truck recall +10.03 [4.14,14.80]; these secondary conditional intervals exclude zero. Mean car recall is almost unchanged, but its interval allows loss. The stricter claim “improves without sacrificing class recall” is not established.

**MFCC:** F1 changes −0.98 points [−2.00,0.72]; this interval rules out the proposed ≥3-point practical F1 gain under this conditional estimator. BA is nearly unchanged. A +2.33-point truck-recall change accompanies −1.87 points in car recall; this is not a general improvement. These secondary comparisons are not multiplicity-adjusted confirmation.

The always-car reference yields 49.19% F1, 50% BA and zero truck recall. Both learned candidates remain below it in F1 while detecting trucks; neither is a reliable classifier. BEATs truck precision is only 5.06%, reflecting many false truck predictions under the target’s 96.83% car prevalence.

## Ranking, probability errors and residual domain gap

| Representation / arm | ROC-AUC | Truck AP | Brier | Log loss | ECE10 |
| --- | --- | --- | --- | --- | --- |
| BEATs768 / baseline | 61.56 | 5.68 | 0.4862 | 3.6628 | 0.5215 |
| BEATs768 / nested_C | 70.03 | 8.15 | 0.3713 | 1.1710 | 0.5069 |
| MFCC26 / baseline | 64.59 | 6.31 | 0.4339 | 1.4110 | 0.5550 |
| MFCC26 / nested_C | 65.45 | 6.21 | 0.4178 | 1.2807 | 0.5582 |

These are descriptive means, without group-resampled intervals. BEATs ranks target classes better (.6156→.7003 AUC) and has lower Brier/log loss; however, ECE remains approximately .507. No probability calibrator was fitted. MFCC ranking changes little and AP slightly declines. The pooled truck prevalence reference for AP is 3.17%; higher AP does not by itself establish useful precision at the fixed decision rule.

| Representation | Original source-minus-target F1 gap | Candidate gap |
| --- | --- | --- |
| BEATs768 | 19.11 [11.16, 25.91] | 20.30 [10.43, 27.88] |
| MFCC26 | 24.88 [19.49, 34.03] | 25.20 [20.43, 32.89] |

The BEATs gap does not shrink: its point estimate is 19.11→20.30 points because source F1 improves more than target F1. The MFCC gap is 24.88→25.20. These gaps compare different class priors and aggregation schemes; they are descriptive domain differences, not isolated acoustic mechanisms.

## Recording-group findings and worst cases

### BEATs768

| Target group | Car/truck | F1, original → candidate | Car recall | Truck recall |
| --- | --- | --- | --- | --- |
| 2024-01-17 | 626/36 | 38.16 → 42.19 | 46.75 → 51.45 | 80.00 → 89.54 |
| 2024-01-16 | 468/32 | 37.49 → 38.72 | 45.81 → 46.37 | 70.62 → 78.54 |
| Nine linked 2023 dates | 6290/187 | 33.51 → 33.74 | 45.58 → 44.95 | 66.77 → 77.31 |
| 2024-02-09 | 426/1 | 28.64 → 29.95 | 42.30 → 45.09 | 100.00 → 100.00 |

### MFCC26

| Target group | Car/truck | F1, original → candidate | Car recall | Truck recall |
| --- | --- | --- | --- | --- |
| 2024-01-17 | 626/36 | 23.00 → 23.89 | 22.12 → 23.32 | 92.78 → 93.61 |
| 2024-01-16 | 468/32 | 28.04 → 27.82 | 28.40 → 27.84 | 87.40 → 88.65 |
| Nine linked 2023 dates | 6290/187 | 32.07 → 30.95 | 41.58 → 39.36 | 72.94 → 75.74 |
| 2024-02-09 | 426/1 | 30.03 → 28.67 | 43.78 → 41.14 | 86.67 → 90.00 |

The largest connected group contributes 6,477/8,066 excerpts. BEATs gains only +0.23 F1 points there and loses 0.63 points of car recall, although truck recall improves. The February group contains one truck: its repeated 100% BEATs recall is one event, not evidence of broad truck robustness.

| Representation | Mean per-model worst group/class recall, original → candidate | Paired gain [95% CI] | Absolute minimum, original → candidate |
| --- | --- | --- | --- |
| BEATs768 | 30.22 → 36.49 | +6.27 [-1.16, 12.62] | 4.63 → 4.63 |
| MFCC26 | 16.17 → 17.57 | +1.40 [-1.49, 5.09] | 0.00 → 0.00 |

Worst-group improvement remains uncertain. The absolute worst cases persist: 4.63% car recall for BEATs and 0% truck recall for MFCC. The averages therefore do not justify automatic adoption or a robustness claim.

## Verification and limits

Seven pre-run tests passed. Independent verification reconstructed all 60 C choices from the 900 original IDMT inner-prediction records, checked all 107 distinct pipelines and 120 source/target evaluations, all 480 target-group score records, scaler training means/variances, saved probabilities, metrics, bootstrap samples and intervals. The 60 original H1 target probability arrays replay exactly (maximum error 0); all source predictions also replay exactly. **8,079 protected H1/diagnostic artifacts remain byte-identical.**

M3 Pro CPU, 18 GiB, one BLAS thread: inference/checks 19.73 s; complete prediction/scoring/bootstrap 29.13 s; independent verification 26.42 s. No new fits, feature extraction, GPU use or downloads. Peak RAM was not measured.

Intervals use 10,000 paired draws (seed 314159) over four whole target groups, six source folds and five training selections; no draw lacked a class. They are conditional on fixed, overlapping source fits and a few uneven conservative groups, with uncertain coverage. Original recording/vehicle independence and pretrained-data overlap remain unknown. Checkpoint provenance retains the original mirror/official-checksum qualification. This is exposed development evaluation, not new independent confirmation.

## Decision and preparation for the existing source × propagation design

Keep H1 unchanged and retain source-selected regularization as a documented sensitivity, not a replacement baseline. There is evidence of a partial BEATs transfer benefit in balanced accuracy/recall, but no established primary F1 gain or resolution of worst-group failures. Close this bounded check; do not launch another C/threshold sweep on MELAUDIS.

The next physical comparison remains the existing [H2 four-cell proposal](proposed_experiments.md#7-h2--compare-source-and-propagation-contributions-e4e5):

| Cell | Source | Propagation |
| --- | --- | --- |
| F00 | S0: audited simple procedural dry source | P0: moving direct path, spreading/delay/Doppler |
| F10 | S1: paired source-envelope/fluctuation/component-balance diversity | Same P0 |
| F01 | Same S0 | P1: same path/trajectory plus ground reflection and atmospheric filtering |
| F11 | Same S1 | Same P1 |

Before implementing that comparison, freeze the source/backend identity, numerical factor ranges and paired source/trajectory manifests, and a common head/decision rule. Keep bandwidth, class/event budget, environment and sensor treatment identical across cells; source/path effects and their interaction must all be reported. The current proposal’s RMS matching conditions out absolute attenuation/SNR effects and must remain explicit. The released synthetic banks have insufficient generation lineage to relabel them as these controlled cells. No unavailable anechoic corpus or BVP access is required for the proposed procedural comparison.

If a later synthetic-training comparison uses IDMT-selected regularization, declare the additional real-label validation access and apply the same rule to every cell. Do not describe it as a zero-real-label experiment or select separate C values from target results. Source dominance remains a hypothesis; this study manipulated the classifier alone and supplies no causal source-versus-propagation ranking. No simulator was implemented or modified in this follow-up.

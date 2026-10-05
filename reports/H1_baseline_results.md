# H1 baseline results

2026-10-02. The minimum frozen H1 run is complete: six IDMT held-out groups, five training selections, 190 cars + 190 trucks per single-origin fit, frozen BEATs and MFCC controls. All 260 scaler/heads passed independent artifact, metric and known-group-separation checks. MELAUDIS is exposed development evaluation, not untouched confirmation.

**BEATs loses 19.11 macro-F1 percentage points across datasets. A blanket synthetic-training deficit is not supported:** procedural-only training exceeds matched IDMT training on MELAUDIS, while AudioLDM-only falls below it. Absolute performance is weak and four uneven target groups limit uncertainty. H1 does not establish whether source or propagation mismatch dominates.

## Evidence and frozen scope

- [Executed protocol](ARTIFACT_INDEX.md#artifact-0e3678c5ed62), [exact file selections and lock](ARTIFACT_INDEX.md#artifact-884f21ce4e24), [canonical configuration](ARTIFACT_INDEX.md#artifact-4c1be1b0afbd).
- [Qualified admission and missing metadata](dataset_admission_provenance.md).
- [Complete metrics and per-fit/group records](ARTIFACT_INDEX.md#artifact-fec2538f04ed), [run summary/hashes](ARTIFACT_INDEX.md#artifact-6b412e9fcddb), [independent verification](ARTIFACT_INDEX.md#artifact-5c3533e5d819).
- [Commands](../experiments/h1_baselines/REPLAY.md), [pre-fit prose clarification](../experiments/h1_baselines/notes/documentation_clarifications.md). Frozen document snapshots remain unchanged.

The same E0 fitted scaler/head is evaluated on its held-out IDMT group and on all admitted MELAUDIS excerpts. No validation, hyperparameter search, threshold tuning, target statistics, best-seed selection or score-based exclusions occur. Raw audio, R0 results and military evaluation sources are unchanged. No simulator interventions or encoder training were performed.

## Primary scores

Values are percentages. Brackets are 95% whole-group/selection percentile intervals, conditional on these fitted models. Source score averages six held-out-group F1s; target score averages those models on all 8,066 target excerpts. These are not ensembles. Synthetic rows average five actual fits, not thirty artificial repetitions. Balanced accuracy and recall are descriptive means.

| Representation / training | Evaluation | Macro-F1 [95% interval] | Balanced accuracy | Car recall | Truck recall |
|---|---|---:|---:|---:|---:|
| BEATs768 / IDMT only | Held-out IDMT | 53.08 [47.97, 58.31] | 65.87 | 66.15 | 65.60 |
| BEATs768 / IDMT only | MELAUDIS | 33.96 [27.05, 40.77] | 57.38 | 45.51 | 69.24 |
| BEATs768 / procedural only | MELAUDIS | 45.78 [38.29, 54.45] | 58.31 | 71.31 | 45.31 |
| BEATs768 / AudioLDM only | MELAUDIS | 23.47 [12.11, 25.77] | 54.27 | 25.34 | 83.20 |
| BEATs768 / IDMT repeated twice | MELAUDIS | 33.93 [27.13, 40.48] | 57.18 | 45.47 | 68.88 |
| BEATs768 / IDMT + procedural | MELAUDIS | 36.68 [29.95, 43.92] | 59.83 | 50.44 | 69.22 |
| BEATs768 / IDMT + AudioLDM | MELAUDIS | 33.18 [25.43, 37.35] | 55.62 | 43.61 | 67.63 |
| MFCC26 / IDMT only | Held-out IDMT | 56.23 [51.25, 60.82] | 71.00 | 69.66 | 72.34 |
| MFCC26 / IDMT only | MELAUDIS | 31.34 [22.88, 35.50] | 58.47 | 39.35 | 77.59 |
| MFCC26 / procedural only | MELAUDIS | 36.69 [31.99, 41.67] | 40.35 | 56.32 | 24.38 |
| MFCC26 / AudioLDM only | MELAUDIS | 36.75 [35.75, 43.06] | 49.06 | 52.18 | 45.94 |
| MFCC26 / IDMT repeated twice | MELAUDIS | 31.43 [22.85, 35.58] | 58.29 | 39.57 | 77.02 |
| MFCC26 / IDMT + procedural | MELAUDIS | 22.42 [18.17, 26.82] | 51.85 | 24.55 | 79.14 |
| MFCC26 / IDMT + AudioLDM | MELAUDIS | 39.76 [38.03, 45.17] | 56.47 | 57.03 | 55.91 |

MELAUDIS has **96.83% cars**. Always predicting car yields 49.19% macro-F1, 50% balanced accuracy, 100% car recall and 0% truck recall. Every learned arm scores below that constant predictor in target macro-F1, although several have balanced accuracy above 50%. Macro-F1 still depends on prevalence: the balanced-training fixed decision rule produces many false truck predictions. Procedural BEATs exceeds real-only by 11.82 F1 points but only 0.93 balanced-accuracy points. This is not evidence of generally reliable recognition. No threshold was adjusted after observing it.

Within-domain BEATs balanced accuracy is 65.87%, compared with 71.00% for MFCC. The weak within-domain baseline and constrained file budget limit mechanistic interpretation. Historical paper replay scores used different tasks, partitions and preprocessing and are not comparable to these binary results.

## Paired gap and mixing estimates

Values are macro-F1 percentage points. Positive domain/synthetic gap means the real-source reference exceeds its comparator. These conditional intervals have uncertain coverage with four uneven target groups; secondary comparisons are not multiplicity-adjusted confirmation claims.

| Contrast | BEATs estimate [95% interval] | MFCC estimate [95% interval] |
|---|---:|---:|
| Within IDMT minus cross-dataset | 19.11 [11.16, 25.91] | 24.88 [19.49, 34.03] |
| Real minus procedural on target | -11.82 [-23.04, -1.63] | -5.35 [-14.86, 0.67] |
| Real minus AudioLDM on target | 10.49 [3.85, 24.49] | -5.41 [-18.65, -0.91] |
| Real + procedural minus real | 2.71 [1.24, 4.74] | -8.93 [-12.19, -2.01] |
| Real + AudioLDM minus real | -0.78 [-7.22, 2.27] | 8.41 [4.67, 20.73] |
| Real + procedural minus repeated real | 2.74 [1.18, 4.95] | -9.01 [-12.30, -1.98] |
| Real + AudioLDM minus repeated real | -0.75 [-7.00, 2.17] | 8.33 [4.60, 20.79] |

The H1a domain-gap intervals exceed the prespecified 5-point practical margin for both representations. This establishes a descriptive loss under this protocol. Priors, fleet and recording context also change; acoustics are not separately identified.

For H1b, the BEATs procedural gap's upper bound is negative, contradicting an expected positive procedural-training deficit against this real baseline. The AudioLDM gap is positive, but its lower bound (3.85 points) is below the 5-point margin: a confidently ≥5-point deficit remains unresolved. MFCC reverses the AudioLDM ordering and also has a negative procedural point estimate. Do not select the representation that supports the working hypothesis.

BEATs procedural mixing gains 2.71 points, with a conditional interval above zero but crossing the broader proposal's 3-point intervention margin. AudioLDM mixing is inconclusive. MFCC procedural mixing hurts and AudioLDM mixing helps. Conventional real corruption augmentation was not run, so no advantage over ordinary augmentation is established. Repeating real features alone barely changes results. Mixed gains cannot be attributed to specific physics.

Paired BEATs-minus-MFCC real-only target F1 is 2.62 points, interval [−3.42, 14.21]. A general pretrained-representation advantage is not established. Every arm's representation contrast is in metrics.json.

## Source groups

Rows average five training selections. These are three locations across six recording dates, not six independent locations/known physical vehicles.

| Held-out IDMT site/date | Car/truck support | BEATs F1 | BEATs balanced accuracy | MFCC F1 |
|---|---:|---:|---:|---:|
| Schleusinger-Allee 2019-11-12 | 491/79 | 62.84 | 72.63 | 63.10 |
| Schleusinger-Allee 2019-11-13 | 666/110 | 57.64 | 71.37 | 60.80 |
| Fraunhofer-IDMT 2019-10-23 | 254/14 | 48.56 | 62.82 | 48.06 |
| Langewiesener-Strasse 2019-11-19 | 1365/152 | 49.33 | 65.09 | 53.66 |
| Fraunhofer-IDMT 2019-10-22 | 302/10 | 44.32 | 57.94 | 49.74 |
| Langewiesener-Strasse 2019-11-18 | 824/146 | 55.77 | 65.39 | 62.01 |

## Target groups

Real-only BEATs results average the 30 fitted models. This does not create thirty independent recordings per group.

| MELAUDIS conservative group | Car/truck support | Macro-F1 | Car recall | Truck recall |
|---|---:|---:|---:|---:|
| 2024-01-17 | 626/36 | 38.16 | 46.75 | 80.00 |
| 2024-01-16 | 468/32 | 37.49 | 45.81 | 70.62 |
| Nine linked dates in 2023 | 6290/187 | 33.51 | 45.58 | 66.77 |
| 2024-02-09 | 426/1 | 28.64 | 42.30 | 100.00 |

The February group has one truck: 100% average truck recall is one event repeatedly evaluated, not strong robustness evidence. The largest component supplies 6,477/8,066 events. Across individual real-only BEATs fits, minimum observed source class/group recall is 40%; minimum target class/group recall is 4.63%. MFCC minima are 30% and 0%. Means conceal severe failures.

## Verification and runtime

Eighteen admission regression tests, seven H1 tests and two existing BEATs adapter tests passed. The only shared adapter change removes an unnecessary benchmark/plotting import by using a local SHA256 helper; encoder loading/inference are unchanged. A deterministic artificial input gave bit-identical frozen encoder outputs.

The independent verifier checked all 260 model/prediction hashes, recomputed source/target confusion matrices and macro-F1/balanced accuracy using scikit-learn, checked known group/file/duplicate separation, verified every scaler against its exact training features, and replayed probabilities at three fixed target positions per model. These checks verify computation/known links, not unknown original-source independence. All prediction probabilities, training IDs, calibration metrics and per-group outcomes remain inspectable.

Apple M3 Pro, 18 GiB RAM, CPU only: provenance audit 63.18 s; 14,630-excerpt BEATs/MFCC cache 383.07 s (BEATs forwards 354.18 s); 260 fits, predictions and uncertainty 33.15 s. No GPU/cloud use. NumPy 1.26.4, SciPy 1.17.1, scikit-learn 1.4.2, Torch/torchaudio 2.11.0, Librosa 0.11.0, SoundFile 0.13.1. Lock SHA256: `3e427a2f4ee084ecdd4102dd3a9863a55cc06496abdb9688258ec26cbaaf901e`.

The budget counts retained excerpts, not independent physical vehicles/templates. AudioLDM's 190 trucks recur in every selection; only 190-of-192 cars vary. Synthetic generation backend/source lineage and pretrained data overlap remain unknown. The bounded waveform search found no cross-corpus matches but cannot rule out arbitrary cropped/modified reuse. Four conservative target groups yield conditional, uncertain-coverage intervals. R0 exposure means all target results are development observations.

## Next decision

H1 provides an auditable gap measurement and a counterexample to assuming all synthetic training is worse. It does not establish source dominance, propagation accuracy, a performance ceiling, model-level recognition or field readiness. Address the weak baseline and prevalence/recording-context confounds using source-only diagnostics and a separately frozen sensitivity study before interpreting simulator-component changes. Do not optimize thresholds/simulator settings on these target results and call the same target untouched confirmation.

This minimum H1 completes E0, E1, released-bank E2 and the MFCC part of E3. Official AST, a newly trained CNN, conventional augmentation, full-real sensitivity and E4–E9 remain unrun. Later controlled interventions require backend-verified source/trajectory manifests and a paired factorial. The released banks alone cannot identify source versus propagation effects.

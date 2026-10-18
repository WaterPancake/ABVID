# Target panel and prevalence-robust re-scoring

2026-10-11. **Post hoc descriptive re-analysis of exposed development predictions.**
No model was refitted, selected or thresholded. Every number below comes from the
archived H1, H2 and attenuation/latency prediction files, each checked against its
recorded SHA256. Nothing here is independent confirmation.

Code: [`abvid.evaluation.ranking`](../src/abvid/evaluation/ranking.py) and
[`abvid.evaluation.panel`](../src/abvid/evaluation/panel.py). Thin layer:
[`experiments/target_panel`](../experiments/target_panel/README.md). Full results:
[`results.json`](ARTIFACT_INDEX.md#artifact-a964f1669a63) (SHA256 `fb813dfe…1ac6`; v1 had identical numbers without commit provenance).

## Why

MELAUDIS has **3.17% trucks**. Always predicting car scores 49.19% macro-F1, above
every learned arm. With a fixed argmax boundary, macro-F1 there mostly records which
side of the boundary the target lands on. The H2 direct-path collapse was a boundary
displacement with near-chance ranking.

The metrics added here separate ranking from threshold placement:

| Metric | Reads | Constant-predictor reference |
|---|---|---|
| AUROC | ranking, threshold-free | 50% |
| Average precision (AP) | ranking weighted toward the rare class | the prevalence (3.17% MELAUDIS, 10.26% IDMT mean) |
| Truck recall at 5% car false-alarm rate | one ROC point. Its threshold is placed using target cars, so it is **descriptive, not deployable** | 5% |
| Balanced accuracy (BA) | the archived fixed decision rule | 50% |

Intervals are 95% paired percentile intervals from 10,000 draws:
- **Whole-group resampling:** the four conservative MELAUDIS groups, or the six IDMT
  held-out groups for the source row.
- **Selection resampling:** H1 seeds × folds, or the H2 banks.
- **Pairing:** draws are shared within each study, so contrasts are paired. Seed
  20261011; no draw was rejected.

With four uneven groups, coverage is uncertain and intervals are skewed. The largest
group holds 6,477 of 8,066 events. The 3-point practical margins in earlier protocols
were declared for macro-F1 and are **not** applied retroactively to AUROC.

## Panel

One row per dataset role. Rows are never pooled. Values are percentages; trainers are
the H1 real-data arm (190 cars + 190 trucks per fit, 30 fits).

| Dataset | Role | Task | Groups | Support (neg/pos) | AUROC | AP | Truck recall @5% car FPR | BA (fixed rule) |
|---|---|---|---|---|---|---|---|---|
| IDMT_V1 | source_within_domain | car/truck, BEATs768 IDMT-trained | 6 | 3902/511 | 71.49 [65.01, 77.07] | 31.09 [17.13, 44.89] | 23.67 [13.14, 33.76] | 65.87 [61.34, 70.14] |
| MELAUDIS_V1 | exposed_development | car/truck, BEATs768 IDMT-trained | 4 | 7810/256 | 61.56 [56.05, 73.32] | 5.68 [3.95, 16.22] | 11.88 [8.40, 21.76] | 57.38 [53.87, 64.29] |
| IDMT_V1 | source_within_domain | car/truck, MFCC26 IDMT-trained | 6 | 3902/511 | 80.22 [76.78, 83.67] | 37.14 [22.29, 50.64] | 30.38 [19.33, 39.78] | 71.00 [65.41, 75.33] |
| MELAUDIS_V1 | exposed_development | car/truck, MFCC26 IDMT-trained | 4 | 7810/256 | 64.59 [60.05, 77.57] | 6.31 [4.26, 19.03] | 13.70 [9.85, 26.95] | 58.47 [55.96, 63.82] |
| DCASE2024_T10_DEV | sealed_confirmation | car/cv proxy (single-vehicle minutes) | 6 | 1361/228 | sealed | sealed | sealed | sealed |
| MAVD_traffic | pending_audit | car/truck | — | —/— | — | — | — | — |
| ABVID_roadside_v1 | planned_collection | car/truck, camera-labelled | — | —/— | — | — | — | — |

DCASE 2024 Task 10 is admitted and sealed ([report](dataset_admission_dcase2024_t10.md)).
MAVD needs an overlap/provenance audit first, because P02's compiled test material
includes it and R0 scored that material. The roadside row is the collection plan below.

## H1 re-scored

| Representation / training arm | MELAUDIS AUROC | AP lift (AP ÷ prevalence) | Truck recall @5% car FPR | BA |
|---|---:|---:|---:|---:|
| BEATs768 / IDMT only | 61.56 [56.05, 73.32] | 1.79 | 11.88 | 57.38 |
| BEATs768 / procedural only | 64.15 [55.78, 73.41] | 2.22 | 16.56 | 58.31 |
| BEATs768 / AudioLDM only | 55.35 [43.28, 56.97] | 1.13 | 4.69 | 54.27 |
| BEATs768 / IDMT repeated | 61.14 [55.72, 72.69] | 1.75 | 11.59 | 57.18 |
| BEATs768 / IDMT + procedural | 65.45 [59.67, 77.71] | 2.15 | 14.54 | 59.83 |
| BEATs768 / IDMT + AudioLDM | 59.03 [52.77, 65.90] | 1.58 | 10.08 | 55.62 |
| MFCC26 / IDMT only | 64.59 [60.05, 77.57] | 1.99 | 13.70 | 58.47 |
| MFCC26 / procedural only | **35.89** [33.79, 42.11] | 0.72 | 2.42 | 40.35 |
| MFCC26 / AudioLDM only | 48.10 [45.03, 62.89] | 0.95 | 3.75 | 49.06 |
| MFCC26 / IDMT + procedural | 54.23 [50.64, 69.93] | 1.16 | 7.02 | 51.85 |
| MFCC26 / IDMT + AudioLDM | 58.13 [54.23, 72.90] | 1.26 | 6.20 | 56.47 |

Paired contrasts, in percentage points. The macro-F1 column is copied from the
[H1 report](H1_baseline_results.md):

| Contrast | BEATs macro-F1 | BEATs AUROC | MFCC macro-F1 | MFCC AUROC |
|---|---:|---:|---:|---:|
| IDMT within − MELAUDIS | +19.11 [11.16, 25.91] | +9.93 [−2.00, 16.24] | +24.88 [19.49, 34.03] | +15.63 [1.68, 21.39] |
| Procedural only − IDMT only | +11.82 [1.63, 23.04] | +2.59 [−8.48, 9.48] | +5.35 [−0.67, 14.86] | **−28.70** [−37.70, −24.30] |
| AudioLDM only − IDMT only | −10.49 [−24.49, −3.85] | −6.21 [−25.23, −0.28] | +5.41 [0.91, 18.65] | −16.49 [−20.46, −11.48] |
| IDMT + procedural − repeated | +2.74 [1.18, 4.95] | +4.31 [2.73, 7.40] | −9.01 [−12.30, −1.98] | −10.04 [−13.94, −5.63] |
| IDMT + AudioLDM − repeated | −0.75 [−7.00, 2.17] | −2.11 [−8.56, −0.14] | +8.33 [4.60, 20.79] | −6.15 [−8.96, −1.08] |

**What changes:**
- **Procedural only − IDMT only, BEATs:** H1's counterexample ("procedural-only beats
  real") does not survive as a ranking result. The +11.82 F1 gain becomes +2.59 AUROC
  with an interval crossing zero; the F1 gain came mostly from boundary placement
  under the prevalence shift.
- **Synthetic-only arms, MFCC:** the F1 "gains" reverse. Procedural-only MFCC ranks
  trucks *below* cars (AUROC 35.89%).
- **IDMT + AudioLDM − repeated, MFCC:** the +8.33 F1 mixing gain becomes a −6.15 AUROC
  loss.
- **IDMT within − MELAUDIS, BEATs:** the cross-dataset ranking gap is smaller and less
  certain than the F1 gap; the IDMT and MELAUDIS prevalences differ (10.26% and 3.17%).
  BA still shows a gap of +8.49 [1.09, 13.12] points.

**What holds:**
- **IDMT + procedural − repeated, BEATs:** remains positive: +4.31 [2.73, 7.40] AUROC.
  Its lower bound is below 3 points.
- **AudioLDM only − IDMT only, BEATs:** remains harmful.
- **Absolute performance:** stays weak. The best H1 arm reaches 65.45% AUROC and about
  2.2× prevalence in AP. Across H1 and H2, no arm exceeds 71.5% AUROC.

## H2 and the latency control re-scored

| Representation / cell | AUROC | AP lift | BA |
|---|---:|---:|---:|
| BEATs768 F00 (S0, direct path) | 50.68 [47.88, 60.34] | 1.01 | 50.12 |
| BEATs768 F01 (S0, + ground/air) | 68.96 [58.74, 71.07] | 2.12 | 62.99 |
| BEATs768 F10 (S1, direct path) | 46.55 [42.91, 54.50] | 0.89 | 50.03 |
| BEATs768 F11 (S1, + ground/air) | 66.96 [57.54, 70.14] | 1.99 | 58.31 |
| MFCC26 F00 | 71.48 [67.07, 83.62] | 2.72 | 50.00 |
| MFCC26 F01 | 70.00 [68.33, 76.93] | 2.10 | 50.00 |
| MFCC26 F10 | 69.81 [65.54, 81.76] | 2.45 | 50.00 |
| MFCC26 F11 | 68.68 [66.20, 75.61] | 2.00 | 49.96 |

| Contrast | Macro-F1 (earlier report) | AUROC (this analysis) |
|---|---:|---:|
| BEATs D = source − path (Q10 − Q01) | −34.90 [−43.56, −29.17] | −22.41 [−25.55, −10.28] |
| BEATs path marginal | +32.08 [24.77, 42.10] | +19.35 [6.70, 22.17] |
| BEATs source marginal | −2.82 [−4.94, 0.06] | −3.06 [−5.78, −1.30] |
| MFCC D | +0.01 [0.00, 0.02] | −0.19 [−8.38, 6.90] |
| BEATs latency only − attenuation only | +31.76 [24.77, 41.06] | +13.39 [8.42, 15.67] |
| BEATs both − latency only | +0.03 [−0.14, 0.21] | −0.01 [−0.09, 0.16] |
| MFCC latency only − attenuation only | — | +4.09 [−1.51, 6.85] |

These re-scored values reproduce the AUROCs already published in the H2 reports:
- direct-path BEATs at 48.6%, the mean of F00 and F10;
- MFCC at 69.61% with latency only and 65.51% with attenuation only.

That independently cross-checks the new metric code.

**Interpretation:**
- **BEATs:** the direction of the H2 result holds. The implemented path, specifically
  its filter latency, moves BEATs from chance ranking to about 68% AUROC. The effect
  size shrinks by roughly a third once threshold displacement is removed.
- **MFCC:** the earlier summary "MFCC nearly always predicts truck" hid usable ranking.
  Every MFCC cell ranks the real target at about 69–71% AUROC, as high as or higher
  than the best BEATs cell. MFCC's failure is boundary calibration, not separability,
  and neither intervention changes its ranking.
- **Reliability:** neither representation is reliable at any fixed rule.

## Per-group caution

Single-group AUROC for the H1 BEATs real arm is 73.0, 62.8, 59.5 and 88.6 for the
four MELAUDIS groups. The last value comes from **one truck** and carries no
information; for H2 F01 the same group gives 7.2. Per-group values are in
`results.json` and must not be read as group-level evidence.

## Reporting rule proposed for future studies

1. **Per-dataset panel rows.** Report each dataset role on its own row, as above.
   Exposed, sealed, pending and planned roles are explicit, and pooling is refused in
   code (`panel.validate`).
2. **Co-primary metrics.** Use macro-F1 (continuity with earlier reports), balanced
   accuracy and AUROC, with AP against prevalence. Treat recall at a fixed false-alarm
   rate as descriptive. A future frozen protocol must declare its own AUROC margin.
   This analysis does not choose one.
3. **Worst-group reporting.** Keep it, and flag any group that has fewer than about 10
   examples of a class.

## Step 4: own roadside collection (planned, not started)

The only target fully under project control. Minimum specification, for a separate
protocol and authorization:
- **Sessions:** at least 8 independent recording sessions across at least 4 sites and
  at least 2 dates. Session IDs are recorded at capture time and never inferred later.
- **Video labels:** frame-accurate pass-by times, class
  (car / van / truck / bus / motorcycle, with written definitions; truck by axle count),
  lane, direction and speed band. Vehicle-absent intervals are logged for detection work.
- **Sensors:** one fixed reference microphone model across all sessions plus one
  secondary sensor. Record a calibration tone, gain and position metadata, weather and
  road surface.
- **Support target:** at least 300 trucks overall and at least 20 per session, so
  whole-session intervals are informative.
- **Sealing:** sessions are assigned to a sealed confirmation partition at random,
  before any model is scored on them.
- **Privacy:** blur faces and plates, check local recording law, keep raw video outside
  the checkout, and keep only annotations in project storage.

## Limits

These are post hoc metrics on data that were already exposed, computed after the
earlier results were known. They change how earlier evidence should be read; they do
not upgrade it to confirmation. The always-car and always-truck references are 50%
AUROC and AP equal to prevalence. No AUROC decision rule existed when these studies
were frozen.

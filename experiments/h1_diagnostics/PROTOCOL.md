# IDMT-only source diagnostics

Protocol `h1_source_diagnostics_v1`, 2026-10-03. The user authorized the five proposed checks and an individual findings report plus downstream implications. This protocol is frozen before new feature extraction, fitting or diagnostic scoring. The completed H1 artifacts and settings remain unchanged.

## Scope, data and exposure

The only new data processing/evaluation is **IDMT real → IDMT real**, car versus truck. Use all 4,413 admitted SE_CH34 excerpts (3,902 cars/511 trucks), six connected site/date groups at three sites. Retain known source/duplicate/channel groups from admission v1.3.1. No MELAUDIS/synthetic waveform, target feature array, target model prediction or military recording enters any new operation. H1 source-only model/prediction members may be read for replay verification. Pretrained encoder exposure remains a qualification.

All IDMT groups were previously evaluated in H1. These are development diagnostics, not newly untouched confirmation. Five seeds `[42,123,456,789,1024]`. Reuse H1 outer group assignments and exact 190/class selections. New smaller/larger samples are nested prefixes within each class of the same deterministic group-round-robin ordering. The smallest five-group pool supports **359/class**; every four-group inner training pool supports at least 213 trucks, so 190/class is feasible throughout.

Save a source-only manifest, exact selections, all validation/test roles, input hashes, executable config, Git HEAD/dirty status, software versions, code snapshots and freeze timestamp. All outputs use new directories. Never change admission labels/exclusions in response to scores.

## Fixed model and observation

Use the same hash-verified official BEATs backbone and fixed MFCC settings as H1. Encoders remain frozen. Mean both stored sE8 channels. Resample with SciPy resample_poly, Kaiser beta 5, center crop, no amplitude normalization, padding, denoising, augmentation or new source generation. StandardScaler is fitted independently on each fit's training features. LogisticRegression uses the H1 solver, tolerance and stopping limit; only explicitly listed C values change. Fail on nonconvergence.

Three observation profiles: (a) control: native→8 kHz→16 kHz, center 2 s; (b) bandwidth/resampling route: native→16 kHz, center 2 s; (c) context: native→8 kHz→16 kHz, center 1 s. The shorter input is processed at its true length, with no zero/repeat padding, and the same token-mean pooling. Verify the variable-length helper matches the existing adapter for a 2 s artificial signal. Identical class/events/heads are used for paired comparisons. The full-band contrast changes the resampling route as well as the available upper spectrum; do not claim it isolates a single spectral mechanism. Longer-than-2 s context cannot be tested from these released excerpts.

## D1 — Training versus held-out groups and head regularization

For each representation, six outer groups × five selection seeds, fit the unchanged 190/class, core8_2s, C=1 baseline. Save training and held-out confusion/F1/recall, balanced accuracy, ROC-AUC, average precision, Brier and log loss. Compare balanced accuracy/AUC primarily when discussing fit gaps because training is balanced and evaluation is naturally imbalanced. A train/test F1 gap alone is not proof of overfitting.

Bounded regularization diagnostic: within each outer training pool, leave out each of its five groups for inner validation and fit on 190/class from the other four. Compare C=[.01,.1,1], using the same inner samples for every C. Select C by mean inner-group macro-F1 at threshold .5, ties to smaller C. Refit on the exact outer 190/class samples and evaluate the outer group once. Report all selected values and the paired difference against fixed C=1. Outer labels/features do not enter scaler fitting or C selection. Inner selection sees additional source validation labels; its data access differs from the original untuned H1 fit even though each final fit still has 190/class.

## D2 — Training-size learning curve

Same six folds, five seeds, core8_2s and representations. N/class=[25,50,100,190,300,359]. Each smaller class selection is a prefix of the larger selection; all group boundaries remain intact. Primary curve fixes C=1, preserving the H1 fitting rule. Secondary control sets C(N)=190/N. In installed scikit-learn 1.4.2, lbfgs uses mean-loss L2 strength 1/(C × sample count); this control keeps that strength equal to the 190/class reference. It does not freeze the training-estimated feature scales or create new recording groups.

Report training/held-out metrics for every budget, 359-minus-190 and 359-minus-25 paired contrasts, plus seed/group uncertainty. No best-N selection from the outer results. The larger-real-data curve is a diagnostic reference, not a replacement for H1's matched synthetic/real budget. A plateau within this range is not a universal sample-efficiency limit.

## D3 — Controlled preprocessing comparisons

At fixed N=190/class and C=1, compare native16_2s and core8_1s separately against core8_2s, with identical files and folds. Report absolute metrics, paired changes and every group. Do not combine the winning settings into an additional post-hoc arm. No profile is chosen from target scores. Full-band real results cannot be compared as though bandwidth matched the released 8 kHz procedural simulator.

## D4 — Per-class discrimination, decision threshold and calibration

Report baseline ROC-AUC (truck positive), average precision with each group's truck prevalence reference, precision/recall, F1, balanced accuracy, Brier, log loss and ten-bin equal-width truck-probability ECE. These diagnose different properties and should not be collapsed into a single success score.

Threshold-only control fixes C=1 and reuses its inner out-of-group predictions from D1. Select t in [.05,.10,...,.95] by mean inner-group macro-F1; ties prefer closest to .5, then smaller t. Apply that threshold to the unchanged outer C=1 model, predicting truck iff p>t (ties to car). Outer evaluation never selects a threshold. Report selected thresholds and paired F1/BA/class-recall changes; probabilities and therefore AUC, AP, Brier, ECE and log loss must remain identical. Threshold selection is not probability calibration. Do not combine nested C and threshold choices in this minimum pass.

## D5 — Group/location dependence

Show baseline group-level confusion, class support and the mean/range over five seeds; report worst class/group recall rather than only a pooled score. Summarize recorded direction, road-condition and posted speed-limit tokens without treating them as measured RPM, actual vehicle speed or weather.

Additional fixed control: leave an entire site out (both dates), train on 190/class from the other two sites, C=1, core8_2s, five seeds, both representations. The smallest remaining-site pool has 213 trucks. Evaluate each of the site's two groups separately and give each original group equal weight in the aggregate. Compare to the corresponding leave-one-date-group-out predictions. This changes which site is seen and reduces training-group diversity from five to four; it does not isolate a pure location effect. There are three sites, not six.

## Metrics, inference and stopping

Primary summary: mean macro-F1 over the six held-out groups and five selections, alongside balanced accuracy and per-class recall. Also report ROC-AUC/AP, probability metrics, pooled out-of-fold scores, constant-car/truck controls and individual group outcomes. Preserve natural evaluation priors.

95% paired percentile intervals: 10,000 draws, seed 314159, resample six whole groups and five selection IDs with replacement. Pair draws across all arms. Also report a **site-block sensitivity**: resample three sites, carrying both dates together, with paired selection resampling. No clip bootstrap, no independent-fold training claim. These are conditional-on-fits development intervals with uncertain coverage at six groups/three sites, not full retraining confidence or multiple-comparison-adjusted confirmation.

Use .03 macro-F1 as the previously proposed practical-effect reference, not a pass requirement. Report negative/ambiguous effects and class tradeoffs. Do not expand the grid after seeing results. No automatic threshold/head/preprocessing change is promoted into H1 or H2. The report must address each D1–D5 independently, then explain what remains uncertain and what should change in the next freeze.

## Runtime, artifacts and checks

Apple M3 Pro CPU, 18 GiB RAM, four Torch threads, one BLAS thread. Estimate 4–8 min for 4,413 excerpts × three profiles, 1–4 min for the bounded fixed/nested heads, and 1–3 min for verification/reporting; measure actual times. No downloads or GPU needed.

Artifacts: source-only frozen manifest/selection lock, 3-profile BEATs/MFCC cache and waveform hashes, every fit's train/evaluation IDs and fitted scaler/head, inner choices, outer probabilities, training/held-out/group metrics, paired/bootstrap samples, plots, independent verification, and `reports/H1_source_diagnostics.md`.

Tests must cover source-only access, group separation, nested sample prefixes, no outer contribution to selection, threshold tie rules, unchanged probability metrics under thresholding, and 1 s/2 s finite encoder output. Independently recompute results from saved probabilities and verify scaler means against training-only features. Check source-only baseline replay against H1's saved source prediction members and ensure previous artifact hashes remain unchanged. Computation checks do not certify unknown physical-vehicle identity or audible annotation correctness.

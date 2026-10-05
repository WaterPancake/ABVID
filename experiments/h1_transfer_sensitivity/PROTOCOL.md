# Frozen regularization transfer sensitivity

Protocol `h1_regularization_transfer_v1`, 2026-10-04. The user authorized this bounded
follow-up to the source-only diagnostic. Freeze exact existing model identities,
source selections, target manifest, feature-cache hashes and analysis before any
new candidate target inference. This is a development protocol fixed after the
original H1 target results were known, not a prospective untouched-target study.

## Question and exact contrast

Does the regularization selected entirely within IDMT improve IDMT → MELAUDIS
classification compared with the unchanged H1 C=1 head? BEATs768 is primary;
MFCC26 is the prespecified comparator. Evaluate both and retain negative findings.

Each representation has 6 source outer folds × 5 seeds (42,123,456,789,1024):

- `baseline`: exact diagnostic C=1 scaler/head, already verified to replay H1 exactly.
- `nested_C`: exact saved scaler/head whose C in {.01,.1,1} was selected by mean
  macro-F1 across five IDMT-only inner held-out groups. No new C selection occurs.

There are 120 arm/model evaluation records and 107 distinct fitted pipelines:
13 MFCC candidate cases already selected C=1 and share the baseline pipeline.
These are repeated model evaluations of one fixed target, not 120 datasets.

## Data, roles and fixed processing

Source pool: 4,413 admitted IDMT SE_CH34 car/truck excerpts in six conservative
site/date groups at three locations. Every outer fit reuses H1's exact 190 cars
and 190 trucks from five groups; the sixth is its held-out source evaluation.
For candidate selection, each inner fit used 190/class from four groups and the
remaining source group for validation. Candidate selection therefore had access
to additional IDMT validation labels compared with the fixed untuned reference.

Target: all 8,066 H1-admitted MELAUDIS FF exact `1V-Car`/`1V-Truck` excerpts
(7,810 cars, 256 trucks), four conservative connected provenance groups.
Use the exact H1 target IDs/order; do not resample a balanced target or change labels.
Motorcycle and all other categories remain excluded. No target group is used for
fitting, validation, threshold selection, feature normalization or calibration.
Unknown original sessions/vehicle IDs and pretrained-data overlap remain unknown.

Processing stays mean of stored channels, center two seconds, native → 8 kHz →
16 kHz, no amplitude normalization/padding/augmentation. Reuse the original H1
feature cache after verifying hashes. BEATs uses the previously pinned backbone
and token-mean 768-dimensional output; MFCC uses the pinned 26 mean/std features.
Preserve encoder provenance qualifications, including the checkpoint mirror's
unverified official checksum. No extraction, training or downloads are needed.

Reuse each training-only StandardScaler and L2 logistic head, lbfgs, tolerance 1e-6,
maximum 5,000 iterations, intercept, no class weights. Predict truck iff p(truck)>.5
(ties to car). Change neither bandwidth/window/budget nor thresholds, and do not
combine diagnostic settings. Synthetic banks are not evaluated or refitted.

## Metrics, uncertainty and interpretation

Primary target score: compute macro-F1 on the full natural-prior target separately
for every fitted head, then average six folds and five selections. Primary effect:
candidate minus baseline for BEATs. Retain the same convention as H1. Source score
is the mean of six held-out-group F1s across five selections. The source-minus-target
F1 gap is descriptive: class priors and aggregation differ between domains.

Report target/source F1, balanced accuracy, accuracy, class precision/recall/F1,
confusions, ROC-AUC and AP (truck positive), Brier, log loss and ten-bin truck
probability ECE. Report each target group's support, mean/range across models,
and both the mean per-model worst group/class recall and the absolute minimum.
Show always-car/truck controls without treating their F1 as a class-neutral chance level.

For confusion-derived scores, source-target F1 gaps, worst-group/class recall and
paired effects: 10,000 percentile draws, seed 314159. In each draw jointly resample
five selection indices and six source outer-fold indices, and independently four
whole target groups, with replacement. Use identical draws for both arms and
representations. Sum target group confusions before per-model scoring; reject and
count class-missing draws. Source scoring averages sampled held-out group metrics.
Worst recall in each draw takes the minimum class recall among sampled target groups
within each sampled model, then averages models. All estimates are conditional on
the fixed fits; folds have overlapping training sets, groups are conservative proxies,
and four uneven target groups give uncertain coverage. No clip bootstrap or independent
model-population claim. ROC-AUC/AP and probability metrics are descriptive means,
without group-resampled intervals or confirmatory claims in this minimum pass.

Use the existing .03 macro-F1 practical reference. A lower paired 95% bound above
zero supports a conditional positive gain; above .03 supports that practical gain.
An upper bound at/below zero contradicts a positive gain; below .03 rules out the
prespecified practical gain under this estimator; otherwise uncertainty remains.
“Improves without sacrificing class recall” additionally requires nonnegative
paired lower bounds for both class recalls and balanced accuracy (zero tolerance,
not an invented allowed recall loss). Report point decreases as tradeoffs even if
intervals cross zero. Inspect worst-group recall separately; no automatic promotion.
MFCC and nonprimary metrics are secondary, without multiplicity-adjusted claims.

Expected outcome: a modest target gain is plausible from the source diagnostic,
but no benefit or reversal is equally reportable. This does not test simulation
components, source dominance, independent new-site robustness or model recognition.
Do not expand the grid, retrain, recalibrate or pick the best target-performing C.

## Integrity, runtime and deliverables

Before inference: validate immutable H1/diagnostic lock hashes, saved source-only
selection/model hashes, exact train/inner/outer roles and file/group/duplicate/hash
separation from target. Verify each scaler mean/variance against its training-only
cached features. Freeze source and target metadata and every model reference.

After inference: independently recompute every score from saved probabilities,
replay probabilities from saved models, recompute nested C choices from saved inner
IDMT predictions, check baseline target replay against H1, check source replay against
diagnostics, validate aggregate/paired intervals, and verify protected artifacts unchanged.

Apple M3 Pro CPU, 18 GiB, one BLAS thread. Expected prediction/statistics/verification
runtime 1–3 minutes excluding documentation; no GPU. Measure wall time; peak RAM is
not assumed. Record Git HEAD/dirty state, code, configuration, versions, seeds,
dataset/model/cache hashes and exact partitions.

Save under `experiments/h1_transfer_sensitivity/`: frozen config/protocol/manifests,
model references, per-model source/target predictions and group metrics, bootstrap
indices/samples, summary and independent verification. Write
`reports/H1_regularization_transfer.md`, update the research log and only the roadmap/
proposal status. After reporting either outcome, prepare the existing source × path
factorial's remaining input requirements; do not implement or tune a simulator here.

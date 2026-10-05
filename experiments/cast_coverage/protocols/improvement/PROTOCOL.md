> Historical protocol/command reference. Commands retain their original paths
> and run from `ABVID_ARCHIVE_ROOT`, not from the reorganized code checkout.
> See the [experiment registry](../../../README.md).

# CAST coverage improvement — development protocol v1

User objective: **at least 80% held-group marginal coverage and at least 2.5%
relative reduction in descriptor distance against each control, in both car
and truck**. This goal authorizes a new version; the earlier 5% criterion and
failed v0 result remain unchanged historical evidence. Retain the additional
0.5–2 family spread checks and all numerical/provenance requirements.

## Access and preservation

Only IDMT H1 outer fold 0, seed 42's existing **380 training IDs** (190/class,
five conservative source groups) may calibrate the model. The first ablation
uses the same 50 parents as CAST v0. No resampling or replacement admissions.
Use the pinned source-only manifest, H1/source locks and exact ancestry/path
validation before reading any input. Keep MELAUDIS, mixed caches, supplied
synthetic banks, military and reserved audio excluded. Reuse the existing
environment read-only and keep all implementation/artifacts under this new
directory, plus a new final report. Earlier CAST code and runs are immutable.

Group `connected_4001f06f57cfeee7` (491 cars, 79 trucks) remains excluded from
calibration. It was exposed by H1 and the completed CAST check; future scores
are **development comparisons**, never untouched confirmation. Do not derive
parameters, EQ, scales, sample choices or selection rules from its audio or
descriptors. Open its stored descriptors only after a candidate is frozen.
Changing model hypotheses after the known v0 result is human development
exposure; explicitly retain this limitation even with disjoint bank ancestry.

## Experiment sequence

1. Freeze the new access/target contract and snapshot all parent source hashes.
2. Implement smooth noise spectral interpolation while keeping eight controls,
   the eight-harmonic model, five envelope knots, existing optimizer budget,
   preprocessing and loss. Existing noise weights represent point power density
   after division by the old band widths; interpolate log power at band centers
   and shape independently seeded white noise by its square root. Normalize
   component RMS exactly as before. This removes hard band discontinuities.
3. Pass deterministic, gradient, alias/boundary, provenance and synthetic
   recovery checks before fitting real observations. Retain the existing
   fixture thresholds. Fit the same 50 parents and compare checking loss and
   training-group coverage with the unchanged v0 result.
4. If source-side evidence supports the revision, extend fitting to all 380
   locked training parents. An independently versioned 16-point noise spectrum
   or nine-knot envelope may be considered only after examining training
   residuals. Preserve unsuccessful configurations and implementation repairs.
5. Select a bounded prior using leave-one-training-group-out development
   comparisons. Recompute every prior statistic and descriptor scale from
   each fold's allowed training parents. The candidate joint prior keeps full
   vectors and equal-probability source groups. Compare temperatures
   `[1.0, 1.1, 1.25, 1.5]` in transformed parameter coordinates, with hard
   renderer bounds, against matched prototype and independent-scalar-marginal
   controls. Temperature applies to the donor bank in all arms. Temperature 1
   is unchanged exemplar replay. Record all parents and transformations.
6. Freeze the selected renderer, bank, prior, metrics, seeds and complete
   generated schedule before the outer development comparison. Generate all
   outputs before reading held descriptors. Do not fit held waveforms.

The initial implementation begins with steps 1–3. Any later method revision
gets a new immutable method/fit snapshot; completed fits are resumable only
under exactly matching code, configuration and source hashes.

## Scoring and selection

Keep the existing four descriptor families, empirical marginal Wasserstein-1
score, 5th–95th percentile coverage, class-balanced aggregation, five seeds
`[42,123,456,789,1024]`, 50 outputs/class/seed/arm and original outer training-only
descriptor scales. Do not widen the percentile interval, discard difficult
clips, change family weighting, increase only joint sample counts, or weaken
controls to hit the goal. Preserve separate class and family results, tails,
spread, counts and all failure records.

For source-group prior selection, minimize the maximum class shortfall from
coverage 0.8 and gain 0.025, then mean W1, requiring family spread in [0.5,2].
Exact selection code is frozen before that experiment. If no candidate meets
source criteria, report the shortfall and use training residuals to justify a
new version rather than choosing with outer held results.

Completion requires the actual outer class-specific means to meet coverage
≥0.8 and relative W1 gain ≥0.025 against **both** matched controls, alongside
the retained spread/numerical/provenance checks. Recompute scores and replay
outputs under the pinned environment. Report a scientific failure if these
conditions are not reached; never turn a target into an assumed result.

The result can support this descriptor-level development objective only.
Classifier training, real-world transfer, physical source recovery and new
military milestones remain out of scope. New random seeds are not new real
ancestry. Uncertain physical quantities remain unknown.

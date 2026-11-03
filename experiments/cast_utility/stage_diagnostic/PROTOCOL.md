# Bounded CAST reconstruction and sampling diagnosis

Protocol `cast_stage_diagnosis_v1`, 2026-10-25, authorized by the user's request
to run the bounded CAST diagnosis. Frozen before new rendering/feature scores.
This is source-training, post hoc engineering evidence, not held evaluation,
classification utility, physical source recovery, or independent confirmation.

## Fixed population and access

Use **only CAST-5A role `utility_s42_0`**: its exact 380 training parents,
190 car and 190 truck, from five conservative IDMT_V1 CH34 groups. Reuse the
existing audited smooth8/v7/v8/v9/v10/v11 parameter records and flags. Validate
the 5A design, pinned 4A context, exact source metadata, source-role guards,
parent bundles and known provenance links before use. Session/vehicle metadata
remain unknown where originally unknown. No parent replacement or flag exclusion.

Raw access is limited to those 380 IDMT training sources. Original waveform
preprocessing is replayed and its float32 shape hash must equal the existing
5A real-feature cache's waveform hash. Cached originals, ordinary augmentation
and existing CAST population features may be read only for this same role.
No held waveform/feature rows are selected; no MELAUDIS, DCASE, military,
MVD, other target, or new dataset audio is accessed. Source-only cache containers
may be decompressed to select allowlisted rows; no held rows are used in numerical
calculations or exported.

## Primary comparison: all 380 parents

1. **Original:** existing frozen BEATs768/MFCC26 real features.
2. **Smooth8:** regenerate both historical checking realizations from the
   unchanged per-parent fitted controls and renderer, with shared fit phase.
   Both raw checking arrays must exactly match independently anchored original
   `components.npz` arrays. No optimizer is invoked.
3. **Calibrated v11:** regenerate both historical checking realizations from
   the final calibrated same-parent controls, using ResolutionRenderer. Do not
   reuse later saved v9/v11 WAVs whose independent historical hashes were absent.
4. **V11 fresh randomness:** unchanged final controls with fresh deterministic
   phase/noise namespaces `cast_stage_diagnosis_v1:{file_id}:fresh:{0,1}`,
   `sampling=True`, purpose `sampling`. The two draws share controls, but each
   has its own phase/noise seed. This separates checking-stream dependence from
   population parameter changes. These draws never enter a training/coverage bank.
5. **Population:** the role's existing 380 CAST-5A v15 feature rows, unchanged.
   These are unconditional and have no one-to-one parent correspondence.
6. **Conventional augmentation:** the role's existing 380 RA feature rows,
   reordered to their original parents, as an alignment reference.

All generated inputs use identical DC removal/unit RMS in float64 then float32,
matching CAST-5A's input profile. Existing real/RA/population features retain
their original preprocessing. No amplitude, bandwidth, head or feature choice
is tuned. Checking realizations are two sensitivity views, not independent samples.

## Secondary calibration localization: exactly 50 parents

Before new numerical scores, choose the lexicographically first five training
file IDs in each class and source group (5 groups × 2 classes × 5 = 50).
Render both checking realizations at v7, v8, v9 and v10. Compare all six stages
on this exact subset, also taking smooth8/v11 from the primary arrays. Use
SmoothRenderer at v7/v8/v9, its temporal41 configuration at v10, and
ResolutionRenderer at v11. This is a fixed diagnostic subset, not a class/group
population estimate. All original flags and group/class sample counts remain.

Total bounded new render count: **2,680 two-second observations** (2,280 primary
and 400 secondary). No new population is fitted or sampled. In addition,
replay originals for 380 hash checks and extract only **10 original controls**
(lexicographically first ID per class/group) to verify the baseline adapter.

## Frozen features, heads and diagnostics

Use the existing frozen BEATs checkpoint/upstream code and H1 MFCC26 recipe,
the existing real-only and CAST-only heads for `utility_s42_0`, and each head's
saved scaler. No head, scaler, threshold, prior or waveform parameter is fitted.
BEATs CPU extraction retains H1 batch size 8 and four Torch threads. CAST
rendering uses its exact one-thread deterministic environment. The 10 controls
must agree with cached real features at `atol=rtol=1e-5`; report actual errors.
This is an implementation tolerance, not scientific acceptance.

Report each realization separately and their mean/range:

- Frozen-head AUROC, balanced accuracy, car recall, truck recall, confusion,
  signed class margin and label agreement with the same head's original-parent
  predictions. Population has no paired-original agreement/margin change.
- Geometry after the R head's training-only StandardScaler: class-direction
  cosine, class separation ratio, per-class centroid displacement divided by
  real class separation, within-class RMS spread ratio; paired standardized
  real-feature RMS error for parent reconstructions/RA only.
- All five source groups and class counts; every parent, checking realization,
  flags, wave/feature hash and seed namespace retained. Two realized waveforms
  of one recording never count as additional independent recordings.

Use descriptive paired differences on these same training parents. Do not
bootstrap clips or label the five conservative groups independent sessions.
No p-value, new success threshold, model selection, or transfer claim is made.
Intermediate calibration changes co-vary; these comparisons localize the
implemented pipeline's changes, not isolated physical causal mechanisms.

Interpretation: a large gap already at smooth8 supports a renderer/fitting-
objective limitation; a later deterioration localizes additional loss to that
stage. Check-to-fresh deterioration implicates realization dependence jointly
with the changed phase/noise sampling recipe. A remaining fresh-parent-to-
population difference includes the prior, transformed controls, finite sample
composition and their interactions; it does not uniquely identify one block.

## Freeze, preservation and stopping

Validate canonical H1 and grouped JSON configs through `abvid.protocol` and
resolve every storage location through `abvid.paths`. Before numerical work,
publish a create-only external design bundle containing the protocol, exact
parent/control/subset IDs, per-stage parameters/flags and ancestry, source/model/
cache hashes, implementation snapshots, environments and initial Git state.
All generated waveforms/features/results stay in a new external artifact run.
Historical bundles, pinned runners, configs, tests, locks and harness baselines
remain unchanged. Small reports may be added to the checkout.

Missing parents, invalid ancestry, hash/environment drift, nonfinite results,
failed smooth8 exact replay or failed original-feature controls stop the run.
Retain an explicit failure record; do not omit/replace parents, change tolerances,
or refit. Implementation corrections require a new pinned run revision preserving
the failed attempt. Verify deterministic waveform replay, immutable input hashes,
the full numerical score recomputation and independent metric checks before delivery.

Commands use `python -B -m abvid.cast.stage_diagnostic STAGE --run artifact:...`.
Stages are `freeze`, `render`, `features`, `score`, `verify`. Freeze/render/verify
use the CAST readiness environment; features/score use `.venv-baseline`.

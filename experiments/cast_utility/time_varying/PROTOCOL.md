# CAST time-varying spectral weights: reconstruction check

Protocol `cast_tv_weights_v1`, written 2026-10-31 before any fit, render or
score. User authorization: "Let's try the time varying renderer weights."

## Question

The smooth8 renderer keeps its 8 harmonic weights, 8 noise-band weights and the
harmonic/noise mix fixed for a whole 2 s clip; only pitch (3 knots) and overall
loudness (5 knots) change over time. The stage diagnosis found that most of the
real classifier's class structure is lost at reconstruction (BEATs AUROC on the
380 training parents 100 → 68.16), and the spectrogram comparison found that the
fit flattens the engine band's frame-to-frame variation most in trucks BEATs no
longer recognizes.

Does letting those three spectral controls vary over the clip preserve the
real-trained classifier's class structure better at reconstruction?

## Fixed design

- **Population:** exactly the stage diagnosis's 380 training parents of CAST-5A
  role `utility_s42_0` (190 cars, 190 trucks, admitted IDMT_V1, five conservative
  groups). Training data only: no held group, target or sealed data.
- **Originals:** rebuilt from source and required to match the CAST-5A cached
  wave hashes.
- **Arms:**
  - `smooth8`: the existing historical fits, rendered on check realizations 0
    and 1 (`artifact:cast/diagnostics/cast_stage_diagnosis_20261025_v1`). Not refit.
  - `tv3`: new fits with `src/abvid/cast/time_varying.py`. Harmonic weights,
    noise-band weights and harmonic fraction each get 3 knots, linearly
    interpolated (59 controls instead of 25). Everything else is unchanged from
    `smooth8_v1`: the same initialization (every knot starts at the smooth8
    initial value), objective, optimizer, 300 steps, 4 starts, winner rule,
    seeds, and check noise streams.
- **Gate:** with equal knots the new renderer must reproduce `SmoothRenderer`
  (maximum absolute difference ≤ 1e-5) on every parent's initial controls,
  before any fit result is used.
- **Classifiers:** the saved real-only (R) and CAST-only (C) heads for this role,
  frozen BEATs768 and MFCC26, the original train-fitted scalers and decision
  rule. No new heads, scalers or tuning.

## Measures

- **Primary:** BEATs R-head AUROC on the 380 reconstructions, mean of check
  realizations 0 and 1, `tv3 − smooth8` (smooth8 = 68.16).
- **Decision rule, fixed now:**
  - **Promising:** primary ≥ +5.0 points, and BEATs truck recall also rises.
  - **No meaningful change:** primary within ±2.0 points.
  - **Otherwise:** small or ambiguous; report as such.
- **Secondary (descriptive):**
  - BA and truck/car recall.
  - MFCC.
  - C-head scores.
  - Per-group results.
  - Per-clip verdict flips against the originals.
  - Multi-resolution check loss per parent, for both arms against the same
    targets.
  - Frame-to-frame variation kept at 100–400 Hz in BEATs's filterbank, for
    both arms, in the spectrogram groups.

## Limits stated in advance

- **Training-data diagnostic:** the R head is overfit to these parents (100% on
  the originals). Better alignment here is not better recognition of held
  recordings and is not a utility or transfer result; that needs a separately
  frozen utility protocol.
- **Extra capacity:** 59 vs 25 controls can also copy recording-specific
  detail. A gain does not show that the added detail is vehicle information.
- **One role and five groups:** no confidence interval or independence claim.
  Fitted controls remain effective-observation parameters, not physical
  vehicle states.

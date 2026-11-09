# CAST reconstruction on held recordings

Protocol `cast_held_reconstruction_v1`. **Frozen 2026-11-01 on user approval
("run it")**, before any fit, render or score. Drafted 2026-10-31. The run
records this file's SHA256 and refuses to continue if it changes.

## Question

When CAST fits and re-renders a clip, does it lose class information that
generalizes to new recordings, or only detail a classifier memorized?

The stage diagnosis scored CAST re-renders with a classifier that was trained on
those same clips. Its AUROC fell from 100 to 68 (BEATs), but that classifier is
overfit (100% on its training clips) and may rely on recording fingerprints.
The time-varying renderer recovered only 3 of those points. Scoring
re-renders of clips the classifier never saw separates the two explanations:
- **If CAST keeps what generalizes:** held re-renders should score about as
  well as the held originals (about 71 AUROC on average).
- **If CAST loses what generalizes:** held re-renders should also fall sharply.

## Fixed design

- **Clips:** all 4,413 admitted IDMT_V1 car/truck clips (511 trucks), using
  the same core8 2 s profile as before. Originals are rebuilt from source and
  must match the CAST-5A cached wave hashes.
- **Roles and classifiers:** the 30 CAST-5A roles (6 folds × 5 training
  selections) and their saved real-only (R) and CAST-only (C) heads. Encoders
  are frozen BEATs768 and MFCC26, with the original train-fitted scalers and
  decision rule. No new heads, scalers or tuning.
- **Fits:**
  - **smooth8:** fit every clip once with the packaged smooth8 fitter and the
    `smooth8_v1` settings, then render check realizations 0 and 1. A fit
    depends only on the clip, not on the role, so every role's held group and
    its training parents are both covered.
  - **tv3 (secondary):** fit the fold-0 held group (570 clips) with the
    time-varying renderer from `cast_tv_weights_v1`, under the same rules.
- **Two sides per role:**
  - **Held:** the role's held group (one whole connected group), which its
    heads never saw.
  - **Train:** the role's 380 training parents, which its heads were fitted on.

## Gates (all must pass before any new score is used)

1. **Originals:** every rebuilt original matches its CAST-5A cached wave hash.
2. **Held originals:** R and C held metrics recomputed from the cached original
   features reproduce the CAST-5A per-head results exactly.
3. **Fitter replay:** fresh smooth8 renders of role `utility_s42_0`'s 380
   training parents match the stage diagnosis's smooth8 check waves (maximum
   absolute difference ≤ 1e-6).
4. **Renderer parity (tv3):** the equal-knot parity check is ≤ 1e-5, as in
   `cast_tv_weights_v1`.

## Measures

- **Primary:** BEATs R-head held AUROC, re-render minus original
  (`D_held`).
  - Re-render AUROC is the mean of check realizations 0 and 1.
  - Average over the 30 roles, with folds weighted equally.
  - 95% interval from the CAST-5A paired fold × selection bootstrap
    (`abvid.learning.utility_stats`, 10,000 draws, seed 20261101), plus the
    site-block sensitivity.
- **Decision rule, fixed now:**
  - **Keeps the generalizable cue:** the interval's lower bound is above −3
    points.
  - **Loses the generalizable cue:** the interval's upper bound is below −10
    points.
  - **Otherwise:** partial; report the estimate and interval.
- **Secondary (descriptive):**
  - **Retained above-chance signal:** (AUROC − 50) / (AUROC_original − 50), held
    side vs. train side.
  - **Memorization gap:** train-side drop minus held-side drop, per role.
  - MFCC.
  - Balanced accuracy, truck and car recall.
  - Per-fold results, flagging the two folds with only 10 and 14 trucks.
  - C heads.
  - Per-clip verdict flips.
  - tv3 vs. smooth8 on the fold-0 held group.

## What each outcome would mean

| Outcome | Reading | Next step it supports |
|---|---|---|
| Keeps | The 100 → 68 training-side drop was mostly memorized recording detail. Re-renders keep what generalizes | Stop renderer work. CAST's utility failure lies in population sampling or in what the real data can teach |
| Loses | Fitting and re-rendering destroys class information that generalizes | Renderer and objective work is justified, including a held-side re-test of tv3 |
| Partial | Both effects are present | Size them with the memorization gap before choosing |

## Data rules

- **No new data roles:** held IDMT groups are already-exposed development data,
  and no MELAUDIS, DCASE, MVD or military data is used.
- **Fitting held clips:** held clips are fitted only to be scored by heads
  that did not train on them.
- **Where reconstructions may go:** these conditional reconstructions never
  enter a training set, synthetic bank, population or coverage evaluation.
- **Grouping:** sessions and physical vehicles stay `unknown`; grouping uses
  the existing conservative connected groups.

## Limits stated in advance

- **Few groups:** six groups at three sites; intervals are conditional on these
  groups and are not independent-session evidence.
- **Not a utility test:** preserving the real head's held decisions does not
  mean CAST-generated training data would help. That needs a separate utility
  protocol.
- **Linear heads only:** frozen encoders with linear heads; other
  representations may behave differently.

## Cost

About 5,000 fits at about 16 s each on 8 workers (≈ 2.8 h), plus features for
about 10,000 renders (≈ 40 min). About 1.3 GB of renders, stored outside the
checkout.

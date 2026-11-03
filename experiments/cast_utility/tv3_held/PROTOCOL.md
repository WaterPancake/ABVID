# Time-varying renderer on held recordings

Protocol `cast_tv3_held_v1`. **Frozen 2026-11-02 on user approval ("run it")**,
before any new fit or score. Drafted 2026-11-02. The run records this file's
SHA256 and refuses to continue if it changes.

## Question

Does letting CAST's spectral balance change during a clip (tv3) win back class
information that generalizes to new recordings?

`cast_held_reconstruction_v1` found that smooth8 re-renders cost the real-trained
BEATs head 9.9 AUROC points on held groups (71.3 → 61.4). tv3 was fitted only on
held group 0, where it won back 3.6 of 13.5 points. That was one group with no
interval.

## Fixed design

- **Clips:** the same 4,413 admitted IDMT_V1 car/truck clips (511 trucks) and
  core8 2 s profile. Originals are rebuilt from source and must match the
  CAST-5A cached wave hashes.
- **New tv3 fits:** fit the 3,843 clips outside held group 0. Use the tv3
  settings and fitting code recorded in `cast_held_reconstruction_v1`:
  - 3 knots for the harmonic weights, noise-band weights and mix;
  - the same fitter, starts, budget and noise streams as smooth8.

  Render check realizations 0 and 1.
- **Reused from `cast_held_reconstruction_v1`:** its 570 fold-0 tv3 fits and
  its smooth8 fits and features for all clips.
- **Classifiers:** the 30 CAST-5A roles (6 folds × 5 training selections) and
  their saved heads. Encoders stay frozen BEATs768 and MFCC26, with the
  original train-fitted scalers and decision rule. No new heads, scalers or
  tuning.
- **Comparison:** on each role's held group, the tv3 re-render vs. the smooth8
  re-render of the same clips, scored by the same head.

## Gates (all must pass before any new score is used)

1. **Originals:** every rebuilt original matches its CAST-5A cached wave hash.
2. **Reused inputs:** every reused bundle verifies. Metrics recomputed from
   the reused features must reproduce the `cast_held_reconstruction_v1`
   per-head results exactly. This covers original, smooth8 and the fold-0 tv3
   results.
3. **Fitter replay:** refitting 8 of the reused fold-0 clips with tv3
   reproduces their stored waves (maximum absolute difference ≤ 1e-6).
4. **Renderer parity:** the equal-knot parity check is ≤ 1e-5 for every new fit.
5. **Features:** re-extracted features of 16 reused fold-0 tv3 renders match
   the stored ones (≤ 1e-5).

## Measures

- **Primary:** BEATs R-head held AUROC, tv3 minus smooth8 (`G_held`), on held
  groups 1–5. Group 0 does not count toward the decision because its tv3
  result has already been seen.
  - Each re-render's AUROC is the mean over check realizations 0 and 1.
  - Average over the 25 roles of folds 1–5, with folds weighted equally.
  - The 95% interval comes from the paired fold × selection bootstrap
    (`abvid.learning.utility_stats`, 10,000 draws, seed 20261102). Report the
    site-block sensitivity alongside it.
- **Decision rule, fixed now:**
  - **Helps:** the interval's lower bound is above 0 and the estimate is at
    least +3 points (about a third of the smooth8 drop).
  - **Not enough:** the interval's upper bound is below +3 points.
  - **Otherwise:** ambiguous; report the estimate and interval.
- **Secondary (descriptive):**
  - **All groups:** the same contrast on all six groups.
  - **Remaining drop:** tv3 minus original on held groups, and the retained
    above-chance signal, (AUROC − 50) / (AUROC_original − 50).
  - **Training side:** tv3 vs. smooth8 on each role's 380 training parents,
    and tv3's memorization gap.
  - **Other measures:**
    - MFCC;
    - balanced accuracy, AP, macro-F1, truck and car recall;
    - per fold, flagging folds 2 and 4 (14 and 10 trucks);
    - fit and check loss per clip, tv3 vs. smooth8;
    - per-clip verdict changes vs. smooth8.
  - **C heads:** scored for completeness but not interpreted. They were trained
    on smooth8 samples and break on tv3 renders.

## What each outcome would mean

| Outcome | Reading | Next step it supports |
|---|---|---|
| Helps | Fixed spectral weights were a real cause of the lost generalizable cue | Use tv3 as the renderer in any future CAST population or utility protocol |
| Not enough | Fixed weights are a minor part of the loss | Stop extending this renderer's time variation; look for the loss elsewhere, or move to population and data work |
| Ambiguous | Can't tell at this precision | Report only; no renderer decision |

## Data rules

- **No new data roles:** held IDMT groups are already-exposed development data,
  and no MELAUDIS, DCASE, MVD or military data is used.
- **Fitting held clips:** held clips are fitted only to be scored by heads that
  did not train on them.
- **Where reconstructions may go:** these conditional reconstructions never
  enter a training set, synthetic bank, population or coverage evaluation.
- **Grouping:** sessions and physical vehicles stay `unknown`; grouping uses
  the existing conservative connected groups.

## Limits stated in advance

- **Few groups:** five decision groups at three sites. The interval is
  conditional on these groups and is not independent-session evidence.
- **Not a utility test:**
  - **What it shows:** whether tv3, which uses 59 controls instead of 25,
    reproduces a given clip more faithfully.
  - **What it doesn't show:** that tv3 populations would make better
    training data. That needs new populations, new C heads and a separate
    protocol.
- **8 kHz route:** this tests the weak 8 kHz recipe (real-head ceiling about
  71), not the B1 recipe.
- **Linear heads only:** frozen encoders with linear heads; other
  representations may behave differently.

## Cost

- **Fitting:** about 6 hours for 3,843 fits on 8 workers. The 570 fold-0 tv3
  fits took 54 minutes.
- **Features and scoring:** a few minutes each.
- **Storage:** about 1 GB of renders, outside the checkout.

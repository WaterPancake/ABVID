# B1 reference on MELAUDIS

Protocol `b1_melaudis_v1`. **Frozen 2026-11-04 on user approval ("run it")**,
before any MELAUDIS feature was extracted for the B1 classifiers or any MELAUDIS
score was computed. Drafted 2026-11-03. The run records this file's SHA256 and
refuses to continue if it changes.

## Question

Does the B1 reference recipe (`b1_reference_v1`), trained on IDMT only, rank
MELAUDIS trucks above cars better than H1's recipe did?

Within IDMT, B1 raised held-group AUROC by 13–15 points over H1. That recipe was
chosen within IDMT, so the first question outside IDMT is whether the gain
carries over to another dataset.

**MELAUDIS is not untouched.** H1 already scored IDMT-trained classifiers on
it. BEATs reached AUROC 61.6, MFCC 64.6
([target panel](../../reports/archive/target_panel.md)). The result is exposed
development evidence. The B1 recipe was chosen without MELAUDIS, and nothing
here is tuned on it.

## Fixed design

- **Classifiers:** the two frozen all-IDMT B1 heads from
  `artifact:baseline_strength/b1_reference_20261103_v1`:
  - BEATs layer mean, C 0.001, SHA256 `ff49f793…`;
  - MFCC26, C 1, SHA256 `0d770904…`.

  Neither is refitted, and the threshold is not changed: argmax at 0.5.
- **Target:** all 8,066 H1-admitted MELAUDIS_V1 car/truck events (7,810 cars,
  256 trucks) in the four conservative groups. Supports are 626/36, 468/32,
  6,290/187 and 426/1.
- **Audio:**
  - **Source:** members are read from the original, read-only
    `MELAUDIS_Vehicles.rar` (SHA256 `e9e9af45…`) into a temporary work folder.
    Each member must match its admission hash.
  - **Route:** the B1 full-band route (channel mean, 48 kHz → 16 kHz, centred
    2 s). The events are exactly 2 s long, so nothing is cropped.
  - **Clean-up:** the extracted copies are deleted after feature extraction;
    only a hash manifest is kept.
- **Features:** the same extraction as B1 (BEATs layer mean, MFCC26).
- **Comparison arm:** H1's saved MELAUDIS predictions from its IDMT-only real
  arm: 5 selections × 6 folds = 30 heads per representation, on the 8 kHz
  route with 190 per class and C = 1. These are already exposed. Only their
  saved predictions are rescored; no H1 model is loaded.

## Gates (all must pass before any MELAUDIS score is used)

1. **Inputs:** each of these matches its recorded SHA256:
   - the B1 reference results and both head files;
   - H1's `admitted.jsonl`, against the H1 lock;
   - the MELAUDIS RAR;
   - each H1 prediction file, against its model record;
   - the target-panel results.
2. **Audio:** every extracted member matches its admission `source_file_sha256`
   and `decoded_audio_sha256`.
3. **Feature path:** the extraction code reproduces B1's cached features for
   the first 16 IDMT clips (≤ 1e-5).
4. **Scorer replay:** rescoring H1's real-arm predictions reproduces the
   target panel's published MELAUDIS point estimates exactly, overall and per
   group.
5. **Heads:** the frozen heads, applied to B1's cached IDMT features,
   reproduce their recorded training AUROC exactly.

## Measures

- **Primary:** MELAUDIS AUROC for BEATs, B1 head minus H1 real arm.
  - **Pooling:** AUROC is pooled over events, as in the target panel. The H1
    arm averages its 30 heads.
  - **Interval:** a 95% paired percentile bootstrap. It resamples whole
    MELAUDIS groups, redrawing any draw that lacks a class, and it resamples
    H1's selections and folds. 10,000 draws, PCG64 seed 20261104,
    `abvid.evaluation.ranking`.
- **Decision rule, fixed now:**
  - **Positive:** the interval's lower bound is above 0. B1 then counts as
    positive development evidence and becomes a candidate for the one-time
    DCASE pass. That pass still needs an overlap audit, its own frozen
    protocol and your written authorization.
  - **No practical gain:** the interval's upper bound is below +3 points.
  - **Otherwise:** ambiguous; report the estimate and interval.
- **Secondary (descriptive):**
  - **MFCC:** the same contrast.
  - **For both arms and representations:**
    - AUROC, AP and AP lift (AP ÷ prevalence);
    - truck recall at 5% car false-positive rate;
    - balanced accuracy, macro-F1, truck and car recall at the fixed rule;
    - per group, flagging the 1-truck group.
  - **Dataset gap:** B1's IDMT held-group AUROC (84.9 BEATs, 86.7 MFCC)
    minus its MELAUDIS AUROC. The heads differ: IDMT uses the per-fold heads
    and MELAUDIS the all-IDMT head.

## What each outcome would mean

| Outcome | Reading | Next step it supports |
|---|---|---|
| Positive | B1's gain is not only an IDMT effect | Overlap audit and a DCASE protocol for your decision |
| No practical gain | The gain stayed in IDMT; site/dataset shift dominates | Data work (MVD admission, roadside collection) before more modeling |
| Ambiguous | Four groups can't resolve it | Report only; MVD admission becomes the way to get more groups |

## Data rules

- **No other data:** no DCASE, MVD, synthetic or military data is read.
- **MELAUDIS role:** target-only development evaluation. It is never used for
  fitting, selection, thresholds or normalization.
- **Grouping:** sessions stay `unknown`; grouping uses the existing
  conservative connected groups.

## Limits stated in advance

- **Four uneven groups:** one group holds 6,477 of the 8,066 events, and one
  has a single truck. Intervals are conditional on these groups and coarse.
- **Not a clean recipe contrast:** the B1 heads train on all six IDMT groups;
  each H1 head trains on five groups and 190 per class. The contrast compares
  recipes as used, not one factor at a time.
- **Already exposed:** H1's MELAUDIS results were seen before this protocol,
  so a positive result is development evidence, not confirmation.
- **Fixed threshold:** at 3.2% trucks, macro-F1 at the fixed rule mostly
  measures threshold placement. The ranking metrics are the ones to read.

## Cost

- **Audio:** about 3 GB extracted temporarily, then deleted.
- **Features:** about 5 minutes of BEATs and MFCC extraction.
- **Scoring:** seconds.
- **Output:** create-only under `artifact:baseline_strength/b1_melaudis_20261103_v1/`.

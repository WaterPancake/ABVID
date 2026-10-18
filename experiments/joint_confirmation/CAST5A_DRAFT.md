# CAST-5A executable protocol: DRAFT, NOT FROZEN

Draft 2026-10-11 for the [CAST-5A stage](../../CAST_ROADMAP.md#cast-5a-test-classification-utility-within-idmt).
It fills in the executable choices that the roadmap leaves open, using proposed
defaults. Items marked **DECISION** need the user before freezing. Nothing here has
been run.

## Population and roles

- Six conservative IDMT connected-group folds and five H1 training selections
  `[42, 123, 456, 789, 1024]`, with 190 unique real parents per class per
  fold/selection. These are the exact H1 lock selections. The three-site blocks are
  a sensitivity analysis.
- **Parent fits:** the union of training parents is **3,245** recordings. CAST-4A
  (running since 2026-10-11 15:59) fits 1,864 of them, and **1,496 more** are needed
  for 5A. The same four-start smooth8 + v7–v11 chain is reused, and parents are
  reused only within valid training roles.
- **Held-out data:** each fold's held IDMT group is opened only after that fold's
  heads are saved and hashed. Held groups were exposed in H1, so results are
  development evidence.

## Input profile (shape-only, common to all arms)

H1 `core8_2s` route (channel mean, native → 8 kHz → 16 kHz, centred 2 s), then DC
removal and unit-RMS normalization for real, augmented and generated audio alike.
Observations with RMS below 1e-7 are rejected and counted, as in the CAST pilot.
Because this profile differs from H1, every control is recomputed and no historical
H1 score is reused.

## Arms (rows per class per head)

| Arm | Rows/class | Definition |
|---|---:|---|
| R | 190 | real parents |
| RR | 380 | the same parents twice |
| RA | 190 + 190 | real plus one conventional augmentation per parent (below) |
| RU | 190 + 190 | real plus an unfitted same-renderer prior (below) |
| RC | 190 + 190 | real plus 190 CAST v15 samples from this fold's training-only population |
| U | 190 | unfitted renderer only |
| C | 190 | CAST v15 only |

That is 7 arms × 2 representations (frozen BEATs768 primary, MFCC26) × 6 folds ×
5 selections = **420 heads**. Each uses a train-only StandardScaler and L2 logistic
regression with C = 1, the H1 convergence settings and a fixed argmax rule.
Generated and augmented feature banks are shared between single-origin and mixed arms.

**RA, the conventional augmentation (default accepted 2026-10-11).** One augmented
copy per parent:
- **EQ:** three random peaking filters, centre log-uniform 100–3,000 Hz, Q
  uniform 0.7–2, gain uniform ±6 dB.
- **Additive noise:** IDMT background recordings (`-BG` files) at an SNR uniform
  10–30 dB. Backgrounds are drawn only from the fold's training groups, with
  ancestry recorded, and never from a held group.

Scalar gain is omitted because the shape-only profile removes it. Augmentation seeds
are derived from the selection seed and parent ID.

**RU, the unfitted prior (default accepted 2026-10-11).** The same CAST renderer is
sampled uniformly within the fitter's declared parameter box. Its only class
information is firing-frequency ranges declared before use and taken from literature
(paper ID, page and section to be cited). It uses no fitted IDMT statistics. A
class-independent version is a negative control only.

## Contrasts and decisions

- **Primary:** BEATs RC − RA paired macro-F1, 3-point practical margin, paired
  whole-group/selection bootstrap (10,000 draws).
  - A lower 95% bound above 3 supports the gain.
  - An upper bound below 3 rules it out.
  - A crossing interval is inconclusive.
- **Required supporting contrasts:** RC − RR, RC − R, RC − RU and C − U.
- **Also reported:** AUROC, AP against prevalence, balanced accuracy, both recalls
  and worst-group recall. Flag groups with fewer than 15 trucks (two IDMT groups
  hold 10 and 14).
- **Precision:** from the [precision planning](../../reports/CAST5A_precision_planning.md),
  a true gain of about 5.6–6.6 points is needed for an 80% chance of clearing the
  margin. A true 3-point gain will probably be inconclusive. Accepted by the user
  2026-10-11; the design is unchanged.
- **Class-recall regression limit (accepted 2026-10-11):** RC is not called better if
  its mean truck recall falls more than 5 points below RA.

## Outcome for the joint batch

- **RC beats RA** (lower bound above 3): freeze the RC procedure and run CAST-5B on
  exposed MELAUDIS, then carry R, RA and RC heads to the joint confirmation.
- **Otherwise:** CAST either enters the joint batch only as a pre-declared secondary,
  or is dropped before the joint freeze. Decide before the freeze, not after access.

## Costs (estimates, to be measured on the first fold)

- 1,496 additional fits at about 5.5 per minute with 4 workers: roughly 4.5 hours.
- 420 linear heads: minutes.
- BEATs feature extraction for the generated and augmented banks: roughly 15–30 minutes
  on CPU, from the H1 rate of about 0.025 s per excerpt.

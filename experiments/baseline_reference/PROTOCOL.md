# B1 reference recipe

Protocol `b1_reference_v1`. **Frozen 2026-11-03 on user approval ("run it")**,
before the recipe's own scores were computed. Drafted 2026-11-03. The run
records this file's SHA256 and refuses to continue if it changes.

## Why

Every synthetic-to-real comparison so far (H2, CAST) used the weak H1 route. That
route band-limits the audio to 4 kHz and uses 190 recordings per class with weak
regularization, and it reaches about 71 held-group AUROC on IDMT. B1
([report](../../reports/archive/B1_baseline_strength.md)) found a recipe near 84
and recommended freezing it as the real-only reference before any
cross-dataset test. This protocol does that:
- **Fix the recipe** in one place (`src/abvid/learning/reference.py`).
- **Check it** against B1's saved fits.
- **Record its IDMT score** and uncertainty.
- **Save frozen all-IDMT heads** for a later cross-dataset protocol.

## The recipe (fixed now)

- **Route:** channel mean, native → 16 kHz directly (no 8 kHz step), centred
  2 s, Kaiser β = 5. This is B1's `full` route.
- **Representations:** reported separately, never pooled.
  - **MFCC26:** the H1 config.
  - **BEATs layer mean:** the mean of the 12 layer means of the frozen
    `BEATs_iter3_plus_AS2M` checkpoint.
- **Training data:** every recording of the training groups, with
  `class_weight = "balanced"`.
- **Head:** a train-only StandardScaler plus lbfgs logistic regression
  (tol 1e-6, max_iter 5,000), with the fixed argmax rule.
- **Regularization:** C ∈ {0.001, 0.01, 0.1, 1}. C is chosen by mean inner
  leave-one-group-out AUROC over the training groups only; ties go to the
  smaller C. B1's nested search chose C 1 or 0.1 for MFCC and 0.001 for BEATs,
  so a single fixed C would be wrong for one of them.

## Data

- **Clips:** the 4,413 admitted IDMT car/truck excerpts, with B1's cached
  `full` route features (SHA256 `aab4b3aa…`).
- **Folds and groups:** the six H1 outer folds (one conservative connected
  group held out each); site labels as in B1.
- **No other data:** no MELAUDIS, DCASE, MVD, synthetic or military data is
  read.

## Gates (all must pass before any recipe score is used)

1. **Inputs:** B1 features and results match their recorded SHA256
   (`aab4b3aa…`, `46399d34…`).
2. **Fit parity:** for both representations and all four C values, the
   packaged recipe's inner and outer fits reproduce B1's saved metrics exactly.
   That is 6 folds × (1 outer + 5 inner) × 4 C × 2 representations = 288 fits.
3. **Feature parity:** re-extracting the first 16 clips from source audio with
   the packaged route matches B1's cached features (≤ 1e-5).

## What is reported

For each representation:
- **Per fold:**
  - the chosen C;
  - held-group AUROC, balanced accuracy, AP and macro-F1;
  - truck and car recall;
  - training-side AUROC, as the overfitting check.
- **Mean over folds:** the worst group, and 95% percentile intervals over
  folds and over site blocks (10,000 draws, PCG64 seed 20261103).
- **The gain over H1's reference:** the recipe minus H1's reference (BEATs
  layer 12, `core8`, 190 per class, C = 1), paired by fold, with the same
  intervals.
- **All-IDMT heads:** fitted on all six groups, with C chosen by inner
  leave-one-group-out over the six. They are saved with SHA256 hashes and not
  scored on anything here.

There is no decision rule. If the gates pass, the recipe is the reference. Later
protocols that say "real-only reference" use `b1_reference_v1` unchanged; any
change is a new version.

## Limits stated in advance

- **Optimistic IDMT score:** the route, budget and representations were picked
  after B1's outer results were seen. Only C is chosen inside the training
  groups. So the IDMT score is development evidence, not an untouched
  estimate. The untouched test is the later cross-dataset protocol.
- **Few groups:** six groups at three sites; intervals are conditional on them.
- **Narrow search:** frozen encoders and linear heads only. IDMT clips are
  about 2 s.
- **CAST and H2 need full-band renders:** CAST renders at 8 kHz internally,
  so comparing it against this reference needs a full-band renderer, or an
  explicit band-limited control.

## Cost

About 340 logistic-regression fits and 16 feature extractions, a few minutes
on CPU. Outputs are create-only under
`artifact:baseline_strength/b1_reference_20261103_v1/`.

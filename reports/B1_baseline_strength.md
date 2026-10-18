# B1: baseline strength within IDMT

Completed 2026-10-12 under the frozen protocol
[`baseline_strength_v1`](../experiments/baseline_strength/PROTOCOL.md) (SHA256
`7f83741d…ab2f`). This is development evidence on previously exposed IDMT groups. No
MELAUDIS, DCASE or synthetic data was used.

**The weak H1 baseline was mostly self-inflicted.** Selecting the representation, route,
budget and regularization by fully nested leave-one-group-out search raises held-out-group
car/truck AUROC from **71.5 to 84.1: +12.6 points, 95% CI [8.3, 17.8]** (site-block
interval [8.1, 20.5]). Outer groups were never used for selection.
- Every one of the six held-out groups improves, by 7 to 24 points.
- Balanced accuracy rises from 65.9 to 74.9.
- AP rises from 31.1 to 45.7.
- Macro-F1 at the fixed argmax rule rises from 53.1 to 59.9; the remaining gap there is
  threshold calibration at about 10% truck prevalence.

## What was compared

**96 configurations**, all reported in `results.json`:

| Factor | Levels |
|---|---|
| Encoder | BEATs layer 12 (= H1), layer 6, mean of the 12 layers; PANNs CNN14; MFCC26; BEATs-12 + MFCC26 |
| Route | `core8` (H1: native → 8 kHz → 16 kHz) or `full` (native → 16 kHz) |
| Budget | 190 per class (H1 selections, 5 seeds) or all training-group recordings with balanced class weights |
| L2 strength | C ∈ {0.001, 0.01, 0.1, 1} |

Heads were frozen-encoder StandardScaler + logistic regression, totalling **4,608 fits with
0 failures**. The run took 4.3 minutes after a 27-minute feature extraction.

**Encoder checks:**
- CNN14 was implemented from the official PANNs checkpoint with a strict load. Its front
  end matches librosa, and it tags IDMT clips as "Vehicle" and "Car".
- BEATs per-layer outputs come from forward hooks; layer 12 equals the frozen H1 embedding
  exactly.

## Results (held-out-group AUROC, %)

| | Reference (H1) | Nested procedure |
|---|---:|---:|
| Mean over 6 groups | 71.49 | **84.10** |
| Per group | 79.1 · 79.9 · 68.6 · 71.2 · 59.0 · 71.1 | 90.5 · 87.1 · 92.3 · 79.1 · 76.2 · 79.3 |
| Selected configuration | — | MFCC26 · full · all data in 5 of 6 folds (C 1 or 0.1); BEATs-avg · full · all data · C 0.001 in 1 fold |

**Single-factor and combined contrasts** for BEATs layer 12 against the reference. These
are post hoc but over pre-declared factors; paired fold intervals, with site-block in
brackets.

| Change from H1 | AUROC | Δ (95% CI) [site] |
|---|---:|---:|
| Full-band route only | 77.40 | +5.91 (3.11, 9.00) [3.23, 10.66] |
| Strong L2 only (C = 0.001) | 75.50 | +4.01 (1.73, 7.38) [1.65, 8.12] |
| All training data only (C = 1) | 69.05 | −2.44 (−6.61, 1.23) [−9.31, 1.07] |
| Full + all data + C = 0.01 | 84.16 | **+12.67 (9.28, 16.18)** [9.37, 19.17] |
| BEATs layer mean, full + all data + C = 0.001 | 84.89 | +13.40 (9.59, 18.03) [9.64, 20.39] |
| MFCC26, H1 route (= H1 diagnostics) | 80.22 | +8.73 (4.33, 13.38) [3.66, 16.81] |
| MFCC26, full + all data + C = 1 (best, post hoc) | 86.72 | +15.23 (8.39, 23.48) [8.08, 28.26] |

**Factor averages over all other factors (AUROC):**
- **Route:** full 79.3 vs core8 75.6.
- **Budget:** all data 78.4 vs 190 per class 76.5.
- **C:** 0.001 → 79.7, 0.01 → 79.0, 0.1 → 76.7, 1 → 74.4.
- **Encoder:** MFCC26 83.0 · BEATs-avg 79.1 · BEATs-6 78.8 · BEATs-12+MFCC 78.3 ·
  BEATs-12 77.5 · CNN14 68.0.

Inner and outer scores agree closely (best configuration: inner 86.2, outer 86.7), so the
selection is not overfitting the outer groups.

## Interpretation

1. **The H1 setup discarded the most useful information and underregularized.**
   - The `core8` route throws away everything above 4 kHz, and restoring it is worth
     about 4–6 points.
   - BEATs fits its training selection perfectly
     ([H1 diagnostics](H1_source_diagnostics.md)), so strong L2 is worth about 4 points.
   - More data helps only once it is regularized. Combined, BEATs gains about 13 points.
2. **Simple MFCC statistics generalize across IDMT groups at least as well as frozen
   BEATs or CNN14.** Within IDMT, a pretrained general-audio encoder gives no advantage.
   Whether that holds across datasets is untested.
3. **Implication for the simulator studies (a hypothesis, not tested here).** CAST renders
   at 8 kHz internally, and H2 rendered at 8 kHz before resampling. Both were scored on the
   `core8` route. If the cues above 4 kHz are what separates trucks from cars across
   recording groups, band-limited synthesis cannot supply them. That could partly explain
   why neither CAST nor H2 added value. A full-band generator is needed to test this.
4. **Macro-F1 stays modest** because the fixed 0.5 threshold is not calibrated to the
   source prevalence. Ranking quality (AUROC, AP) is what improved.

## Limits

- **Development only.** All six IDMT groups were exposed in H1. The recipe was chosen
  within IDMT (nested) and has not been tested on another dataset.
- **Few groups.** Six groups at three sites; intervals are conditional on those groups.
- **Narrow search.** The search covered frozen encoders and linear heads only; fine-tuning,
  non-linear heads, longer context and augmentation are untested. IDMT clips are about
  2 s long.
- **Post hoc rows.** Rows other than the nested procedure are post hoc descriptions over
  pre-declared factors, not independent estimates.

## Recommendation

Adopt a **B1 baseline recipe** as the reference for future comparisons, frozen before any
cross-dataset test:
- full-band input;
- all training-group data with balanced class weights;
- strong L2 (C ≈ 0.001–0.01);
- two representations: MFCC26 and BEATs layer mean.

Revisit synthetic-data questions only with full-band synthesis and this recipe as the
real-only reference. A frozen cross-dataset check of the recipe would be a natural, cheap
candidate for the eventual DCASE pass.

## Provenance

- Features SHA256: `aab4b3aa…8d93`.
- Results SHA256: `46399d34…a62e` ([index](ARTIFACT_INDEX.md#artifact-71bdc7bfa2d8)).
- Executed from the `learned-audio-signature` checkout (`1937c89`; code identical to
  `main` `2a1d40e` plus two Markdown files), dirty tree.
- numpy 1.26.4, torch 2.11.0, scikit-learn 1.4.2, macOS arm64, CPU.

Code:
- [`run.py`](../experiments/baseline_strength/run.py)
- [`grouped_search`](../src/abvid/learning/grouped_search.py)
- [`cnn14`](../src/abvid/representations/cnn14.py)
- `beats.extract_layer_embeddings`

Tests: `tests/test_baseline_strength.py`, 7 passing including two archive checks.

> Historical protocol/command reference. Commands retain their original paths
> and run from `ABVID_ARCHIVE_ROOT`, not from the reorganized code checkout.
> See the [experiment registry](../../../../README.md).

# CAST prior v2: separate spectral and envelope spread

This is a new source-development method version. The original smooth8 fitter,
all 380 fits, prior v1 experiments and their failures are immutable. The outer
IDMT group has not been opened in this improvement goal.

## Evidence and hypothesis

The completed, audited full-bank v1 grid fails the joint-versus-marginals
distance margin at every temperature. At temperature 1, joint source coverage
is 81.676% car and 80.777% truck, but relative W1 gains versus marginals are
-5.952% and -3.853%. Mean spectral/band spread ratios are 1.39/1.61 car and
1.28/1.48 truck, whereas envelope/modulation ratios are 0.96/0.81 and 0.97/0.83.
Uniform expansion therefore moves different descriptor families in conflicting
directions. Spectral residuals remain visible in full training reconstructions.

Before changing renderer capacity, test separate spread controls for spectral
parameters and the amplitude envelope. This modifies the training-only prior;
it neither refits real audio nor changes any descriptor or acceptance rule.

## Frozen candidate grid

- Spectral temperature: `[0.5, 0.65, 0.8, 1.0]`.
- Envelope temperature: `[1.0, 1.25, 1.5]`.
- All twelve Cartesian pairs, same settings for car and truck.
- Spectral coordinates: log harmonic spacings, centered log harmonic weights,
  centered log noise weights and harmonic-fraction logit (first 20 coordinates).
- Envelope coordinates: five log envelope knots with their mean removed.
- For each class, compute the center as an equal-weight mean of group means,
  using only the current fold's training parents. Transform each complete
  vector about this center, invert and enforce the unchanged renderer bounds.
- `(1, 1)` is exact original-vector replay, including its original envelope
  scale. All other pairs use the same gauge removal/inversion as prior v1.

Transform the donor bank identically before joint, prototype and independent
scalar-marginal sampling. Keep equal group sampling, all original parents,
five seeds, 50 samples per class/seed/arm, and common random streams across
candidates and matched arms. Retain direct donor ancestry and explicitly record
all class-center calibration ancestors for every sample.

## Selection, access and acceptance

Use the exact same five leave-one-training-group-out folds from the full
380-parent bank. Recompute descriptor scales within each fold. Every candidate
must retain all 1500 samples per fold and all real validation parents.

Rank by: any family-spread failure; maximum class shortfall from coverage 0.8
or relative W1 gain 0.025 against either control; mean class W1; distance of
the two temperatures from `(1, 1)`; spectral temperature; envelope temperature.
The final tie-breaks are deterministic, not a new acceptance rule.

Require the existing **80% coverage and 2.5% relative W1 gain versus both
matched controls in each class**, family spread in [0.5, 2], and full technical
verification. Keep original four descriptor families, percentile interval,
weights, sample budgets and outer scales. A failed grid is retained as a
scientific failure, without an outer comparison. Selection cannot use outer
descriptors, real-world target data, classification or held latent fitting.

Only a fully audited passing source candidate may be frozen for the already
exposed outer IDMT development comparison. Generate its complete schedule and
outputs before reading held descriptors. No fresh-confirmation, physical-source
recovery or classification claim follows from success on this objective.

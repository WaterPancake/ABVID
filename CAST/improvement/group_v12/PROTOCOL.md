# Source-group variation prior, v12

The audited v11 source bank is fixed: 380 original IDMT training IDs,
190 per class, five source groups, 69 effective controls. Its source-only
variance diagnostic assigns 27–39% of spectral/band descriptor variance
and 8–15% of temporal descriptor variance to group mean differences.
These are descriptive components, not causal recording effects or proof
of independent vehicles. The diagnostic reads no outer observations.

The prior v11 outer result is already known (77.984% car / 76.054% truck
coverage). This motivates further development but supplies no fitted
quantity, scale, candidate choice or new descriptor inspection here.
All previous data-access restrictions and scientific criteria remain.

## Fixed candidates and transformations

Compare exactly three candidates with temperature 1:

1. `spectrum16`: exact original v11 joint/prototype/marginal sampling.
2. `group_recombined`: recombine full parent residual vectors with group
   mean shifts, keeping the empirical group-effect variance.
3. `group_predictive`: the same residuals, plus symmetric group shifts
   scaled by sqrt((G+1)/(G-1)), with G training groups in that fold.

No additional scale grid is permitted in this version. For each class
inside each source training fold, transform spacing with log, harmonic
and noise simplexes with centered log ratios (floor 1e-8), harmonic
fraction with logit (input clip [1e-4,1-1e-4]), and envelope knots with
centered log ratios (floor 1e-8). Compute each group's equal-parent mean
mu_g and the equal-group mean mu. If parent i belongs to group g, define
r_i = z_i - mu_g. Its virtual children are:

    z_child = mu + r_i + sign * a * (mu_h - mu)

For recombination, h is uniform over training groups, sign=+1 and a=1.
For predictive sampling, h is uniform, sign is equally -1/+1, and
a=sqrt((G+1)/(G-1)). This fixed group-count factor combines empirical
sample-variance correction and uncertainty of a mean under an exchangeable
group approximation. It is an engineering hypothesis, not an estimated
Bayesian posterior. Group means also contain finite-parent sampling noise;
recombining effects can break real context dependencies.

Invert with softmax for simplexes, bounded exponentials for spacing and
envelopes, and sigmoid with logit clipped to [-15,15]. Retain each residual
parent's original mean log envelope as its amplitude gauge. Project to the
unchanged spacing [10,400] and envelope [0.1,3] bounds; retain and count every
projection. Original ambiguity flags and physical unknowns remain intact.

The virtual population is finite and balanced: equal original source group,
equal parent within that group, equal virtual child. Joint sampling takes
one complete vector. Marginals independently sample each of the 69 scalar
coordinates from this same population, then normalize simplexes as before.
The prototype is the coordinate mean under these same weights. Keep the
original donor/phase/noise streams; derive child choices from a separate
fixed namespace. Record direct residual donors and every class/group-mean
ancestor, their hashes, child choices, projections and population hash.
Freeze each fold's statistics and prototype before generating any samples.
Virtual children are derivatives, not new independent recordings.

## Unchanged evaluation and access gates

Use only the existing five source groups and their saved descriptors and
parameters. No new fitting, raw source reads, MELAUDIS, supplied synthetic
banks, military/reserved data or classifier training. Old immutable loaders
also verify saved source waveform hashes; they do not open outer audio.
Recompute all prior statistics and descriptor scales inside each training
fold, excluding its validation group and the outer group. Retain every
parent and every generated example.

Use the same four equally weighted descriptor families, W1 definition,
5th–95th quantiles, family spread [0.5,2], seeds [42,123,456,789,1024],
50 samples/class/seed/arm, folds, group/parent weights, and selection rule.
Require coverage >=80% and W1 reduction >=2.5% against both matched controls
in each class. Controls depend on this 69-coordinate representation and
cannot establish a representation-independent advantage.

Numerical tests cover analytic transformed moments and prototype weighting,
exact reference draws, deterministic waveforms, bounds/projection records,
complete ancestor sets and excluded/tampered input rejection. Preserve
all original numerical/determinism/provenance checks. Replay all 22,500
source records and 450 scores and require all 150 v11 reference scores exact.

Only a newly selected candidate that passes source criteria and full replay
may receive a new outer development comparison. If spectrum16 wins, preserve
its existing outer result without rerunning it. A new outer run freezes all
group statistics, original ancestors, scales and the complete 1,500-example
schedule, then generates every waveform before reading held descriptors.
Replay all 1,500 outputs and all 570 raw held descriptors. The exposed group
is development evidence only; no independent confirmation is claimed.

Keep all scientific failures and implementation defects. A numerical defect
must be fixed without relaxing its acceptance test before freezing the run.

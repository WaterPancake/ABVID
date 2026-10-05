# Capacity-v3 fixed source comparison

Freeze this comparison before generating any candidate. Compare capacity16x9
with smooth8 using exactly the same pilot parents, then the exact same 380
parents only if paired checking loss and source-only evidence justify expansion.
No outer descriptor or raw outer audio is read by this source module.

There is one prior per renderer: temperature 1, complete-vector empirical joint
sampling, equal-group class prototype and independently resampled scalar
marginals. All three arms use the same fitted bank within that renderer. The new
schema has 37 coordinates (3 spacing, 8 harmonic, 16 noise, 1 mixture, 9 envelope).
The old schema remains 25. Independent marginal draws renormalize each simplex.
Joint draws preserve all coordinates of the selected parent. Each draw retains
every scalar donor and original group ancestry. No prior calibration or new
spread grid is introduced by this comparison.

Leave out each of the five source training groups in turn. Remove its parents
before sampling and calculate descriptor scales from only the remaining real
training observations. Use the unchanged descriptor families, floors,
quantiles, seed schedule, 50 samples per class/seed/arm, equal family/fold/seed
weights, and [0.5, 2] spread bound. All sample arrays and descriptors must replay
exactly and every score must recompute. Replay must match the complete expected
schedule, refusing duplicate, missing, additional or reordered samples.

Retain the v1 random namespace so the matched renderers share phase/noise
streams and joint parent choices. The first 25 scalar-marginal RNG draws share
selection seeds, but coordinate meanings diverge after the larger noise block;
do not claim identical marginal parameter vectors between parameterizations.

The goal remains >=80% mean marginal coverage and >=2.5% relative W1 reduction
versus each control in each class. Use the unchanged source ranking: spread
failure, maximum class coverage/margin shortfall, mean joint W1, temperature.
Ranked candidates may fail; ranking is not acceptance. Preserve every score,
flag and failed candidate. All generated records, priors, configs, source code,
inputs, tests and runtime are hashed and stored. A source candidate must pass
all criteria and exact audits before any later outer development comparison.

Source groups have already informed design. These measurements are exploratory
development, not fresh confirmation, independent vehicle recognition or
classification transfer. Prior outer results remain unchanged.

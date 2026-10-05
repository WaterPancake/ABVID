# Nested spectral resolution, v11

V10's five source folds still have spectral and band W1 around 0.72–0.77,
versus envelope 0.47–0.52 and modulation 0.21–0.22. Car log-spectrum coverage
is 79.94% on these source folds. These source residuals motivate the next
fixed representation check. V10's exposed outer aggregate is already known
(77.21% car / 75.57% truck coverage); no outer observations, per-feature
residuals, or scores enter calibration or candidate selection here. This is
continued exploratory development after exposure, not fresh confirmation.

Use exactly the sixteen nested spectral points already specified in v3:
the eight original band centers, their seven midpoints, and 4000 Hz.
Use the fixed positive Voronoi reference widths. Lift the old log-power
density by interpolation; component normalization preserves the old waveform
within the original 1e-5 nested-render tolerance. Keep temporal41 controls,
spacing and harmonic weights. The schema is 3+8+16+1+41 = 69 coordinates.
The original eight analysis bands and 64 spectrum bins do not change.

Starting from all 380 audited v10 source parents, optimize only the sixteen
noise logits and harmonic fraction. Use the unchanged expected-power
calibration objective, two saved checking streams, 150 Adam updates, step
size 0.03, betas (0.9,0.999), epsilon 1e-8, gradient clip 100 and fraction
projection to [0,1]. Save initialization, all 151 losses, best parameters,
gradients, streams, hashes, flags, runtime and two reconstruction WAVs per
parent. The checking streams are calibration data. No failed parent is
removed. Audit all 380 calibrations and 760 WAV sample arrays exactly before
source generation. Earlier artifacts are immutable.

First require known expected-spectrum recovery with the existing mixture
error <0.1 and noise-simplex L1 <0.35 thresholds and objective reduction,
nested render agreement, finite gradients, deterministic replay, component
endpoints, valid 69-coordinate sampling and exclusion/tamper rejection.
Retain all original numerical, determinism and provenance tests.

Compare the exact v10 temporal41 bank with the new spectrum16 bank. All
joint/prototype/independent-marginal arms receive the same bank within each
parameterization. Scalar marginal sampling changes dimension to 69; report
its representation dependence explicitly. Uniform group and parent weights,
source folds, seeds, 50/class/seed/arm, phase/noise namespace, four equal
descriptor weights, scales, 5th–95th quantiles and ranking are unchanged.
Both classes must meet 80% coverage, 2.5% gain against both matched controls
and family spread [0.5,2]. Replay all 15,000 generated records and 300 scores;
the 150 v10 reference scores must remain exact.

Only the frozen 380 source parents and saved training descriptors may inform
this method. No raw dataset reads, MELAUDIS, supplied synthesis banks,
military/reserved audio, classifier training or target-derived adjustment.
If the unchanged source ranking retains v10, do not repeat its outer check.
A new selected candidate must pass all source criteria and full audit before
any new frozen outer development comparison. All outcomes, including failure,
remain reportable evidence; none implies independent vehicle generalization.

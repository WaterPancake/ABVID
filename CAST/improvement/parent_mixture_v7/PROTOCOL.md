# Per-parent bounded mixture calibration, v7

The fixed -1 mixture offset reduces full-source W1 by 7.8% car and 9.4% truck,
but its source coverage is 79.896%/78.888% and its marginal-control gains remain
-1.962%/-0.459%. V6's exact audit must pass before this experiment. A global
offset does not account for different component spectra or target balance in
each training recording.

Test one fixed closed-form calibration of the harmonic fraction per training
parent, retaining every other fitted control and the original smooth8 renderer.
No new raw source, outer descriptor, classifier or neural model is introduced.
Keep the original fit artifacts unchanged and create a separate derived bank.

Read the parent's two saved checking harmonic/noise components and calculate
their mean Welch power in the original eight bands (Hann, nperseg 2048,
noverlap 1024, 16 kHz). Divide out their original mixture power factors to get
unweighted component band powers H and N. Compare their expected power mixture,
ignoring the stochastic cross term, with that parent's saved observed band
proportions y. Let h and n be H and N normalized to unit band power.

    beta = clip(dot(y-n, h-n) / dot(h-n, h-n), 0, 1)
    new_fraction = beta*sum(N) / ((1-beta)*sum(H) + beta*sum(N))

This is bounded least squares in expected normalized band power. It is not a
claim of exact waveform reconstruction or recovery of physical engine power.
If the original mixture is exactly 0 or 1, a component has zero power, or
dot(h-n,h-n) <= 1e-10, retain the original fraction and flag the undefined or
unidentifiable calibration. Do not omit that parent. Reject nonfinite or
invalid power/target data. Store original/new fractions, component powers,
target, expected-band objectives, flags, parent ID/group, and source hashes.

The saved checking components are calibration inputs for this derived bank.
The original independent checking-loss claims apply to the original fits;
do not reuse them as independent validation of the calibrated vectors.
Generation uses the unchanged separate sampling streams and source-group
exclusion. Calibration uses each parent's own observations only, with no
cross-parent statistics. Remove source-validation parents before any prototype,
marginal or joint sampling. Earlier source-driven method design remains exposed
development and does not establish independent validation.

Compare two fixed banks: original winner-only and the per-parent calibrated
bank. All three arms use the same bank within each candidate, with the original
equal-group/parent weighting, complete-vector joint sampling and independent
scalar marginals. No temperature, offset or calibration-strength grid is used.
The winner-only arm must reproduce all 150 audited v1 full-bank T1 scores.

Keep the original 380 IDs, metrics, scales, quantiles, five seeds, 50 samples
per class/seed/arm and all weighting. Acceptance remains >=80% mean marginal
coverage and >=2.5% relative W1 gain against each control in each class, with
family spread [0.5,2]. Use the unchanged source maximum-shortfall ranking.
Replay every waveform, derived parameter, parent, calibration input and score.
Preserve every failure. No outer evaluation without a passing audited source
candidate; any later outer result remains exposed development evidence.

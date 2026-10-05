> Historical protocol/command reference. Commands retain their original paths
> and run from `ABVID_ARCHIVE_ROOT`, not from the reorganized code checkout.
> See the [experiment registry](../../../../README.md).

# Source-only band-bias calibration, v5

Both the capacity and equivalent-fit experiments improved some absolute metrics
but failed the matched-control margin. The audited smooth8 reconstructions have
a consistent source-group band bias: car low-band observation/reconstruction
ratios are below one in every group, while 500–1500 Hz ratios exceed one in
every group. Truck 500–1500 Hz ratios also exceed one in every group. The
training-only evidence is `diagnostics/source_spectral_bias_evidence.json`.

Test a fixed, bounded first-order correction to the existing smooth8 noise
weights. This is an empirical observation correction, not recovered dry-source
physics, and its improvement is not assumed. No new fit or outer input is used.

For each class and source-training fold, calculate the mean observed and mean
reconstructed eight-band power proportions within each included group, using
the saved unit-RMS original and first fixed checking reconstruction. Form each
group's log(observed/reconstructed) ratio with floor 1e-8. Average these log
ratios equally over included groups and clamp each coordinate to [-ln(2),ln(2)].
Retain all calibration parent IDs, groups and fitted-artifact hashes.

Apply the already defined group-balanced spectral temperature to the original
winner-only bank, keeping envelope temperature 1. Then multiply each noise
weight by exp(strength × class correction), and renormalize its simplex.
Harmonic spacing, harmonic balance, mixture fraction and envelope are unchanged
by the correction. This approximates a noise-spectrum correction; the full
rendered mixture need not follow the requested band ratio exactly.

Freeze eight cases: spectral temperature [0.5, 0.65, 0.8, 1.0] × correction
strength [0, 1]. Strength zero preserves the prior-v2 vectors bit-for-bit.
The four zero-correction cases must exactly reproduce the already audited
prior-v2 envelope-temperature-1 score records. No finer search is authorized
by this version. Every arm uses the same transformed/calibrated donor bank.

Leave each of the same five source groups out before calculating centers,
calibration ratios, descriptor scales or sampling. Calibration IDs must equal
that fold's training bank exactly; missing, duplicate or extra calibration IDs
fail closed. The original outer group must be absent from all ancestry.

Preserve the exact 380 source IDs, renderer, original fits, descriptor families,
scales, quantiles, 5 seeds, 50 samples/class/seed/arm and weighting. Acceptance
remains >=80% mean marginal coverage and >=2.5% relative W1 reduction against
each control in each class, with family spread [0.5, 2]. Rank by spread failure,
maximum class shortfall, mean W1, distance from the identity setting, spectral
temperature, then correction strength. Ranking does not imply acceptance.

Freeze code, paired training bands and all inputs before generation. Replay
every waveform, parameter, donor, calibration ancestor and descriptor, and
recompute all calibration statistics/scales/scores. Only a passing audited
source candidate permits an outer development comparison. No target statistics,
reserved audio, labels beyond car/truck, or classifier are introduced.

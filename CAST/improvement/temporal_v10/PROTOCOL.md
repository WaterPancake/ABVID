# Temporal-resolution test, v10

V9 improved its per-parent spectral objective by about 50% but source margins
remain negative; v8 still ranks first. V8's five-knot envelope calibration
improved all 380 same-parent envelope errors, yet modulation still favors the
independent-coordinate control. Test temporal representation at the resolution
already measured by the unchanged 40 consecutive 50 ms RMS frames.

Replace only the five envelope controls with 41 controls at the 40 frame
boundaries (0 to 2 s, every 50 ms). All original five knot positions are nested
in this grid. Keep the smooth8 renderer, eight noise weights, harmonic weights,
spacing and v9's calibrated mixture/noise parameters. No new raw input or
descriptor is introduced. Config/parameter schema is separately versioned:
3 spacing + 8 harmonic + 8 noise + 1 fraction + 41 envelope = 61 coordinates.

Lift the original envelope exactly by linear interpolation. Let B be the
frame-average linear basis on the original internal 16,000-sample timeline,
and D the second-difference matrix. Scale the parent's observed RMS trajectory
y to the lifted envelope's original mean amplitude. Solve the fixed convex
quadratic objective

    mean((B @ knots - scaled_y)**2) + 0.01 * mean((D @ knots)**2)

with original [0.1,3] bounds and equality preserving mean(B@knots). The fixed
curvature penalty removes the alternating-boundary ambiguity and discourages
rapid oscillation; it is not selected by a grid. Use SLSQP with exact gradient,
lifted initialization, 300 iterations maximum and ftol 1e-14. Retain the lifted
original and an explicit flag on solver failure, nonfinite output, bounds,
mean-amplitude discrepancy >1e-8 or objective increase. Retain all boundary
cases and all 380 parents. The frame-average model approximates RMS; actual
waveforms determine evaluation scores. These are calibration reconstructions.

Numerical tests must verify nested-waveform agreement, affine-trajectory
recovery, regularized-objective non-increase, preserved gauge/bounds, solver
failure retention, invalid inputs, deterministic valid 61-coordinate sampling
and complete ancestry. Keep all previous numerical/provenance criteria.

Compare three banks: exact v8 reference, exact v9 reference, and v9 plus the
41-control temporal calibration. Every arm receives the same bank within its
candidate. The new independent control samples every one of the 61 scalar
coordinates separately with original simplex normalization. Keep equal group
and parent weights, seeds and waveform namespace. The first 20 marginal draws
share their meanings across schemas; envelope positions differ after that.
Do not describe their full vectors as identical. Joint parent choices and
phase/noise streams remain shared.

Use the same five source folds, 50/class/seed/arm, five seeds, four equally
weighted descriptor families, 5th–95th coverage quantiles, scales, criteria
and selection rule. Both classes need >=80% coverage, >=2.5% W1 gain against
both controls, and family spread [0.5,2]. Do not change controls or criteria to
make the new parameterization pass. Replay all 22,500 samples, 450 scores and
380 local calibrations. Both reference banks must reproduce all 300 v9 source
score records exactly. V9's full fit and source audits must pass first.

Source descriptors/checking streams are already exposed calibration data.
Calibration uses each parent's own observations only; remove validation-group
parents before sampling/statistics. Repeated source-based method design is
exploratory development. No outer access until a candidate passes unchanged
source criteria and complete audit. A later outer comparison remains exposed
development, not fresh confirmation or classification/physical-source evidence.

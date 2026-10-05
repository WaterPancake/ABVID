> Historical protocol/command reference. Commands retain their original paths
> and run from `ABVID_ARCHIVE_ROOT`, not from the reorganized code checkout.
> See the [experiment registry](../../../../README.md).

# Direct temporal-envelope calibration, v8

V7 passed exact replay but failed the scientific criteria. Coverage is
80.426% car / 79.698% truck; gains against independent marginals are
-1.650% / -0.438%. Temporal-family coverage remains only 77.9–80.0%, and
the joint sampler loses on modulation in both classes. Added envelope knots
under the original spectral fitting objective previously failed the margin.

Test one fixed calibration of the existing five envelope knots to the parent's
observed 40-frame (50 ms) RMS trajectory. Retain v7's mixture calibration and
all spectral controls. This tests the fitting target without changing renderer
capacity, sampling, evaluation metrics or acceptance thresholds.

Let B be the mean linear-interpolation basis in each of 40 consecutive frames
on the original 16,000-sample internal timeline. Let old be the fitted five
knots and y be the saved observed unit-RMS envelope. Rescale y to have the
same mean amplitude as B@old. Solve the convex quadratic problem

    minimize mean((B@knots - scaled_y)**2)
    subject to 0.1 <= knots <= 3
               mean(B@knots) == mean(B@old)

Use deterministic SLSQP with the exact gradient, initial old knots, ftol 1e-12
and maximum 300 iterations. These are solver controls, not a parameter grid.
Retain the original knots and an explicit failure flag if convergence,
finite values, bounds, the 1e-8 mean-amplitude constraint or non-increase of
the declared objective fails. Retain boundary hits. No parent is discarded.
The mean-amplitude constraint preserves each parent's existing envelope scale
gauge, so independent-coordinate controls are not changed by arbitrary new
per-parent gain choices. The linear frame-average model approximates RMS;
generated waveforms, not this surrogate, determine all evaluation scores.

Store original/new controls, the target, scale, objective before/after,
solver diagnostics, flags, input/parent hashes and preserved gauge. The
source descriptors and earlier checking components are calibration inputs.
Do not present their reconstruction errors as independent validation.

Compare v7 mixture-only versus mixture-plus-envelope banks with the unchanged
joint, prototype and independent-scalar controls on each common bank. The
mixture-only candidate must reproduce all 150 corresponding v7 score records.
Use the same 380 source IDs, five omitted-group folds, five seeds, 50 samples
per class/seed/arm, scales, four equally weighted descriptor families and
5th–95th percentile coverage. Both classes must meet 80% coverage and 2.5%
relative W1 gain against each control, with family spread [0.5,2]. Keep the
unchanged maximum-shortfall ranking. Calibration is local to each parent;
remove validation-group parents before any sampling or class statistics.

Require the successful complete v7 replay audit first. Freeze this code and
both banks before generation, replay all 15,000 waveforms, calibration inputs,
ancestry and 300 scores, and preserve all failures. No new raw dataset reads,
outer descriptors, classifier or military data. Source method design has seen
these five groups repeatedly; source selection remains exploratory. Outer
access requires a passing audited source candidate and remains development.

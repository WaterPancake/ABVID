> Historical protocol/command reference. Commands retain their original paths
> and run from `ABVID_ARCHIVE_ROOT`, not from the reorganized code checkout.
> See the [experiment registry](../../../../README.md).

# Joint expected-spectrum calibration, v9

V8 meets source coverage but fails the margin: 80.626%/80.448% coverage and
-1.254%/-1.267% gain against independent scalar sampling (car/truck). Spectral
and band-power distances dominate W1. The v5 class-average noise correction
and v7 single-mixture calibration did not remove this mismatch. Test one fixed
per-parent joint calibration of the eight smooth-noise controls and harmonic
fraction. Keep v8's envelope and original spacing/harmonic weights unchanged.

Use the original smooth8 waveform renderer and its two deterministic checking
noise streams as calibration inputs. Compute the expected Welch power

    P = m * harmonic_power + (1-m) * noise_power(weights)

from separately rendered unit-mixture components, including the fixed envelope
and upsampler. Ignore their stochastic cross term. Welch uses the unchanged
16 kHz, periodic Hann, 2048 window, 1024 overlap, segment-mean removal and
one-sided power factors. Aggregate into the original 64 spectrum bins and
eight bands, normalize each to unit total, then fit the equal-weight mean of
squared log-probability errors for these two representations. Use the existing
spectral probability floor. This is a declared calibration objective; the
four-family evaluation metrics, scales and weights remain unchanged.

Initialize from v8. Optimize eight softmax logits and m with exactly 150 Adam
steps, learning rate 0.03, betas (0.9,0.999), epsilon 1e-8, gradient norm clip
100; project m to [0,1] after every update. Save the best objective including
initialization, all 151 objective values, initial/final parameters, gradients,
streams, input/parent hashes, runtime and boundaries/ambiguities. No class or
cross-parent statistic enters calibration. Every parent remains; technical
nonfinite errors stop the stage with a retained error record. Checking streams
are explicitly calibration data, not independent validation of these vectors.

First pass numerical tests against SciPy Welch, deterministic exact replay,
component endpoints, finite gradients and known expected-spectrum recovery
(mixture error <0.1 and noise-simplex L1 <0.35 where identifiable). Retain all
previous numerical/recovery/provenance criteria. Freeze code, configuration,
380 source IDs and inputs before calibration. Never change previous artifacts.
Calibrate all 380 parents; audit their exact fitted results before generation.

Compare v8 versus the new derived bank under unchanged joint, prototype and
independent-scalar sampling, all arms receiving the same bank. Source selection
uses the same five omitted-group folds, five seeds, 50/class/seed/arm, equal
weights, 5th–95th percentile coverage, scales and [0.5,2] family spread bounds.
Both classes must meet 80% coverage and 2.5% relative W1 gain against each
control. Preserve the maximum-shortfall ranking and all failed candidates.
The 150 v8 score records must reproduce exactly; replay all generated waveforms,
descriptors, direct/calibration ancestors and scores. A source pass and complete
matching audit remain prerequisites to any outer development comparison.

Only the audited 380 H1 training parents and saved source descriptors are used.
No new raw source or outer observations are opened. Original source licence
and provenance restrictions remain. Source groups have informed repeated method
design; these are exploratory results, not independent confirmation, physical
source recovery, classification or real-world transfer.

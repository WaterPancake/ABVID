# CAST capacity v3: sixteen spectral controls and nine envelope knots

This is a new, separately frozen renderer version. Preserve smooth8, both prior
versions, all fits, scores and failure records. The target stays at 80% mean
marginal coverage and 2.5% relative W1 gain against both matched controls in
each class, with family spread in [0.5, 2] and all technical gates retained.

## Source evidence and fixed revision

The audited 380-parent smooth8 bank retains median band-energy L1 residuals
around 0.25, envelope RMSE around 0.12 and systematic log-spectrum deficits
around 1 kHz. Uniform prior expansion failed. The separate-spread prior grid
also has no candidate meeting coverage and margin together; its exact audit
must finish successfully before new real fitting is authorized by this protocol.

Test one fixed capacity increase: retain eight harmonics and three harmonic
spacing knots, increase smooth noise controls from eight to sixteen, and
increase envelope knots from five to nine. This is a combined resolution test;
separate causal contributions of the two changes remain untested.

Use the original eight spectral centers, the seven arithmetic midpoints between
them, and a final control at 4000 Hz. DC and exact Nyquist output remain zero.
The sixteen positive reference widths are the Voronoi intervals bounded by
0 Hz, successive center midpoints and 4000 Hz. Controls are simplex values
representing point power times these widths. Interpolate log power, shape the
same independently seeded white noise, and normalize component RMS as before.

Lift old eight-control initializations by interpolating their log power density
at the new centers, multiplying by reference widths and renormalizing. Lift
five envelope knots to nine by interpolation. This nests the old smooth shape
to floating-point tolerance and keeps the original initial signal family.

## Unchanged settings and acceptance

Retain the original preprocessing, sample rates, four starts, 300 Adam steps,
learning rates, multiresolution spectral objective, separate two-realization
fit/check streams, gain ambiguity and all synthetic thresholds. Unit tests
must verify finite gradients, valid bounds, no aliasing, deterministic replay,
component reconstruction and the lifted signal equivalence. Run all seven
original fixture types with lifted controls. Do not weaken their frequency,
noise-vector or loss-improvement tolerances to accommodate the larger model.

After synthetic acceptance and a passing prior-v2 audit, fit the same original
50 pilot parents. Save originals, all starts, fitted vectors, audio, residual
plots, failures, runtime and hashes. Compare the original checking objective
against each clip's smooth8 result and unoptimized initialization. Retain every
parent, every failure and all identifiability flags.

Use the same source-only 380-ID H1 selection, five source groups, classes,
descriptor definitions, outer scales and 5 seeds × 50 samples per class/arm.
All matched arms use the same new 37-coordinate parameterization. Keep complete
vector joint sampling, group-balanced prototype and independent scalar-marginal
controls. No held latent fitting, target access or classifier is introduced.

If the paired pilot and source-only evidence support this fixed revision,
expand within the same 380 parents and perform the full training-group source
check before any outer comparison. Prior tuning, if necessary, must receive
its own frozen source-only configuration before generation. Do not select a
renderer, loss, prior or correction using outer descriptor values.

An outer candidate requires passing source criteria and complete replay audits.
The already exposed IDMT outer group remains development evidence; success is
neither new independent confirmation nor classification/physical-source proof.

> Historical protocol/command reference. Commands retain their original paths
> and run from `ABVID_ARCHIVE_ROOT`, not from the reorganized code checkout.
> See the [experiment registry](../../../../README.md).

# Empirical equivalent-fit alternatives, v4

The fixed capacity revision improved source reconstruction, W1 and coverage,
but failed the joint-versus-marginals margin. Test parameter ambiguity before
another large refit. Reuse the audited 380-parent smooth8 bank; no new raw audio
or outer descriptor access is required. Preserve all earlier results.

For each parent, retain the original winning fit and all alternative starts
with fitting-stream loss at most 1.05 times the best fitting loss. This is the
original, already frozen diagnostic ambiguity bound, not a tuned threshold.
Never select starts using held descriptors or checking-loss outcomes. There
are 732 eligible car starts and 703 truck starts; 379 of 380 parents have
multiple eligible starts. These alternatives describe numerical ambiguity;
they are not independent recordings, physical explanations, or a calibrated
Bayesian posterior.

Compare two fixed priors: original winner-only and the eligible-start mixture.
Use the same alternatives in all three arms within each prior:

- Joint: choose group uniformly, parent uniformly within group, and eligible
  start uniformly within parent; preserve its entire 25-coordinate vector.
- Prototype: arithmetic mean within parent over starts, within group over
  parents, and within class over groups; renormalize simplex coordinates.
- Marginals: independently choose group, parent and eligible start for each
  scalar coordinate, then renormalize the two simplex blocks.

Parents with more eligible starts do not receive greater probability. Keep
the original v1 parent-selection and waveform namespaces. Use a separately
hashed stream for eligible-start choice so the parent schedule remains matched.
Save every coordinate's parent and start index, all prototype ancestors, the
eligible bank, code and parent-fit hashes. Winner-only scores must reproduce
the previously audited full-bank temperature-1 source comparison exactly.

Leave each of the same five training groups out before any sampling or prior
statistics. Keep the exact 380 source IDs, renderer, fitted vectors, descriptor
families, scales, quantiles, 5 seeds, 50 samples/class/seed/arm and equal
family/fold/seed weights. The original outer scales remain untouched.

Acceptance remains >=80% mean marginal coverage and >=2.5% relative W1 gain
against both controls in each class, with each family spread ratio in [0.5, 2].
Use the existing maximum-shortfall source ranking. A ranked candidate can fail.
Require complete waveform/parameter/descriptor/schedule replay and score
recomputation before considering any outer development comparison. Preserve
every failure; never tune the threshold after seeing this comparison.

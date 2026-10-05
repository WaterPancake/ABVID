> Historical protocol/command reference. Commands retain their original paths
> and run from `ABVID_ARCHIVE_ROOT`, not from the reorganized code checkout.
> See the [experiment registry](../../../../README.md).

# Scoped group effects, v14 — preregistered next experiment

This protocol was preregistered before v14 implementation, generation or
outer comparison. It records the source-only hypothesis after v13's reference
selection. It does not change any completed result or acceptance criterion.
Require v13's complete source replay before running this experiment.

## Source evidence and fixed question

The already verified source-only v11 variance decomposition gives these
mean between-group fractions in transformed parameters:

| Parameter block | Car | Truck |
|---|---:|---:|
| Log spacing | 0.064 | 0.093 |
| Harmonic centered log ratios | 0.023 | 0.038 |
| Noise centered log ratios | 0.307 | 0.212 |
| Mixture logit | 0.026 | 0.242 |
| Envelope centered log ratios | 0.114 | 0.129 |

Group variation is much weaker in the fitted harmonic weights and spacing
than in the noise controls. V12 applies group effects to every block. Its
source joint W1 improves for trucks but worsens for cars relative to v11;
v13's more balanced sampling worsens joint W1 in both classes. These source
facts motivate testing the scope of the group transformation, with the
original v12 draw schedule and matched controls held fixed.

These are effective observation parameters. The variance fractions do not
identify physical source versus microphone/environment components, and
bounded/ambiguous fitted parameters remain ambiguous. No outer descriptor,
residual, fitted vector or performance-derived parameter enters this method.

## Exactly three source candidates

1. `group_predictive`: exact v12 reference, with group effects on all blocks.
2. `spectral_context`: apply the exact v12 group transformation only to the
   16 noise controls and mixture fraction. Retain the residual parent's
   spacing, harmonic weights and envelope exactly as originally serialized.
3. `spectrotemporal_context`: apply the same transformation to noise,
   mixture and envelope. Retain original spacing and harmonic weights exactly.

Use the same class-agnostic scope for both classes. No temperature/width grid,
new child seeds, extra parameters or optimizer changes are permitted. Each
parent has the same 2G virtual children as v12, with the same group/sign
ordering and sqrt((G+1)/(G-1)) factor. Unchanged blocks are copied from the
original residual parent. Modified blocks are exactly those of the existing
v12 virtual child. Record which blocks are restored. Do not report discarded
group-transform projections as applied projections in restored blocks; keep
the original parent boundary/ambiguity flags and full ancestry separately.

Recompute the prototype from the same scoped virtual population using equal
groups, equal parents within group and equal children. Sample joint and scalar
marginal arms from that population, using the exact original v12 donor,
child, phase and noise schedules. Record all group-center and residual-parent
ancestors. Freeze every scoped child-population hash, prototype, projected
coordinate count and parameter-scope declaration before generation.

## Unchanged acceptance and access

Use exactly the 380 admitted source parents, five source groups and saved
source descriptors. Exclude each source validation group before prior
statistics, and always exclude the outer group. Preserve MELAUDIS, mixed
caches, supplied synthesis, military/reserved data, datasets, splits,
environments, existing code and all baseline results.

Keep 50/class/seed/arm, seeds [42,123,456,789,1024], four equally weighted
families, training-only scales, 5th–95th quantile coverage, joint family
spread [0.5,2], >=80% coverage and >=2.5% relative W1 gain over both matched
controls in each class. Retain the exact source ranking function. The scalar
marginal control remains dependent on the 69-coordinate representation.

Preserve all existing numerical, determinism and provenance tests. Add tests
for exact restored-block identity, exact reference samples, unchanged direct
donor/child/waveform streams, correctly recomputed population means and
projection records, deterministic rendering, and excluded/tampered ancestors.
Replay all 22,500 source records and 450 scores, with all 150 reference scores
exact. Retain every failure and sample.

Only a newly selected candidate passing all source criteria and full replay
can enter a separately frozen outer development comparison. Generate all
1,500 outer waveforms before reading held descriptors; replay them and all
570 raw held descriptors. A retained reference must not reopen the outer
group. Previously exposed outer results remain development context, not
independent confirmation or evidence of classification/real-world transfer.

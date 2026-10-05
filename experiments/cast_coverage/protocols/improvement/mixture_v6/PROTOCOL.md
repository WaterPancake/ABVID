> Historical protocol/command reference. Commands retain their original paths
> and run from `ABVID_ARCHIVE_ROOT`, not from the reorganized code checkout.
> See the [experiment registry](../../../../README.md).

# Effective harmonic/noise allocation, v6

The band-bias correction alone reduced absolute W1 but failed the margin; its
full audit must pass before this source generation. Examine a different cause
of persistent low-band excess: effective harmonic/noise allocation. The
source-only saved-component diagnostic at
`../diagnostics/smooth8_component_allocation_r1` found that a harmonic logit
offset of -1 lowers same-parent band error for 161/190 cars and 165/190 trucks,
with positive median improvement in every class/group. Offset -2 overcorrects
several groups and is excluded from this fixed experiment. No new raw source or
outer audio was opened by that diagnostic; it read saved training waveforms.

This is a development-driven hypothesis, not an unbiased estimate from new
source groups. The five groups have already informed renderer/prior design.
Do not call the source selection score fresh generalization evidence.

Reuse the original audited 380-parent smooth8 winner bank. First apply the
existing spectral temperature with envelope temperature one. Then apply only
the fixed harmonic-fraction odds multiplier exp(offset):

    new_fraction = exp(offset)*fraction / (1-fraction + exp(offset)*fraction)

This preserves exact zero/one endpoints. Offset zero returns the original
controls bit-for-bit. Other parameter coordinates are unchanged by this
allocation adjustment. It is an effective observation correction, not a claim
about the true physical engine/noise decomposition.

Freeze six cases: spectral temperature [0.65, 0.8, 1.0] × offset [0, -1]. Every
matched arm receives the same transformed bank; retain group/parent weighting,
complete-vector joint sampling and independent scalar marginals. The three
zero-offset cases must exactly reproduce prior-v2 envelope-temperature-one
scores (450 records). No additional offset grid is included.

For each source fold, remove that validation group before calculating class
centers and sampling. Preserve the outer group exclusion and all calibration
ancestor records. The offset is a fixed candidate constant; class centers are
the only fitted prior statistics. Record the all-source design exposure above.

Keep the original 380 IDs, fits, renderer, descriptor definitions, scales,
quantiles, five seeds, 50 samples/class/seed/arm and family/fold/seed weights.
Acceptance remains >=80% mean marginal coverage and >=2.5% relative W1 gain
against each control in each class, with family spread [0.5, 2]. Rank by spread
failure, maximum class shortfall, mean W1, distance from identity, spectral
temperature, then offset. Rank does not imply acceptance.

Freeze all code and inputs before generation and replay every sample, ancestry,
center, descriptor, scale and score. Preserve every failed case. A passing
audited source candidate is required before any outer development evaluation;
the outer group has already been exposed in H1 and CAST v0 and is not fresh
confirmation. No classification or cross-dataset transfer is authorized here.

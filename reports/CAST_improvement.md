# CAST coverage improvement — development record

**Status: the stated held-group development objective is achieved and fully
audited.** The source-selected v15 width **T=1.1** reaches **82.019% car /
81.286% truck mean marginal coverage**, above the 80% target. W1 reductions
against matched marginals are **12.295% / 17.207%**, and against prototypes
**20.946% / 24.048%**, exceeding the required 2.5% in each class. All joint
family spreads remain in [0.5,2].

All **306 tests pass**. Complete v15 replay verifies **30,000 source samples,
600 source scores, 150 exact T=1 reference scores, 1,500 outer samples and
570 raw held descriptors**, including all prior statistics, training-only
ancestry and original inputs. The frozen v11 bank retains its complete
380-calibration/760-reconstruction audit. Historical results, datasets,
splits, environments and tracked ABVID work are unchanged.

The improvement is broader coverage, with a trade-off: class-balanced outer
W1 is 4.460% worse than the narrower v14 reference. Two car synthesis seeds
remain below 80%; the declared mean over all five fixed seeds passes.
This is one repeatedly exposed IDMT development group, not fresh independent
confirmation, classification accuracy or real-world generalization. The
marginal control remains dependent on the chosen 69-coordinate representation.

The objective is at least **80% mean marginal coverage in each class** and at
least **2.5% relative reduction in descriptor W1 against each of the prototype
and independent-marginal controls, in each class**. The original family-spread
range [0.5, 2], numerical checks and provenance requirements are retained. The
historical v0 experiment and its earlier 5% criterion remain unchanged in
[CAST_generalization.md](CAST_generalization.md). These are acoustic descriptor
criteria, not classification accuracy.

## Method, data and preservation

The new implementation is isolated under [CAST/improvement](../CAST/improvement).
Smooth8 replaces the original hard noise bands with continuous interpolation of
log power at eight fixed frequency centers. It retains eight harmonics, three
harmonic-spacing knots, five envelope knots, the original four-start/300-step
optimizer budget, preprocessing and multi-resolution spectral objective.
Parameters describe effective recorded observations; RPM, physical vehicle
identity, load and geometry remain unknown.

The pilot uses the same 50 parents as v0: five recordings per class and source
group. The expansion uses exactly H1 fold 0 / seed 42's 380 training IDs,
190 per class, across the same five source groups. More excerpts do not create
new independent sessions or vehicle identities. All inputs retain their raw
hashes, class/group ancestry and fixed CH34 preprocessing.

The outer group `connected_4001f06f57cfeee7` is excluded from fitting and prior
selection. It was already exposed in H1 and CAST v0; any eventual comparison
is development evidence, not fresh confirmation. The v10 comparison is the
first new outer comparison in this improvement work; its access receipt records
the exact transition after generation and source verification. MELAUDIS, mixed H1 caches, military/reserved
audio, supplied synthetic banks, original datasets, splits, environments and
baseline artifacts are unchanged. The source access contract and decisions are
in [PROTOCOL.md](../CAST/improvement/PROTOCOL.md) and
[RESEARCH_LOG.md](../CAST/improvement/RESEARCH_LOG.md).

## Completed reconstruction evidence

All seven original synthetic fixtures and the gain-ambiguity check passed with
the original tolerances. The first incomplete synthetic run is retained with
its failure record: inconsistent normalization caused near-exact tonal replay
losses to differ. The corrected run uses consistent normalization and checking
batch shape; no acceptance threshold was relaxed and no real fitting preceded
the repair.

All 50 smooth8 pilot fits completed. Every checking loss improved over its v0
counterpart. Median paired relative reductions were **1.486% car** and **1.452%
truck**. Median checking losses changed from 1.402128 to 1.372945 for cars and
from 1.366418 to 1.352880 for trucks. These small gains are in-sample
reconstruction results.

Ambiguity remains substantial: equally good starts disagree in **42/50** fits
(22 cars, 20 trucks). Median band-energy L1 residuals are 0.265 car and 0.256
truck. The [training residual plot](../CAST/improvement/diagnostics/pilot_training_residuals/training_residuals.png)
shows remaining spectral mismatch around 1 kHz and modest temporal/modulation
residuals. These findings motivate targeted model checks; they do not identify
the underlying vehicle physics.

The [local audio review](../CAST/improvement/diagnostics/pilot_audio_review/index.html)
links all fits and displays 15 systematic examples: best/median/worst checking
loss plus the first input ID in each class/group. Original and reconstructed
audio share a playback gain, and all alternative starts, diagnostics, parameters,
failure flags and source hashes remain available. All 291 page links were
verified. Derivatives are for local research under the recorded source licence,
CC BY-NC-ND 4.0; this report does not authorize redistribution.

All **380/380** full-bank fits subsequently completed without numerical/input
failures, and every raw observation, reconstructed waveform, component and
checking start passed replay. Median checking losses are 1.375286 car and
1.371373 truck; median reductions versus each clip's best unoptimized checking
start are 20.05% and 21.60%. Equally good starts disagree in **345/380** fits
(174 car, 171 truck), reinforcing the lack of physical identifiability. The
full [fit summary](../CAST/improvement/runs/cast_smooth8_v1_20261004_r1/fit_full_summary.json)
and [replay audit](../CAST/improvement/runs/cast_smooth8_v1_20261004_r1/fit_full_verification.json)
retain all records.

## Source-group generation evidence

These comparisons leave out one of the five **training** groups at a time.
Prior statistics and descriptor scales are recomputed from the other groups.
They use five seeds, three matched arms and 50 generated samples per class,
seed and arm. The pilot contains only five validation recordings per class in
each fold. Fitter design has already used these development groups.

| Renderer, same 50 parents | Car joint W1 | Truck joint W1 | Car coverage | Truck coverage |
|---|---:|---:|---:|---:|
| v0 | 0.753893 | 0.711373 | 78.760% | 79.859% |
| smooth8 | 0.705477 | 0.678869 | 80.754% | 81.691% |

Smooth8 improves absolute joint W1 and coverage. It still loses on W1 to the
independent-marginal control. The generic maximum-shortfall rank in this
two-renderer ablation slightly favors v0; it is preserved as such. This
ablation was not used to select an outer candidate. Expansion of smooth8 is
supported by the paired reconstruction and absolute source-distance evidence.

A second source-only experiment tests the already declared uniform spread
settings, applied to every matched arm in transformed parameter coordinates:

| Temperature | Car coverage | Truck coverage | Car gain vs marginals | Truck gain vs marginals |
|---|---:|---:|---:|---:|
| 1.0 | 80.754% | 81.691% | -5.880% | -2.886% |
| 1.1 | 83.571% | 84.176% | -7.336% | -4.739% |
| 1.25 | 86.509% | 87.405% | -9.556% | -7.373% |
| 1.5 | 90.467% | 90.023% | -13.823% | -11.297% |

**All four candidates fail the combined source criteria.** Wider sampling
raises coverage but worsens distance. Temperature 1.25 violates the car band
spread bound; 1.5 violates spectral/band spread bounds in both classes and
loses to both controls. Temperature 1 ranks best under the fixed rule. Its 150
score records reproduce the earlier experiment exactly. See the
[comparison plot](../CAST/improvement/diagnostics/pilot_temperature_comparison/source_comparison.png),
[group/family plot](../CAST/improvement/diagnostics/pilot_temperature_comparison/source_families_groups.png)
and [machine-readable failure](../CAST/improvement/evaluations/smooth8_pilot_temperature_grid/scientific_failure.json).

The full-bank source grid is also complete and audited:

| Temperature | Car coverage | Truck coverage | Car gain vs marginals | Truck gain vs marginals |
|---|---:|---:|---:|---:|
| 1.0 | 81.676% | 80.777% | -5.952% | -3.853% |
| 1.1 | 85.117% | 83.643% | -8.468% | -5.187% |
| 1.25 | 88.305% | 87.052% | -13.505% | -8.519% |
| 1.5 | 91.362% | 90.219% | -21.784% | -14.380% |

Temperature 1 again ranks best, with W1 0.619467 car and 0.611255 truck. It
beats the prototype by 16.07% and 15.39%, but fails against independent
marginals. All four candidates fail the combined source criteria, so the
continuation correctly stopped without reading outer descriptors. This result
is preserved in the [full source summary](../CAST/improvement/evaluations/smooth8_full_source_grid/summary.json),
[failure record](../CAST/improvement/evaluations/smooth8_full_source_grid/scientific_failure.json)
and [figures](../CAST/improvement/diagnostics/smooth8_full_source_grid_figures/source_comparison.png).
The [full audio review](../CAST/improvement/diagnostics/smooth8_full_source_grid_audio/index.html)
links all 380 fitted parents.

At temperature 1, full-source spectral/band spread ratios are 1.39/1.61 car
and 1.28/1.48 truck, while envelope/modulation ratios are 0.96/0.81 and
0.97/0.83. This supports testing separate spread controls before refitting a
larger renderer. [Prior v2](../CAST/improvement/prior_v2/PROTOCOL.md) freezes
twelve combinations: spectral temperature [0.5, 0.65, 0.8, 1.0] and envelope
temperature [1.0, 1.25, 1.5]. Every matched arm receives the same transformed
donor bank, with class centers and descriptor scales fitted inside each source
fold. All center-calibration ancestors are explicitly retained.

**All twelve candidates failed.** The selected least-shortfall candidate,
spectral temperature 0.8 and envelope temperature 1.0, has car/truck coverage
76.81%/75.99% and gains against marginals of -2.08%/-0.82%. The 0.8/1.5
candidate reaches 81.73%/80.95% coverage but still loses by 4.98%/3.22% against
marginals. The 0.5/1.0 candidate gives small positive marginal gains,
0.34%/0.60%, but coverage drops to 65.57%/65.17%. The tested spread adjustments
therefore do not solve both requirements. See the
[complete grid](../CAST/improvement/diagnostics/smooth8_block_v2_source_figures/source_grid.png),
[summary](../CAST/improvement/evaluations/smooth8_block_v2_full_source/summary.json)
and [exact audit](../CAST/improvement/evaluations/smooth8_block_v2_full_source/audit_verification.json).
The identity setting reproduces all 150 corresponding v1 score records exactly.

The fixed [capacity revision](../CAST/improvement/capacity_v3/PROTOCOL.md)
increases spectral controls from eight to sixteen and envelope knots from five
to nine. Persistent source reconstruction residuals motivate this combined
resolution test; it does not isolate the contribution of each change. Its
107-test fitting gate and all seven original synthetic recovery cases passed
with unchanged thresholds in 67.46 s.

All fifty real pilot fits completed and passed exact raw/component/waveform and
alternative-start replay. Forty-nine improve on smooth8; median paired checking
loss reductions are 0.759% car and 0.857% truck. One car becomes 0.182% worse.
Spectral, band and envelope median errors improve in both classes, but truck
modulation error worsens. Forty-three of fifty fits retain disagreeing nearly
equal starts. See the [paired plots](../CAST/improvement/capacity_v3/diagnostics/capacity16x9_pilot_source_paired/paired_reconstruction.png),
[residuals](../CAST/improvement/capacity_v3/diagnostics/capacity16x9_pilot_source_residuals/training_residuals.png)
and [audio review](../CAST/improvement/capacity_v3/diagnostics/capacity16x9_pilot_source_audio/index.html).
All 291 audio-review links were checked; it contains 30 players and all fifty fits.

| Fixed pilot renderer | Car W1 | Truck W1 | Car coverage | Truck coverage | Car gain vs marginals | Truck gain vs marginals |
|---|---:|---:|---:|---:|---:|---:|
| smooth8 | 0.705477 | 0.678869 | 80.754% | 81.691% | -5.880% | -2.886% |
| capacity16x9 | 0.694845 | 0.672773 | 83.776% | 83.244% | -6.916% | -2.088% |

The larger renderer improves absolute source W1 and coverage, but fails the
margin. The unchanged maximum-shortfall rule ranks smooth8 first; both fail.
All 15,000 samples and 300 scores passed exact replay, and all 150 smooth8
scores exactly reproduce the earlier pilot comparison. The
[source plot](../CAST/improvement/capacity_v3/diagnostics/capacity16x9_pilot_source_figures/source_comparison.png)
and [audit](../CAST/improvement/capacity_v3/evaluations/capacity16x9_pilot_source/audit_verification.json)
preserve this result. No full capacity-v3 expansion or outer comparison ran.

The subsequent source-only hypothesis retained nearly equal fitted alternatives.
The existing 380-parent smooth8 bank has 732 car and
703 truck starts within the already frozen 5% fitting-loss ambiguity bound;
379 parents have more than one. These are effective parameter alternatives,
not independent recordings or a calibrated Bayesian posterior. A matched
complete-vector versus independent-scalar comparison must use the same
alternatives and equal group/parent weighting in every arm.

That [fixed comparison](../CAST/improvement/alternatives_v4/PROTOCOL.md) is now
complete and fails. The alternative mixture slightly improves joint W1 to
0.614612 car and 0.607530 truck, with coverage 82.832%/81.792%. Gains against
the prototype are 17.965%/18.180%, but gains against marginals are
**-4.978%/-5.279%**. All 15,000 samples and 300 scores pass exact replay; the
150 winner-only scores exactly reproduce v1. See the
[figure](../CAST/improvement/alternatives_v4/diagnostics/smooth8_alternatives_v4_figures/source_comparison.png)
and [audit](../CAST/improvement/alternatives_v4/evaluations/smooth8_alternatives_v4_full_source/audit_verification.json).
At that stage there was no passing source candidate or new outer comparison.

The [source-only calibration](../CAST/improvement/calibration_v5/PROTOCOL.md)
tests the consistent paired spectral bias across training groups. It estimates
group-balanced observed/reconstructed band-power ratios inside each training
fold, bounds the correction, and applies it to every arm's same noise-control
bank. The fixed eight cases combine correction strength zero/one with the
already tested spectral temperatures 0.5/0.65/0.8/1.0, leaving envelope
temperature one. All eight candidates fail. The unchanged source ranking
selects the uncorrected temperature-0.8 case. At temperature one, correction
reduces W1 to 0.604041 car and 0.597586 truck and gives coverage
82.024%/81.261%, but marginal gains worsen to -6.378%/-4.486%.
The [complete grid](../CAST/improvement/calibration_v5/diagnostics/smooth8_bias_v5_figures/source_grid.png)
and [audit](../CAST/improvement/calibration_v5/evaluations/smooth8_bias_v5_full_source/audit_verification.json)
retain all cases. All 60,000 sample replays and 1,200 scores passed; the 600
zero-correction scores exactly reproduce prior v2. No outer comparison ran.

The subsequent [component diagnostic](../CAST/improvement/diagnostics/smooth8_component_allocation_r1/summary.json)
examines effective harmonic/noise allocation. Scaling saved components with a
harmonic logit offset of -1 reduces same-parent band-L1 error in 161/190 cars
and 165/190 trucks, with positive median improvement in every class/group.
Median absolute band-L1 reductions are 0.09216/0.09070. Offset -2 overcorrects
several groups. This approximate same-source diagnostic neither changes stored
fits nor supplies a held score. Its initial ambiguous audio-access metadata
label is preserved with an erratum; the corrected r1 distinguishes saved
training-waveform reads from raw dataset reads, with identical numerical results.

[Mixture v6](../CAST/improvement/mixture_v6/PROTOCOL.md) tested six fixed
cases: spectral temperature 0.65/0.8/1.0 and harmonic-logit offset 0/-1. Every
case failed. The selected unit-temperature, offset -1 case reduces W1 by
7.8% car and 9.4% truck relative to unadjusted joint sampling, reaching
0.571374/0.553647. Coverage is 79.896%/78.888%; gains against the prototype
are 19.831%/21.191%, while gains against marginals remain -1.962%/-0.459%.
The [grid](../CAST/improvement/mixture_v6/diagnostics/smooth8_mixture_v6_figures/source_grid.png)
and [exact audit](../CAST/improvement/mixture_v6/evaluations/smooth8_mixture_v6_full_source/audit_verification.json)
retain the failure. All 45,000 samples and 900 scores replay exactly; all
450 zero-offset reference scores match prior v2. No outer comparison ran.

The [per-parent mixture calibration](../CAST/improvement/parent_mixture_v7/PROTOCOL.md)
is complete and also fails. It changes one coordinate per training parent using
saved component powers and observed band proportions. Its selected bank gives
W1 0.573288/0.560044, coverage **80.426%/79.698%**, prototype gains
19.595%/20.448%, and marginal gains **-1.650%/-0.438%** (car/truck).
All 380 calibrations and 15,000 sample records replay exactly, with 300 scores
recomputed and 150 original-bank scores identical to v1. See the
[source comparison](../CAST/improvement/parent_mixture_v7/diagnostics/smooth8_parent_mixture_v7_figures/source_comparison.png),
[calibration and audio review](../CAST/improvement/parent_mixture_v7/diagnostics/smooth8_parent_mixture_v7_audio/index.html)
and [audit](../CAST/improvement/parent_mixture_v7/evaluations/smooth8_parent_mixture_v7_full_source/audit_verification.json).

Median effective harmonic fractions fall from 0.335 to 0.156 for cars and
0.347 to 0.164 for trucks. Twelve cars/five trucks reach a boundary, while
33 cars/15 trucks receive a noise-dominant spacing-ambiguity flag; all are
retained. Saved checking components are calibration inputs for these derived
vectors, so their old checking losses cannot validate the new bank independently.
Original fits remain unchanged. Ten audio examples follow first-ID-per-class/group
selection, independent of fit quality.

The [direct-envelope experiment](../CAST/improvement/envelope_v8/PROTOCOL.md)
is now complete. It calibrates the five existing envelope knots to observed
loudness trajectories, preserving their bounds and each parent's mean amplitude
gauge. There were zero solver failures and one truck envelope boundary hit;
all parents remain. Same-parent envelope reconstruction error improves for
all 380 clips. Median envelope RMSE changes from 0.11169 to 0.09208 for cars
and 0.11168 to 0.09158 for trucks. Modulation error improves for 162/190 cars
and 168/190 trucks. This is calibration reconstruction, not independent validation.

| V8 source candidate | Car W1 | Truck W1 | Car coverage | Truck coverage | Car gain vs marginals | Truck gain vs marginals |
|---|---:|---:|---:|---:|---:|---:|
| Mixture-only reference | 0.573288 | 0.560044 | 80.426% | 79.698% | -1.650% | -0.438% |
| Mixture plus envelope | 0.568256 | 0.564910 | 80.626% | 80.448% | -1.254% | -1.267% |

The second candidate ranks first and passes both coverage and spread checks,
but fails the marginal-control margin. Truck absolute W1 worsens despite better
same-parent reconstruction. All 150 mixture-only scores exactly reproduce v7.
The [source figure](../CAST/improvement/envelope_v8/diagnostics/smooth8_envelope_v8_figures/source_comparison.png),
[same-parent residuals/audio](../CAST/improvement/envelope_v8/diagnostics/smooth8_envelope_v8_audio_r1/index.html)
and [exact audit](../CAST/improvement/envelope_v8/evaluations/smooth8_envelope_v8_full_source/audit_verification.json)
retain this result. The audio review uses the first sorted ID in each class/group,
independent of quality, and links ten examples with three matched players each.

The average also masks substantial group variation. For omitted group
`connected_72766641e1daf436`, joint-vs-marginal gains are -20.266% car and
-9.009% truck; `connected_c6148cf0cafb5753` gives -5.102%/-7.886%.
Other groups can favor joint sampling. This is evidence of unstable development
generalization, not a reason to remove or reweight the difficult groups. Spectral
and band-power distances still dominate the total W1, while both classes lose
the modulation comparison. Further work should diagnose these residuals before
freezing another model revision. A passing source candidate is still required
before another outer development comparison.

The [joint expected-spectrum calibration](../CAST/improvement/spectrum_v9/PROTOCOL.md)
fits each parent's noise weights and mixture together, preserving v8's envelope
and the original harmonic spacing/weights. It uses separate component Welch
powers and ignores the stochastic cross term. All 190 numerical/provenance
tests passed before the bank freeze. All 380 parents completed in 98.92 s,
with zero numerical failures; median paired calibration-objective reductions
are 50.20% car / 51.68% truck. All 380 optimizer histories/parameters and 760
reconstruction waveforms replay exactly in 98.06 s.

Actual same-parent log-spectrum RMSE improves for all 380 parents: medians
0.33519→0.26774 car and 0.31930→0.25129 truck. Band L1 improves in 190/190
cars and 189/190 trucks, with medians 0.12751→0.03301 and 0.10827→0.02781.
Temporal residuals have mixed results. These are calibration reconstructions;
the original checking streams are calibration inputs, not independent validation.
See the [audio/residual review](../CAST/improvement/spectrum_v9/diagnostics/smooth8_spectrum_v9_audio/index.html).

The 192-test source gate passed, but both candidates fail scientifically.
V9 W1 is 0.562936/0.560583 and coverage 80.494%/81.438%; gains against the
prototype are 19.902%/18.401%, while gains against marginals are
**-1.408%/-2.387%**. The unchanged rule selects v8. All 15,000 samples and
300 scores replay exactly in 127.32 s; all 150 v8 scores are unchanged. See
the [comparison](../CAST/improvement/spectrum_v9/diagnostics/smooth8_spectrum_v9_figures/source_comparison.png)
and [audit](../CAST/improvement/spectrum_v9/evaluations/smooth8_spectrum_v9_full_source/audit_verification.json).
The [bank run](../CAST/improvement/spectrum_v9/runs/cast_spectrum_v9_20261004)
retains every history, parameter vector, ancestry record and reconstruction.

The next [temporal-resolution comparison](../CAST/improvement/temporal_v10/PROTOCOL.md)
uses 41 envelope controls at the existing 50 ms frame boundaries, initialized
by the exact nested five-knot curve. A fixed curvature penalty limits rapid
oscillation; original bounds and mean-amplitude gauge remain. V8 and v9 are
both retained as exact references. All three arms use the same bank within
its parameterization, including all 61 independent scalar coordinates in the
new control. This tests preserved temporal dependence under a richer envelope
representation; it cannot establish general superiority over other independent
models. Eleven focused checks pass after tightening solver termination to fix
a synthetic affine-recovery failure, without changing the acceptance tolerance.
The fixed source comparison passes, with all reference records unchanged:

| Source candidate | Car W1 | Truck W1 | Car coverage | Truck coverage | Car gain vs marginals | Truck gain vs marginals |
|---|---:|---:|---:|---:|---:|---:|
| v8 reference | 0.568256 | 0.564910 | 80.626% | 80.448% | -1.254% | -1.267% |
| v9 spectral | 0.562936 | 0.560583 | 80.494% | 81.438% | -1.408% | -2.387% |
| temporal41 | 0.554572 | 0.553121 | 83.330% | 83.981% | +9.266% | +8.491% |

Joint W1 improves 1.49% car / 1.33% truck relative to v9. Much of the matched
margin arises because independently sampled 50 ms envelope controls lose
continuity. Against the earlier v9 marginal scores, temporal41 joint gains
would be only about +0.10% car and -1.02% truck. These are different
parameterizations; neither the richer matched-control pass nor this diagnostic
comparison establishes universal generator superiority. No metric, quantile,
split, sample budget, class/group weight or acceptance threshold changed.

The source gate passed 203 tests; all 205 tests including outer guards pass in
the [complete test record](../CAST/improvement/temporal_v10/outer_gate/tests.xml).
The [source summary](../CAST/improvement/temporal_v10/evaluations/smooth8_temporal_v10_full_source/summary.json)
and [complete audit](../CAST/improvement/temporal_v10/evaluations/smooth8_temporal_v10_full_source/audit_verification.json)
record 22,500 exact sample replays, 380 rederived local calibrations and 450
recomputed scores. All 300 v8/v9 reference scores remain unchanged. The fixed
source ranking selects temporal41. The [source plots](../CAST/improvement/temporal_v10/diagnostics/smooth8_temporal_v10_figures/source_comparison.png)
and [audio/residual review](../CAST/improvement/temporal_v10/diagnostics/smooth8_temporal_v10_audio/index.html)
retain ten systematic examples. Same-parent envelope RMSE improves for all 380
parents (medians 0.09178→0.04139 car, 0.09172→0.04188 truck); modulation L1
improves for 184/190 in each class. Three envelopes reach their bounds; no
solver failed or parent was removed. These remain calibration diagnostics.

## Frozen temporal41 outer development comparison

After the source pass and full audit, the selected bank/configuration and all
1,500 generated samples were frozen before reading the unchanged 570-record
outer descriptor cache. Every generated waveform, parameter choice, stream,
ancestor and descriptor replays exactly. Every raw held recording was
reprocessed, all scores independently recomputed, and historical artifacts and
tracked ABVID edits checked unchanged. See the [verification](../CAST/improvement/temporal_v10/evaluations/temporal_v10_outer_development/verification.json)
and [generation/access chronology](../CAST/improvement/temporal_v10/evaluations/temporal_v10_outer_development/held_access_receipt.json).

| Outer development measure | Car | Truck | Requirement |
|---|---:|---:|---:|
| Joint W1 | 0.697096 | 0.640628 | lower is better |
| Joint mean marginal coverage | 77.212% | 75.571% | >=80% each |
| Relative W1 gain vs prototype | 23.104% | 26.155% | >=2.5% each |
| Relative W1 gain vs matched marginals | 11.885% | 8.731% | >=2.5% each |
| Joint family spread, minimum–maximum | 0.774–1.226 | 0.856–1.017 | [0.5,2] |

The retained [scientific failure record](../CAST/improvement/temporal_v10/evaluations/temporal_v10_outer_development/failure.json)
reports both coverage failures, zero numerical failures and zero dropped
records. Coverage remains short by 2.788 and 4.429 percentage points. Absolute
joint W1 improves from the original 0.953320/0.804047; the matched independent
controls also change with the new representation. The [outer plots and audio review](../CAST/improvement/temporal_v10/diagnostics/temporal_v10_outer_review/index.html)
show fixed first-ID originals and seed-42/index-0 generated examples. Generated
audio is an unconditional draw, not a reconstruction of its neighboring held
example. Five generation seeds measure sampler variability; they are not five
independent real sessions. This one exposed site/date group cannot establish
physical-vehicle independence, cross-dataset transfer or classification quality.

The next source-only hypothesis is sixteen nested spectral controls, motivated
by the still-large spectral/band source W1 (about 0.72–0.77, versus modulation
0.21–0.22). It preserves temporal41, all eight analysis bands and every criterion.
See [v11 protocol](../CAST/improvement/resolution_v11/PROTOCOL.md). Outer aggregate
exposure is disclosed; no held residuals, parameters, EQ or descriptor scales
enter this next calibration or selection. All 224 tests passed before v11 calibration.
All 380 parents completed without numerical failures in 147.73 s; median
paired expected-objective reductions are 66.64% car / 59.66% truck. The
[bank](../CAST/improvement/resolution_v11/runs/cast_resolution_v11_20261005)
retains every optimization history and 760 reconstruction WAVs. All 380 optimizations and 760 WAV sample arrays replay exactly in 132.92 s.
The unchanged source rule selects spectrum16: W1 0.548904/0.549782,
coverage 84.119%/84.571%, prototype gains 20.490%/19.437%, and matched
marginal gains 10.028%/8.340%. All family spreads pass. Joint W1 improves
only 1.02% car / 0.60% truck against temporal41 despite the much larger
calibration-objective reductions. Both matched marginal W1 scores also improve;
this incremental pass is not caused by worsening that control. The
[source summary](../CAST/improvement/resolution_v11/evaluations/smooth16_temporal41_v11_full_source/summary.json)
retains both candidates. The [full source audit](../CAST/improvement/resolution_v11/evaluations/smooth16_temporal41_v11_full_source/audit_verification.json)
passed all 15,000 generated records and 300 scores in 255.74 s; all 150
temporal41 reference scores are exact. Both source figures were generated and the
[coverage/margin plot](../CAST/improvement/resolution_v11/diagnostics/smooth16_temporal41_v11_figures/source_comparison.png)
was visually inspected. The [local audio/residual review](../CAST/improvement/resolution_v11/diagnostics/smooth16_temporal41_v11_audio/index.html)
contains ten first-ID examples, common playback gains and complete parameters.
All 41 links and 40 example artifact hashes were checked. Actual reconstruction
log-spectrum RMSE improves for all 380 parents: medians 0.27004→0.15524 car
and 0.25123→0.15602 truck. Band L1 improves for 154/190 per class; most
envelope/modulation errors slightly worsen. Fourteen mixture estimates reach
a boundary, and 67 noise-dominant fits carry spacing-unidentified flags.
No failure or ambiguity was discarded.

## Frozen spectrum16 outer development comparison

The newly selected candidate was separately frozen after source replay. Its
1,500 generated waveforms and sidecars were complete before held-descriptor
access. The [complete outer verification](../CAST/improvement/resolution_v11/evaluations/resolution_v11_outer_development/verification.json)
replayed every generated sample, reprocessed all 570 raw held recordings,
recomputed scores and checked all ancestry plus historical/ABVID preservation.

| Outer development measure | Car | Truck | Requirement |
|---|---:|---:|---:|
| Joint W1 | 0.694522 | 0.643825 | lower is better |
| Joint mean marginal coverage | 77.984% | 76.054% | >=80% each |
| Relative W1 gain vs prototype | 22.272% | 25.177% | >=2.5% each |
| Relative W1 gain vs matched marginals | 9.271% | 9.604% | >=2.5% each |
| Joint family spread, minimum–maximum | 0.777–1.225 | 0.857–1.046 | [0.5,2] |

Both coverage criteria fail. The [failure record](../CAST/improvement/resolution_v11/evaluations/resolution_v11_outer_development/failure.json)
retains these failures, zero numerical failures and zero dropped records.
Relative to temporal41, coverage rises only 0.771 car / 0.483 truck percentage
points; car W1 improves slightly, while truck W1 worsens. The
[outer plots and audio review](../CAST/improvement/resolution_v11/diagnostics/resolution_v11_outer_review_r1/index.html)
use fixed first-ID originals and seed-42/index-0 generated draws. They are
unpaired examples, not held-waveform reconstructions.

Further spectral capacity has diminishing distribution-level returns in these
experiments. In the source folds, worst-group coverage is 79.348% car and
79.254% truck even though the means exceed 84%. The next justified work is to
inspect source-group distribution variation and seed sensitivity, then freeze
one prior/sampling hypothesis with all acceptance criteria and group weights
unchanged. No new prior or wider scoring interval has been introduced. Do not
use outer residuals, means, EQ, scales or sample choices to tune that revision.
This v11 result left the goal scientifically unmet.

## Source-group prior v12

A [source-only variance diagnostic](../CAST/improvement/diagnostics/spectral16_source_group_variation/summary.json)
attributes 27–39% of spectral/band descriptor variance and 8–15% of temporal
variance to differences between group means. Its
[plot](../CAST/improvement/diagnostics/spectral16_source_group_variation/group_variance.png)
uses equal source-group and parent weights. These descriptive components
include finite-parent sampling noise and do not identify causal recording
effects or independent physical vehicles.

The fixed [v12 protocol](../CAST/improvement/group_v12/PROTOCOL.md) keeps all
380 v11 fits and the renderer unchanged. It compares exact v11 sampling,
recombination of full within-group residual vectors and group mean shifts,
and symmetric group shifts scaled by sqrt((G+1)/(G-1)). G is the training
group count, not a tuned parameter. Every arm draws from the same finite
virtual population, with equal original group/parent weights. All center
ancestors, projections and uncertainty flags remain recorded. No additional
real parent, target-derived statistic or scalar-temperature search is used.

| Source candidate | Car W1 | Truck W1 | Car coverage | Truck coverage | Car / truck gain vs marginals |
|---|---:|---:|---:|---:|---:|
| V11 reference | 0.548904 | 0.549782 | 84.119% | 84.571% | 10.028% / 8.340% |
| Group recombined | 0.552698 | 0.557845 | 82.476% | 84.167% | 9.797% / 7.668% |
| Group predictive | 0.561161 | 0.524022 | 84.745% | 87.134% | 9.564% / 16.014% |

All three pass the source criteria. The unchanged rule selects group predictive
because its class-balanced mean W1 is lowest (0.542591 versus 0.549343 for v11).
Truck W1 improves while car W1 worsens. Prototype gains are 18.317% car and
23.162% truck, and all family spreads pass. The minimum source-fold mean
coverage remains 79.646% for cars; trucks reach 85.993%. Mean source success
does not imply every group meets 80%, or establish the outer objective.
The new marginal control also changes: W1 rises from 0.610083/0.599809
to 0.620504/0.623941. The larger truck control margin therefore combines
a better joint distribution with a worse matched marginal distribution.
The class-balanced absolute joint W1 reduction is only about 1.23%.

All 241 tests passed before generation. A pre-freeze numerical endpoint
rounding defect was fixed without relaxing the test; see the
[implementation record](../CAST/improvement/group_v12/implementation_defects.json).
The [complete source results](../CAST/improvement/group_v12/evaluations/spectrum16_group_v12_full_source/summary.json)
retain all candidates. [Full replay](../CAST/improvement/group_v12/evaluations/spectrum16_group_v12_full_source/audit_verification.json)
passed for all 22,500 samples, 450 scores and fold-specific prior statistics.
All 150 v11 reference scores remain exact. Source generation/scoring took
283.48 s after 73.96 s of upstream verification; replay took 337.70 s. The
selected new candidate proceeds to one separately frozen outer comparison.

## Frozen group-predictive v12 outer development comparison

The new source-selected candidate was frozen on all 380 allowed parents,
including all group centers, bounded virtual children, prototype, seeds and
complete direct/statistical ancestry. All 1,500 waveforms were generated
before reading the held descriptor cache. The empirical group-count factor
is sqrt(6/4) on these five source groups; no outer statistic determines it.

| Arm | Car W1 | Car coverage | Truck W1 | Truck coverage |
|---|---:|---:|---:|---:|
| Joint | 0.694225 | 79.359% | 0.639702 | 78.997% |
| Marginals | 0.785686 | 67.468% | 0.775717 | 66.509% |
| Prototype | 0.896279 | 23.666% | 0.871750 | 26.088% |

Both classes still fail coverage. Marginal gains 11.641%/17.534%, prototype
gains 22.544%/26.619%, and every joint family spread pass. Relative to v11,
coverage rises by 1.376 pp car and 2.943 pp truck; absolute joint W1 improves
slightly in both classes. No sample is dropped. The
[complete scores](../CAST/improvement/group_v12/evaluations/group_v12_outer_development/scores.json),
[failure record](../CAST/improvement/group_v12/evaluations/group_v12_outer_development/failure.json),
[frozen group statistics](../CAST/improvement/group_v12/evaluations/group_v12_outer_development/group_effect_statistics.json)
and [full replay verification](../CAST/improvement/group_v12/evaluations/group_v12_outer_development/verification.json)
retain the result and all numerical/provenance evidence.

The [source comparison](../CAST/improvement/group_v12/diagnostics/spectrum16_group_v12_figures/source_comparison.png)
and [group/family plot](../CAST/improvement/group_v12/diagnostics/spectrum16_group_v12_figures/source_families_groups.png)
show all candidates. The [source audio review](../CAST/improvement/group_v12/diagnostics/spectrum16_group_v12_audio/index.html)
contains ten fixed matched examples, with original parents, fitted reconstructions
and three generated draws; all 62 links and 61 artifact hashes pass verification.
The [outer review](../CAST/improvement/group_v12/diagnostics/group_v12_outer_review/index.html)
shows fixed held originals and unconditional generated examples, not held refits.
No independent vehicle/session generalization or classification result is claimed.

The next source-only diagnostic examines finite sampling at the unchanged
50-example budget. Its [saved results](../CAST/improvement/diagnostics/group_v12_source_seed_variation/summary.json)
show within-fold coverage seed SD averaging 3.175 pp car / 2.532 pp truck.
Random source-group counts span 7–20 / 7–22 instead of the expected 12.5 per
inner-fold group. Phase/noise variation is also present, so these numbers do
not isolate donor imbalance as the cause. They motivate a balanced-sampling
hypothesis; no future gain is assumed and no outer descriptors inform it.

## Balanced finite sampling v13

The fixed [balanced-sampling protocol](../CAST/improvement/balanced_v13/PROTOCOL.md)
changes allocation of the same 50 draws: balance source-group counts, balance
parent use inside each selected group, and independently balance virtual-child
indices. The marginal arm gets independent plans per coordinate. The parameter
population, renderer, prototype, phase/noise streams and scientific criteria
remain unchanged. All plans are frozen before generation.

| Source candidate | Car W1 | Truck W1 | Car coverage | Truck coverage | Car / truck gain vs marginals |
|---|---:|---:|---:|---:|---:|
| Exact v12 reference | 0.561161 | 0.524022 | 84.745% | 87.134% | 9.564% / 16.014% |
| Balanced allocation | 0.567073 | 0.541720 | 84.619% | 87.805% | 6.526% / 13.458% |

Both pass the source thresholds, but the unchanged rule retains v12. Balanced
sampling worsens joint W1 in both classes and the class-balanced mean by 2.18%
(0.554397 versus 0.542591). Prototype scores are identical. The result does
not support replacing the sampler, and **no new outer evaluation runs**.
All candidates are retained in the
[source summary](../CAST/improvement/balanced_v13/evaluations/group_balanced_v13_full_source/summary.json).
All 259 tests passed before generation. The [full replay](../CAST/improvement/balanced_v13/evaluations/group_balanced_v13_full_source/audit_verification.json)
passed for all 15,000 samples, 300 scores and allocation plans, with all 150
v12 reference scores exact. The [progression decision](../CAST/improvement/balanced_v13/evaluations/group_balanced_v13_full_source/progression_decision.json)
retains the reference and prohibits another outer check. The verified outer
result after this comparison was v12. The [source comparison](../CAST/improvement/balanced_v13/diagnostics/group_balanced_v13_figures/source_comparison.png),
[group/family plot](../CAST/improvement/balanced_v13/diagnostics/group_balanced_v13_figures/source_families_groups.png)
and [fixed audio review](../CAST/improvement/balanced_v13/diagnostics/group_balanced_v13_audio/index.html)
preserve the rejected variant. All 73 review links and 71 artifact hashes
pass verification. The ten cards explicitly identify separate residual parents
for the two allocation strategies; each card has six audio clips with one
common playback gain. No unchanged-parent pairing is implied.

## Scoped group effects v14

The [preregistered scope comparison](../CAST/improvement/context_v14/PROTOCOL.md)
preserves the original donor, virtual-child, phase and noise streams while
testing which fitted controls receive the v12 group transformation. Source
group variation is appreciably stronger in noise controls than harmonic
weights or spacing. Both new candidates therefore restore the original
parent's harmonic controls exactly; one also restores its envelope. These
blocks remain effective observation controls, not identified physical source
or microphone components. Prototypes are recomputed from each candidate's
equally weighted virtual population, and all 69 marginal coordinates share
that same population.

| Source candidate | Car coverage | Truck coverage | Mean joint W1 |
|---|---:|---:|---:|
| Exact v12 reference | 84.745% | 87.134% | 0.542591 |
| Noise and mixture scope | 84.552% | 86.447% | 0.546445 |
| Noise, mixture and envelope scope | 84.628% | 86.821% | 0.541608 |

All three pass the unchanged source criteria. The fixed ranking selects
`spectrotemporal_context_T1`: class-balanced mean W1 improves by 0.181% over
the reference, while mean coverage falls slightly in both classes. Its
car/truck gains against matched marginals are 9.829%/16.010%, and against
prototypes 18.288%/23.365%. This is a small source improvement, not a
demonstration that outer coverage has improved. The worst source-fold mean
car coverage is still 79.580%, despite the higher overall mean.

The [source results](../CAST/improvement/context_v14/evaluations/group_context_v14_full_source/summary.json)
retain every candidate. All 278 tests pass, including 19 new numerical,
ancestry and access checks. The [complete replay](../CAST/improvement/context_v14/evaluations/group_context_v14_full_source/audit_verification.json)
passes for all 22,500 generated records and 450 scores, all 380 parent inputs,
frozen group statistics and training-only scales. All 150 v12 reference scores
are exact. Restored harmonic controls avoid 335 car / 154 truck spacing
projections across the source folds; original parent ambiguity and boundary
flags remain intact. Repeated virtual children are not independent recordings.

The [source comparison plot](../CAST/improvement/context_v14/diagnostics/group_context_v14_figures/source_comparison.png),
[family/group plot](../CAST/improvement/context_v14/diagnostics/group_context_v14_figures/source_families_groups.png)
and [ten fixed audio cards](../CAST/improvement/context_v14/diagnostics/group_context_v14_audio/index.html)
retain all three variants. Both plots were visually checked. All 62 audio
review links, 61 artifact hashes and 50 waveform exports were verified.

The [progression decision](../CAST/improvement/context_v14/evaluations/group_context_v14_full_source/progression_decision.json)
permitted a separately frozen outer development comparison. All 1,500
waveforms were generated before reading held descriptors. The verified
[scores](../CAST/improvement/context_v14/evaluations/context_v14_outer_development/scores.json)
fail coverage in both classes:

| Class | Joint W1 | Joint coverage | Gain vs marginals | Gain vs prototype |
|---|---:|---:|---:|---:|
| Car | 0.694337 | 78.849% | 11.694% | 22.544% |
| Truck | 0.639688 | 78.725% | 16.771% | 26.681% |

Both gains and all joint family spreads pass. Compared with v12, coverage
falls by 0.510 percentage points for cars and 0.272 for trucks; the small
source W1 improvement did not solve outer coverage. V12 remained the best
verified outer coverage result at this stage. The [full outer replay](../CAST/improvement/context_v14/evaluations/context_v14_outer_development/verification.json)
passes for all 1,500 generated waveforms and 570 raw held descriptors, every
score, source-selected statistics and complete training ancestry. The
[fixed outer review](../CAST/improvement/context_v14/diagnostics/context_v14_outer_review/index.html)
shows both classes and all three arms. This scientific failure is retained
unchanged, without a new fit, discarded example or revised threshold.

## Source-fold width calibration v15

A [new source-fold protocol](../CAST/improvement/width_v15/PROTOCOL.md) was
preregistered before implementation or generation. It tests four fixed widths of the source-selected
noise/mix/envelope population and adds a requirement that every source fold's
mean coverage reach 80% in each class. All existing source criteria and actual
outer acceptance remain required. This strengthens internal selection instead
of letting a high source average hide the known failing fold. It does not
guarantee coverage in another group, and v14's already exposed outer failure
is explicitly recorded in the preregistration.

V15 implementation passes 28 focused tests and the complete 306-test gate.
The four-width source comparison selected T=1.1 and passed its complete
replay before the outer comparison. Only source folds determine the width;
no outer result enters this selection.
The width transform changes only noise/mix/envelope variation around the
equally weighted virtual-population center. T=1 is an exact v14 identity;
spacing and harmonic weights, donor/child draws and waveform streams remain
unchanged at every width. Matched prototypes are recomputed for every
population. Missing/duplicated folds or seeds fail selection before any outer
access. This is calibration of an observation distribution, not additional
renderer capacity or measured vehicle physics.

| Width | Mean car coverage | Mean truck coverage | Minimum car fold | Minimum truck fold | Mean joint W1 | Original criteria | Added fold gate |
|---|---:|---:|---:|---:|---:|---|---|
| T=1 reference | 84.628% | 86.821% | 79.580% | 85.912% | 0.541608 | Pass | Fail |
| **T=1.1 selected** | **87.246%** | **89.318%** | **83.280%** | **88.814%** | **0.568944** | **Pass** | **Pass** |
| T=1.25 | 89.958% | 91.643% | 87.133% | 90.545% | 0.626211 | Pass | Pass |
| T=1.5 | 92.091% | 93.176% | 91.061% | 91.622% | 0.756869 | Fail | Pass |

The [complete source results](../CAST/improvement/width_v15/evaluations/context_width_v15_full_source/summary.json)
expose the trade-off. T=1.1 improves mean coverage by 2.618/2.498 percentage
points and clears the weakest source folds, while class-balanced mean W1
is 5.047% worse than T=1. Its car/truck gains against matched marginals are
9.139%/15.824%, and against prototypes 14.400%/19.239%, all above 2.5%.
The original source ranking would retain T=1; the preregistered extra fold
gate excludes it. Among eligible widths, the unchanged original ranking
selects T=1.1. T=1.5 is rejected despite its high coverage: both classes
lose to the prototype and exceed the spectral spread limit. Coverage alone
is insufficient.

The selected width records 71 car / 84 truck envelope-coordinate projections
across source virtual populations; the upstream 18/37 projections and original
parent flags are retained separately. No parent or generated sample is removed.
These are repeated virtual-coordinate counts, not independent recordings or
counts of failed fits. No width is selected using an outer descriptor or fit.

The [full source audit](../CAST/improvement/width_v15/evaluations/context_width_v15_full_source/audit_verification.json)
replays all 30,000 generated records and 600 scores, all 380 parent inputs,
every prior statistic and training-only scale. The [reference check](../CAST/improvement/width_v15/evaluations/context_width_v15_full_source/identity_replication.json)
verifies all 150 T=1 v14 scores exactly. The [source decision](../CAST/improvement/width_v15/evaluations/context_width_v15_full_source/progression_decision.json)
permits only the selected T=1.1 candidate for the next outer comparison.

The [source coverage/margin plot](../CAST/improvement/width_v15/diagnostics/context_width_v15_figures_r1/source_comparison.png)
shows both the mean and minimum-fold coverage. The [family/group plot](../CAST/improvement/width_v15/diagnostics/context_width_v15_figures_r1/source_families_groups.png)
and [ten fixed audio cards](../CAST/improvement/width_v15/diagnostics/context_width_v15_audio/index.html)
expose all four widths. All 60 audio exports, 72 links and 71 artifact hashes
were verified. A presentation-only revision moved the legend above the bars;
both original and revised figures are retained and scientific inputs are unchanged.

## Frozen width-v15 outer development comparison

The source-selected **T=1.1** population was frozen before generating all
1,500 waveforms and then reading the same 570 held observations. The
[access receipt](../CAST/improvement/width_v15/evaluations/width_v15_outer_development/held_access_receipt.json)
records that order. No held recording is fitted and no width is chosen from
its scores. The [complete results](../CAST/improvement/width_v15/evaluations/width_v15_outer_development/scores.json)
meet every unchanged numerical criterion. The [full replay](../CAST/improvement/width_v15/evaluations/width_v15_outer_development/verification.json)
passes for all 1,500 generated waveforms and all 570 raw held descriptors,
with scores, source selection, group statistics and training ancestry
recomputed. The [failure record](../CAST/improvement/width_v15/evaluations/width_v15_outer_development/failure.json)
records no numerical failure, zero dropped records and a scientific pass.

| Class | Arm | W1 | Mean marginal coverage |
|---|---|---:|---:|
| Car | **Joint T=1.1** | **0.719913** | **82.019%** |
| Car | Marginals | 0.820840 | 69.969% |
| Car | Prototype | 0.910661 | 23.626% |
| Truck | **Joint T=1.1** | **0.673616** | **81.286%** |
| Truck | Marginals | 0.813613 | 68.909% |
| Truck | Prototype | 0.886901 | 25.670% |

| Class | Gain vs marginals | Gain vs prototype | Coverage >=80% | Both gains >=2.5% |
|---|---:|---:|---|---|
| Car | 12.295% | 20.946% | Pass | Pass |
| Truck | 17.207% | 24.048% | Pass | Pass |

Joint spread ratios also remain inside the unchanged [0.5,2] bounds:

| Class | Log spectrum | Bands | Envelope | Modulation |
|---|---:|---:|---:|---:|
| Car | 1.378 | 1.323 | 0.882 | 1.052 |
| Truck | 1.302 | 1.105 | 0.943 | 1.043 |

Relative to the narrower v14 reference, coverage increases by 3.170
percentage points for cars and 2.561 for trucks. Relative to the preceding
best-coverage v12 result, gains are 2.660 and 2.289 points. The width trade-off
is retained: W1 is 3.684%/5.304% worse than v14, or 4.460% worse in the
class-balanced mean. This meets the specified coverage-plus-control-margin
objective, not a claim that every acoustic matching measure improves.

| Synthesis seed | Car joint coverage | Truck joint coverage |
|---|---:|---:|
| 42 | 87.025% | 82.143% |
| 123 | 79.258% | 81.474% |
| 456 | 82.934% | 80.299% |
| 789 | 79.406% | 80.956% |
| 1024 | 81.473% | 81.558% |

Two car seeds remain below 80%; none is removed or rerun. The declared
acceptance uses the mean of all five fixed seeds. Their ranges describe
synthesis variation, not uncertainty over independent recording groups.
The single outer group has been repeatedly exposed during development.
There is no new-site, new-vehicle, classification-transfer or field-validation
claim, and the 69-coordinate marginal control remains representation dependent.

The [outer review](../CAST/improvement/width_v15/diagnostics/width_v15_outer_review/index.html)
provides eight fixed audio examples, both comparison plots, fitted-parameter
ancestry, playback gains, scores, failures and replay evidence. Generated
examples are unconditional draws, not fits to those held originals. Original
CAST pilot/generalization results and every unsuccessful revision remain
available; no failed source candidate or synthesis seed was removed.

## Verification, runtime and reproduction

The latest complete numerical/provenance suite has 205 passing tests. There
were 176 passing tests at the v8 source freeze.
It had 113 passing tests at the capacity source
freeze and 122 at the equivalent-fit source freeze, recorded in the latter's
[test log](../CAST/improvement/alternatives_v4/evaluations/smooth8_alternatives_v4_full_source/tests.log).
Added checks enforce
class-specific thresholds, reject duplicated or missing generated schedules,
and reject failed source candidates before any held access.

The [pilot fit audit](../CAST/improvement/runs/cast_smooth8_v1_20261004_r1/fit_pilot_verification.json)
reprocessed all 50 raw inputs, replayed every saved waveform and component,
and recomputed all fitted and initial checking starts with the unchanged 1e-5
loss tolerance. The three completed source experiments passed exact donor,
waveform and descriptor replay: 15,000 samples/300 scores for the renderer
ablation, and 30,000 samples/600 scores each for the pilot and full spread
grids. Scales and scores were
independently recomputed from the frozen source-only records.
Prior v2 additionally passed all 90,000 exact sample replays and 1,800 score
recomputations, including every direct donor and class-center ancestor.

| Completed stage | Measured wall time |
|---|---:|
| Synthetic acceptance, four CPU workers | 64.92 s |
| 50-parent smooth8 pilot, four CPU workers | 407.22 s |
| Additional 330 parents, reusing the 50 pilot fits | 2650.70 s |
| Pilot raw/fit/alternative-start replay audit | 7.88 s |
| Full 380-parent raw/fit/alternative-start replay audit | 27.06 s |
| Pilot spread-grid generation and scoring | 193.54 s |
| Pilot spread-grid exact replay audit | 190.22 s |
| Full-bank spread-grid generation and scoring | 119.66 s |
| Full-bank spread-grid exact replay audit | 114.90 s |
| Prior-v2 twelve-case generation and scoring | 420.29 s |
| Prior-v2 exact replay audit | 384.83 s |
| Capacity-v3 synthetic acceptance, four CPU workers | 67.46 s |
| Capacity-v3 fifty-parent pilot, four CPU workers | 459.37 s |
| Capacity-v3 pilot exact fit replay | 3.47 s |
| Capacity-v3 matched source generation/scoring | 67.97 s |
| Capacity-v3 matched source exact replay | 63.56 s |
| Equivalent-fit v4 full source generation/scoring | 192.63 s |
| Equivalent-fit v4 full source exact replay | 66.04 s |
| Band-bias v5 full source generation/scoring | 240.76 s |
| Band-bias v5 full source exact replay | 239.75 s |
| Mixture v6 full source generation/scoring | 186.36 s |
| Mixture v6 full source exact replay | 167.70 s |
| Per-parent mixture v7 source generation/scoring | 69.29 s |
| Per-parent mixture v7 exact replay | 71.37 s |
| V7 bank verification/calibration preparation | 7.78 s |
| Direct-envelope v8 bank verification/calibration preparation | 13.19 s |
| Direct-envelope v8 source generation/scoring | 93.89 s |
| Direct-envelope v8 exact replay | 93.47 s |
| Spectral v9 verified bank preparation/test gate | 20.41 s |
| Spectral v9 380-parent calibration, four workers | 98.92 s |
| Spectral v9 complete optimizer/WAV replay | 98.06 s |
| Spectral v9 source generation/scoring | 120.90 s |
| Spectral v9 source exact replay | 127.32 s |
| Temporal v10 source generation/scoring | 178.48 s |
| Temporal v10 complete source replay | 219.98 s |
| Complete 205-test gate | 15.82 s |
| Spectral-resolution v11 bank preparation/test gate | 81.92 s |
| Spectral-resolution v11 380-parent calibration | 147.73 s |
| Spectral-resolution v11 complete optimizer/WAV replay | 132.92 s |
| Spectral-resolution v11 source bank verification/preparation | 120.93 s |
| Spectral-resolution v11 source generation/scoring | 234.99 s |
| Spectral-resolution v11 complete source replay | 255.74 s |
| Temporal v10 outer preparation/generation/scoring | 77.92 s |
| Temporal v10 outer waveform generation (included above) | 12.90 s |
| Temporal v10 complete outer replay/raw descriptor audit | 92.50 s |
| Spectral-resolution v11 outer preparation/generation/scoring | 264.38 s |
| Spectral-resolution v11 outer waveform generation (included above) | 23.93 s |
| Spectral-resolution v11 complete outer replay/raw descriptor audit | 245.26 s |
| Group prior v12 upstream source verification | 73.96 s |
| Group prior v12 source generation/scoring, including 241-test gate | 283.48 s |
| Group prior v12 complete source replay | 337.70 s |
| Group prior v12 outer preparation/generation/scoring | 212.79 s |
| Group prior v12 outer waveform generation (included above) | 19.44 s |
| Group prior v12 complete outer replay/raw descriptor audit | 208.85 s |
| Balanced v13 upstream source verification | 83.26 s |
| Balanced v13 source generation/scoring, including 259-test gate | 185.02 s |
| Balanced v13 complete source replay | 442.26 s |
| Scoped v14 upstream source verification | 147.91 s |
| Scoped v14 source generation/scoring, including 278-test gate | 436.65 s |
| Scoped v14 complete source replay | 427.61 s |
| Scoped v14 outer generation and scoring | 234.40 s |
| Scoped v14 generated waveform subset of that runtime | 17.55 s |
| Scoped v14 complete outer replay | 203.20 s |
| Width v15 upstream source verification | 87.46 s |
| Width v15 source generation/scoring, including 306-test gate | 407.36 s |
| Width v15 complete source replay | 675.56 s |
| Width v15 outer generation and scoring | 413.78 s |
| Width v15 generated waveform subset of that runtime | 31.79 s |
| Width v15 complete outer replay | 355.23 s |

The recorded platform is macOS arm64 with Python 3.11.13, Torch 2.13.0,
NumPy 2.4.6, SciPy 1.17.1, SoundFile 0.14.0 and Matplotlib 3.11.1. Fitting uses
four CPU workers with one Torch thread each. Some analysis timings include
contention with the expanded fit. The full stage's highest recorded worker
peak RSS was 713,605,120 bytes (680.55 MiB); this is a single worker's lifetime
high-water mark, not measured aggregate machine memory. Combined pilot and
expansion fitting took 3057.92 s for 760 s of input audio, about 4.02 wall seconds
per fitted audio second under this workload. Do not extrapolate these values
to deployment or other hardware. The v10 outer run peaked at 699,383,808 bytes
RSS; its replay at 998,014,976 bytes. Source figure generation overlapped part
of that replay, so its wall time includes the observed concurrent workload.
The v11 outer run overlapped its source figure/audio generation and peaked at
731,332,608 bytes RSS; its complete replay peaked at 934,625,280 bytes.
V12 outer generation/scoring peaked at 742,080,512 bytes RSS and its full
replay at 1,033,207,808 bytes; source presentation work overlapped outer
preparation. V14 outer generation/scoring peaked at 752,861,184 bytes RSS,
and its complete replay at 1,289,682,944 bytes. Its source figure/audio work
overlapped outer preparation. These are per-process high-water marks, not
aggregate machine memory. V15 generation/scoring peaked at 644,448,256 bytes
RSS, and replay at 985,399,296 bytes. Source figure/audio generation overlapped
outer preparation, and a presentation-only figure refresh overlapped part of
the outer replay. These timings include that observed contention.

From the ABVID root, with the original local inputs, frozen parent artifacts
and pinned environment available:

```sh
bash CAST/improvement/reproduce.sh smooth8_reproduce 4
```

The fresh-run command retains results, checks synthetic acceptance before raw
fitting, audits fits and source selection, and creates plots/audio review. It
refuses outer access when source selection fails and returns nonzero for a
verified scientific failure. The wrapper's syntax and invalid-ID handling are
checked; the current experiment executes its stages individually. Detailed
commands are in [README.md](../CAST/improvement/README.md).

The capacity pilot workflow is also reproducible with
`bash CAST/improvement/capacity_v3/reproduce.sh NEW_RUN_ID 4`; see its
[instructions](../CAST/improvement/capacity_v3/README.md). The current run used
the same stages individually and the analysis continuation completed all seven
commands, returning the scientific-failure code 3.

The completed temporal source workflow is documented in
[temporal-v10 README](../CAST/improvement/temporal_v10/README.md):

```sh
bash CAST/improvement/temporal_v10/reproduce.sh NEW_SOURCE_ID
```

It runs tests, source evaluation, replay, plots and systematic audio review.
It returns code 3 for a retained scientific failure and does not open outer data.
To additionally execute the conditional outer development comparison and audit:

```sh
bash CAST/improvement/temporal_v10/reproduce_outer.sh NEW_SOURCE_ID NEW_OUTER_ID
```

This requires the unchanged audited v9 bank and all frozen parent artifacts.
The second command requires source success before outer access, creates all
1,500 generated waveforms before reading the frozen held descriptors, replays
every generated sample and reprocesses all 570 held recordings. It returns code
3 for a verified outer scientific failure. Shell syntax and invalid-ID rejection
were checked; the current experiment executes these same stages individually.
All unsuccessful candidates are preserved. No classifier, cross-dataset
transfer or independent physical-vehicle generalization is claimed.

The v11 full workflow, including calibration, all audits and the conditional
outer development comparison, is documented in [resolution-v11 README](../CAST/improvement/resolution_v11/README.md):

```sh
bash CAST/improvement/resolution_v11/reproduce_outer.sh NEW_RUN_ID NEW_OUTER_ID
```

It requires the unchanged audited v10 source bank and its frozen ancestors.
A failed source candidate or a retained temporal41 reference stops outer access.
A verified outer coverage failure returns code 3 with complete artifacts. The
current result used these exact stages individually; shell syntax and invalid-ID
rejection passed. Original CAST pilot/generalization commands remain unchanged.

The latest completed group-prior workflow, with unchanged v11 fitted ancestors,
is documented in [group-v12 README](../CAST/improvement/group_v12/README.md):

```sh
bash CAST/improvement/group_v12/reproduce_outer.sh NEW_SOURCE_ID NEW_OUTER_ID
```

It runs all tests, freezes group statistics, audits 22,500 source samples and
conditionally runs and audits the selected new outer candidate. It also creates
source/outer plots and fixed audio reviews. Code 3 preserves a scientific
failure or unchanged-reference selection. Shell syntax and invalid-ID rejection
were checked; all actual stages were executed individually. That result left
the goal unmet; its small numerical shortfall was not treated as a pass.

The subsequent balanced-sampling comparison is reproduced with:

```sh
bash CAST/improvement/balanced_v13/reproduce_outer.sh NEW_SOURCE_ID NEW_OUTER_ID
```

Its [instructions](../CAST/improvement/balanced_v13/README.md) describe the
unchanged population and frozen allocation plans. On the observed result,
it returns code 3 after source replay and presentation because v12 is retained;
no new outer directory or held access is created. The actual source and audit
stages were executed individually, with shell syntax and invalid-ID handling
checked. The subsequent [scoped group-effects workflow](../CAST/improvement/context_v14/README.md)
is reproduced with:

```sh
bash CAST/improvement/context_v14/reproduce_outer.sh NEW_SOURCE_ID NEW_OUTER_ID
```

It requires the unchanged v13 source audit and v11 fitted bank. The completed
source stages pass their full replay and select a new candidate. Outer
generation, verification and presentation are conditional on that decision.
The current run executes these same stages individually; shell syntax and
invalid-ID rejection are checked.

The current width-calibration workflow has its own [instructions](../CAST/improvement/width_v15/README.md):

```sh
bash CAST/improvement/width_v15/reproduce_outer.sh NEW_SOURCE_ID NEW_OUTER_ID
```

It requires completed v14 delivery and frozen ancestors. All 30,000 source
samples and 600 scores must replay exactly, including the 150 T=1 reference
scores. A source failure, failed minimum-fold gate or unchanged reference
stops before outer access. All actual outer thresholds remain unchanged.
The current run executes the same stages individually; shell syntax, Python
parsing and invalid-ID rejection passed.

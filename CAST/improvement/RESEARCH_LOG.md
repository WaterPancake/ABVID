# CAST improvement research log

## 2026-10-04 — Goal and access

The user set a new goal: at least 80% held-group acoustic coverage and a 2.5%
relative distance improvement. Interpret the margin against both existing
controls, separately for car and truck. Preserve the original failed v0 run
and its earlier 5% criterion. The new protocol retains metrics, percentile
intervals, outer scales, class/seed budgets and spread limits.

No military, reserved, MELAUDIS or mixed-cache access. The largest newly
authorized training selection in this protocol is H1 fold 0 / seed 42's exact
380 IDs. Group 0 is excluded from all fitting and source selection. Its future
score is explicitly exposed development, following the previous CAST result.

## Eight-control spectral smoothing

Implemented continuous log-power interpolation between the original noise-band
centers, with component RMS normalization and separately seeded noise. Retain
the original harmonic/envelope controls, optimizer budget and objective.

The first frozen synthetic run exposed inconsistent input normalization between
the fitting and artifact paths in near-exact tonal cases. Its incomplete run
and failure record are preserved. A probe using the same normalized target for
both paths reproduced checking loss exactly. The repaired version also uses
two-point interpolation (avoiding batch-dependent matrix reductions) and checks
each vector with the same batch shape as replay. Acceptance tolerances were
unchanged. No real fitting had started before this repair.

`runs/cast_smooth8_v1_20261004_r1` passed all seven original synthetic fixtures
and the gain-ambiguity check. The complete current test suite has 81 passes.
All 50 real pilot fits completed with zero numerical/input failures, in
407.22 s on four single-thread CPU workers. Every clip's checking loss improved
over its v0 fitted reconstruction; median paired gains were 1.486% car and
1.452% truck. This is reconstruction evidence only.

## Same-parent source-group comparison

`evaluations/smooth8_vs_v0_pilot_source` generated 15,000 clips across the two
renderers, five leave-one-training-group-out folds, three arms and five seeds.
It never opened outer held descriptors. Source-side prior scales and parent
sets were recalculated separately inside each fold.

| Renderer | Car mean W1 | Truck mean W1 | Car coverage | Truck coverage |
|---|---:|---:|---:|---:|
| v0 | 0.753893 | 0.711373 | 78.760% | 79.859% |
| smooth8 | 0.705477 | 0.678869 | 80.754% | 81.691% |

The smooth renderer improves absolute joint distance in both classes, but
independent marginals still have lower distance: the joint relative margins
are -5.88% car / -2.89% truck. Neither candidate passes the full source goal.
The generic maximum-shortfall rank slightly favors v0 because its worst
relative-control shortfall is smaller; this is recorded unchanged. This
ablation is not an outer-candidate selection. Continue investigating smoothing
because it improved every paired reconstruction, both absolute group distances
and both coverage scores, as contemplated in protocol step 4.

Each inner held class contains only five real clips in this 50-parent ablation.
The expanded bank is needed to assess whether those sparse empirical validation
distributions explain the relative-control result. No new outer comparison has
been performed or used to choose the expansion.

## Full training bank — in progress

Fit the remaining 330 exact H1 training parents with the frozen smooth8 fitter,
reusing the 50 completed artifacts. Four CPU workers; expected wall time about
40–50 minutes from the observed per-fit throughput. Each completion is saved
individually with its parameters, all starts, checking loss, audio, plot, flags,
runtime, source ancestry and hashes. Next: train-group temperature comparisons
under the previously declared grid, then freeze an outer development candidate.

## Replay audit and the next decision

The same-parent source ablation passed replay of all 15,000 generated waveforms,
descriptors, parameter choices and donors, with all 300 scores and training-only
fold scales recomputed. Every one of the 50 pilot fits also passed raw-source
preprocessing and saved-waveform/component replay. All four fitted starts and
all four unoptimized starts had their checking losses recomputed with the
unchanged 1e-5 tolerance. This additional audit took 7.88 s.

The expanded test suite has 87 passes, including class-specific acceptance
boundaries, duplicate/missing generated schedules and rejection of a failed
source candidate before any held access. The outer runner now enforces the
already declared source-pass and replay-audit prerequisites. If the full source
grid fails, inspect training residuals and preserve that scientific failure;
do not use the outer group to choose a temperature.

The pilot source family comparison suggests a coverage/distance tradeoff:
complete-vector sampling covers more envelope and band variation, while the
marginal control can give lower W1. More spread alone may worsen the latter.
This is evidence for evaluating prior shape jointly with coverage, not evidence
that the outer target has been reached. Diagnostic plots are kept under
`diagnostics/`, with paired fit residuals separated from generation scores.

## Pilot spread sensitivity while the full fit runs

The separately recorded `evaluations/smooth8_pilot_temperature_grid` tests the
already declared temperatures on the completed 50-parent bank. This is a
source-only sensitivity analysis; the outer candidate still requires the full
380-parent source comparison. All four temperatures apply to every matched arm.

| Temperature | Car coverage | Truck coverage | Car gain vs marginals | Truck gain vs marginals |
|---|---:|---:|---:|---:|
| 1.0 | 80.754% | 81.691% | -5.880% | -2.886% |
| 1.1 | 83.571% | 84.176% | -7.336% | -4.739% |
| 1.25 | 86.509% | 87.405% | -9.556% | -7.373% |
| 1.5 | 90.467% | 90.023% | -13.823% | -11.297% |

No candidate passes. At 1.25 the car band spread exceeds 2; at 1.5 both classes
have excessive spectral/band spread and lose to both controls. Temperature 1
is selected by the preregistered rank. Its 150 per-fold/class/arm/seed score
records match the earlier same-parent ablation exactly. Increasing all control
variation is therefore not sufficient on this pilot.

Generation/scoring took 193.54 s. Independent replay passed all 30,000 waveform,
descriptor and donor checks, all 600 scores, and all training-only fold scales
in 190.22 s. These timings include contention with the live four-worker fit.
No outer held descriptors were opened. The plotted result is under
`diagnostics/pilot_temperature_comparison/`.

The new `diagnostics/pilot_audio_review/index.html` links all 50 audited fits and
shows 15 systematic examples with 30 audio players. All 291 local links were
checked; no external assets are used. The fresh-run `reproduce.sh` command
sequences fitting, audits, source selection, figures and conditional outer
verification. Its shell syntax and invalid-ID rejection have been checked;
the current run is executing the same stages individually, with full-bank and
outer stages still pending at this log entry.

## Queued full-bank analysis

`continue_full.py` is waiting on the existing full-fit summary and does not
launch any fitter. On verified completion it will run the full fit replay,
the predeclared full-bank temperature grid, exact source replay, diagnostic
figures and the systematic audio review. Only a source candidate passing all
criteria can trigger the outer comparison and its verification. Each command,
start/end time and return code is preserved in an atomic continuation receipt.
Scientific failures remain failures and stop the pipeline with exit code 3.

The interim [report](../../reports/CAST_improvement.md) explicitly separates
completed reconstruction/source evidence from the pending outer objective.
All eleven local report links were checked when it was created.

## Full fitting completed and audited

All 380 fits completed (50 reused, 330 new) without numerical/input failures.
The expansion took 2650.70 s on four workers. Maximum recorded worker peak RSS
was 713,605,120 bytes; this is not aggregate machine memory. The independent
replay audit passed all 380 raw/preprocessing/waveform/component checks and all
fitted/unoptimized checking starts in 27.06 s, preserving the 1e-5 tolerance.

Full-bank median checking losses are 1.375286 car and 1.371373 truck. Median
gains versus the unoptimized checking baseline are 20.05% and 21.60%.
Equally good starts disagree in 174/190 car and 171/190 truck fits; this
continued ambiguity must accompany any parameter interpretation.

The continuation has started `smooth8_full_source_grid` using exactly the
predeclared four temperatures. Fitting has terminated successfully; do not
restart it or launch a duplicate source evaluation.

## Full source failure and separate-spread prior v2

The full-bank v1 grid completed in 119.66 s. Its exact audit passed all 30,000
waveform/descriptor/donor replays and 600 recomputed scores in 114.90 s.
Temperature 1 ranks best: car/truck coverage 81.676%/80.777%, with W1
0.619467/0.611255. Gains against the prototype are 16.07%/15.39%, but gains
against independent marginals are -5.952%/-3.853%. All four temperatures fail.
The continuation exited with the declared scientific-failure code 3 after
writing full diagnostics and the 380-fit audio review. No outer comparison ran.

Full-bank spectral spread is broader than real validation spread (spectral/band
ratios 1.39/1.61 car and 1.28/1.48 truck), whereas envelope/modulation spread is
narrower (0.96/0.81 and 0.97/0.83). Freeze a new prior version with separate
spectral [0.5, 0.65, 0.8, 1.0] and envelope [1.0, 1.25, 1.5] temperatures before
changing renderer capacity. Apply each transformation identically to all matched
arms, recompute centers inside each training fold, and retain every calibration
ancestor. All original criteria, scales, budgets, seeds and fits remain intact.

The new six focused prior checks passed; the complete source test gate has
93 passes. `smooth8_block_v2_full_source` is generating all twelve candidates
under the new immutable source/protocol snapshot. Its outcome and exact replay
audit remain pending. Do not edit `block_prior.py`, `block_evaluation.py`,
`tests/test_block_prior.py` or `prior_v2/PROTOCOL.md` after this freeze.

## Prior-v2 failure audited; fixed capacity revision

All twelve source-only spread candidates failed. The selected 0.8 spectral /
1.0 envelope candidate has car/truck coverage 76.81%/75.99% and marginal gains
-2.08%/-0.82%. Generation took 420.29 s; the 384.83 s exact audit passed all
90,000 generated records and 1,800 scores. The identity candidate reproduces
150 v1 scores exactly. The audited heatmap has been visually checked. No outer
comparison ran. All failed candidates remain preserved.

Source residuals justify the separately versioned capacity-v3 probe: sixteen
log-spectrum controls and nine envelope knots, with the previous signal family
nested by deterministic interpolation. The original objective, optimization
budget, synthetic tolerances, data IDs and evaluation criteria remain fixed.
Seven initial numerical checks pass; original synthetic recovery is next,
before any new real fitting. This combines two resolution changes and does
not identify their separate contributions.

## Capacity-v3 synthetic acceptance

Frozen run `capacity_v3/runs/cast_capacity16x9_v3_20261004` passed all seven
original synthetic fixture types and the unchanged gain-ambiguity check. The
107-test gate passed before fitting. Synthetic acceptance took 67.46 s on four
workers. Single-tone frequency error was 0.1972 Hz, changing-spacing error
0.9643 Hz, and noise-control L1 error 0.1162 (bound 0.35). Ambiguous harmonic
explanations remain flagged. The same 50-parent real pilot is now running.

During separate sampling-unit implementation, a mechanical dimension edit also
changed `sha256` to nonexistent `sha376`; the tests caught this before source
evaluation or code freeze. Restoring SHA-256 repairs the implementation without
changing any threshold or random-stream specification. Fitting is unaffected.

## Capacity-v3 pilot complete: coverage improves, margin still fails

All fifty fits completed in 459.37 s, with no numerical failures. Exact raw,
component, waveform and all-start checking replay passed in 3.47 s. Checking
loss improved in 49/50 clips, with median paired reductions of 0.759% car and
0.857% truck. One car worsens by 0.182%. Truck modulation median error also
worsens despite improved spectral/band/envelope residuals. All cases remain.

Matched source comparison took 67.97 s; all 15,000 records and 300 scores replay
exactly in 63.56 s. The 150 smooth8 score records exactly match the earlier
pilot. Capacity car/truck W1 is 0.694845/0.672773, coverage 83.776%/83.244%, and
marginal gains -6.916%/-2.088%. Both variants fail; the unchanged ranking selects
smooth8. The seven-command continuation completed successfully and exited with
scientific-failure code 3. All 291 audio-gallery links are valid (30 players),
and paired/residual/source figures were visually inspected. No outer run occurred.

Decision: preserve the fixed capacity revision and defer its 330-clip expansion
while testing the clearer remaining margin issue: fitted-vector ambiguity. The
existing full smooth8 bank contains 732 car and 703 truck starts within the
already frozen 5% fit-loss ambiguity bound, across 379 parents with alternatives.
A source-only empirical alternative mixture can test whether choosing just one
parameter explanation loses useful uncertainty, without another renderer change
or raw-data access. Every arm must use the same alternatives with equal parent
and group weights. These are not independent samples or a calibrated posterior.

## Equivalent-fit mixture v4 preregistered

Nine focused tests pass for the exact winner identity, deterministic alternative
choices, group/parent weighting despite unequal alternative counts, fit-only
eligibility and protected-group rejection. The two fixed priors are winner-only
and the uniform mixture of starts within the existing 5% fitting-loss ambiguity
bound. All arms use the same alternatives. Parent-selection and waveform streams
remain unchanged, and winner scores must reproduce the full-bank T1 reference.
The source evaluation freezes its implementation before generation, includes all
380 parents, and cannot open outer descriptors. No new real fitting is launched.

## Equivalent-fit mixture v4 failed; exact audit passed

The fixed mixture is selected by the unchanged source rule but fails the goal:
car/truck coverage is 82.832%/81.792%, W1 is 0.614612/0.607530, and gains against
marginals are -4.978%/-5.279%. Generation/scoring took 192.63 s. All 15,000
samples and 300 scores replay exactly in 66.04 s, with all 150 winner scores
identical to the v1 full-bank reference. The source test gate passed 122 tests.
Two additional auditor checks prove that caching seed-independent prototype
means preserves complete sampling records; every waveform is still rendered.
No outer comparison ran.

## Source-only spectral-bias calibration v5 preregistered

Across all five training groups, car low-band observed/reconstructed power
ratios are below one while car/truck 500–1500 Hz ratios exceed one. This
consistent paired bias remains despite added capacity and motivates a bounded
noise-weight correction estimated inside each training fold. The correction is
first-order and may not exactly correct the full harmonic/noise mixture.

Freeze eight cases: spectral temperatures [0.5,0.65,0.8,1] with correction
strength [0,1], retaining envelope temperature 1. Estimate class log band-power
ratios with equal group weighting, clamp to +/-ln2, then apply them to every
matched arm's common donor bank. All calibration parents equal the fold's
training parents exactly. Ten focused checks pass for unchanged zero-correction
vectors, correction direction/bounds, unchanged other controls and fail-closed
calibration ancestry. All acceptance rules, metrics and outer exclusions remain.

## Band-bias v5 complete; allocation hypothesis v6

The eight-case calibration source experiment took 240.76 s and all cases failed.
The original uncorrected spectral-temperature 0.8 case ranks first. At unit
temperature, correction reduces joint W1 from 0.619467/0.611255 to
0.604041/0.597586 (car/truck) but marginal gains worsen to -6.378%/-4.486%;
coverage is 82.024%/81.261%. Full exact replay passed 60,000 samples and
1,200 scores in 239.75 s; all 600 zero-correction references match prior v2.
No outer comparison ran.

Source component scaling found a more specific allocation issue: harmonic logit
offset -1 improves same-parent band error for 161/190 cars and 165/190 trucks,
with a positive median in every class/group. Median absolute band-L1 reductions
are 0.09216/0.09070. Offset -2 overcorrects several groups. This is an approximate
saved-component diagnostic, not a new fit or held score. Its first metadata
label said real_audio_opened=false ambiguously; that artifact and a metadata
erratum are retained. The r1 diagnostic explicitly records saved training audio
reads and no raw dataset reads, with all per-parent numbers bit-exact unchanged.

V6 freezes spectral temperatures [0.65,0.8,1] × harmonic-logit offsets [0,-1],
with the same donor transformation for every arm. Nine focused checks pass,
including zero-offset identity, endpoint preservation and fixed-grid rejection.
The five source groups have informed this hypothesis; source scores remain
exploratory development. V6 generation starts only after the completed v5 audit.


## Effective-mixture v6 complete; per-parent v7 preregistered

All six source cases fail. Unit spectral temperature and offset -1 ranks first:
car/truck W1 0.571374/0.553647, coverage 79.896%/78.888%, marginal gains
-1.962%/-0.459%. Absolute joint W1 improves 7.8%/9.4% over original full-bank
sampling, but this does not establish the requested matched-control advantage.
Generation/scoring took 186.36 s. Exact audit passed all 45,000 samples and 900
scores in 167.70 s, with 450 zero-offset reference scores unchanged. The
145-test source gate passed; two additional outer-access guards passed. The
source-grid figure was visually checked. No outer action occurred.

V7 specifies one bounded least-squares harmonic-fraction calibration for each
training parent using its two saved checking components and observed bands.
All other controls and original fits remain unchanged. Undefined and boundary
cases are flagged and retained. Checking components are explicitly calibration
inputs for these derived vectors, not independent validation. One fixed bank
is compared with the original bank under every matched arm; no strength/grid
search is added. The old full-bank T1 scores must reproduce exactly.

V7's nineteen focused numerical/sampling/ancestry checks passed. They cover
unequal component amplitudes, both checking realizations, mixture boundaries,
undefined fits, unchanged other controls, original schedule identity in all
three arms, altered calibration rejection, and held-group rejection before
sampling or file access. The full source comparison is running.

## Per-parent mixture v7 failed with exact calibration/sample replay

The full source suite passed 166 tests. Both banks failed. The derived bank
ranks first: car/truck W1 0.573288/0.560044, coverage 80.426%/79.698%,
marginal gains -1.650%/-0.438%, prototype gains 19.595%/20.448%. Source
generation/scoring took 69.29 s. Replay rederived all 380 parent calibrations
and all 15,000 samples/300 scores in 71.37 s. All 150 original-bank scores
match v1 exactly. Calibration hits the mixture boundary for 12 cars / 5 trucks;
33 cars / 15 trucks become noise-dominant and receive spacing ambiguity flags.
All remain in the experiment. Calibration and source figures were inspected;
the audio gallery contains ten fixed class/group examples with three players
each, explicitly labelled calibration reconstruction, not independent validation.

## Direct envelope v8 preregistered

V7 temporal coverage is 79.366%/78.015% for envelope and 79.963%/77.889%
for modulation (car/truck). Joint modulation W1 is worse than marginals in both
classes. V8 keeps the existing five knots but fits their frame-mean amplitudes
directly to observed RMS trajectories, with the original bounded control range
and each parent's original mean envelope amplitude preserved as a gauge. Ten
focused tests pass, covering representable/nonrepresentable trajectories,
solver failure retention, ancestry, unchanged other controls and leakage.
Both calibrated and matched reference banks are frozen before generation.
No outer audio or descriptors have been read.

## Direct envelope v8 complete: source coverage passes, margin still fails

All 176 tests passed. Bank verification/calibration took 13.19 s; generation
and scoring took 93.89 s. All 380 local calibrations and 15,000 samples/300
scores replay exactly in 93.47 s, with all 150 v7 reference scores unchanged.
No solver failed. One truck reaches the envelope bound; no parent was removed.

The selected bank has car/truck W1 0.568256/0.564910, coverage 80.626%/80.448%,
prototype gains 19.769%/20.046%, and marginal gains -1.254%/-1.267%. Family
spreads pass. Both source candidates fail the combined criteria. All 380
same-parent envelope RMSE values improve; modulation L1 improves for 162/190
cars and 168/190 trucks. These training reconstruction gains do not translate
to a passing distribution comparison; truck W1 worsens. The two omitted groups
72766641e1daf436 and c6148cf0cafb5753 produce negative matched margins in both
classes, despite positive results for some other groups. No new outer read ran.

Source figures and temporal residuals were inspected. An additive r1 audio/plot
folder only reduces crowded axis ticks, preserving the first rendering and all
scientific numbers. The one-command reproducibility wrapper runs all stages
and reports scientific failure with exit 3. Its syntax and invalid-ID guard
were checked; the current run uses the identical individual stage commands.
The next work should diagnose spectral/modulation mismatch across all source
groups before freezing another revision, keeping difficult groups and the same
criteria. Goal status remains active and unmet.

## Joint expected-spectrum v9 preregistered

V8 passes source coverage but not the matched marginal-control margin. The
spectral families still dominate distance, and the earlier class-wide bias
correction/single-mixture calibration failed. V9 jointly calibrates the eight
noise weights and harmonic fraction against each parent's observed spectrum
and band powers. It preserves v8's envelope, spacing, harmonic weights and
waveform renderer. Expected component power ignores the random cross term;
two original checking streams are explicitly calibration inputs. One fixed
150-step Adam budget fits the equally weighted mean squared log probabilities
of the original 64 spectrum bins and 8 bands. Evaluation criteria do not change.

Fourteen focused checks pass: independent SciPy Welch agreement, component
endpoints, known-mixture recovery with the original noise L1 bound, finite
boundary gradients, deterministic complete optimizer replay, matched sampling
schedule and complete ancestry, and rejection of excluded/tampered parents.
The complete suite runs again before freezing all 380 calibration inputs and
code. All per-parent histories and two reconstruction waveforms will be kept.
A full calibration replay audit precedes source generation; no outer access
is authorized by a failed source candidate.

V9 bank calibration completed for all 380 parents, with zero numerical errors,
in 98.92 s (four CPU workers; maximum recorded worker RSS 376,782,848 bytes).
Median expected-objective reductions are 50.20% car / 51.68% truck. Full replay
repeated every optimization/history/parameter and both waveforms exactly in
98.06 s: 380 calibrations and 760 WAV sample arrays. Original tracked ABVID
changes remain unchanged. The bank freeze passed 190 tests; two additional
outer-access guards passed before the source comparison. The conditional outer
runner has not been invoked. Source evaluation follows the unchanged schedule.

## Spectrum v9 scientific failure; exact source audit passed

The spectral calibration bank improves both absolute W1 scores to
0.562936 car / 0.560583 truck and gives coverage 80.494% / 81.438%, but
marginal-control gains are -1.408% / -2.387%. The unchanged ranking selects
v8; both candidates fail. Generation/scoring took 120.90 s. All 15,000
waveforms/descriptors/ancestry and 300 scores replay exactly in 127.32 s, with
all 150 v8 scores bit-exact unchanged. No outer comparison ran. Actual
same-parent log-spectrum error improves for all 380 parents; band L1 improves
for 190/190 cars and 189/190 trucks. Calibration reconstruction is not held
validation and did not produce the required distribution-level margin.

## Temporal-resolution v10 preregistered

Test 41 envelope controls at the existing 50 ms descriptor frame boundaries,
retaining the original five-knot curve as a nested initializer. Preserve the
old bounds and amplitude gauge; fit observed source RMS with a fixed 0.01
curvature penalty. Keep v9's spectral controls. The 61-coordinate independent
control samples every scalar with unchanged simplex normalization. Compare
with exact v8 and v9 references; require all 300 reference scores to replay.
Numerical tests caught affine error 5.19878209e-6 at solver ftol 1e-12, just
outside the unchanged 5e-6 tolerance. Tightening solver ftol to 1e-14 repairs
the numerical issue before real temporal calibration or freeze. All eleven
focused checks now pass, including nested-waveform agreement, finite gradients,
affine recovery, failure retention, gauge/bounds and complete donor ancestry.

## Temporal v10 source pass; conditional outer comparison frozen next

The source gate passed 203 tests; the additional outer guards bring the
complete suite to 205 passing tests (15.82 s). Temporal41 meets all unchanged
source criteria: car/truck coverage 83.330%/83.981%, marginal gains
9.266%/8.491%, prototype gains 21.337%/19.944%, and every family spread
within [0.5,2]. Generation/scoring took 178.48 s. The complete 219.98 s
audit replayed 22,500 samples, 450 scores and 380 calibrations; all 300
v8/v9 reference scores remain exact. No outer descriptor or audio has been
read in this improvement work at this point.

The gain is representation dependent: joint W1 improves only about 1.49% car
and 1.33% truck relative to v9, while the 61-coordinate independent control
loses temporal coherence. Report both changes; do not describe this as a
universal improvement over the strongest earlier baseline. Freeze this
source-selected candidate and the complete 1,500-sample outer schedule
before reading the exposed IDMT outer development descriptors. Retain the
original scales, seed schedule, quantiles, sample counts, class weights and
all acceptance criteria. A successful source result does not complete the
held-group goal.

## Temporal v10 verified outer scientific failure

Frozen before the held descriptor read, temporal41 improves outer W1 to
0.697096/0.640628 and coverage to 77.212%/75.571%. Matched marginal gains
11.885%/8.731%, prototype gains 23.104%/26.155%, and every family spread
pass. Both coverage criteria fail; the goal is not complete. Outer preparation,
generation and scoring took 77.92 s, including 12.90 s for waveform generation.
Full replay/raw-descriptor audit took 92.50 s and passed all 1,500 generated
samples, 570 original recordings, scores, ancestry and preservation checks.
Source diagnostics confirm all 380 envelope errors improved; 184/190 per class
improve modulation. Plots/audio retain fixed systematic examples.

## Nested spectral-resolution v11 preregistered

V10 source residuals, not held residuals, justify testing the already specified
v3 sixteen-point spectral grid with temporal41. Source spectral/band W1 is
0.72–0.77 versus 0.21–0.22 modulation; car source spectrum coverage is 79.94%.
Use identical 150-step expected-spectrum calibration of noise weights and
mixture, preserving other controls. V10 is an exact reference. The new schema
has 69 coordinates; matched marginal sampling remains representation dependent.
Eight focused spectral numerical tests pass, including existing recovery
thresholds, nested waveforms, finite gradients and exact replay. No new raw
source or outer data has been read for v11. Outer aggregate exposure remains
explicit; no held fitting, EQ, scales or selection rules are permitted.

V11 passed all 224 tests before freeze. All 380 calibrations completed with
zero numerical failures in 147.73 s using four workers (maximum worker RSS
381,353,984 bytes; not aggregate memory). Median paired expected-objective
reductions are 66.64% car / 59.66% truck. These are in-sample calibration
objectives, not distribution or held-out results. All histories, parameters,
ancestry, flags and 760 reconstruction WAVs are retained; complete replay is
running before source evaluation. No new outer access has occurred for v11.

V11's full calibration audit passed: all 380 optimizations, histories and
parameter/ancestry records, plus all 760 WAV sample arrays, replay exactly in
132.92 s. Historical tracked ABVID work remains unchanged. Source evaluation
now compares the frozen 69-coordinate model against the exact temporal41
reference, using the same folds, arms, seeds, scales and selection rule.

## Spectral-resolution v11 source pass; replay running

The unchanged source rule selects spectrum16. Coverage is 84.119% car /
84.571% truck, marginal gains 10.028%/8.340%, prototype gains
20.490%/19.437%, and every family spread passes. Joint W1 is
0.548904/0.549782, a modest improvement over temporal41's
0.554572/0.553121. This version also improves both marginal-control W1
scores; the matched margin is still representation dependent. All 224 tests
passed before source generation. Complete 15,000-sample/300-score replay
and 150-record temporal41 identity verification are running. No new outer
comparison is permitted before that audit passes.

V11's complete source audit passed in 255.74 s: 15,000 exact generated
records, 300 independently recomputed scores, verified calibration inputs,
unchanged tracked ABVID state, and all 150 temporal41 reference scores exact.
The selected spectrum16 bank now proceeds to one separately frozen outer
development comparison, with all 1,500 generated waveforms complete before
reading held descriptors. This is a new candidate, not a rerun of temporal41.

## Spectral-resolution v11 verified outer scientific failure

The new source-selected frozen candidate improves outer coverage modestly to
77.984% car / 76.054% truck, still below 80%. Joint W1 is
0.694522/0.643825; car improves slightly versus v10 but truck worsens.
Marginal gains 9.271%/9.604%, prototype gains 22.272%/25.177%, and every
family spread pass. No parent or held example was dropped. Preparation,
generation and scoring took 264.38 s (23.93 s waveform generation), overlapping
source plot/review generation. Peak process RSS was 731,332,608 bytes.

The complete independent replay passed in 245.26 s: all 1,500 generated
samples, 570 raw held descriptors, scores, sampling/calibration ancestry and
historical/ABVID preservation checks. Replay peak RSS was 934,625,280 bytes.
The numeric goal remains unmet, short by 2.016 and 3.946 percentage points of
coverage. This is exposed one-group development evidence, not confirmation.

Source reconstruction log-spectrum RMSE improves for all 380 parents; band L1
improves for 154/190 per class. Most temporal errors slightly worsen after
spectral recalibration. Source worst-group coverage is 79.348% car and
79.254% truck, despite mean source coverage above 84%. These source-only
results motivate inspecting group/distribution and sampler variability before
another fixed-capacity increase. No new prior, selection rule or wider interval
has been introduced, and no outer residuals are permitted to tune a revision.

Final v11 presentation QA checked all source and outer audio links/hashes and
visually inspected the plots. An additive outer-review r1 moves labels inside
bars near the 80% line to avoid overlap; it changes no values, examples or
scientific artifacts. The first rendering and delivery receipt are retained.
The final report links r1 and the conditional reproduction command.

## Source-group prior v12: frozen hypothesis, source run started

A source-only diagnostic of the 380 audited v11 parents assigns 27–39% of
spectral/band descriptor variance and 8–15% of temporal variance to group
mean differences, under equal group/parent weights. These are descriptive
components with no physical or causal interpretation. No outer observations
were read for this diagnostic or prior design.

The fixed protocol compares exact spectrum16 with group recombination and
symmetric group-effect prediction, retaining the same within-group residuals,
all 69 coordinates, matched virtual populations, seeds, budgets, metric
weights, quantiles and selection rule. Group-count extrapolation is fixed,
not selected from a scale grid. All center ancestors and projections remain
recorded. Only a new passing source winner can trigger an outer comparison.

The 17 focused numerical, replay, ancestry and outer-access guard tests pass.
A pre-freeze exp(log(bound)) endpoint-rounding defect was repaired without
changing its strict acceptance test; implementation_defects.json preserves it.
The full existing test suite is the source-generation gate. No new real fitting
or raw outer audio access has occurred.

V12 source selection chooses group_predictive under the unchanged rule.
All 241 tests pass. Car/truck coverage is 84.745%/87.134%; gains versus
matched marginals are 9.564%/16.014% and versus prototypes 18.317%/23.162%.
All family spreads pass. Joint W1 is 0.561161/0.524022: truck improves and
car worsens versus the exact v11 source reference. The fixed class-balanced
mean W1 improves from 0.549343 to 0.542591, so the new candidate ranks first.
All three candidates pass the source criteria. The full replay audit is
required before any new outer access; these remain source-development results.

V12 source audit passed in 337.70 s: all 22,500 generated records, 450 scores,
fold-specific centers/prototypes/projections, complete ancestors and training
scales replay exactly. All 150 v11 reference scores are exact. Tracked ABVID
state is unchanged. The selected group_predictive candidate now proceeds to
one frozen outer development comparison; source figures/audio reviews run
alongside its preparation. Every outer output is generated before the first
held descriptor read. Existing outer evaluations remain unchanged.

## Group prior v12 outer scientific failure; audit running

The frozen new outer comparison reaches 79.359% car and 78.997% truck
coverage. W1 is 0.694225/0.639702, modestly better than v11 in both classes.
Marginal gains are 11.641%/17.534%, prototype gains 22.544%/26.619%, and all
family spreads pass. Coverage still fails both classes by 0.641 and 1.003
percentage points. Generation/scoring took 212.79 s (19.44 s generation),
with overlapping source presentation jobs and 742,080,512 bytes peak RSS.
No recordings or generated samples were dropped. Complete replay is pending.

A separate source-only finite-sampling diagnostic uses the audited v12 source
archive, never outer descriptors. Within-source-fold coverage seed standard
deviation averages 3.175 pp car / 2.532 pp truck. Direct group counts range
7–20 car / 7–22 truck instead of the expected 12.5 per group in a 50-sample
inner-fold batch. Distinct direct parents average 42.84/39.32. Variation also
includes waveform streams and cannot be attributed entirely to parent counts.
This motivates a fixed balanced sampling experiment at the unchanged budget,
with the same empirical population and matched controls. It is a hypothesis,
not evidence of a future held improvement.

V12 complete outer verification passed in 208.85 s: all 1,500 generated
waveforms, all 570 raw held descriptors, scores, complete center/donor ancestry,
frozen access sequence and historical/ABVID preservation. Peak audit RSS was
1,033,207,808 bytes. The coverage failure is verified and remains reported;
the numeric goal is not achieved.

## Balanced finite sampling v13 frozen

The source-only seed/donor diagnostic motivates one fixed balanced allocation
strategy against exact v12. Balance the 50 direct group counts, balance parent
use conditionally within each group, and independently balance virtual-child
indices. Apply independent full plans per scalar coordinate to the marginal
arm; preserve the exact prototype and waveform streams. The population,
380 original parents, all parameters/bounds, weights, seeds, sample budgets,
quantiles, metric scales and selection rule remain unchanged. Freeze all
plans before generation. No outer value is a fitted statistic or selector.

All 18 focused balance, determinism, population identity, provenance and
outer-access guard tests pass. The complete test suite will gate generation.
Only a new selected passing candidate with complete replay can enter the
outer comparison. The already evaluated v12 reference will not be repeated.

V13's complete 259-test gate passes. Balanced sampling passes the source
thresholds but is not selected: W1 worsens to 0.567073 car / 0.541720 truck,
versus v12's 0.561161/0.524022. Coverage is 84.619%/87.805%; marginal gains
6.526%/13.458%, prototype gains 17.457%/20.567%, and all spreads pass. The
fixed class-balanced W1 ranks the exact v12 reference first (0.542591 versus
0.554397). No new outer comparison is authorized by this result. Full source
replay is running; all unsuccessful method evidence remains preserved.

V13 complete source replay passed in 442.26 s: all 15,000 records, 300 scores,
allocation plans, group statistics, all ancestors and scales match exactly.
All 150 v12 reference scores are exact. The reference is retained, so the
progression decision forbids a new outer evaluation. V12 remains the best
fully verified outer result. The source generation/scoring stage took 185.02 s
after 83.26 s of upstream checks, with a 259-test gate taking 11.72 s.

A v14 protocol is preregistered only, not implemented or run. Source group
variance is weak in spacing/harmonic weights and stronger in noise controls;
this motivates two fixed scopes that restore each residual parent's original
harmonic controls while applying unchanged v12 group effects to noise/mix,
or noise/mix/envelope. It preserves the original random draws and every
criterion. Full matched source selection/replay remains mandatory before any
new outer comparison. No v14 result is assumed.

Final v13 presentation QA visually inspected both plots and verified all 73
review links, 71 artifact hashes and 60 playback WAV hashes. The ten cards
explicitly distinguish the different residual parents selected by each
sampler and use a common gain within each six-audio card. No new outer run
occurred. The latest numeric result remains v12's audited coverage failure.
All live execution handles are closed. The v14 scoped-effect protocol is the
next concrete work item; no missing external input currently blocks progress.

## Scoped group effects v14 implemented

Current artifacts confirm the preceding goal turn made progress: v12's
complete outer replay passed with coverage 79.359%/78.997%, and v13's fully
replayed balanced-allocation comparison retained the v12 reference. There
is no missing input or external blocker.

The preregistered v14 implementation now compares exact group_predictive,
spectral_context (noise and mixture shifts only), and spectrotemporal_context
(noise, mixture and envelope). Original parent spacing/harmonic weights are
copied exactly in both new scopes; spectral_context also retains the exact
original envelope. All v12 donor/child/phase/noise schedules remain unchanged.
Prototypes are recomputed from each matched finite population. Applied and
discarded projection counts are kept distinct; original ambiguity remains.

All 19 focused restored-block, mean-weighting, exact reference, deterministic
waveform, ancestry and outer-access guard tests pass. The old preregistration
text is preserved; only its opening status was changed to historical tense.
No hypothesis, numerical threshold or access rule changed. The complete test
gate and fixed source experiment are now starting. No v14 outer access has
occurred.

V14's complete 278-test gate passed in 22.07 s. Source selection chooses
spectrotemporal_context: W1 0.560524 car / 0.522692 truck, coverage
84.628%/86.821%, gains vs marginals 9.829%/16.010% and vs prototypes
18.288%/23.365%. Every joint family spread passes. All three candidates pass
source thresholds. The fixed mean-W1 rule favors the new scope by only
0.18% (0.541608 vs 0.542591); coverage is slightly lower than v12 in both
classes. This modest source improvement is not an assumed outer pass.

Upstream source verification took 147.91 s; source generation/scoring took
436.65 s including the test gate. Full replay of all 22,500 samples, 450
scores, scoped prototypes/projection records and 150 reference scores is
running. No v14 outer descriptors or audio have been opened.

V14 full source replay passed: 22,500 samples, 450 scores, 150 exact v12 reference scores, all 380 parents and ancestry/statistics verified in 427.61 s. The source progression receipt permits the selected spectrotemporal_context_T1 candidate. Its separate outer run is starting after this audit, with all 1,500 generated waves frozen before held descriptor access. Source figures and fixed audio presentation are generated concurrently; timings include that workload. A transient bookkeeping write used the immutable-artifact save helper against mutable WORK_STATE.json and correctly refused overwrite; the state update was then made explicitly without changing scientific inputs or artifacts.

## V14 complete: scoped prior failed outer coverage

The complete audit passed for all 1,500 generated waveforms and all 570 raw held descriptors in 203.20 s. Outer coverage is 78.849% car / 78.725% truck, with all matched margins and family spreads passing; v12 remains the best coverage result. Generation/scoring took 234.40 s, including 17.55 s for waveforms. Every sample, failure and original parent remains. Source and outer plots were visually checked; 62 source and 19 outer review links were verified, along with artifact hashes. The v14 delivery receipt and exact report snapshot preserve the completed negative result.

The v15 protocol was preregistered before implementation or generation. It addresses source evidence: the v14 car source-fold minimum is 79.580% despite a passing mean. It adds a minimum-fold coverage gate and tests only [1, 1.1, 1.25, 1.5] dispersion on noise/mix/envelope with unchanged harmonic controls. Actual outer acceptance and every original source criterion remain required. V14 outer results were already known; this is recorded explicitly. No missing input blocks this source-only next step.

## V15 width calibration implemented

After v14 delivery was complete, implemented the preregistered [1,1.1,1.25,1.5] widths around equal-group/parent/child transformed centers, preserving harmonic controls and direct donor/child/phase/noise streams. All 28 new focused tests pass, including analytic transformed variance, matched prototypes, exact T=1 waveforms, bounded projections, complete source-score schedule rejection and both outer-access gates. The stronger internal minimum-fold coverage requirement is separate from the unchanged original criteria. The full source command is starting; no v15 outer input has been opened.

V15 full gate passed all 306 tests before generation. The source comparison is now executing all 30,000 preregistered records; no width is selected from partial results, and no outer input has been accessed. Presentation/reproduction entry points are additive and their syntax/invalid-ID rejection checks pass.

V15 selected T=1.1 by the preregistered stronger source-fold rule. It has mean coverage 87.246% car / 89.318% truck, minimum-fold coverage 83.280% / 88.814%, and all original margins/spreads pass. W1 is 5.047% worse than the narrower T=1 reference; the coverage trade-off is explicit. T=1.5 fails original spread/prototype-margin checks despite high coverage. Full source audit passed 30,000 exact records and 600 scores in 675.56 s, with 150 exact v14 T=1 reference scores and all 380 parent inputs/statistics verified. The frozen T=1.1 outer comparison is running only after this evidence; source figures/audio overlap its preparation. No further candidate or width is being selected from outer results.

## V15 complete: stated numeric development goal achieved

The source-selected T=1.1 outer result has 82.019% car / 81.286% truck mean marginal coverage. W1 gains are 12.295% / 17.207% against matched marginals and 20.946% / 24.048% against prototypes. All unchanged family-spread bounds pass. The complete outer audit passes all 1,500 bit-exact waveforms and 570 raw held descriptor replays, with scores/statistics/ancestry recomputed and historical artifacts preserved. Audit runtime is 355.23 s; generation/scoring is 413.78 s, including 31.79 s waveform generation. Source presentation and one legend-only refresh overlapped part of these timings.

All 306 tests and all source audits pass. Both source and outer plots are visually verified; 72 source and 19 outer audio-review links and their stored artifact hashes are valid. Original v12-v14 report snapshots remain exact. The final report explicitly retains the 4.460% worse W1 versus narrower v14, both below-target car seeds, every rejected candidate and the repeatedly exposed single-group limitation. The criterion is the mean over all five fixed seeds; no seed or parent was dropped. This achieves the requested numerical goal, not classification transfer or independent real-world generalization. No further experiment is required for this goal.

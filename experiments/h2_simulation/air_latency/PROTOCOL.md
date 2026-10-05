> Historical protocol/command reference. Commands retain their original paths
> and run from `ABVID_ARCHIVE_ROOT`, not from the reorganized code checkout.
> See the [experiment registry](../../README.md).

# H2 air-filter attenuation × latency control

Protocol `attenuation_latency_20261004_v1`. This is a further development control
within the active ground/air investigation. The earlier ground/air outputs and
their null gain findings remain immutable. It is motivated by their observed
interaction and the documented 11-tap filter implementation; it is not a blind
preregistration or independent confirmation. Freeze this design before generating
or scoring either new arm. Known endpoint results do not license tuning new arms.

## Question and exact interventions

Does the previously observed joint-path transfer benefit arise from atmospheric
amplitude shaping, from the implementation's finite-FIR latency, or from both?
Keep ground reflection **on in every cell** and retain both source levels.

| Cell | Air amplitude shaping | Air-filter latency | Implementation |
|---|---|---|---|
| A0L0 | Off | Off | Reuse verified ground-only G1A0 and its existing heads. |
| A1L0 | On | Off | Same 11-tap coefficients, centered offline FIR on direct path and both reflected legs. |
| A0L1 | Off | On | Replace each air filter by a unit impulse at tap 5; preserve direct/leg placement. |
| A1L1 | On | On | Reuse verified ground+air G1A1 and its existing heads. |

The existing causal filter evaluates `sum_k h[n,k] x[n-k]`. The amplitude-only
control evaluates `sum_k h[n,k] x[n+5-k]`, with exactly the same geometry-dependent
coefficients at output sample n. The delay-only control evaluates `x[n-5]` at
each corresponding filter position. Thus L toggles finite implementation latency,
including its placement relative to ground filtering and time-varying geometry.
For static geometry, it adds five direct-path samples and ten reflected-path
samples, changing relative delay by five samples (0.625 ms at 8 kHz).

The centered filter is an explicit offline numerical control, not a deployment
implementation or a newly calibrated physical model. Zero-pad outside the full
ten-second grid; retain the original interior crop. Validate its static complex
response, not only magnitude, and check that centered air responses remain
positive on the tested frequency/distance grid before calling them amplitude-only.
Report any dynamic coefficient-time effects as part of this operational contrast.
Do not edit, phase-correct or replace either historical renderer in place.

## Population and fixed controls

Use the exact 240 H2 source templates, 480 dry sources and 1,200 paired geometry
slots already verified in the parent experiment. Reuse its 30 validation-only
pilot geometries. S0 and S1, class-shared source/trajectory pairing, source RMS,
ground parameter, temperature, humidity, pressure and ideal mono microphone all
remain fixed. No added background, sensor effects, augmentation or random draws.

Cells × two source levels × five banks × two representations give 80 heads:
40 new, 40 reused by hash. The joined bank contains 19,200 observations, 9,600
new. Per cell/bank/representation: train on 190 car and 190 truck observations
from 19 source templates/class × ten paths; validate on 50/class from five
disjoint templates/class × ten paths. Preserve every source parent's role.
Car=0, truck=1; motorcycles remain excluded. Templates are not real vehicles.

Keep ten-second 8 kHz float64 rendering, the exact geometry-defined two-second
crop, −26 dBFS crop RMS, then 8→16 kHz resampling to 32,000 float32 samples.
Stop the complete bank on nonfinite, silent, invalid-shape or peak >.98 output;
no clipping, redraws or selective rejection. All members of a geometry must pass
together. Preserve all final observations. To avoid duplicating another 7.4 GB,
retain complete new-render hashes rather than ten-second render arrays; regenerate
every full float64 waveform exactly in a fresh process and compare its hash before
fitting. Reused endpoints must match their existing full-waveform/observation hashes.

Representations are the unchanged locally hashed BEATs768 checkpoint and MFCC26;
no encoder training. Each head uses only its own 380 training rows for StandardScaler
and L2 logistic regression: C=1, intercept, lbfgs, tol=1e−6, max_iter=5000, no class
weights, float64 inputs and its bank seed. Banks: 42, 123, 456, 789, 1024. Truck iff
p>0.5. Lock all 80 heads before target evaluation. Validation is diagnostic, not
model selection. Reproduce every reused prediction vector.

Test exactly the same 8,066 MELAUDIS crops (7,810 cars, 256 trucks; four conservative
groups, original sessions unknown). Primary input is the original native-level
cache. Reuse the parent's fixed −26 dBFS target cache as a secondary diagnostic;
do not extract a new target population or choose normalization from its scores.
No IDMT tuning, reserved audio, new dataset, adaptation or threshold optimization.

## Estimands, falsification and uncertainty

For each source level and bank, Q is complete-target macro-F1. Average bank scores,
then equally average the two fixed source levels; do not pool predictions or resample S.

```text
attenuation = ((Q_A1L0-Q_A0L0) + (Q_A1L1-Q_A0L1))/2
latency = ((Q_A0L1-Q_A0L0) + (Q_A1L1-Q_A1L0))/2
interaction = Q_A1L1-Q_A0L1-Q_A1L0+Q_A0L0
latency_minus_attenuation = Q_A0L1-Q_A1L0
```

Sole primary contrast: native BEATs mean_S latency_minus_attenuation. A CI wholly
above/below zero supports a conditional ordering. Practical dominance additionally
requires a ≥3-point lower-bound advantage and a positive lower-bound marginal
benefit for that factor; reverse signs for attenuation dominance. Crossing zero is
inconclusive, not equivalence. Report all conditional effects and source interactions.

Secondary diagnostic: if the paired 95% interval for Q_A1L1−Q_A0L1 lies wholly
inside [−.03,+.03], latency-only is practically equivalent to the implemented joint
condition at this declared tolerance. Failure to fit inside that interval does not
prove a difference. If latency-only fails to recover the gain while amplitude-only
does, that contradicts the latency explanation. If neither new cell does, report
an amplitude × latency interaction rather than claiming an isolated cause.

Use the same 10,000 PCG64-314159 paired whole-group/whole-bank draws as the parents,
identical for all cells, source levels, representations and target variants. Report
macro-F1, balanced accuracy, accuracy, per-class precision/recall/F1, confusion
matrices, AUROC, truck AP, Brier, log loss, ECE, predicted-truck rate, group results
and worst group/class recall. Secondary intervals are descriptive; four uneven
groups give conditional, limited uncertainty. All validation path pairs are scored
with matched source level and bank, without refitting. Reuse existing gain findings;
no new gain sweep. Independently reconstruct primary contrasts from integer counts.

## Admission, hardware, artifacts and completion

Freeze exact observation/fit/target manifests, parent hashes, design and executable
code. Before full generation, require: centered-FIR scalar-stencil agreement on
moving coefficients; delayed-identity behavior; static complex response at the
existing 5/20/50 m and reversed-height fixtures across 125/500/1500/3000 Hz;
endpoint equality; exact repeated rendering; and full pilot replay. Stop on failure.
After execution, independently check every model's roles/scaler/settings, all
target and cross-path predictions, paired intervals, and all preserved parent files.

Use the same M3 Pro/18 GiB, four render workers, four Torch CPU threads, one BLAS
thread and batch eight. About 1.3 GB is needed for new final observations, plus
features/results; full renders are verified by exact replay without duplicating
their arrays on disk. Estimate 15–30 minutes each for generation and replay,
5–10 minutes for features, then measure actual times. No user download or purchase.

Deliver frozen configuration/manifests, full-render hashes, retained observations,
complete replay evidence, features, 80 fixed heads, every prediction/metric,
an independent audit, figures and `reports/H2_air_latency_control.md`. Update the
roadmap only with the resulting attribution and its limits. Do not start H3–H5.

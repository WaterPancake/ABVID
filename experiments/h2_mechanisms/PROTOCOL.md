# H2 follow-up: ground, air and direct-path collapse

Protocol `ground_air_collapse_20261004_v1`. This is an authorized development
follow-up to [H2](../../reports/H2_source_path_results.md), whose results are already
known. It is not a blind preregistration or independent confirmation. The original
H2 code, sources, renders, features, heads and results remain immutable.

## Questions and estimands

1. Did ground reflection, atmospheric absorption, or their interaction produce
   the H2 propagation benefit?
2. Why do direct-only heads label almost all real excerpts as truck: a software
   or label error, gain sensitivity, poor class ranking, or displacement outside
   the synthetic training distribution?

Cross ground G=0/1 with air A=0/1 at **both** frozen source levels S0/S1. G0A0
reuses H2 P0; G1A1 reuses H2 P1. Add G1A0 (ground only) and G0A1 (air only).
There are eight cells, five banks and two representations: 80 heads total,
including the 40 immutable H2 heads and 40 new heads. No source level, seed or
target group is selected from the new results.

For each source level and bank, compute complete-target macro-F1 Qga. Average
the resulting scores over banks and, for the primary comparison, equally over
the two fixed source levels. Do not pool predictions or resample source levels.

```text
ground = ((Q10-Q00) + (Q11-Q01))/2
air = ((Q01-Q00) + (Q11-Q10))/2
interaction = Q11-Q10-Q01+Q00
ground_minus_air = Q10-Q01
```

The sole primary contrast is BEATs ground_minus_air averaged over source levels.
Report conditional effects and interactions at each S, and MFCC as a secondary
control. A positive/negative interval excluding zero supports a conditional
ordering; a practical 3-point ordering additionally requires the corresponding
marginal benefit's lower bound above zero. An interval crossing zero is
inconclusive. A nonpositive benefit upper bound contradicts a useful benefit.
An effect of these interventions is not an intrinsic fraction of physical mismatch.

## Exact populations and unchanged controls

Reuse the H2 freeze `source_path_20261004_v1`, execution `source_path_20261004_r1`,
all 240 base templates / 480 dry sources, all 1,200 class-shared geometry slots,
and banks 42, 123, 456, 789, 1024. Each cell/bank uses 19 training templates/class
times ten paths = 190/class, and five disjoint validation templates/class times
ten paths = 50/class. All derivatives of a source parent retain its role.
Car=0, truck=1; motorcycles remain excluded. These are generated templates, not
independent real vehicle identities.

The complete joined bank has 19,200 observations: 15,200 training and 4,000
validation. Exactly 9,600 observations are new; the other 9,600 and their heads
are reused by hash and exact ID mapping. No real example trains a head.

The target is exactly the H1/H2 8,066 MELAUDIS observations: 7,810 cars / 256
trucks in four conservative provenance groups. Original sessions remain unknown.
Native target preprocessing and cached features remain the primary evaluation.
No military audio, reserved source, IDMT validation selection or new dataset is
used. All target-derived summaries are diagnostic, never fitting statistics.

Source synthesis, engine priors, trajectories, speed/direction, atmosphere,
raw ground parameter 20,000, heights and ideal mono sensor are unchanged. Their
definitions and unknown ground calibration are in [H2's protocol](../h2/PROTOCOL.md).
Use the same repaired renderer primitives, 31-tap delay interpolation, 40-tap
ground FIR and 11-tap air FIR. Ground is a reflected-path toggle; air affects
the direct path and both reflected legs. Retain all implemented filter latency:
the air factor includes its finite-filter implementation, not attenuation alone.

Each ten-second float64 render uses 8 kHz. Apply the same geometry-defined
two-second crop and -26 dBFS scalar crop RMS, then the identical 8→16 kHz
resampling to 32,000 float32 samples. No extra noise or augmentation. All eight
class/source members of a geometry's two new path conditions pass together;
nonfinite, silent, peak >.98, shape or renderer failure stops the bank. Zero
redraws or selective exclusions. Source/receiver RMS normalization still removes
absolute attenuation/SNR as tested benefits.

Frozen BEATs768 and MFCC26 are identical to H1/H2. Each new scaler/L2 logistic
head uses its own 380 training rows: C=1, lbfgs, tol=1e-6, max_iter=5000,
intercept, no class weighting, float64 features and replicate random seed.
Truck iff probability >.5. No tuning, early stopping, fine-tuning or calibration.
Validation remains diagnostic. Lock all new heads before target inference.

## Prespecified collapse diagnostics

Run every check, including negative findings; do not choose a remedy by target F1.

1. Verify class mapping, balanced fits, training-only scaler statistics, coefficient
   sign/score reconstruction and probabilities. Reproduce all original native
   endpoint predictions. Report train/validation/target score distributions,
   class recalls, predicted-truck fraction, AUROC/AP and calibration. Ranking
   success with a displaced decision boundary differs from failed class ranking;
   do not select a new threshold from the target.
2. For every fitted head, score its same-source-level, disjoint validation
   templates through **all four** propagation settings. This separates failure
   under an unseen simulated path from failure specific to real audio. No refits.
3. Apply one label-independent target gain intervention: normalize each existing
   16 kHz two-second crop to -26 dBFS RMS in float64, then cast to float32. Use
   the same fixed heads. The target's crop/order/bandwidth do not change. This is
   a diagnostic applied after resampling; H2 training normalizes before resampling,
   so tiny final-RMS differences remain explicit. No gain grid, clipping, peak
   rejection, label selection or target normalization statistics. Finite float
   samples above 1 are retained and their counts/peaks reported; this is not an
   integer audio export. Zero/silent inputs stop the diagnostic.
4. On every synthetic validation observation, apply fixed -12 and +12 dB scalar
   gains without clipping; retain 0 dB as the existing observation. Extract the
   same features and score only its matched fixed head. No new fitting. This is
   a controlled gain-sensitivity check, not additional independent validation.
5. Measure waveform RMS/peak and normalized spectral power in fixed bands
   [0,200), [200,500), [500,1000), [1000,2000), [2000,4000] Hz, plus spectral
   centroid and 85% rolloff (Welch, 16 kHz, Hann, nperseg=1024, overlap=512).
   Compare classes/domains descriptively. Report each head's standardized-feature
   norm, fraction beyond its training 95th-percentile norm, decision-score
   quantiles and mean per-coordinate linear contributions. MFCC mean-C0 is
   explicitly identified; unnamed BEATs coordinates get no physical attribution.
   These associations cannot uniquely identify a real engine or sensor mechanism.

The fixed gain intervention can falsify the claim that scalar level mismatch
alone explains collapse: persistent car failure and low ranking after matching
show a residual mismatch. Improvement establishes sensitivity to this input
change, not a selected deployable preprocessing rule. Synthetic gain stress
tests whether the same direction of bias can be induced without changing source
identity, spectrum shape or path. A complete explanation may remain conditional;
report unresolved source/environment/sensor confounds instead of asserting causality.

## Uncertainty, checks and artifacts

Report all H2 metrics by cell/source/seed/group, with class support and worst
group/class recall. Reuse H2's 10,000 paired whole-group/whole-bank bootstrap
draws, PCG64 seed 314159, linear 2.5/97.5 percentiles. Apply identical draws to
all cells, source levels, representations and native/gain-matched target variants.
Report gain-matched minus native effects with paired intervals. Four uneven
incompletely reconstructed groups give weak conditional uncertainty. Secondary
intervals are descriptive; they are not multiple opportunities to pick a winner.

Before bulk rendering: freeze this design and exact manifests; validate toggle
independence, endpoint equality, static tone response and scalar-reference
agreement for both new conditions, including a moving circular-buffer wrap.
Use existing analytic tolerances (0.5 dB, 1% where applicable). Hash all parent
artifacts and new executable code/environment. Verify all new complete waveforms
by fresh-process exact replay and all reused endpoints against their original
hashes before fitting. Independently reconstruct metrics and bootstrap contrasts
from confusion counts after evaluation. Preserve every failed run.

Deliver a protocol/config/locks, joined manifests, new renders/features/heads,
native and diagnostic probabilities, individual diagnostic findings, paired
statistics, figures, an independent audit and `reports/H2_ground_air_collapse.md`.
Update the roadmap status narrowly; no later milestone starts automatically.

Use the existing M3 Pro / 18 GiB: four rendering workers, one BLAS thread,
four Torch CPU threads, batch eight. Reuse sources/endpoints. New complete
renders/observations need about 7.4 GB; features/diagnostics are smaller. Check
free disk space before generation. Estimate 20–40 minutes each for generation
and complete replay and 10–15 minutes for all new embeddings, then measure actual
time. No cloud/GPU purchase, access request or user download is required.

# H2 follow-up: ground, air and direct-path collapse

**Test domain: controlled synthetic → previously exposed MELAUDIS development data.**
Completed 2026-10-04 local time. This report preserves the first mechanism follow-up;
it does not replace [the original H2 result](../../reports/H2_source_path_results.md).

**The large BEATs gain requires the implemented ground and air branches together.
Neither branch alone reproduces it. Scalar loudness mismatch does not explain the
direct-path collapse. A newly quantified filter-latency confound prevents attributing
the joint gain specifically to physical atmospheric absorption.**

| BEATs training path | Native-real macro-F1 | Native-real car recall |
|---|---:|---:|
| Moving direct path | 3.38% | 0.30% |
| Direct + air branch | 3.40% | 0.32% |
| Direct + ground branch | 3.65% | 0.59% |
| Direct + ground + air branches | 35.46% | 48.10% |

These values equally average the two fixed source levels and five banks. Here and
in the figures, **“air” means the implemented finite-FIR branch, including latency**.
It is not an attenuation-only intervention. The interaction is **+31.80 macro-F1
points, conditional 95% CI [24.76, 41.14]**. The primary ground-minus-air contrast is
only **+0.26 [−0.07, 1.22] points**; neither factor satisfies the prespecified
practical-dominance criterion. All cells and both target-level variants retain a
zero-recall source/bank/group/class case. This is not reliable recognition or
independent confirmation. Joint-path balanced accuracy is 60.65%, but its macro-F1
is still below the 49.19% always-car reference on this heavily imbalanced target.

## Findings for each check

### 1. Reproducibility, roles and software consistency — passed

All 9,600 new complete waveforms replay exactly; all 9,600 reused endpoints match
their original full-waveform and observation hashes. All 80 heads preserve car=0,
truck=1, balanced training, disjoint source-template roles, training-only scalers,
fixed hyperparameters and convergence. Reconstructed logistic scores and saved
probabilities agree; all 40 original native-target prediction vectors are preserved.
The independent audit reproduced every target, cross-path and gain evaluation,
and independently reconstructed 504 F1/balanced-accuracy/truck-rate intervals from
integer confusion counts. This rules out the checked software/ID/scaling mistakes;
it does not establish that every dataset annotation is correct.

### 2. Ground versus air — interaction, not a standalone winner

Adding ground without air gives +0.27 [−0.05, 1.23] F1 points. Adding air without
ground gives +0.014 [−0.004, 0.048] points. Adding ground when air is present gives
+32.07 [24.75, 42.10]; adding air when ground is present gives +31.81 [24.79, 41.14].
The positive interaction appears at both source levels: +34.32 points at S0 and
+29.27 at S1, with both conditional intervals above zero.

The ground and air marginal averages, +16.17 and +15.91 points, therefore must not
be interpreted as two independently useful 16-point improvements or as physical
shares of the domain gap. They average over an almost ineffective standalone
condition and a strongly effective joint condition. MFCC remains near its
always-truck reference throughout. Full conditional estimates are retained below
and in the statistics artifact.

### 3. Matched synthetic validation — perfect, but insufficient

Every head obtains 100% macro-F1 on its matched synthetic validation templates.
These templates are disjoint from fitting, but share the specified generator
families and parameter distributions. This is synthetic generalization within
that construction, not evidence of independent real-vehicle recognition.

### 4. Unseen simulated path — some sensitivity before reaching real audio

BEATs heads trained on the direct path obtain 100% F1 on direct/air-only validation,
80.01% on ground-only and 89.91% on the joint path. MFCC direct-path heads obtain
81.57% on ground-only and 47.14% on the joint path. Thus path changes can move the
representation even with source templates held out. The remaining real-audio
failure is much larger for BEATs: its direct-path real F1 is 3.38%. Every path pair
was evaluated; none was selected as a new training recipe.

### 5. Fixed target RMS matching — no rescue of the direct-path collapse

Setting each existing real crop to −26 dBFS changes direct-path BEATs F1 from
3.3825% to 3.3875%: **+0.0051 points, CI [−0.0316, 0.0302]**. Car recall remains
about 0.31%. MFCC direct-path car recall is 0% after matching. The joint BEATs
condition also does not improve: its F1 changes by −0.69 [−0.97, 0.56] points.

Original real-car RMS has median −28.02 dBFS and a 5–95% range of −35.35 to −19.40;
real trucks have median −23.78, range −32.37 to −16.86. Median gains are +2.02 dB
for cars and −2.22 dB for trucks. All 8,066 original crop and MFCC identities were
verified, and all normalized waveforms were reconstructed by an independent scalar
formula. No clipping or selective exclusion occurred. One normalized float crop
peaks at 1.02015; it is retained as declared, not exported as clipped integer audio.

**Interpretation:** scalar gain mismatch alone is contradicted as a sufficient
explanation for this collapse. This does not establish universal level invariance
or test microphone frequency response, clipping, SNR, or additive background.

### 6. Synthetic ±12 dB gain stress — does not induce the direct-path failure

Both direct-path representations retain 100% car and truck recall at −12, 0 and
+12 dB, on every tested source level/bank. Across other paths, the largest mean
class-recall loss is one point: BEATs ground-only truck recall becomes 99% at
+12 dB. The floating-point stress waveforms are not clipped; their maximum peak
is 1.30902. These checks reinforce the negative loudness finding within the tested
range and source population; they do not prove invariance under all recording gains.

### 7. Real feature displacement and class ranking — the main observed failure

For direct-path BEATs heads, the mean car decision score moves from **−9.31 in
synthetic training to +6.83 on real cars**; positive scores predict truck. Real
truck scores average +6.76. All real observations exceed their respective head’s
training 95th-percentile feature-norm threshold, before and after RMS matching.
For real cars, about 27.49% of standardized coordinates lie outside the training
coordinate ranges. Those are descriptive distances, not a calibrated OOD test.

BEATs direct-path AUROC is 48.61% [45.63, 57.34], rising only to 51.70%
[49.40, 65.05] after matching. Its problem includes poor class ordering, not just
an inconvenient threshold. Joint-path BEATs improves AUROC to 67.96% and lowers
the mean real-car score to +0.12, but does not eliminate the feature-distribution
mismatch or worst-group failures. No target threshold was fitted.

MFCC differs: its direct-path AUROC is 70.64% [66.36, 82.80] despite predicting
almost every event as truck. Its real car/truck mean scores, +9.64/+10.79, show
some ordering on the wrong side of the fixed decision boundary. Mean C0 contributes
only +1.22 to the real-car score shift of +16.54; the other coefficients account
for the rest. This is not a C0-only or scalar-gain explanation, and no new
calibration rule was selected.

### 8. Waveform spectrum — residual differences, with unknown physical origin

Direct synthetic cars have mean 0–4 kHz centroid 355 Hz, joint-path synthetic cars
472 Hz, and real cars 416 Hz. The joint path changes their low/mid-band balance,
but matching a centroid is not evidence that the full signal distribution matches.
The mean 2–4 kHz power share is approximately 0.01% for direct synthetic cars
versus 1.88% for real cars; corresponding truck values are 0.29% versus 2.48%.
All measurements use the same processed bandwidth. Real high-band energy could
come from vehicle source detail, background, sensor/processing, or mixtures of
these. This study does not uniquely identify its origin. BEATs coordinates are
not assigned engine/ground/sensor meanings.

### 9. Air-filter implementation inspection — a concrete attribution confound

This inspection was added after the factorial result; it does not add predictions,
refit heads or change frozen code. The protocol had already specified that air
included finite-FIR latency. The [renderer](../../experiments/h2/renderer.py)
constructs symmetric 11-tap filters and applies one to the direct route and one
to each reflected leg. Each has a five-sample static group delay: direct gets
five samples, reflected gets ten, introducing **0.625 ms of extra relative delay
at 8 kHz**. Even the zero-distance/no-absorption filter is a five-sample delayed
identity, to numerical error below 1e−14.

The saved static response inspection gives a concrete witness. At 20 m horizontal
range and 500 Hz, direct-path air attenuation is only about 0.047 dB. The implemented
joint transfer nevertheless differs from a zero-phase, magnitude-only air transfer
by 13.55 dB, because the extra delay changes interference. A delay-only transfer
is within 0.047 dB of the implemented joint transfer at that point. These are
static transfer calculations, **not** classification ablations, and cannot prove
how much of the F1 interaction was caused by delay.

The current experiment therefore identifies a joint *implemented-branch* effect.
It does **not** isolate physical absorption from numerical filter latency. A
separate attenuation × latency control with ground fixed is needed before making
that attribution; the present outputs remain frozen while that control is prepared.

## What changes downstream

The loudness-only explanation can be deprioritized for this collapse. Broad
representation mismatch remains directly observed, with different ranking failures
for BEATs and MFCC. The propagation benefit must be interpreted as a coupled
filter/path result until latency is separated from attenuation. The next controlled
comparison should address that specific implementation confound before adding
source complexity, architecture changes or target-driven tuning. The broad
source-versus-propagation hypothesis remains open beyond these interventions.

## Population, controls and limits

The [frozen protocol](../../experiments/h2_mechanisms/PROTOCOL.md) fixes 240 base
templates, 480 dry sources, 1,200 paired geometries, two source levels and four
paths: 19,200 joined observations, half new. Each head uses 190 training events
per class from 19 templates/class and 50 validation events/class from five
disjoint templates/class. Banks are 42, 123, 456, 789 and 1024. Car=0 and truck=1;
motorcycles are excluded. These templates are not independent recorded vehicles.

Ten-second renders use 8 kHz; the geometry-defined two-second crop receives
−26 dBFS RMS normalization before 8→16 kHz resampling. BEATs768 and MFCC26,
training-only StandardScaler and C=1 L2 logistic regression remain fixed. No
encoder fine-tuning, new backgrounds, sensor randomization, target model selection
or threshold tuning occurs. Receiver normalization conditions out absolute
attenuation/SNR benefits. Ground calibration remains unknown. The unchanged BEATs
checkpoint is verified by its local mirror hash; an official upstream checksum
identity remains unverified, as recorded in the parent experiment.

The exact target has 7,810 cars and 256 trucks in four conservative, uneven
provenance groups; original sessions remain unknown. Paired 95% intervals use
10,000 whole-group/whole-bank draws, PCG64 seed 314159. The two source levels are
fixed and equally averaged, not resampled. Only native BEATs ground-minus-air F1
is primary; other intervals are descriptive. Group support, including one group
with a single truck, limits the uncertainty assessment. All findings are on exposed
development audio, not untouched confirmation. No military milestone gate advances.

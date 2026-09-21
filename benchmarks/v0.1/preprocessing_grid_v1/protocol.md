# Preprocessing grid v1: preregistered before feature extraction / scoring

Domain: **native real → real, nested unseen recording sessions**. User-requested
scope: park startup audio, then grid-search preprocessing over all eligible
development windows. Preserve historical startup annotations and all original
audio; do not relabel startup as idle. Old benchmark snapshots remain unchanged.

Use every complete reviewed **2 s window at 1 s hop**, channel 0 / 16 kHz,
with **no per-session cap and no RMS/audibility gate**. Exclude startup and all
protected recording sessions before any feature extraction. No rejected gaps,
reserved T90M/JLTV, consumed Sherman/PDSounds, unreviewed clips or procedural
audio enter the grid. Check catalog/sidecar agreement, hashes and reviewed bounds.
All 7 tracked / 5 wheeled development sessions must remain represented. Longer
recordings have more windows but not greater total session training weight.

Grid: DC subtraction {off,on} × high-pass {none,30,50,80 Hz} × RMS target
{off,−26,−20 dBFS}: **24 pipelines**, including an exact unchanged-input control.
Order is DC subtraction → high-pass → scalar RMS gain. High-pass is a second-order
Butterworth run forward/backward within each window (effective fourth-order
magnitude response, zero phase); no other recording context is used. This is
**offline**, not a causal streaming claim. RMS gain is capped at ±12 dB and then
limited to a 0.99 peak; peak safety can override the lower gain cap. Silence below
1e−8 RMS keeps unity gain. This does not reconstruct clipped inputs or improve SNR.

Keep the frozen PANNs checkpoint and 16→32 kHz frontend unchanged. Primary head:
35 AudioSet vehicle/mechanical scores. Secondary, separately reported comparator:
53 classical features. The pretrained "Engine starting" AudioSet feature is not
a ground-truth startup tag; retain the fixed ontology feature list so condition
exclusion does not simultaneously change the model. No new encoder, PCEN frontend,
denoising network, fine-tuning, augmentation, fusion or context-length selection.

For each outer held-out tracked/wheeled session pair, inner leave-one-session-per-
class-out validation selects **both preprocessing and logistic C** from
{0.001,0.01,0.1,1,10}. Standardization and equal-class/equal-session weighting use
training samples only. Fixed threshold 0.5. Candidate C selection matches the
existing mean-BA, median-BA, smaller-C rule. Across pipelines rank inner mean BA,
inner median BA, fewer operations, lower high-pass cutoff, then fixed grid order.
Never select preprocessing using outer-fold accuracy. Save chosen pipeline/C per
outer fold and its checkpoint. Report the unchanged pipeline with inner-selected
C on the **same new dataset**, not only comparison against historical capped data.

Keep all 24 fixed-pipeline outer results as exploratory diagnostics (including
fixed-C ensembles), but **the largest outer score is not the grid-search result**.
Headline result is the inner-selected pipeline evaluated once on each outer pair.
Report per-class/per-session/worst-session recall, BA, accuracy, F1, confusion
matrices, descriptive class-stratified session-bootstrap intervals and paired
differences versus unchanged input. Overlapping windows and repeated outer folds
are not independent samples. This remains development, not locked confirmation.

Version manifests, removed-interval history, catalog, source/window hashes,
config, seed, git commit/dirty state, code/dependency versions, split membership,
feature caches, processing diagnostics and saved heads. Verify saved-head replay,
startup/protected exclusions, numerical stability and regression tests. Do not
open the reserved evaluation pair even if one exploratory pipeline looks good.

Configuration: `configs/benchmark_preprocessing_grid_v1.yaml`.

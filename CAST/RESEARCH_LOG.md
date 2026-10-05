# CAST research log

Experiment: CAST-0 through CAST-2, `cast_pilot_v0_20261004`.

Date: 2026-10-04. Synthetic acceptance preceded all real waveform decoding.

Hypothesis: the bounded effective-observation renderer can improve in-sample
spectral reconstruction over its best unoptimized initialization; physical
identifiability and downstream transfer are separate, untested outcomes.

Config: [resolved configuration](runs/cast_pilot_v0_20261004/config.resolved.json),
SHA256 `88f0f975b9c39c7510c0576f815be1b2cd685070faf9fd045950e496b5b6b5bb`.
Commit: `0d242c6d76ff036747f89576ee15d5f7399f5891`, dirty. Full dirty/source/environment
snapshot and implementation copies are stored inside the run; the commit alone
does not identify these new files.

Dataset split: IDMT_V1 sE8 CH34, exact H1 fold-0 seed-42 real training selection,
lexicographically first five IDs per remaining group/class: 25 cars and 25 trucks
across five conservative groups. No shortages or replacement IDs. Fold-0 held-out
group, target data, mixed feature caches, supplied synthesis banks and military
audio excluded. Test domain: source-real in-sample observation reconstruction;
synthetic acceptance is synthetic-to-synthetic reconstruction, not classification.

Seeds: run 42; starts 42/123/456/789. Component-purpose streams are SHA256-derived.
Each fit uses two fixed noise realizations and two disjoint checking realizations.

Result: 29 CAST tests and seven synthetic fits plus a gain-ambiguity check passed.
Three legacy source-simulator tests passed separately. All 50 real fits completed;
all beat their best unoptimized checking loss by at least 5%. Median paired loss
reductions: car 19.58%, truck 22.08%. All 50 exported reconstructions replayed
bit-for-bit, source/artifact hashes and ancestry verified, checking losses
recomputed, common-gain playback read back successfully.

Runtime: 813.51 s real pilot; 933.75 s complete workflow. CPU only, one Torch
thread. Peak process RSS 0.689 GiB at pilot completion and 0.692 GiB after final
verification. Runtime is measured, not extrapolated from earlier classifiers.

Interpretation: numerical fitting works within this representation; better
in-sample loss is not acoustic coverage, source recovery, or transfer evidence.

Failure modes: zero numerical failures; 43/50 fits have disagreeing controls
among near-equal starts, two are noise-dominant, one has weak salience under the
declared heuristic, and two hit spacing-bound diagnostics. Flags overlap.
Synthetic frequency-initialization defects and unsuccessful repairs were retained
under `development/`; final numeric tolerances and iteration budgets were not
relaxed. The fundamental-only single-tone test explicitly fixes known harmonic
order; a second unrestricted tone tests recovery plus ambiguity.

Decision: CAST-2 complete. Do not infer a stable distribution of identifiable
physical parameters or proceed to CAST-3+ from these fits.

Next experiment: none executed or authorized here. Propose identifiability-aware
reporting/equivalence handling, then a separately versioned source-only revision
if warranted. No classifier, target evaluation, full-corpus fit, or military
milestone advancement occurred.

Artifacts: [pilot report](../reports/CAST_pilot.md),
[audio gallery](runs/cast_pilot_v0_20261004/index.html),
[verification](runs/cast_pilot_v0_20261004/verification.json).

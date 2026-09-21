# Benchmark v0.1 native-real 7/5 result

Date: 2026-09-20  
Status: development evaluation complete; gates 2-4 failed

## Evidence boundary

This is a `real -> real` nested unseen-recording-session development result. It is
not a locked confirmation, a vehicle-presence result, or field validation. The
T90M/JLTV future confirmation pair and consumed Sherman/PDSounds pair have zero
overlap with this manifest and were not evaluated.

The run used 593 two-second windows from 7 tracked and 5 wheeled sessions. The
manifest SHA-256 and dataset version are
`e013532104c209e3c8f6906bb3c9daa3c46c2d6fb2e9fed82a65f4d2699a42c4`.
The tracked [dataset lock](../benchmarks/v0.1/native_real_7t5w/dataset_lock.json)
records every session and source hash; the [artifact snapshot and reproduction commands](../benchmarks/v0.1/native_real_7t5w/README.md)
preserve the manifest, reviewed intervals, full results and split assignments. The local window manifest is
`data/benchmark_v0_1_native_real_7t5w/real_manifest.jsonl`.

The protocol was fixed in
[`benchmark_v0_1_native_real_protocol.md`](benchmark_v0_1_native_real_protocol.md)
before these scores were generated. The recorded Git commit is
`0d242c6d76ff036747f89576ee15d5f7399f5891`; the worktree was dirty, so the dataset,
protocol, code, and result hashes are the stronger identity for this run.

## Results

The table reports the fixed regularization ensembles for the two comparators and
the preregistered inner-threshold-selected equal fusion as the primary model.

| Model | Mean balanced accuracy | Tracked recall | Wheeled recall | Worst held-out session-context recall | Both held-out sessions correct |
|---|---:|---:|---:|---:|---:|
| Classical MFCC/spectral | 45.24% +/- 20.29% | 68.58% | 21.89% | 0.00% | 14.29% |
| Semantic PANNs | **53.40% +/- 15.73%** | **81.04%** | 25.77% | 0.00% | 20.00% |
| Equal semantic/classical fusion (primary) | 39.70% +/- 14.07% | 51.98% | **27.42%** | 0.00% | 0.00% |

These are means and sample standard deviations across 35 outer tracked/wheeled
session pairs. Each fold's full metrics, confusion matrix, selected hyperparameters,
session predictions, split membership, and model state are retained under
`runs/benchmark_v0_1_native_real_7t5w/`.

A descriptive, stratified 10,000-replicate bootstrap over the primary model's
per-session mean recalls gives a 95% percentile interval of **28.67%-51.22%** for
session-balanced accuracy, **37.74%-65.24%** for tracked recall, and
**11.21%-47.68%** for wheeled recall. Seed 42 was used. This interval was not used
for model or threshold selection.
The 35 folds share training and test sessions and are not 35 independent experiments.
This descriptive interval resamples 7 tracked and 5 wheeled session means; it does
not include uncertainty from refitting models or acquiring different recordings.

## Gate assessment

| Gate | Result | Status |
|---|---:|---|
| At least five development sessions per class | 7 tracked / 5 wheeled | **Pass** |
| Mean balanced accuracy at least 75% | 39.70% primary fusion | Fail |
| Mean recall at least 70% for both classes | 51.98% / 27.42% | Fail |
| Every held-out session at least 50% window recall | Minimum 0.00% | Fail |

No model is frozen, and the reserved confirmation pair must remain untouched.

## What changed

The enlarged corpus exposes poor generalization beyond the earlier 4/3 development
corpus. The two new modern military wheeled sessions are usually
predicted as tracked. Under the primary fusion, mean recall across pairing contexts
is 7.59% for the M1126 Stryker convoy and 12.28% for the M1151 HMMWV. The semantic
comparator is more stable on tracked sessions but still recalls only 3.13% of the
Stryker windows and 9.60% of the HMMWV windows on average.

One hypothesis is that the existing representations rely on civilian engine/source
signatures and broad heavy-vehicle cues. These scores alone do not identify the
acoustic cause; changed labels, session composition and threshold instability also
need examination. The result shows that the historical equal-fusion and
threshold-selection rule does not transfer to the expanded session distribution.
The old 77.35% result remains valid only for its seven-session development corpus.

## Next development step

Keep this result immutable as Benchmark v0.1's first 7/5 baseline. Before defining a
new protocol version:

1. audit per-session loudness, bandwidth, operating-state, recording-device, and
   vehicle-count distributions for confounds;
2. inspect feature and probability distributions without using protected sources;
3. add more independent heavy wheeled sessions, especially single-vehicle pass-bys,
   so HMMWV/Stryker do not define that domain alone;
4. only then preregister any representation, aggregation, or calibration change as
   a new development protocol and rerun all 35 session pairs.

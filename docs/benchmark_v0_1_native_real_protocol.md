# Benchmark v0.1 native-real preregistration

Date: 2026-09-20  
Status: preregistered before evaluating the 7-tracked/5-wheeled corpus

This document freezes the development protocol for the refreshed native-real
benchmark. The machine-readable authority is
[`configs/benchmark_v0_1_native_real.yaml`](../configs/benchmark_v0_1_native_real.yaml).

## Dataset construction

- Discover only local targets whose sidecar has `admitted_to_corpus: true`.
- Keep every original recording in exactly one `recording_session`.
- Use only human-reviewed allowlist intervals; do not infer an SNR for native audio.
- Select channel 0, resample to 16 kHz, and make two-second windows at a one-second
  hop, capped at 64 evenly spaced windows per session.
- Reject any manifest with incomplete provenance, unreviewed whole recordings, or
  overlap with consumed Sherman/PDSounds or reserved T90M/JLTV sources.

## Evaluation

The primary result is equal-probability late fusion of the fixed 35-output semantic
PANNs ensemble and the existing 53-feature classical ensemble. Classical-only and
semantic-only ensembles are comparators. The PANNs Cnn14 encoder remains frozen.

Every outer fold holds out one complete tracked session and one complete wheeled
session. Model and threshold selection use only pairwise inner folds among the
remaining sessions. Training weights give equal mass to each class, each session
within its class, and each window within its session.

The fixed regularization candidates are `0.001, 0.01, 0.1, 1, 10`. The primary
fusion threshold candidates are `0.20` through `0.60` in increments of `0.05`.
No candidate, feature family, or threshold will be changed after viewing these
development results without declaring a new protocol version.

## Acceptance gates

On this same manifest and protocol:

1. at least five independent sessions per class;
2. mean outer-fold balanced accuracy at least 75%;
3. mean tracked and wheeled recall each at least 70%;
4. every held-out session has at least 50% window recall.

Passing these four development gates authorizes freezing a candidate rule. It does
not authorize Milestone 7. T90M/JLTV remain untouched until a frozen rule is ready
for their one-shot confirmation.

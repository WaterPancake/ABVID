# Native-real 7/5 failure diagnosis

Date: 2026-09-20. Domain: **real -> real, nested unseen-session development**.
This is post-hoc explanation, not new model selection or a locked test.

Reproduce with:

```bash
uv run python scripts/diagnose_native_benchmark.py --output benchmarks/v0.1/diagnostics_7t5w_v1
```

Use a new output directory for reruns. The [complete report](../benchmarks/v0.1/diagnostics_7t5w_v1/report.md),
[machine-readable results](../benchmarks/v0.1/diagnostics_7t5w_v1/results.json), and
[window diagnostics](../benchmarks/v0.1/diagnostics_7t5w_v1/window_diagnostics.json)
retain acoustic measurements, operating-state breakdowns, probe contributions, hashes,
and splits. Saved models reproduce the original per-class recalls exactly in all 35
folds for all three branches. No source labels, checkpoints, or native benchmark
artifacts were changed; no protected audio was loaded.

## Findings

1. **Threshold selection is unstable, but not the sole failure.** Selected wheeled
   thresholds span 0.20-0.60; 12 of 35 folds choose the upper boundary, 0.60. Inner
   selected balanced accuracy averages only 50.40%. Holding fusion at 0.50 gives
   41.66% balanced accuracy, versus 39.70% for the original rule, but wheeled recall
   falls from 27.42% to 11.64%. This post-hoc comparison does not justify replacing
   the frozen threshold rule or claiming a successful model.
2. **Equal fusion sometimes dilutes the stronger branch.** On T-72/BMP-3, semantic
   recall is 95.0%, classical 7.2%, fusion 16.9%. On Maserati, the respective recalls
   are 81.4%, 0.7%, and 62.6%. The branches disagree on an average 37.97% of test
   windows per fold. The semantic comparator's overall 53.40% balanced accuracy is
   still inadequate; removing fusion alone would not solve generalization.
3. **Modern wheeled errors are broad, not confined to ambiguous states.** Stryker
   fusion recall is 4.4% at idle, 0% during acceleration, and 6.4% at steady speed.
   HMMWV gives 13.5%, 17.9%, and 10.4%, respectively. Sparse states and overlapping
   windows make these descriptive breakdowns, not independent experiments.
4. **Recording characteristics differ, but are not established causes.** HMMWV
   and Stryker have median digital levels of -30.4 and -25.1 dBFS, and median
   spectral bandwidths of 476 and 541 Hz. Several tracked recordings have similar
   or narrower bandwidths. Maserati is narrower still (226 Hz), yet its semantic
   recall is much higher. Stryker's within-session loudness-versus-fusion-correctness
   Spearman correlation is approximately -0.023: simply favoring louder windows
   does not explain that session's errors. Native recording SNR remains unknown.
5. **The semantic head uses context-sensitive proxies.** For Stryker, standardized
   AudioSet `Air brake` and `Accelerating, revving, vroom` outputs contribute toward
   tracked predictions; `Truck` contributes toward wheeled. Maserati's revving
   output strongly contributes toward wheeled. These are mean linear-probe logit
   contributions relative to training means, not proof those sounds are present
   or causal evidence about tracks versus wheels. Averaging logits also does not
   explain the probability ensemble exactly.

## What remains unresolved

Device, distance, and simultaneous vehicle count are not systematically annotated.
Stryker is documented as a convoy, but exact counts are unknown. Provider metadata
is not a substitute for microphone identity. Twelve sessions cannot isolate these
confounds statistically, and dependent windows must not inflate confidence.

Next: characterize fixed-model sensitivity using controlled recorded-background
mixing and explicitly simulated microphone responses. Keep the native reference
unchanged. Any future changes to features, weighting, aggregation or thresholds
require a new development protocol; neither T90M nor JLTV may inform those choices.

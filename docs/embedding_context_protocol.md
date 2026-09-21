# Full embeddings, then longer context: preregistered development follow-up v1

Written before evaluating the new heads. Domain: **native real → real, unseen
recording sessions**. This is development, not another locked confirmation.

1. On the exact frozen 593-window, 7-tracked/5-wheeled manifest, compare the
   frozen Cnn14 full 2,048-dimensional embedding with the existing 35 AudioSet
   scores. Use the existing session/class-balanced scaler and logistic probes,
   C = 0.001/0.01/0.1/1/10, seed 42, nested leave-one-session-per-class-out.
   Primary comparison: fixed equal-probability C ensemble, threshold 0.5 (matches
   the original semantic comparator). Also report inner-selected C without
   selecting between these rules using outer results. No fusion or fine-tuning.
2. Only after step 1, evaluate context for **both** representations, regardless
   of step 1's result. Primary matched set: 2/4 s, all 12 sessions. Supplemental
   matched set: 2/4/8 s, 11 sessions; StuG has no reviewed interval ≥8 s.
   Take centers from the original capped native windows, retaining only centers
   whose longest window fits wholly inside the same reviewed condition interval.
   All lengths in a set have identical centers, labels, train/test sessions and
   sample counts. Fit fresh heads separately for every set/length. A 2 s matched
   control is mandatory; do not compare an 8 s subset to the original 593 windows
   as evidence of a duration effect. Do not concatenate, repeat, cross gaps or
   pad to manufacture longer reviewed intervals.
3. As a separate part of the earlier recommendation 3, compare the original
   16→32 kHz path against direct original-source→32 kHz on all 593 original
   two-second windows, using both representations and identical session splits.
   This tests bandwidth/resampling-path changes without changing context. It
   does not reconstruct frequencies already absent from a source recording.

Keep channel 0 and the original 16 kHz extraction followed by PANNs' 32 kHz
resampling fixed. No normalization, filtering, bandwidth or label changes in
the context experiment. Step 3 changes only the source extraction sample rate.
No hypothesis-driven dropping of difficult sessions or state-specific tuning.

Report balanced accuracy, both recalls, F1, per-fold confusion matrices,
per-session mean/minimum recall, worst session-context recall, and descriptive
class-stratified session-bootstrap 95% intervals. Paired differences bootstrap
session-level recall differences, not correlated overlapping windows or the
35 overlapping fits as independent observations. These small-sample intervals
are descriptive, not population guarantees. Report all conditions, not only
the best one. Any new development selection still requires a separately frozen
confirmation rule and must not consume T90M/JLTV during this study.

Inputs are hash-verified by `FrozenBenchmark`, including source audio, frozen
catalog and caches. Save config, code hashes, dirty git state, versions, source
and window hashes, context manifests, checkpoints, splits and metrics in new
versioned directories. Existing baseline artifacts remain unchanged. Memoized
fits may reuse only exactly identical training partitions within one feature
matrix; test cached versus uncached numerical equivalence.

Configuration: `configs/benchmark_embedding_context_v1.yaml`.

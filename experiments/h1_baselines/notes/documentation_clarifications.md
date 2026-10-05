> Historical protocol/command reference. Commands retain their original paths
> and run from `ABVID_ARCHIVE_ROOT`, not from the reorganized code checkout.
> See the [experiment registry](../../README.md).

# H1 documentation clarification before fitting

2026-10-02, after the metadata lock and during fixed feature extraction, before any head fit or target prediction. The live proposal now explicitly matches EXPERIMENTAL_PROTOCOL §5: average source held-out-group F1 rather than pooled event F1; positive-truck binary Brier plus log loss, not the earlier broad proposal's multiclass Brier/ECE. DATASET_PROTOCOL now explicitly accepts provider FLOAT WAVs as used since the full integrity audit and says provider-labelled event rather than certified original event. These changes clarify already frozen executable rules; no data membership, grouping, class, preprocessing, model, budget, seed or estimator changed. The original document snapshots in the lock are retained unchanged.

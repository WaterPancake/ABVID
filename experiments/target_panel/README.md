# Target panel re-scoring

Post hoc, descriptive re-scoring of archived, already exposed predictions with
prevalence-robust metrics. It loads prediction files only, never models, features
or audio, and fits nothing. See the [report](../../reports/target_panel.md).

```bash
# Requires ABVID_ARCHIVE_ROOT (or configs/local.json). Output directory is create-only.
uv run --extra cast python experiments/target_panel/rescore.py
```

[`config.json`](config.json) pins the inputs, the bootstrap draws and seed, and the
output `artifact:target_panel/rescore_20261011_v2`. The output records the config
and script hashes. To re-run, change `output` to a new directory; existing results
are never overwritten.

The config deliberately lives here rather than in `configs/experiments/`. Adding a
registry entry changes the frozen-file canary and `tests/test_protocol.py`, so that
step is left for human review.

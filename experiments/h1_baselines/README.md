# H1 Baselines

Within-domain, cross-dataset and released synthetic baselines.

- Canonical wrapper: [configuration](../../configs/experiments/h1_baselines.json).
- Original contract: `archive:experiments/h1/config.json` (SHA256 pinned by the wrapper).
- Original runner directory: `archive:experiments/h1`.
- [Original command sequences](REPLAY.md) retain their historical paths. Run them
  from `ABVID_ARCHIVE_ROOT`, using the environment described there. New destinations
  must be distinct from completed runs; never rewrite frozen manifests or results.
- Shared numerical code now lives in `src/abvid/`.

```bash
abvid validate configs/experiments/h1_baselines.json --check-files
abvid replay configs/experiments/h1_baselines.json
```

`replay` prints the original command; `--execute` explicitly runs it. The bundled
wrapper is a regression/help entry point, not a new training protocol. The complete
historical workflow remains in the original command sequences above.

- [Report: H1_baseline_results.md](../../reports/H1_baseline_results.md)

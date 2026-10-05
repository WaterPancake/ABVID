# R0 Reproduction

Published splits and released checkpoint audit.

- Canonical wrapper: [configuration](../../configs/experiments/r0_reproduction.json).
- Original contract: `archive:experiments/reproduction/README.md` (SHA256 pinned by the wrapper).
- Original runner directory: `archive:experiments/reproduction`.
- [Original command sequences](REPLAY.md) retain their historical paths. Run them
  from `ABVID_ARCHIVE_ROOT`, using the environment described there. New destinations
  must be distinct from completed runs; never rewrite frozen manifests or results.
- Shared numerical code now lives in `src/abvid/`.

```bash
abvid validate configs/experiments/r0_reproduction.json --check-files
abvid replay configs/experiments/r0_reproduction.json
```

`replay` prints the original command; `--execute` explicitly runs it. The bundled
wrapper is a regression/help entry point, not a new training protocol. The complete
historical workflow remains in the original command sequences above.

- [Report: published_reproduction_status.md](../../reports/published_reproduction_status.md)

# H2 Simulation

Source/path and propagation mechanism controls.

- Canonical wrapper: [configuration](../../configs/experiments/h2_simulation.json).
- Original contract: `archive:experiments/h2/config.json` (SHA256 pinned by the wrapper).
- Original runner directory: `archive:experiments/h2`.
- [Original command sequences](REPLAY.md) retain their historical paths. Run them
  from `ABVID_ARCHIVE_ROOT`, using the environment described there. New destinations
  must be distinct from completed runs; never rewrite frozen manifests or results.
- Shared numerical code now lives in `src/abvid/`.

```bash
abvid validate configs/experiments/h2_simulation.json --check-files
abvid replay configs/experiments/h2_simulation.json
```

`replay` prints the original command; `--execute` explicitly runs it. The bundled
wrapper is a regression/help entry point, not a new training protocol. The complete
historical workflow remains in the original command sequences above.

- [Report: H2_ground_air_collapse.md](../../reports/H2_ground_air_collapse.md)
- [Report: H2_air_latency_control.md](../../reports/H2_air_latency_control.md)

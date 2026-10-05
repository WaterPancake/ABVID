# Cast Coverage

CAST observation-model coverage; transfer remains untested.

- Canonical wrapper: [configuration](../../configs/experiments/cast_coverage.json).
- Original contract: `archive:CAST/improvement/width_v15/PROTOCOL.md` (SHA256 pinned by the wrapper).
- Original runner directory: `archive:CAST`.
- [Original command sequences](REPLAY.md) retain their historical paths. Run them
  from `ABVID_ARCHIVE_ROOT`, using the environment described there. New destinations
  must be distinct from completed runs; never rewrite frozen manifests or results.
- Shared numerical code now lives in `src/abvid/`.

```bash
abvid validate configs/experiments/cast_coverage.json --check-files
abvid replay configs/experiments/cast_coverage.json
```

`replay` prints the original command; `--execute` explicitly runs it. The bundled
wrapper is a regression/help entry point, not a new training protocol. The complete
historical workflow remains in the original command sequences above.



Current numerical implementation includes the v15 width-calibrated population:
`abvid.cast.population.width_prior.WidthPrior`. Rendering, fitting and descriptor
coverage are separate modules. Earlier numerical controls remain where v15 or
regression comparisons depend on them; their original study runners are archived.

The completed v15 result met the declared outer **development** coverage goal
(82.019% car / 81.286% truck, averaged over the fixed seeds). Some individual
car-seed results remain below 80%. This is observation-descriptor coverage, not a
classification-transfer result. See [the CAST report](../../reports/CAST_improvement.md).

Use [the v15 replay instructions](REPLAY.md) for full source/outer replay. The
canonical wrapper prints the source runner's help unless its reviewed arguments
are explicitly changed. Do not treat the exposed outer fold as a new untouched test.

# CAST-4A metadata preparation

The [protocol](PROTOCOL.md), [design](design.json) and
[numerical recipe](numerical_contract.json) freeze a future source-domain
coverage comparison. This entry point only prepares metadata; it has no fitting,
audio generation, feature extraction or scoring command.

From the repository, with the configured external roots:

```sh
.venv/bin/python -m abvid.cli validate configs/experiments/cast_grouped_coverage.json --check-files
.venv/bin/python -m abvid.cli prepare configs/experiments/cast_grouped_coverage.json
.venv/bin/python -m abvid.cli prepare configs/experiments/cast_grouped_coverage.json --execute
.venv/bin/python -m abvid.cast.grouped_design verify --config repo:configs/experiments/cast_grouped_coverage.json
```

`prepare` prints by default. Explicit metadata execution freezes external files
once; subsequent calls verify byte-identical replay and refuse changed or partial
outputs. `abvid replay` rejects this configuration. The preserved historical
replay configurations and locks remain unchanged.

Output: `artifact:cast/grouped_coverage/cast4a_design_20261007_v1`.
It contains exact source rows, nine role manifests, planned seed records,
candidate-only fit inventory, pinned input copies, relevant code snapshot and a
hash lock with environment/commit/dirty status. Audio hashes are copied from the
admission metadata; this pass does not reread audio to verify them.

See [the preparation report](../../../reports/CAST_grouped_coverage_design.md)
for checked counts, limitations and the next execution prerequisites. The planned
27,000 observations have **not** been generated. No coverage or transfer result
is produced by this command.

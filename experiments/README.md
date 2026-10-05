# Research experiment registry

Work in dependency order: admission → R0 published reproduction → H1 magnitude
of the domain gap → H2 controlled simulation mechanisms. CAST is the parallel
source-coverage branch that must pass its own checks before a transfer experiment.

| Directory | Role | Shared implementation |
|---|---|---|
| [admission](admission/README.md) | Decode, metadata and provenance audit | `abvid.data`, `abvid.provenance` |
| [r0_reproduction](r0_reproduction/README.md) | Original published splits/checkpoints | Preserved author/replay snapshots |
| [h1_baselines](h1_baselines/README.md) | Real/cross-domain/synthetic baselines | `abvid.representations`, `abvid.learning`, `abvid.evaluation` |
| [h2_simulation](h2_simulation/README.md) | Source/path, ground/air and latency controls | `abvid.simulation`, `abvid.evaluation` |
| [cast_coverage](cast_coverage/README.md) | Effective-observation fitting and population coverage | `abvid.cast` |

The [canonical configuration format](CONFIGURATION.md) wraps an exact historical
contract without changing scientific settings. Current configurations deliberately
expose regression/help commands; full frozen workflows retain their original code
and environments in the external archive. This restructuring introduces no new
training, target evaluation, simulator mechanism or transfer claim.

The protocol/matrix documents at this level retain their dated scientific history.
The current [roadmap](../ABVID_ROADMAP.md) and result reports determine completed
versus proposed work; an old plan is not authorization to repeat a locked test.

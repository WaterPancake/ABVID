# ABVID

Research on which acoustic simulation components determine synthetic-to-real
vehicle classification performance with pretrained audio representations.

The active work uses civilian car/truck recordings, frozen BEATs features,
published-result reproduction, controlled propagation studies, and CAST source
coverage. Source-model dominance is a hypothesis; the completed controls do not
establish it. See [the roadmap](ABVID_ROADMAP.md) and [repository guide](docs/STRUCTURE.md).
The [refactor report](reports/REFACTOR.md) records preservation and validation checks.

The earlier military tracked/wheeled benchmark and project-local OpenClaw
integration have been retired from this checkout. Historical results are preserved
in Git history and the local archive; their claims have not been reinterpreted.

## Layout

```text
src/abvid/          data, simulation, cast, representations, learning, evaluation, provenance
experiments/        admission, r0_reproduction, h1_baselines, h2_simulation, cast_coverage
configs/            canonical experiment configurations and numerical references
tests/              numerical, leakage, path and configuration regression checks
reports/            literature synthesis and research results
literature/         literature matrix and paper index
vendor/             upstream simulation license, revision and repair history
environments/       separate historical numerical environment pins
```

Raw audio, generated corpora, extracted features, checkpoints and full runs live
outside the checkout. Copy [configs/local.example.json](configs/local.example.json)
to `configs/local.json` and set your storage roots, or use `ABVID_DATA_ROOT`,
`ABVID_ARTIFACT_ROOT` and `ABVID_ARCHIVE_ROOT`.

## Start here

Python 3.11 and uv are required. The CAST and baseline environments deliberately
use different numerical pins; do not combine them when replaying results.

```bash
uv sync --extra cast --group dev
uv run --extra cast abvid list
uv run --extra cast abvid validate configs/experiments/cast_coverage.json
uv run --extra cast python -m pytest tests/cast tests/test_protocol.py
```

For the baseline/simulation profile, use a separate environment:

```bash
UV_PROJECT_ENVIRONMENT=.venv-baseline uv sync --extra baseline --group dev
UV_PROJECT_ENVIRONMENT=.venv-baseline uv run --extra baseline python -m pytest tests/test_audit_integrity.py tests/test_audit_provenance.py tests/test_h1.py tests/test_beats.py tests/test_simulation.py tests/test_protocol.py
```

Commands above run regression checks, not research training. See
[experiment registry](experiments/README.md) for preserved study runners,
[dataset protocol](experiments/DATASET_PROTOCOL.md) for admission/splits, and
[environment notes](environments/README.md) for exact historical replay.

CAST is an effective recorded-observation model. Coverage of source-domain
descriptors does not establish clean engine-source recovery or classification
transfer. MELAUDIS results already inspected remain exposed development evidence.

Original ABVID code is MIT licensed. The separately identified pyroadacoustics
backend is GPL-3.0; its upstream license and provenance are retained under
[vendor/pyroadacoustics](vendor/pyroadacoustics/README.md).

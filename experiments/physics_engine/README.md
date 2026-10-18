# Simple physics engine prototype

The additive `abvid.physics` package drives an engine pulse source from RPM and
torque, demonstrates free-field propagation, and compares reconstructions with
unchanged CAST v0 and smooth8. Read [the protocol](PROTOCOL.md) for assumptions,
data access, fixed budgets and scientific limits.

Use the existing CAST environment; no lock or environment changes are needed.
From the ABVID checkout, run this command with a **new** artifact directory:

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/abvid-physics-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m abvid.physics run --config repo:configs/physics/engine_v0.json --output artifact:physics_engine/physics_v0_reproduction --download
```

The first execution fetches the pinned inventory, release metadata and 16
selected WAVs (about 75 MB). It never downloads the 24 GB full audio release.
Subsequent runs verify cached data before use. Remove `--download` for a strictly
offline run. Storage roots come from `configs/local.json` or the documented
`ABVID_DATA_ROOT`/`ABVID_ARTIFACT_ROOT` environment variables via `abvid.paths`.
Existing runs and corrupt cached inputs are never silently replaced.

Open the output directory's `index.html` for all audio players, plots and result
links. `summary.json` contains measured runtime and aggregate losses;
`audit.json` contains admission findings; per-clip `result.json` files retain
every parameter candidate, CAST start, loss history and scientific failure.

Verify replay, including three complete fits on the first clip:

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/abvid-physics-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m abvid.physics replay --output artifact:physics_engine/physics_v0_reproduction --refit-first
```

Focused tests:

```sh
.venv/bin/python -B -m pytest tests/test_physics_engine.py tests/cast/test_numerical.py tests/cast/test_engine.py tests/cast/test_roles.py tests/cast/test_smooth_fit.py -m 'not archive'
```

The new numerical config is validated by `abvid.physics.config.load`; it does
not change the frozen-replay registry or add another mode to `abvid.cli`.
Original CAST results remain in their existing archive paths.

The test suite's collection-count canary deliberately needs a human baseline
update after reviewing new tests. Do not run the baseline-refresh script as an
agent. No merge or change to the existing frozen hash contract is performed.

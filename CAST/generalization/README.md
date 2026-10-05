# CAST generalization diagnostic

This separate extension implements the minimal CAST-3 sampler and first CAST-4
held-group acoustic coverage check, authorized by the follow-up request “test
generalization.” It leaves the completed CAST-0–2 prototype and its artifacts
unchanged. Read [PROTOCOL.md](PROTOCOL.md) for scope, frozen metrics, sampling
controls, ancestry requirements and limitations.

From the ABVID root:

```sh
bash CAST/generalization/run.sh workflow --run-id cast_generalization_reproduction_01
```

Choose a new run ID. The command freezes metadata, source code and all sampling
choices, runs tests, generates all 1,500 clips from the training bank, then opens
the 570 allowed fold-0 IDMT observations and evaluates coverage. It verifies
every generated waveform and donor choice by exact replay, recomputes every
held observation/descriptor and score, checks the unchanged pilot, and writes
`CAST/generalization/runs/<run-id>/CAST_generalization.md`, plots, JSON and a
local `index.html` audio gallery. Any technical failure is retained and fails
the run. Scientific adequacy failures are reported without changing thresholds.

Requirements: the completed parent `CAST/runs/cast_pilot_v0_20261004`, unchanged
CAST v0 source/config, historical source/H1 manifests and locks, and existing
local IDMT originals must match their frozen hashes. There are no downloads.
The launcher uses the existing root `.venv/bin/python` read-only, with the
versions in [../requirements-lock.txt](../requirements-lock.txt). `CAST_PYTHON`
can select an isolated environment installed using the parent README; Python
and numerical package versions must match. CPU only, one Torch thread. No
existing environment or dataset is modified.

Verification and tests:

```sh
bash CAST/generalization/run.sh verify --run-id cast_generalization_v0_20261004_r1
PYTHONPATH=CAST/generalization/src:CAST/src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest CAST/generalization/tests CAST/tests tests/test_source_simulation.py -q -p no:cacheprovider
```

All new implementation lives here to preserve the original pilot's exact source
hash checks. The new held-data allowlist is explicit and separate; the old
pilot guard still rejects fold 0. All ancestors and calibration failures remain
visible. The only new audio is IDMT group `connected_4001f06f57cfeee7`, previously
exposed H1 development data, at a training-seen site. No MELAUDIS, mixed feature
caches, supplied synthetic bank, military/reserved data, held latent fitting,
classification, other fold or CAST-5+ work is performed.

Joint sampling replays at most 25 fitted vectors per class with fresh random
phase/noise. Prototype and independently sampled scalar marginals are controls.
Every synthesis record contains all real parents/groups and per-coordinate
donors. Neither new seeds nor new scalar combinations imply new independent
vehicles. Peak-safe gallery clips have explicit playback gain; metrics always
use unit-RMS observations. All derivatives remain local evaluation artifacts.

The initial run `cast_generalization_v0_20261004` is preserved after an artifact
verification failure: macOS Finder updated `.DS_Store`. The repaired run ignores
only that OS presentation file when inventorying artifacts; every experiment
file remains hashed. Sampling, scoring, frozen criteria and configuration are
unchanged. See [DECISIONS.md](DECISIONS.md). The held group had already been
decoded in the initial run; the repair is a deterministic engineering replay
of the preregistered experiment, not fresh held-out confirmation.

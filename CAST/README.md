# CAST observation reconstruction prototype

Implements **CAST-0 through CAST-2 only** from [CAST_PLAN.md](../CAST_PLAN.md).
All implementation, configuration, tests, engineering records and run outputs live
in this directory. The additive final report is [reports/CAST_pilot.md](../reports/CAST_pilot.md).
No ABVID dataset, environment, split, checkpoint or historical output is modified.

From the ABVID repository root:

```sh
bash CAST/run.sh workflow --run-id cast_pilot_reproduction_01
```

This command freezes the exact permitted IDMT selection, runs numerical and
provenance tests, verifies synthetic fitting, fits all 50 permitted real clips,
verifies artifacts by deterministic replay, and writes a report and local audio
gallery. Choose a **new run ID** each time. It fails closed before real decoding
if tests or synthetic acceptance fail. It never downloads data. Local raw IDMT
files and the two original H1/source-diagnostic locks must be present and match
the SHA256 values in [the complete config](configs/pilot_v0.json).

The default executable is the existing `../.venv/bin/python`, used read-only.
The historical `experiments/reproduction/.venv` is not used or synchronized.
All imported dependencies and their transitive dependencies are pinned in
[requirements-lock.txt](requirements-lock.txt). To install an isolated environment
instead (installation requires package access):

```sh
uv venv --python 3.11.13 CAST/.venv
UV_CACHE_DIR="$PWD/CAST/.cache/uv" uv pip sync --python CAST/.venv/bin/python CAST/requirements-lock.txt
CAST_PYTHON="$PWD/CAST/.venv/bin/python" bash CAST/run.sh workflow --run-id cast_pilot_isolated_01
```

Exact waveform replay is asserted under the pinned numerical environment. Other
platforms or library versions need their own documented determinism check; the
runner rejects dependency-version drift. Computation is CPU only, one Torch
thread, with no encoder, GPU, TensorFlow, classifier or external service.

Individual stages can be executed sequentially on the same new run ID:

```sh
bash CAST/run.sh inventory --run-id cast_staged_01
bash CAST/run.sh synthetic --run-id cast_staged_01
bash CAST/run.sh pilot --run-id cast_staged_01
bash CAST/run.sh verify --run-id cast_staged_01
bash CAST/run.sh report --run-id cast_staged_01
```

`inventory` reads metadata and hashes only the 50 allowed originals; it does not
decode audio. The other stages preserve their outputs and refuse overwrites.
Implementation changes require a new run. A terminated pilot retains completed
clip artifacts; do not splice different code/config versions into one run.

To run just the fast checks:

```sh
PYTHONPATH=CAST/src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest CAST/tests -q -p no:cacheprovider
```

Open `CAST/runs/<run-id>/index.html` locally to inspect all original/reconstructed
pairs and their plots. The float `original.wav` is the unnormalized fixed-profile
observation; raw stereo originals remain at their locked source paths. Listen to
`*_playback.wav` for the documented common-gain pair. The gallery is a local
artifact, not a published website. IDMT derivative audio is retained for local
evaluation; the recorded provider licence remains CC BY-NC-ND 4.0.

Per-run artifacts include immutable config/lock/selected manifest, source copy,
commit plus dirty snapshot, dependency versions, test evidence, synthetic
acceptance, real fits, failure records, timing/RSS, aggregated results and replay
verification. Per-clip outputs include parameters, ancestry, streams, all start
scores, full optimization history, losses by FFT size, components and float audio.

The model fits **effective observation controls**. Harmonic spacing is not RPM;
noise is not separated tire/background sound; envelope is not throttle or range.
Reconstruction success does not establish independent acoustic coverage or
classification transfer. No CAST-3+ sampler, held-out evaluation, classifier,
target-data access or military milestone advancement is implemented.

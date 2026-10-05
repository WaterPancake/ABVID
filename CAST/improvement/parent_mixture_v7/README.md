# Per-parent mixture calibration

From the ABVID root, using the pinned existing environment and audited smooth8
380-parent bank:

```sh
bash CAST/improvement/parent_mixture_v7/run.sh source --eval-id NEW_SOURCE_ID
bash CAST/improvement/parent_mixture_v7/run.sh audit-source --eval-id NEW_SOURCE_ID
```

Use a fresh ID. Both commands preserve existing outputs. The first runs the
numerical/provenance suite before freezing code, calibration inputs, derived
parameters, metric configuration and original bank. It then executes both banks
with all three matched controls under five leave-one-source-group-out folds.
The second recomputes all 380 local calibrations and replays the complete
15,000-sample schedule, all scores, ancestry and the 150 identity references.
Successful command execution does not mean the scientific criteria pass;
inspect `scientific_status.json` and `summary.json`. No outer command is invoked.

Plots after a successful audit:

```sh
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=CAST/improvement/.cache/matplotlib \
XDG_CACHE_HOME=CAST/improvement/.cache \
PYTHONPATH=CAST/improvement/src:CAST/generalization/src:CAST/src \
.venv/bin/python -m cast_improvement.parent_mixture_figures \
  --eval-id NEW_SOURCE_ID --output-id NEW_FIGURE_ID
```

See [PROTOCOL.md](PROTOCOL.md). The derived bank's checking components are
calibration data, not independent validation. The original fit/check artifacts
remain unchanged. Raw dataset and outer audio are not opened; saved training
waveforms and their components are read. These source results are exploratory
development after repeated method design on the same five source groups.

Calibration plot and systematic audio review (first ID in every class/group):

```sh
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=CAST/improvement/.cache/matplotlib \
XDG_CACHE_HOME=CAST/improvement/.cache \
PYTHONPATH=CAST/improvement/src:CAST/generalization/src:CAST/src \
.venv/bin/python -m cast_improvement.parent_mixture_diagnostics \
  --eval-id NEW_SOURCE_ID --output-id NEW_AUDIO_ID
```

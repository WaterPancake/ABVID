# Direct envelope calibration

Use a fresh ID and the existing pinned environment, audited source fit bank,
and completed v7 evaluation. From the ABVID root, the complete new workflow is:

```sh
bash CAST/improvement/envelope_v8/reproduce.sh NEW_SOURCE_ID
```

This runs tests, generation, exact replay, plots and audio review, then returns
code 3 if the scientific criteria fail (artifacts remain available). Code 0
requires passing source criteria; it still does not execute outer evaluation.
The current experiment ran these same stages individually. Stage commands:

```sh
bash CAST/improvement/envelope_v8/run.sh source --eval-id NEW_SOURCE_ID
bash CAST/improvement/envelope_v8/run.sh audit-source --eval-id NEW_SOURCE_ID
```

The first runs the numerical/provenance suite and freezes both derived banks,
code and source descriptors before five omitted-group folds. The second
recomputes each local calibration, all 15,000 waveforms, 300 scores, and the
150 unchanged v7 mixture-only references. All three arms use the same bank
within each candidate. Check `scientific_status.json`; technical success is
separate from the coverage/margin criteria. Existing results are never replaced.

After the exact replay audit:

```sh
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=CAST/improvement/.cache/matplotlib \
XDG_CACHE_HOME=CAST/improvement/.cache \
PYTHONPATH=CAST/improvement/src:CAST/generalization/src:CAST/src \
.venv/bin/python -m cast_improvement.envelope_figures \
  --eval-id NEW_SOURCE_ID --output-id NEW_FIGURE_ID
```

The fixed method and limitations are in [PROTOCOL.md](PROTOCOL.md). Calibration
uses already observed source parents; source selection is exploratory. No raw
dataset or outer observations are accessed, and no classifier is trained.

The same-parent residual plot and ten systematic audio examples use the same
environment prefix with `-m cast_improvement.envelope_diagnostics --eval-id
NEW_SOURCE_ID --output-id NEW_AUDIO_ID`. They are labelled calibration
reconstruction, not independent validation. All 380 parents remain in the
residual records, including boundary and solver-failure flags.

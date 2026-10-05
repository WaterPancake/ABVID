# Temporal-resolution experiment

This separately versioned experiment uses the audited v9 bank and compares
v8, v9 and the new 41-control temporal calibration. Use the pinned existing
environment and fresh IDs. The complete source workflow is
`bash CAST/improvement/temporal_v10/reproduce.sh NEW_SOURCE_ID`; it creates
source scores, full replay, plots and ten systematic audio examples, returning
code 3 if the scientific criteria fail. Existing artifacts are preserved.
From the ABVID root, individual stages are:

```sh
bash CAST/improvement/temporal_v10/run.sh source --eval-id NEW_SOURCE_ID
bash CAST/improvement/temporal_v10/run.sh audit-source --eval-id NEW_SOURCE_ID
```

The first runs all numerical/provenance tests, freezes the three banks, source
descriptors, renderer schemas and code, then generates 22,500 samples across
the same omitted source groups. The second recomputes all local temporal
calibrations, complete sampling/ancestry/descriptor records and 450 scores.
Both old banks must reproduce all 300 audited v9 score records exactly.
Inspect `scientific_status.json`: technical completion does not mean the
coverage/margin criteria pass. No outer inputs are opened by these commands.

After the replay audit, produce comparison plots:

```sh
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=CAST/improvement/.cache/matplotlib \
XDG_CACHE_HOME=CAST/improvement/.cache \
PYTHONPATH=CAST/improvement/src:CAST/generalization/src:CAST/src \
.venv/bin/python -m cast_improvement.temporal_figures \
  --eval-id NEW_SOURCE_ID --output-id NEW_FIGURE_ID
```

See [PROTOCOL.md](PROTOCOL.md) for the fixed calibration objective, ancestry,
sampling schema, boundaries and unchanged scientific criteria. The saved
source observations informed calibration; these are exploratory development
results. Original fits, previous revisions and dataset/split roles are preserved.

The denser envelope changes what independent coordinate sampling destroys.
Its matched margin measures temporal dependence under this representation;
it is not evidence of universal prior superiority or downstream classification.
The outer runner remains conditional and refuses a failed or unaudited source
candidate before any outer access. It has run once for this version: source criteria passed, but the verified
outer result fails coverage (77.212% car / 75.571% truck). Both margins and
family spreads pass. See [the report](../../../reports/CAST_improvement.md).

The complete conditional workflow, starting with the unchanged audited v9
bank and all frozen parent artifacts, is:

```sh
bash CAST/improvement/temporal_v10/reproduce_outer.sh NEW_SOURCE_ID NEW_OUTER_ID
```

This first runs the complete source workflow. Only a verified source pass
permits outer generation, scoring, full replay and the audio/plot review. A
scientific failure returns code 3 with all evidence retained. The current run
executed the same individual stages; shell syntax and invalid-ID rejection
were checked. No new dependency installation is needed.

# Fixed capacity revision

The sixteen-point noise spectrum and nine-knot envelope retain the original
preprocessing, objective, optimizer budget, data selection and acceptance
thresholds. The numerical method is frozen before real fitting. Every preceding
CAST result remains intact. See [PROTOCOL.md](PROTOCOL.md).

From the ABVID root, with the pinned environment and existing local parent
artifacts available, choose a fresh run ID:

```sh
bash CAST/improvement/capacity_v3/reproduce.sh NEW_RUN_ID 4
```

This single command runs the synthetic gate, fixed pilot, exact audits, paired
and source comparisons, figures and audio review. It preserves scientific
failures and returns code 3 if the pilot source candidate fails. It does not
automatically expand or access outer observations. The individual stages are:

```sh
bash CAST/improvement/capacity_v3/run.sh prepare --run-id NEW_RUN_ID
bash CAST/improvement/capacity_v3/run.sh synthetic --run-id NEW_RUN_ID --workers 4
bash CAST/improvement/capacity_v3/run.sh fit --run-id NEW_RUN_ID --scope pilot --workers 4
bash CAST/improvement/capacity_v3/run.sh audit-fits --run-id NEW_RUN_ID --scope pilot
```

Run each stage only after the preceding command succeeds. The synthetic stage
runs the numerical/provenance suite and all seven original fixture types with
the previous model's signals lifted to the new representation. It preserves
all tolerances. Failed acceptance prevents real fitting. The pilot is exactly
the original 50 parents; no new source IDs are admitted. Completed run stages
cannot be overwritten. Expanded fitting requires source-only pilot evidence.

All per-fit originals, reconstructions, baselines, component waveforms,
parameters, alternative starts, losses, diagnostics, flags and runtime are
retained under `runs/NEW_RUN_ID`. The independent audit replays every saved
waveform and every fitted/initial checking start from verified raw inputs.

After the pilot audit succeeds, reproduce the fixed source-only comparison:

```sh
bash CAST/improvement/capacity_v3/run.sh source --run-id NEW_RUN_ID --scope pilot --eval-id NEW_SOURCE_ID
bash CAST/improvement/capacity_v3/run.sh audit-source --eval-id NEW_SOURCE_ID
bash CAST/improvement/capacity_v3/run.sh diagnostics paired --id NEW_RUN_ID --scope pilot --output-id NEW_PAIRED_FIGURES
bash CAST/improvement/capacity_v3/run.sh diagnostics fits --id NEW_RUN_ID --scope pilot --output-id NEW_RESIDUAL_FIGURES
bash CAST/improvement/capacity_v3/run.sh diagnostics source --id NEW_SOURCE_ID --output-id NEW_SOURCE_FIGURES
bash CAST/improvement/capacity_v3/run.sh diagnostics gallery --id NEW_RUN_ID --scope pilot --output-id NEW_AUDIO_REVIEW
```

The comparison uses [SOURCE_PROTOCOL.md](SOURCE_PROTOCOL.md). It has two fixed
renderers and no prior search. The matched smooth8 reference is the existing
audited bank `CAST/improvement/runs/cast_smooth8_v1_20261004_r1`. Every bank is
filtered by the validation group before sampling; metrics and scales are
unchanged. There is no outer-data action in this source runner. Inspect both
absolute distance and margin against controls; a ranked candidate can fail.

For an already running pilot, queue the audits and analyses without restarting
any fitting:

```sh
.venv/bin/python CAST/improvement/capacity_v3/continue_pilot.py --run-id EXISTING_RUN_ID --eval-id NEW_SOURCE_ID
```

This records each command, output log, return code and timing in the run's
continuation receipt. It waits up to thirty minutes for the existing pilot.
Timeout does not stop the fitter. It never fits or opens outer observations.
Do not launch the same analysis manually while its continuation is active.

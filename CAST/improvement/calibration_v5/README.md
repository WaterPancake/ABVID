# Training-only band-bias correction

This fixed eight-case comparison uses paired source reconstructions to estimate
a bounded noise-spectrum correction. All calibration parents come from the
training portion of each source fold. No new fits or outer observations are
used. The same calibrated bank feeds all three matched arms. See
[PROTOCOL.md](PROTOCOL.md) for the preregistered calculation and unchanged gates.

From the ABVID root with the existing audited artifacts and pinned environment:

```sh
bash CAST/improvement/calibration_v5/run.sh source --run-id cast_smooth8_v1_20261004_r1 --eval-id NEW_CALIBRATION_SOURCE_ID
bash CAST/improvement/calibration_v5/run.sh audit --eval-id NEW_CALIBRATION_SOURCE_ID
```

The first command runs the full test gate before freezing and generating.
The second replays all samples and calibration ancestry and independently
recomputes every ratio, descriptor scale and score. It also requires the 600
zero-correction score records to reproduce the previously audited prior-v2
reference exactly. Audit success is technical; scientific acceptance is stored
separately in the summary and failure record. Preserve failed candidates.

Only after the selected source candidate passes every criterion and its exact
audit, the separate guarded outer runner is available:

```sh
bash CAST/improvement/calibration_v5/outer.sh run --source-eval-id NEW_CALIBRATION_SOURCE_ID --eval-id NEW_OUTER_ID
bash CAST/improvement/calibration_v5/outer.sh verify --eval-id NEW_OUTER_ID
```

It recalculates the selected correction from the full training bank, retains
every calibration ancestor, and writes all 1,500 generated audio samples before
reading the previously exposed outer descriptor cache. The original outer
scales and 570 IDs are unchanged. Verification replays every generated sample
and every raw outer descriptor. The two access-guard tests pass; no outer
execution is implied by the presence of this runner.

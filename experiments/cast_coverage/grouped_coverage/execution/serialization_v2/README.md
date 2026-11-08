# CAST-4A deterministic serialization execution revision

Prepared 2026-10-18. This separately pinned revision implements correction
steps 1–2 in [the v1 run report](../../../../../reports/CAST4A_run_20261011.md#proposed-correction-requires-human-review-not-applied).
Review this revision and its test additions before authorizing correction step 3.
Implementation validation uses synthetic fixtures only; the scientific revision
has not been executed.

[config.json](config.json) pins the original execution configuration, all its
implementation/test references, the two additive modules and the new regression
test file. Scientific input pins, environment, roles, seeds, budgets, numerical
settings and acceptance criteria retain their v1 values. The canonical
`configs/experiments/` registry is unchanged.

| Item | Revision value |
|---|---|
| Configuration | `repo:experiments/cast_coverage/grouped_coverage/execution/serialization_v2/config.json` |
| Entrypoint | `abvid.cast.cast4a_serialization_runner` |
| Output | `artifact:cast/grouped_coverage/cast4a_run_20261018_v2_npy` |
| Generated waveform | `waveform.npy`, float32, 32,000 mono samples at 16,000 Hz |
| Replay | Exact immutable payload-byte and metadata comparison |
| Preserved v1 output | `artifact:cast/grouped_coverage/cast4a_run_20261008_v1` |

The runner is an additive snapshot of v1 with deterministic NPY serialization,
explicit waveform-format/sample-rate metadata and the new configuration loader.
The loader preserves all metadata-only validation and additionally checks the
original configuration hash and unchanged scientific fields. Revision modules
and tests deliberately avoid the original `grouped_*.py` / `test_grouped_*.py`
inventory names, so the original execution pins still validate unchanged.

NPY follows CAST-5A's contiguous float32 array pattern with `allow_pickle=False`
on both writing and reading. It contains no wall-clock PEAK chunk. The runner
checks shape, dtype, finiteness and an exact float32-byte round trip, retains
the waveform hash and JSON payloads, and uses the original complete-generation
gate before any held-role access. The fixed sample rate is recorded in sample
metadata and the revision configuration because NPY has no audio sample-rate
header.

After separate review, the metadata-only entrypoint is:

```sh
/Users/magi/workspace/abvid-artifacts/environments/cast-readiness-20261007/bin/python \
  -m abvid.cast.cast4a_serialization_runner plan
```

`run` and `verify` also remain metadata-only without `--execute`. No production
plan, parent reuse, waveform regeneration, held-descriptor computation or scoring
was performed to prepare this revision. Correction step 3, including reuse of the
1,864 completed v1 parent bundles under the existing same-parent rules, remains
a separately reviewed operation. This revision adds no automatic v1-output
migration or retry in place.

See [the serialization revision report](../../../../../reports/CAST4A_serialization_revision.md)
for validation details. The human-only harness baseline remains unchanged;
test-count drift must be reviewed before merge.

# CAST-4A serialization execution revision

Prepared 2026-10-18 on `cast4a/serialization-revision`. Implements only correction
steps 1–2 from [CAST4A_run_20261011.md](CAST4A_run_20261011.md#proposed-correction-requires-human-review-not-applied).
The original `cast4a_run_20261008_v1` remains a preserved technical incompletion.
This revision has synthetic implementation validation only; no new CAST-4A
coverage result is established.

## Serialization choice

Generated samples use `waveform.npy`: contiguous float32 arrays written with
`numpy.save(..., allow_pickle=False)` and checked with
`numpy.load(..., allow_pickle=False)`. This follows the existing CAST-5A bank
pattern in `src/abvid/cast/utility_arms.py`. NPY avoids libsndfile's wall-clock
PEAK timestamp without private `sf_command` controls or a weaker replay rule.

The runner retains the finite float32 `(32000,)` input check, verifies the exact
dtype, shape and sample bytes after serialization, and saves the original
float32 waveform SHA256. Sample metadata explicitly records `waveform_format:
npy` and `waveform_sample_rate_hz: 16000`; NPY has no audio sample-rate header.
Records, descriptors, planned IDs, rendering and RNG streams remain unchanged.
The checkpoint still compares all serialized payload bytes and metadata exactly.

## Separate execution pins

| Item | New revision |
|---|---|
| ID | `cast4a_grouped_execution_v2_npy` |
| Configuration | `repo:experiments/cast_coverage/grouped_coverage/execution/serialization_v2/config.json` |
| Instructions | [serialization_v2/README.md](../experiments/cast_coverage/grouped_coverage/execution/serialization_v2/README.md) |
| Entrypoint | `python -m abvid.cast.cast4a_serialization_runner` |
| Output | `artifact:cast/grouped_coverage/cast4a_run_20261018_v2_npy` |
| Original configuration SHA256 | `3b415989bc2cbd2fdb1bba9398dcf445f6db55223e2c7aabbd6d5855d2a7380f` |
| Original output, preserved | `artifact:cast/grouped_coverage/cast4a_run_20261008_v1` |

`cast4a_serialization_runner.py` and `cast4a_serialization_config.py` are additive
snapshots of the pinned v1 runner/loader, allowing a small direct source comparison
without patching module globals or changing historical files. Runner differences
are confined to the module description, sample serialization/format metadata and
CLI loader import. All stage orchestration, complete-schedule checks, held-role
guards, exact replay, failure retention and immutable storage behavior remain.

The new loader pins the original configuration and checks that every scientific
field/input/environment remains identical. Its implementation/test inventories
contain every original reference with its original hash, plus the two new
modules and regression file. It checks those bytes before constructing a context.
The revision names avoid the v1 `grouped_*.py` and `test_grouped_*.py` discovery
patterns, preserving the original pin inventory. Both configurations keep the
same canonical design and numerical contract. The new plan records the prior
execution pin and waveform format. CLI commands remain metadata-only unless
`--execute` is supplied.

## Synthetic regression checks

Added `tests/cast/test_cast4a_serialization_revision.py` (13 cases):

- `test_npy_generation_gate_replays_after_second_boundary`: writes one seeded
  synthetic sample through the actual immutable generation/replay gate, sleeps
  1.1 seconds, then replays and compares every saved byte and modification time.
  It verifies float32 sample bytes, metadata and unchanged JSON payloads against
  v1. Only the schedule is reduced to one fixture entry; no real study runs.
- `test_plain_float_wav_replay_changes_only_peak_timestamp`: invokes the preserved
  plain `soundfile.write` v1 serializer, writes an immutable sample bundle, sleeps
  1.1 seconds, and confirms the PEAK timestamp changes while decoded sample bytes,
  descriptors, records and other container bytes match. The old byte-replay gate
  correctly rejects the second container, preserving the first bundle.
- `test_revision_contract_pins_new_runner_and_preserves_v1_inventory`: checks
  all repository implementation/test hashes, the original configuration pin,
  distinct outputs and unchanged v1 inventory using synthetic external metadata.
- `test_revision_contract_rejects_protocol_and_pin_changes` (7 cases): rejects
  the old output, changed numerical contract, prior configuration pin, waveform
  format, implementation hash, test hash or numerical environment.
- `test_revision_cli_stays_metadata_only_without_execute` (3 cases): `plan`,
  `run` and `verify` cannot reach execution without `--execute`.

## Validation

The 13 new regression cases passed in the pinned local CAST environment:
Python 3.11.13, NumPy 2.4.6, SciPy 1.17.1, soundfile 0.14.0, libsndfile 1.2.2,
Torch 2.13.0 and pytest 9.1.1. No network or archive/real-data access was used
by the added checks.

The requested graded selection is:

```sh
uv run --extra cast pytest -m 'not archive' \
  --ignore=tests/test_h1.py --ignore=tests/test_simulation.py \
  --ignore=tests/test_benchmark_prototype.py -q
```

For sandbox-compatible offline validation, this command used
`UV_OFFLINE=1`, `UV_NO_SYNC=1`,
`UV_PROJECT_ENVIRONMENT=/Users/magi/workspace/abvid-artifacts/environments/cast-readiness-20261007`
and `UV_CACHE_DIR=/private/tmp/abvid-cast4a-serialization-uv-cache`.
The existing environment already matches the execution pins. Syncing is disabled
to preserve it and `uv.lock`; the default uv cache is outside writable storage.

Result: **734 passed, 1 failed, 3 skipped, 9 deselected; exit status 1**.
There are 736 selected test cases: 734 passes, one runtime skip and one failure.
The other two skips occur during collection because the CAST environment lacks
the baseline-only `sklearn` and `numba` dependencies. A collection-only check
confirmed `736/745 tests collected (9 deselected)`.

The sole failure is
`tests/test_harness_integrity.py::test_collected_test_count_matches_baseline`:
**736 collected versus 723 expected**, exactly the 13 added regression cases.
The baseline schema and frozen-file hash checks pass. The baseline was not
refreshed. No other selected check failed.

## Scope and review handoff

Only new source, execution configuration, instructions, tests and this report
are added. Existing reports, v1 code/configuration, output references and hashes,
the protocol registry, lockfile, harness baseline, canary, workflows and roadmaps
retain their original bytes. No v1 output was opened or modified.

Correction step 3 has not been performed: no parent bundles were reused or fit,
no 27,000-sample regeneration occurred, and no held descriptors or scores were
computed. No production plan or new scientific output directory was created.
Review this revision and the test diff before authorizing that operation. The
new runner does not automatically migrate v1 outputs; any later reuse of the
1,864 completed parents must follow the existing same-parent audit rules.
The test-count canary is intentionally left visible. The human-only harness
baseline refresh and line-by-line test review remain human actions before merge.

## Commit handoff

The checkout is on `cast4a/serialization-revision`. Staging the six additions
failed because the sandbox cannot write the worktree's Git metadata outside the
writable checkout:

```text
fatal: Unable to create '/Users/magi/workspace/ABVID/.git/worktrees/abvid-cast4a/index.lock': Operation not permitted
```

No commit was created. All changes remain as six new files, with no tracked-file
modifications. Suggested commit message:
`Add separately pinned CAST-4A deterministic NPY serialization revision`.

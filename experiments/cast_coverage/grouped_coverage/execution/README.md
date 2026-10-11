# CAST-4A grouped execution adapter

This additive layer executes the unchanged [frozen design](../PROTOCOL.md).
The canonical registry, design, numerical contract, readiness audit and historical
artifacts keep their original hashes. `config.json` pins this implementation and
its regression tests as an additional execution contract. No scientific result
is established by building or testing this runner.

Use the isolated CAST environment; the shared baseline `.venv` is incompatible:

```sh
/Users/magi/workspace/abvid-artifacts/environments/cast-readiness-20261007/bin/python \
  -m abvid.cast.grouped_runner plan
```

`plan`, `run` and `verify` without `--execute` print validated metadata only.
They do not decode audio, load fit arrays, fit, generate samples or score held
recordings. Planning validates the canonical configuration, source-role freeze,
readiness evidence, implementation hashes and exact numerical environment.

The future scientific commands are:

```sh
# Executes the entire frozen source-only study and resumes unchanged checkpoints.
/Users/magi/workspace/abvid-artifacts/environments/cast-readiness-20261007/bin/python \
  -m abvid.cast.grouped_runner run --execute

# Replays existing outputs; never fills missing fits or generation work.
/Users/magi/workspace/abvid-artifacts/environments/cast-readiness-20261007/bin/python \
  -m abvid.cast.grouped_runner verify --execute
```

These commands were **not executed during implementation validation**. Transfer,
BEATs/classifier training, target-dataset access and method tuning are outside
this contract. The output location is fixed in `config.json`; large artifacts
remain under the external artifact root.

## Execution and recovery

1. Save the code/configuration, environment and initial commit/dirty-state
   snapshot. An OS lock prevents two executions from sharing the output root.
2. Re-audit the 380 historical same-parent parameter records. Reuse only admitted
   records in a current training role; later historical reconstructed audio is
   excluded. Fit the other 1,484 distinct selected recordings once each using
   the unchanged four-start smooth8 and v7–v11 calibration chain. Retain all
   alternatives, histories, flags and calibration ancestry.
3. Rebuild and freeze each fold's complete training bank, group/class statistics,
   prototypes, virtual children and descriptor scales. No population statistics
   are imported from the old study or another fold.
4. Generate the exact 3,000 declared examples per fold. Regenerate and compare
   every waveform, descriptor and ancestry record before recording the generation
   gate. Only this complete replay receipt permits held-recording access.
5. Compute every declared held descriptor and replay it from its guarded original
   audio. Score the six arms for both classes and all five seeds. Save all 540
   planned score slots, including explicit missing/failed slots in the summary.
6. Apply the original per-fold/class criteria and paired whole-fold bootstrap.
   A completed negative result is valid; missing or invalid work prevents a
   complete-analysis interval and cannot be hidden by dropping folds or seeds.

The generation gate protects each fold's held evaluation. Since training roles
overlap, a recording held in one fold may already have been processed as a
training parent for another fold. Same-parent computations are reusable; the
role checks exclude its fit and descriptors from the first fold's population.
This is not a claim that every held recording is globally unopened.

Each stage is a create-only bundle with an exact inventory, payload hashes and
metadata identity. A valid completed bundle is reused without modification.
Interrupted unpublished staging directories are retained. Repeating the same
command resumes missing work under the same inputs/configuration; it never
refits a completed parent or silently replaces a recorded technical failure.
An attempt log records starts, finishes, previous attempts, runtime and retained
staging directories. A hard interruption can leave a start without a finish;
the operating system releases the execution lock automatically.

A checksum/configuration mismatch fails closed. Do not edit existing manifests,
remove a failure to retry it, change seeds or replace examples to obtain a
passing result. Diagnose a terminal technical failure from its saved artifacts;
a corrected implementation requires an explicit, separately pinned execution
revision preserving the original attempt. Fixed criterion failures require no
implementation repair.

## Interpretation

This evaluates descriptor coverage on previously exposed IDMT development
groups, including a separate three-site sensitivity analysis. It does not
establish independent recording sessions, clean physical sources, identified
RPM, source-model dominance, or classification transfer. The full scientific
run and its replay remain necessary before reporting a CAST-4A outcome.

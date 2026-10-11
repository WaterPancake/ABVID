# CAST-4A grouped runner implementation

Updated 2026-10-08. The additive grouped execution runner connects the frozen
CAST-4A protocol to same-parent fitting, fold-local population construction,
generation, descriptor scoring and replay. **The scientific run remains unrun.**
No new real observation has been fitted, no 27,000-sample study population has
been generated, and no new grouped-coverage or classifier-transfer result is
reported here.

The [execution instructions](../experiments/cast_coverage/grouped_coverage/execution/README.md)
and [pinned execution configuration](../experiments/cast_coverage/grouped_coverage/execution/config.json)
provide the new entry point. The canonical design configuration and all existing
numerical, readiness and historical locks remain unchanged.

## What is implemented

| Component | Implemented behavior |
|---|---|
| Configuration and metadata plan | Validate the original canonical design, reconstructed roles, saved readiness evidence, numerical environment and the new implementation/test hashes. Planning opens metadata only. |
| Same-parent calibration | Exact four-start, 300-step smooth8 fitting followed by unchanged v7–v11 mixture, envelope, spectral and temporal calibration. Preserve histories, alternatives, ambiguity/projection flags and known/unknown source metadata. |
| Parent reuse and execution | Re-audit eligible historical parameters, reuse only within current training roles, and fit each missing selected parent once. Never reuse old population statistics or excluded reconstruction waveforms. Bound CPU fitting to four outer workers with one Torch thread each. |
| Fold populations | Require the entire literal 380-parent role; rebuild both width populations and class descriptor scales using training parents only. Freeze banks and statistics before generating. |
| Generation and held access | Require the exact 3,000-record schedule per fold. Replay every generated waveform, descriptor and ancestry record. Bind the gate to the saved population and sample bundle identities before opening held recordings. |
| Scoring | Replay all held descriptors from their original guarded audio. Retain each class/seed/arm score and every failure. Compute existing marginal coverage, scaled W1, spread criteria and matched-control gains. |
| Uncertainty and interpretation | Paired whole-group bootstrap with 10,000 draws and fixed PCG64 seed 314159, separately for six groups and three sites. Incomplete analyses receive no subset-only interval. Pooled means cannot rescue a failed fold/class. |
| Resume and verification | Create-only, hash-verified checkpoints; unchanged completed work is reused. Preserve failed attempts and terminal numerical failures. Verification recreates populations, generation, held descriptors and scores without fitting missing parents or filling absent outputs. |
| Provenance and resources | Save configuration, code/test bytes, initial commit/dirty state, environment, CPU execution settings, attempts, timing/memory receipts, failures and scientific bundle identities outside the checkout. |

The execution plan retains 4,413 admitted IDMT source recordings, nine
partitions, a union of 1,864 selected training parents, 380 eligible cached
parents and **1,484 missing parent fits**. It retains the six arms, five seeds,
50 generated observations/class/seed/arm, 27,000 generated observations and
540 score slots. It adds no new scientific choice, width search, data budget,
class rule, representation, held group or acceptance threshold.

The generation gate applies to current-fold held evaluation. A recording may
already be a training parent in another fold. Reusing its independent
same-parent computation does not admit it to a fold where it is held; the
complete role checks exclude its parameters and descriptors there. No global
claim of previously unopened recordings or independent confirmation is made.

## Verification

Validation uses synthetic signals, synthetic ancestry fixtures, temporary bundle
storage and metadata-only inspection of the frozen study. The full synthetic
parent pipeline is compared with the preserved historical v7–v11 functions in
an isolated process. Fixture tests exercise interrupted and failed work,
checkpoint tampering, complete generation before held access, role rejection,
matched controls, paired uncertainty and replay.

The metadata-only CLI plan passes with configuration SHA256
`3b415989bc2cbd2fdb1bba9398dcf445f6db55223e2c7aabbd6d5855d2a7380f`.
Eight grouped modules and eight grouped test files, including the prior design
module/tests, are pinned. Its output is retained at
`artifact:cast/grouped_coverage/cast4a_execution_plan_validation_20261008_v1/plan.json`.
The declared scientific output directory does not exist.

Combined regression results are retained at
`artifact:cast/grouped_coverage/cast4a_runner_validation_20261008_v1`.
**322 checks passed in 77.38 seconds**, with no failures, errors or skips:
155 newly added checks and 167 existing design, role, numerical and canonical
protocol checks. The new checks comprise 9 parent-calibration, 18 parent-executor,
24 population, 25 metric, 25 storage, 22 configuration and 32 runner checks.
The full-budget synthetic calibration and isolated historical v7–v11 parity
comparison are included. The full 3,000-entry orchestration fixture mocks
sample payloads/numerical work; separate component tests exercise actual
synthetic rendering and calibration. This does not replace a real study run.

The validation summary SHA256 is
`44ffa5dbb7ad631d066f3a4dfc861d906a8143d01dcde19545107a9b0aa0fb2a`;
its payload lock SHA256 is
`4751f28ee27d555eb9e3cf1ee37aeec12623dad1451a04813599b327b6587d6d`.

The repository's test-count canary requires a human baseline refresh after
legitimate suite changes. This work adds regression tests and leaves
`tests/harness_baseline.json`, the canary and the frozen-file hashes unchanged.
[AGENTS.md](../AGENTS.md) states: “Baseline updates are HUMAN-ONLY” and requires
line-by-line human review of test changes before merge. A resulting test-count
failure is reported separately from numerical/regression results; it is not
silently repaired by changing the baseline. The separate canary run has **two
passes and one failure**: the current CAST selection collects **641 tests**
against **467** in the baseline. The baseline-schema and frozen-file hash
checks pass. This is not an all-green CI claim; the shared checkout also
contains other concurrently added tests. Baseline maintenance for both
dependency environments remains a human action before merge.

## Remaining work and scientific limits

The next scientific operation is the fixed CAST-4A source-only execution:
fit the 1,484 missing parents, rebuild the nine populations, generate the frozen
schedule, replay all artifacts and score the declared held roles. End-to-end
real-data runtime and peak aggregate worker memory are not established by the
fixture checks. The runner records resource measurements when that execution
occurs. Runtime estimates from synthetic fixtures are not treated as real-data
measurements.

A completed negative coverage result remains a valid outcome. Numerical or
provenance failures are technical incompletion and cannot establish a physical
source-versus-propagation effect. All held groups remain previously exposed
IDMT development evidence; unknown original sessions and vehicle identities
remain unknown. Classification utility, target-dataset transfer, CAST-4B and
CAST-5 require their own frozen work and are unchanged by this implementation.

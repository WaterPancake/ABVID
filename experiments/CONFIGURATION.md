# Canonical experiment configuration, schema version 1

All experiment registry entries use JSON under `configs/experiments/` and are
validated by `abvid validate`. The wrapper is the common transport format; the
hashed scientific contract remains authoritative for classes/mapping, dataset
roles, preprocessing, seeds, classifier, metrics, intervals and falsification.
Those settings are not copied into a second conflicting source of truth.

Required fields (unknown fields are rejected):

| Field | Meaning |
|---|---|
| `schema_version` | Integer `1` |
| `id` | Stable lowercase identifier |
| `phase` | admission, reproduction, baseline, simulation, source_coverage |
| `description` | Short scope statement |
| `contract.path`, `contract.sha256` | Exact scientific specification and its hash |
| `execution.mode` | `frozen_replay` |
| `execution.environment` | `cast` or `reproduction`; separate frozen Python environments |
| `execution.script`, `script_sha256` | Exact original runner and its hash |
| `execution.module` | Null for a script, or the fixed historical CAST module entry point |
| `execution.arguments` | Literal argv array, never a shell string |
| `reports` | Root-qualified paths to reviewed reports |

Paths use `repo:`, `data:`, `artifact:` or `archive:`. Absolute paths and `..`
traversal are rejected. Local roots belong in ignored `configs/local.json` or
`ABVID_*_ROOT` variables, not scientific configs. `--check-files` verifies contract
and runner hashes. Original runners retain their additional code/manifest locks.

A changed path layout does not create a new scientific protocol. A changed split,
class definition, numerical setting or model requires a newly reviewed contract
and wrapper; never update old locks to make changed code pass verification.
`abvid replay` only prints a command unless `--execute` is supplied. Registry
regression/help defaults do not reproduce a training run.

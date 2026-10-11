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
| `execution.mode` | `frozen_replay` or the restricted `metadata_only` preparation mode |
| `execution.environment` | `cast` or `reproduction` for frozen replay; `cast` for metadata preparation |
| `execution.script`, `script_sha256` | Exact runner and its hash; original archive for frozen replay |
| `execution.module` | Null for a historical script, the fixed historical CAST module, or the exact metadata module below |
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

`metadata_only` is restricted to CAST grouped-coverage protocol preparation. Its
execution fields must be exactly:

```json
{
  "mode": "metadata_only",
  "environment": "cast",
  "script": "repo:src/abvid/cast/grouped_design.py",
  "script_sha256": "<exact SHA256 of the identified source file>",
  "module": "abvid.cast.grouped_design",
  "arguments": ["freeze", "--config", "repo:configs/experiments/cast_grouped_coverage.json"]
}
```

The placeholder hash above must be replaced by a 64-character lowercase SHA256.
Alternative scripts, modules, arguments and extra fields are rejected.
`abvid prepare configs/experiments/cast_grouped_coverage.json` verifies the contract
and script hashes and prints the metadata command. With `--execute`, it runs that
command using the current checkout's `.venv/bin/python` and working directory,
with inherited `PYTHONPATH` removed. This mode prepares metadata only; it does not
authorize feature extraction, fitting, simulation or transfer evaluation.
`prepare` rejects historical/scientific modes, and `replay` rejects `metadata_only`.
Frozen replay continues to use the original archive and its recorded environments.

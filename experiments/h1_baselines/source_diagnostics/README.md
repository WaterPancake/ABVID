> Historical protocol/command reference. Commands retain their original paths
> and run from `ABVID_ARCHIVE_ROOT`, not from the reorganized code checkout.
> See the [experiment registry](../../README.md).

# H1 source-only diagnostics

Completed run `source_diagnostics_20261003_v1`: [findings](../../../reports/H1_source_diagnostics.md).
Test domain is **IDMT real → held-out IDMT real**, car/truck. No new target evaluation.
The [protocol](PROTOCOL.md) was frozen before feature extraction and diagnostic fitting.

This pass checks training/generalization, regularization, sample count, bandwidth/context,
threshold/ranking/probability behavior and location dependence. It does not change the
completed H1 experiment, source admissions or physical simulation components.

## Reproduce

Use the existing pinned Python environment in `experiments/reproduction/.venv`.
The feature extractor requires the locally pinned official BEATs checkpoint/code recorded
in the frozen config. No downloads are performed. Run from the repository root:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

experiments/reproduction/.venv/bin/python experiments/h1_diagnostics/test_diagnostics.py

# The recorded metadata-only freeze already exists. To create another freeze:
experiments/reproduction/.venv/bin/python experiments/h1_diagnostics/freeze_protocol.py \
  --output experiments/h1_diagnostics/frozen/replay

experiments/reproduction/.venv/bin/python experiments/h1_diagnostics/extract_source.py \
  --lock experiments/h1_diagnostics/frozen/replay/lock.json \
  --output experiments/h1_diagnostics/cache/replay

experiments/reproduction/.venv/bin/python experiments/h1_diagnostics/run_checks.py \
  --lock experiments/h1_diagnostics/frozen/replay/lock.json \
  --cache experiments/h1_diagnostics/cache/replay \
  --output experiments/h1_diagnostics/results/replay

experiments/reproduction/.venv/bin/python experiments/h1_diagnostics/verify_results.py \
  --lock experiments/h1_diagnostics/frozen/replay/lock.json \
  --cache experiments/h1_diagnostics/cache/replay \
  --results experiments/h1_diagnostics/results/replay
```

Every output directory must be new; scripts refuse overwrites. To reproduce the exact
recorded selections without creating another metadata freeze, pass
`frozen/source_diagnostics_20261003_v1/lock.json` instead, with new cache/results paths.
The independent verifier decodes only `source_*` members of old H1 prediction archives
for baseline replay; it never decodes old target arrays. Old artifacts are byte-hashed
before/after to establish immutability. The mixed-domain H1 feature cache is not used.

To verify the recorded run without rerunning fits, use its frozen/cache/results paths
in the final command. `build_report.py` intentionally renders the named recorded run
and its two figures using root `.venv` Matplotlib:

```bash
.venv/bin/python experiments/h1_diagnostics/build_report.py
```

## Artifacts

- `frozen/source_diagnostics_20261003_v1/`: source-only manifest, config, exact outer/inner/site
  selections, input hashes, original H1 protection hashes, Git HEAD/dirty state and versions.
- `cache/source_diagnostics_20261003_v1/`: three observation-profile feature arrays, waveform
  hashes, extraction code snapshots and official checkpoint provenance.
- `results/source_diagnostics_20261003_v1/`: 1,757 unique scaler/heads, training IDs/predictions,
  1,080 outer and 900 inner evaluation records, 60 C/threshold selections, scores, bootstrap
  samples/indices, executed code snapshots and independent verification.
- `reports/H1_source_diagnostics.md`: each check, qualifications and downstream implications.

Nested C and threshold controls are separate. Their validation labels are additional
source-domain access compared with untuned H1. All reported intervals are conditional on
the fixed fits and very few recording groups/sites. No setting is automatically promoted
to a new transfer study, and no combined best configuration has been run.

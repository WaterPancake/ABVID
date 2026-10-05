# H1 regularization transfer sensitivity

Completed `regularization_transfer_20261004_v1`: [results](../../reports/H1_regularization_transfer.md).
Domain: **IDMT real → MELAUDIS real**, car/truck; MELAUDIS is exposed development data.

The reference and source-selected C candidates reuse saved diagnostic models. No training,
new feature extraction, threshold selection, bandwidth change or synthetic evaluation occurs.
The [protocol](PROTOCOL.md) and exact identities were frozen before new candidate inference.

## Reproduction

From the repository root, use the existing pinned environment and local H1/diagnostic
artifacts. Output directories must be new; no command overwrites previous runs.

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

experiments/reproduction/.venv/bin/python experiments/h1_transfer_sensitivity/test_transfer.py

experiments/reproduction/.venv/bin/python experiments/h1_transfer_sensitivity/evaluate_transfer.py \
  --lock experiments/h1_transfer_sensitivity/frozen/regularization_transfer_20261004_v1/lock.json \
  --output experiments/h1_transfer_sensitivity/results/replay

experiments/reproduction/.venv/bin/python experiments/h1_transfer_sensitivity/verify_results.py \
  --lock experiments/h1_transfer_sensitivity/frozen/regularization_transfer_20261004_v1/lock.json \
  --results experiments/h1_transfer_sensitivity/results/replay
```

To make a separate metadata-only freeze before inference:

```bash
experiments/reproduction/.venv/bin/python experiments/h1_transfer_sensitivity/freeze_protocol.py \
  --output experiments/h1_transfer_sensitivity/frozen/replay
```

To verify the recorded run, use `results/regularization_transfer_20261004_v1` in the verifier.
To rebuild its report:

```bash
experiments/reproduction/.venv/bin/python experiments/h1_transfer_sensitivity/build_report.py
```

## Artifacts and checks

- `frozen/regularization_transfer_20261004_v1/`: protocol/config, exact model/training/target
  references, source-only C selections, manifests, hashes, Git state and software versions.
- `results/regularization_transfer_20261004_v1/`: 120 source/target probability archives,
  per-model/group metrics, confusion arrays, 10,000 paired bootstrap draws/indices,
  aggregate estimates, executed code and independent verification.
- Models and features stay in their protected original H1/diagnostic directories.
  There are 107 unique heads: 13 MFCC candidate cases reuse C=1.
- Seven boundary/metric tests cover target separation, exact source-only choices,
  decision ties, natural-prior pooling, paired intervals, class-missing draws and
  rejection of changed frozen artifacts. Independent verification checks every scaler,
  probability, group score, nested C choice, aggregate interval and original H1 replay.

The next source/path factorial is preparation only. This run does not create source assets,
render new audio, modify the simulator, or select a target-optimal replacement model.

# H1: frozen common-task baselines

This is the civilian car/truck gap measurement authorized after R0. It does not implement simulator interventions, fine-tune encoders, use motorcycle as a third test class or access military recordings. The [experimental protocol](../EXPERIMENTAL_PROTOCOL.md) is authoritative; [dataset admission](../../reports/dataset_admission_provenance.md) is qualified because original physical-source identities and generation lineage remain unknown.

Use the existing `experiments/reproduction/.venv` (Python 3.11, NumPy 1.26.4, scikit-learn 1.4.2, Torch 2.11, Librosa 0.11). No environment/model download is required. All paths below are relative to the project root. New output directories are required; immutable outputs are not overwritten.

```bash
# Regression checks for selection, group separation, observation shape,
# training-only statistics, metric definitions and paired uncertainty.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 experiments/reproduction/.venv/bin/python -m unittest discover -s experiments/h1 -p 'test_*.py' -q

# Metadata-only freeze, before extracting features or fitting.
experiments/reproduction/.venv/bin/python experiments/h1/freeze.py --output experiments/h1/frozen/new_lock

# Fixed BEATs/MFCC feature cache; hashes and input length are checked.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 experiments/reproduction/.venv/bin/python experiments/h1/extract.py --lock experiments/h1/frozen/new_lock/lock.json --output experiments/h1/cache/new_cache

# E0/E1 followed by E2 and the matched MFCC representation control.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 experiments/reproduction/.venv/bin/python experiments/h1/evaluate.py --lock experiments/h1/frozen/new_lock/lock.json --cache experiments/h1/cache/new_cache --output experiments/h1/results/new_run

# Recompute metrics and verify every scaler, known group boundary and hash.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 experiments/reproduction/.venv/bin/python experiments/h1/verify.py --lock experiments/h1/frozen/new_lock/lock.json --cache experiments/h1/cache/new_cache --results experiments/h1/results/new_run
```

Current run names use `h1_common_budget_v1.3` for lock/cache and `H1_20261002` for results. Each single-origin fit uses 190/class; there are six complete IDMT held-out-group folds and five training selections. MELAUDIS has four conservative connected groups and is exposed development evaluation. No validation set, target fitting or hyperparameter search occurs. The repeated-real and mixed arms use doubled row counts. The two pretrained/generator real-data histories remain qualifications to “synthetic-only task-training.”

`config.json` is the one canonical configuration format. `freeze.py` resolves exact file IDs; `lock.json` pins documents/config/admission and records software/Git state. Its original document snapshots remain unchanged; [documentation_clarifications.md](documentation_clarifications.md) records pre-fit prose clarifications without modifying any executable choice.

Artifacts include feature IDs/waveform hashes, code snapshots, all 260 fitted scaler/heads, probabilities, training IDs, per-group confusion matrices, conditional group-bootstrap samples and an independent verification report. The [results report](../../reports/H1_baseline_results.md) distinguishes measured gaps from untested simulation-component explanations. The official AST/CNN and conventional-augmentation extensions remain deferred, not implicitly completed by R0.

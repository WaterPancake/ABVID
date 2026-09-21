# Native-real benchmark snapshot

This directory preserves the completed 2026-09-20 development evaluation: 593
windows, 7 tracked sessions, 5 wheeled sessions, and 35 outer pairs. The domain is
`real -> real nested unseen recording-session pairs`.

- `dataset_lock.json`: corpus identity, source hashes, licenses and window counts.
- `real_manifest.jsonl`: exact frozen window manifest; audio paths are relative to
  `data/benchmark_v0_1_native_real_7t5w/`, not this snapshot directory.
- `audio_sources.yaml`: reviewed intervals and source roles at preparation time.
- `protocol.yaml`: protocol recorded before development evaluation.
- `results.json`: compact comparison and gate assessment.
- `verification.json`: per-session mean/min/max recall, artifact/checkpoint hashes,
  source code hashes at packaging time and split audit counts.
- `classical/`, `semantic/`, `fusion/`: full metrics, experiment and split snapshots.

The local audio, feature tensors and fitted probe states remain under `data/` and
`runs/`. The base commit is recorded with a dirty-worktree flag; these artifacts
have not been committed. Code hashes identify the implementation at packaging time.

## Reproduction

Run from the repository root with the existing reviewed source audio and validated
PANNs checkpoint available. Use a fresh output directory for each replication.
The original output directories must be retained. The original native-real manifest
hash is `e013532104c209e3c8f6906bb3c9daa3c46c2d6fb2e9fed82a65f4d2699a42c4`.

```bash
uv run python scripts/prepare_real_corpus.py \
  --config configs/real_corpus.yaml --targets data/targets \
  --output data/real_development_benchmark_v0_1_repeat

uv run --extra pretrained python scripts/extract_real_panns_features.py \
  --real-manifest data/real_development_benchmark_v0_1_repeat/real_manifest.jsonl \
  --checkpoint .artifacts/models/panns/Cnn14_mAP=0.431.pth \
  --output runs/benchmark_v0_1_native_real_repeat/panns_features.pt \
  --channel 0 --batch-size 16 \
  --forbid-source-id candidate-target-tracked-sherman-passby-gvn-43670951 \
  --forbid-source-id candidate-target-wheeled-car-start-drive-pdsounds-194 \
  --forbid-source-id candidate-target-tracked-t90m-27guards-2021 \
  --forbid-source-id candidate-target-wheeled-jltv-fort-mccoy-2019

uv run --extra pretrained --extra inspection python scripts/evaluate_semantic_sessions.py \
  --real-manifest data/real_development_benchmark_v0_1_repeat/real_manifest.jsonl \
  --feature-cache runs/benchmark_v0_1_native_real_repeat/panns_features.pt \
  --checkpoint .artifacts/models/panns/Cnn14_mAP=0.431.pth \
  --output runs/benchmark_v0_1_native_real_repeat/semantic --seed 42

uv run --extra pretrained --extra inspection python scripts/evaluate_fusion_sessions.py \
  --real-manifest data/real_development_benchmark_v0_1_repeat/real_manifest.jsonl \
  --panns-feature-cache runs/benchmark_v0_1_native_real_repeat/panns_features.pt \
  --checkpoint .artifacts/models/panns/Cnn14_mAP=0.431.pth \
  --output runs/benchmark_v0_1_native_real_repeat/fusion --channel 0 --seed 42

uv run --extra inspection python scripts/evaluate_classical_sessions.py \
  --real-manifest data/real_development_benchmark_v0_1_repeat/real_manifest.jsonl \
  --feature-cache runs/benchmark_v0_1_native_real_repeat/fusion/real_classical_features.pt \
  --output runs/benchmark_v0_1_native_real_repeat/classical --channel 0 --seed 42
```

These commands use the recorded default regularization and threshold grids. Verify
the regenerated manifest hash before fitting; a mismatch means the source corpus,
labels or preprocessing changed and requires a new dataset version. Both standalone
ensembles use argmax (equivalent to a 0.5 wheeled threshold); only fusion chooses a
threshold through its inner folds.

For the original run, `uv run python scripts/package_native_benchmark.py` rechecks
the locked hashes and saved splits and refuses to replace a differing snapshot.

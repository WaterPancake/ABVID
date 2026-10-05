# Civilian dataset admission: integrity and conservative provenance

This workflow implements steps 1–2 of [DATASET_PROTOCOL.md](../DATASET_PROTOCOL.md). It reads the supplied IDMT audio, MELAUDIS vehicle RAR and AI4TEN synthetic WAVs. It does not train models, generate new audio, assign train/test splits or touch military evaluation sources.

The audit decodes every inventoried WAV, checks container/payload agreement and numerical validity, records native layout and hashes, applies the car/truck metadata selector, links candidate IDMT microphone pairs and quarantines exact-audio annotation conflicts. Identical stereo copies in a MELAUDIS file labelled mono are recorded explicitly; distinct channels remain unresolved. Raw audio is never rewritten. Original source/session independence and synthetic lineage require later checks.

Use the existing reproduction environment (NumPy 1.26.4, SoundFile 0.13.1, libsndfile 1.2.2). macOS `/usr/bin/tar` must support RAR. The main script records its environment and preserves its executed source and protocol snapshot. Both output and work directories must be new; existing artifacts are refused.

```bash
# Failure-mode regression checks; no real corpus or model use.
experiments/reproduction/.venv/bin/python -m unittest discover -s experiments/admission -p 'test_*.py' -v

# Full read-only audit. Choose unused paths for each new run.
experiments/reproduction/.venv/bin/python experiments/admission/audit_integrity.py --output experiments/admission/results/new_run --work-dir experiments/admission/work/new_run --workers 4

# Recompute manifest consistency and verify saved artifact hashes; no inference.
experiments/reproduction/.venv/bin/python experiments/admission/verify_integrity_manifest.py experiments/admission/results/new_run
```

Results from the current run are documented in [dataset_admission_integrity.md](../../reports/dataset_admission_integrity.md). The first scan is retained in `results/integrity_20261002/`; its filename-layout failures prompted a recorded v1.2.1 clarification, whose output is in `results/integrity_20261002_v1_2_1/`.

Main artifacts:

- `inventory.jsonl`: metadata inventory saved before decoding.
- `decoded_files.jsonl`: decoded checks before cross-file annotation reconciliation (v1.2.1).
- `all_files.jsonl`: final per-file decisions, hashes and native format measurements.
- `car_truck_candidates.jsonl`: files passing steps 1–2, **not training admissions**.
- `exclusions.jsonl`: every other file with explicit reasons; originals retained.
- `idmt_pair_links.jsonl`: candidate acquisition-event links across provider microphone variants.
- `exact_equality_groups.json` and `label_conflict_groups.json`: exact byte/sample equality and conflicting annotations; no near-duplicate claim.
- `audioldm_extra_car.json`: the 201st car file's integrity and exact-equality finding.
- `summary.json`: counts, limitations, runtime, input/code/environment and artifact hashes.

Every row has `admitted_for_training: false` and `split_role: null`. A passing row can still share its source with another passing row. The later provenance audit must resolve same-label duplicates and related sessions before any split is frozen. Raw archive extraction stays in the local work directory; it is not a redistribution package.

## Steps 3–4 and H1 admission

The original steps 1–2 commands/snapshots above remain unchanged. The subsequent [provenance report](../../reports/dataset_admission_provenance.md) documents qualified H1 admission, additional waveform conflicts, conservative groups and unknown original metadata.

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 experiments/reproduction/.venv/bin/python experiments/admission/audit_provenance.py --output experiments/admission/results/new_provenance_run
```

The v1.3.1 `admitted.jsonl` is separate from the earlier candidate manifest. Whole released synthetic banks are training-only, not eligible for a claimed held-out template validation. Raw originals remain untouched. See [H1 protocol](../EXPERIMENTAL_PROTOCOL.md).

# ABVID real-data prototype v0.1

This is a runnable model and benchmark packaging of the existing H1 real/BEATs
arm. Its scientific configuration is the canonical
[`h1_baselines.json`](../../configs/experiments/h1_baselines.json) registry and
the hash-locked, executed `h1_common_budget_v1.3` contract. The registry's frozen
historical runner remains unchanged. This consumer reproduces the heads through
the shared `abvid` library and verifies parity before exporting a model bundle.
It introduces no new split, hyperparameter, model selection or scientific arm.

## Fixed scope

- **Task/data:** IDMT_V1, broad car=0 / truck=1. Motorcycles excluded.
- **Training:** exact archived H1 selections; 190 examples/class per fit.
- **Evaluation:** six connected provenance groups, leave one group out; five
  seeds `[42, 123, 456, 789, 1024]`, 30 fits. Each admitted IDMT file appears in
  one held-out fold per seed. Retain the frozen grouping and unknown original
  sessions; check known file, duplicate, group and session separation.
- **Preprocessing:** arithmetic channel mean; native rate → 8 kHz → 16 kHz via
  the frozen polyphase resampler; center 2 seconds / 32,000 samples; no gain
  normalization, padding, augmentation or learned preprocessing on evaluation.
- **Model:** frozen hash-verified BEATs iter3+ AS2M, mean-pooled 768D embeddings;
  fit StandardScaler and L2 logistic regression only on training records. The
  exact C, solver, tolerance and iteration limit come from the frozen contract.
- **Validation/tuning:** none; no threshold search, early stopping or best-head
  selection. The demo head is the first declared fold and seed, independent of
  performance. All 30 heads contribute equally to the benchmark headline.
- **Metrics:** mean group/seed macro-F1, balanced accuracy and accuracy, with
  per-head confusion matrices and per-class metrics. Include always-car using
  the same aggregation. Retain the existing H1 group/selection bootstrap
  macro-F1 interval only after verifying all source predictions match.
- **Evidence status:** exposed development data; no final independent test.
  CAST, MELAUDIS scoring and synthetic augmentation are outside this prototype.
  CAST transfer requires its own frozen comparison.

## Run

Use the existing Python 3.11 baseline environment. On a new installation:

```bash
UV_PROJECT_ENVIRONMENT=.venv-baseline uv sync --extra baseline --group dev
```

The current checkout, archived H1 manifests/cache/reference predictions, raw
IDMT WAVs and BEATs checkpoint must be present under the configured roots from
[`abvid.paths`](../../src/abvid/paths.py). Nothing is downloaded automatically.
The local artifact bundle is not a standalone dataset/checkpoint distribution.

```bash
.venv-baseline/bin/python -m abvid.benchmark build
.venv-baseline/bin/python -m abvid.benchmark predict --audio /absolute/path/to/vehicle.wav
```

Default output: `artifact:benchmarks/abvid_real_beats_v0_1`. A build refuses to
overwrite any existing directory; repeat with a fresh `--output artifact:...`
and use the corresponding `--bundle artifact:...` for prediction.

`predict` returns the chosen class, both classifier probabilities, file hash,
model identity and observation window. It center-crops longer inputs and rejects
short, silent or nonfinite inputs. It is a closed-set vehicle classifier: it
does not detect whether a vehicle is present, and its probabilities have not
been calibrated. Audio should already represent a vehicle event.

## Acceptance checks and artifacts

The build refits classifiers using the verified existing feature cache. Before
fitting it validates the canonical registry, freeze, input hashes, exact package
versions, budgets and role separation. It does not execute archived scripts or
load archived pickle models. Cached target features are never fitted or scored.

Acceptance requires all 30 new source probability sets to match archived H1
within `rtol=1e-10, atol=1e-10`, and every saved numeric head to reload with exact
predictions. A raw-audio check chooses the smallest file ID for each class in
each held-out group (12 samples, first declared seed). Processed waveform hashes
must match exactly, embeddings must match within `rtol=1e-4, atol=1e-4`, class
probabilities within absolute `1e-4`, and predicted labels exactly. These are
implementation parity tolerances, not scientific success criteria. A mismatch
fails the build instead of changing the old contract.

The external bundle contains:

- `heads.npz`: all 30 scaler/linear heads as numeric arrays, no pickle.
- `model.json`: default model, per-head training/evaluation IDs and metrics,
  complete scientific configuration, dataset version, seeds, checkpoint/code
  hashes, package versions, commit and working-tree status.
- `evaluation.json`: aggregate metrics, inherited interval provenance,
  raw-audio and saved-model checks, runtime and model-manifest hash.
- `EVALUATION.md`: readable results and claim boundaries.

CPU is sufficient; the recipe uses four Torch threads and one BLAS thread.
Budget approximately 2 GB of available RAM for checkpoint loading and inference;
this is a planning estimate, not a measured peak. With the cache present, expect
seconds to a few minutes on the development Mac. Runtime is measured in the
generated report. Neither encoder training nor a GPU is required.

This benchmark establishes a working real-data reference for a later controlled
CAST comparison. It does not itself test synthetic-to-real transfer, physical
source recovery, or whether source mismatch dominates propagation mismatch.

The completed local build and standalone prediction checks are recorded in the
[evaluation report](../../reports/ABVID_prototype_v0_1.md).

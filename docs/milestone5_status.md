# Milestone 5 — Real-World Transfer Status

Date: 2026-08-21
Status: first reviewed, grouped transfer experiment complete
Performance claim: **limited to the exact held-out real sessions documented below**

## Implemented protocol

Milestone 5 now has a deterministic native-real corpus preparer and a complete
single-microphone transfer evaluator.

The real-data preparer:

- selects one channel explicitly rather than averaging channels;
- resamples and extracts fixed windows without synthetic augmentation;
- sets native real SNR to `null` rather than inferring an unavailable value;
- caps windows per recording session and spreads selected windows across long files;
- records source and normalized WAV hashes, source URL, license, attribution,
  recording session, operating condition, window boundaries, and review status;
- produces `real_manifest.jsonl`, per-window metadata, and `corpus_audit.json`.

The evaluator implements the roadmap comparisons:

| Experiment | Training domain | Validation domain | Test domain |
|---|---|---|---|
| A — Real only | Fixed real train sessions | Fixed real validation sessions | Fixed held-out real sessions |
| B — Synthetic only | Synthetic train sessions | Synthetic validation sessions | Fixed held-out real sessions |
| C — Synthetic + limited real | B checkpoint, then 1/5/10/25% of real train windows | Fixed real validation sessions | Same fixed held-out real sessions |

All splits are grouped by `recording_session`. The limited-real subsets come only
from already assigned real training sessions and are class-stratified. Requested and
actual fractions are both recorded because small datasets may require at least one
example of each class.

Every completed run writes:

- `experiment.json`, `metrics.json`, and `splits.json`;
- `models.pt` and `domain_features.pt`;
- `learning_curve.csv` and `learning_curve.png`;
- `embedding_projection.csv` and `embedding_projection.png`.

The embedding plot uses the pooled 64-dimensional synthetic-pretrained CNN encoder
and exact centered-SVD PCA. It is explicitly marked
`visualization_only_not_representation_quality_evidence`.

## Commands

Prepare native real windows:

```bash
uv run python -m vehicle_audio.cli prepare-real \
  --config configs/real_corpus.yaml \
  --targets data/targets \
  --output data/real_eval_v2
```

Run the full transfer protocol after the audit passes:

```bash
uv run python scripts/evaluate_transfer.py \
  --synthetic-manifest data/generated/m3_controlled_seed42/manifest.jsonl \
  --real-manifest data/real_eval_v2/real_manifest.jsonl \
  --output runs/m5_transfer_seed42 \
  --real-fraction 0.01 \
  --real-fraction 0.05 \
  --real-fraction 0.10 \
  --real-fraction 0.25 \
  --pretrain-epochs 10 \
  --finetune-epochs 5 \
  --real-only-epochs 10 \
  --batch-size 32 \
  --seed 42 --device cpu
```

`--allow-unreviewed` exists only for engineering smoke tests and marks output
`engineering_smoke_unreviewed`. It does not bypass the minimum independent-session
gate and its results are not reportable research evidence.

## Checked real corpus

Real manifest SHA-256:

```text
84fb1ef0e9f44df7f8d5c48400ea3f9b3bac3de2b3bdf5ae63004b073883c1dd
```

| Property | Value |
|---|---:|
| Real windows | 271 |
| Tracked windows | 170 |
| Wheeled windows | 101 |
| Independent tracked sessions | 4 |
| Independent wheeled sessions | 3 |
| Provenance-complete sources | 7/7 |
| Sources with reviewed segment boundaries | 7/7 |

The corpus satisfies the evaluator's minimum three independent sessions per class,
but it is still smaller than the preferred five-plus sessions per class. In
particular, each validation/test class is represented by only one source session.

## First transfer result

Run: `runs/m5_transfer_seed42`

Git commit: `737933f9a1203c657438015152f75b62e48a80c7`

Combined dataset version:
`a0a38e6d900e6ea85c5eb395583ad2f44a376f1c82abb0553fe61036289b03d5`

The fixed real test set contains 124 windows from two completely held-out sessions:

- tracked: `commons_retromobile_amx30_demo_2015` (64 windows);
- wheeled: `freesound_lmartins_maserati_granturismo_2019` (60 windows).

| Experiment | Real data used for training | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| A — real only | 100% of fixed real train split | 10.5% | 10.2% | 9.5% |
| B — synthetic only | 0 | 33.9% | **32.8%** | **25.3%** |
| C — pretrained + requested 1% real | 2 windows (actual 2.1%) | 16.9% | 16.4% | 14.5% |
| C — pretrained + requested 5% real | 5 windows (actual 5.3%) | 27.4% | 26.6% | 21.5% |
| C — pretrained + requested 10% real | 10 windows (actual 10.5%) | 27.4% | 26.6% | 21.5% |
| C — pretrained + requested 25% real | 24 windows (actual 25.3%) | 17.7% | 17.2% | 15.1% |

Every evaluated model had **0% recall on the held-out Maserati session**. The
synthetic-only model correctly classified 42/64 AMX-30 windows but 0/60 Maserati
windows. The real-only model selected a checkpoint with 98.5% validation balanced
accuracy, then fell to 10.2% on the two unseen sessions. This large validation/test
reversal is evidence of source/session overfitting, not a successful transfer result.

Calibration is also poor: expected calibration error is 0.414 for real-only and
0.505 for synthetic-only on the real test set. Adding the currently available real
windows does not close the sim-to-real gap and generally degrades the synthetic-only
checkpoint.

## Interpretation and next data work

This run demonstrates that the current representation does **not** generalize at
category level across these real source sessions. It should not be generalized to a
broader tracked-vs-wheeled claim because:

- the test set is one recording session per class;
- the real training split contains only one short wheeled session;
- source identity, venue, microphone, and vehicle identity are inseparable in this
  small corpus;
- the failed Wikimedia collection attempts prevented reaching the preferred five
  sessions per class.

The collection manifest retains the throttled T-18, cobblestone-car, and Cologne
night-traffic attempts for later retry. The duplicate MPF repost remains grouped into
one `recording_session`, and Tiger 131 remains `review_required` because its author
and YouTube chain of custody are ambiguous.

Milestone 6 may now proceed as a controlled attempt to improve corruption invariance,
but it must retain this exact Milestone 5 result as the transfer baseline and must not
present invariance training as a substitute for acquiring more independent real
sessions.

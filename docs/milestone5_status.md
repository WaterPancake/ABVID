# Milestone 5 — Real-World Transfer Status

Date: 2026-08-20  
Status: evaluation implementation complete; checked experiment blocked by real-data
coverage and content review  
Performance claim: **none**

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
  --output data/real_eval_v1
```

Run the full transfer protocol after the audit passes:

```bash
uv run python scripts/evaluate_transfer.py \
  --synthetic-manifest data/generated/m3_controlled_seed42/manifest.jsonl \
  --real-manifest data/real_eval_v1/real_manifest.jsonl \
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

## Checked local corpus audit

Dataset version (SHA-256 of `data/real_eval_v1/real_manifest.jsonl`):

```text
4f1e6daf5137f3b93852e9d574a5ad40e3b656306f9e08536d029e2cd740e889
```

| Property | Current value | Required state |
|---|---:|---:|
| Real windows | 115 | Not a substitute for source sessions |
| Tracked windows | 64 | — |
| Wheeled windows | 51 | — |
| Independent tracked sessions | **1** | At least 3; preferably 5+ |
| Independent wheeled sessions | **2** | At least 3; preferably 5+ |
| Provenance-complete sources | 3/3 | All |
| Sources with reviewed segment boundaries | **0/3** | All used sources |

Current sessions:

- tracked: `dvids_romanian_tank_range_2014`;
- wheeled: `commons_goodwood_festival_of_speed_2009`;
- wheeled: `commons_soundsofchanges_private_collection`.

The preflight error is intentional:

```text
Milestone 5 grouped evaluation requires at least three independent recording
sessions per class; found tracked=1, wheeled=2
```

No model was trained and no accuracy was reported on this corpus. Splitting its 115
windows randomly would turn one tracked recording into apparent train/test diversity
and violate the project leakage rule.

## Data needed next

The local catalog contains approval-gated candidates from independent sessions. A
minimal split-capable addition would require two tracked sessions and one wheeled
session, but that would leave only one session per class in each partition. A more
defensible first experiment should target at least five sessions per class.

Candidate IDs already present in the local catalog include:

- tracked: `target-tracked-mpf-firepower`,
  `target-tracked-stug-iiig-lappeenranta`, and
  `target-tracked-bae-mpf-arrival`;
- wheeled sessions beyond the two already present:
  `target-wheeled-vanwall-1957-goodwood-2010`,
  `target-wheeled-maserati-granturismo-exhaust`,
  `target-wheeled-car-cobblestone-pass`, and
  `target-wheeled-cars-night-stopping`.

The two MPF entries are alternate platform copies of the same 2023-01-25
Tankodrome arrival. Their catalog records now share
`us_army_armor_cavalry_collection_tankodrome_2023`, so they count as one session.
The Tiger 131 item has been returned to `review_required`: its Commons license is
CC0, but the author is unknown and the file came from YouTube.

A 2026-08-20 proposal-only source refresh found three independent tracked-session
candidates. They remain outside the catalog until operator content/provenance
review:

- `candidate-target-tracked-retromobile-amx30-2015` (preferred);
- `candidate-target-tracked-t18-khabarovsk-2015` (fallback);
- `candidate-target-tracked-bovington-type59-2014` (weaker fallback; vehicle label
  is disputed on Commons).

The proposal records are:

- `openclaw/vehicle-audio-curator/proposals/2026-08-20-milestone5-balanced-acquisition.yaml`;
- `openclaw/vehicle-audio-curator/proposals/2026-08-20-milestone5-additional-tracked.yaml`.

The collector rechecks provider metadata and licenses at collection time. The
OpenClaw download gate still requires an explicit operator message of the form:

```text
approve download: <source-id>
```

After collection, usable vehicle-only time intervals must be reviewed and stored as
`condition_segments` before the full protocol will run. Milestone 6 should not begin
until this transfer experiment produces a reproducible, reviewed result.

The current workspace is not a Git checkout (`.git` is absent). Before a reportable
Milestone 5 run, version control must be restored or initialized so the required git
commit can be recorded with the experiment.

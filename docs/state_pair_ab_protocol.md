# Fixed-vehicle idle/moving A/B test — frozen diagnostic protocol

Status: **frozen before fitting, operator-approved on 2026-09-21**.
This proposal was written before fitting any state-pair classifier. It does not
change the existing benchmark, annotations, admissions or Milestone 7 status.

## Question

Can frozen pretrained representations distinguish a fixed pair of reviewed vehicle
models when a classifier is trained on idle excerpts and applied to moving excerpts,
and in the reverse direction? This is not an original-versus-decomposed comparison.
The operator confirmed this interpretation: "Idle versus moving recognition."

The task label is member A versus member B of the chosen pair, not a general
tracked/wheeled label. A same-class pair can therefore test a narrower question
than mobility classification. This is a bounded representation-feasibility probe,
not hierarchical identification, open-set classification or a deployed model.

## Pairs selected from metadata, not pair-test performance

| Role | Vehicle A | Vehicle B | Caveat |
|---|---|---|---|
| Primary cross-class | AMX-30 | M1151 HMMWV | Different recording contexts remain confounded with identity |
| Secondary tracked pair | AMX-30 | T-72B3 | Only eight overlapping idle windows for T-72B3 |
| Secondary wheeled pair | M1151 HMMWV | M1126 Stryker convoy | Stryker is multi-vehicle audio |

The current development inventory supplies exactly one recording session for each
of these four model labels. It contains no verified same-individual vehicle pairs.
T-72B3 excerpts must come only from its reviewed intervals, not BMP-3 intervals
within the same source. All source-level split groups must remain intact under
the default protocol.

## State definition and available windows

For this first comparison use **idle versus steady_speed only**. Exclude acceleration,
deceleration, mixed, unknown and archived startup intervals from the probe, without
altering their existing corpus roles. This avoids treating an engine rev as proof of
vehicle movement. These state names are reviewed metadata, not audio-estimated states.

| Model | Idle windows | Steady-speed windows | Independent sessions |
|---|---:|---:|---:|
| AMX-30 | 35 | 64 | 1 |
| T-72B3 | 8 | 29 | 1 |
| M1151 HMMWV | 24 | 45 | 1 |
| M1126 Stryker convoy | 28 | 82 | 1 |

Keep the frozen 785-window parent manifest and exact reviewed per-interval identity
mapping from the [state/identity audit](state_identity_audit.md). Use its unchanged
two-second 16 kHz mono windows; no new processing, decomposition, augmentation,
normalization, duration search or encoder fine-tuning.

## Planned comparison

- A: fit on idle audio for both members; score steady-speed audio for both members.
- B: fit on steady-speed audio for both members; score idle audio for both members.
- Repeat for frozen BEATs 768-dimensional embeddings, PANNs full 2048-dimensional
  embeddings, and the existing PANNs 35-score proxy. Retain every result; do not
  select a best pair after seeing results.
- Fixed head: training-only class-balanced weighted standardization and logistic
  regression, fixed C=1, seed 42, threshold 0.5. No search or calibration against
  destination-state audio. Cache hashes/sample ordering must be verified before reuse.
- Report balanced accuracy, per-member recall, confusion counts, exact train/test
  intervals and counts, and training fit separately as a resubstitution sanity check.
  A constant prediction has 50% balanced accuracy. Report no window-based confidence
  intervals or significance claims; overlapping windows are not independent trials.
- Save configuration, code/input/checkpoint hashes, commit/dirty status, class mapping,
  splits, fitted heads, predictions and numerical replay checks. No result is a
  milestone gate assessment, confirmation test or measured performance ceiling.

## Approved diagnostic exception

**Default: session-disjoint training and test, as required by AGENTS.md.** The present
fixed-pair inventory cannot support this: each member has just one session, so both
directions would share source sessions. A runner must refuse that split by default.
More independent reviewed sessions for the same models and states are needed.

An alternative requires an **explicit operator-approved diagnostic-only exception**:
use non-overlapping idle/moving intervals from the same sessions. If approved, label
every result `real -> real within-session cross-state diagnostic`, explicitly list
the shared sessions, verify no sample overlap, and keep the outputs separate from
all unseen-session benchmark results. Do not call those excerpts an independent test.
Temporal separation does not remove shared engine, microphone, background or source
context, and a high score cannot prove genuine vehicle-identity recognition.

On 2026-09-21 the operator explicitly approved: "Allow diagnostic-only exception."
This authorizes only the within-session, cross-state diagnostic above. The dedicated
CLI requires `--allow-within-session-diagnostic`; without it the run is refused.
All results explicitly record shared sessions and non-overlapping source intervals.
The logistic heads use class weights N/(2*N_class), lbfgs with max_iter=5000, and
training-only weighted StandardScaler. The pair-member order in the table fixes
labels A=0 and B=1; probability >=0.5 selects B. Fitted heads are saved as float64
arrays in JSON and replayed independently. Reuse only hash-verified cached features
from preprocessing_grid_v1 (unchanged control) and beats_comparison_v2, aligned to
the pinned parent manifest. No trained benchmark classifier is reused.

At protocol freeze no state-pair model has been fitted. Results belong only in a
dedicated `runs/state_pair_ab_v1` diagnostic directory, not a benchmark release.
The normal session-split guards, reserved T90M/JLTV pair and consumed Sherman/PDSounds
exclusions remain unchanged. No new data has been admitted.

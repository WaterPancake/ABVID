# State and vehicle-identity audit

Completed against the current 785-window non-startup development corpus.
**Scope: native-real development metadata audit, not model evaluation or a new
listening/video review.** No corpus labels, admissions, sidecars, protected media,
model artifacts or previous results were changed. No decomposition was run.

## Outcome

The audit covered **12 recording sessions (7 tracked / 5 wheeled), 53 approved
intervals, 840.6 seconds of approved source audio, and all 785 windows**. Intervals
were checked against the catalog and worksheet; source/window hashes, current
sidecars, audio bounds, startup exclusions and protected-session exclusions passed
the existing fail-closed development loader. Every window belongs to exactly one
active interval. Summed overlapping window durations are not independent audio time.

There are **four same-model, same-session idle/motion candidate pairs**, but
**zero documented same-individual vehicle pairs**. This is a limit of the annotations,
not a finding that different individual vehicles necessarily appear in each clip.

| Same-model candidate | Idle audio | Motion-labeled audio | Main caveat |
|---|---:|---:|---|
| AMX-30 | 37 s | 86 s | Physical-vehicle continuity and camera geometry not annotated |
| T-72B3 | 9 s | 31 s | Road-march source also contains BMP-3; never pair across those models |
| M1151 HMMWV | 26 s | 60 s | B-roll may contain different vehicles/takes; continuity not annotated |
| M1126 Stryker | 30 s | 102 s | Explicit multi-vehicle convoy; not an isolated engine reference |

The idle subset is **130 windows across only 2 tracked / 4 wheeled sessions**.
Military-wheeled idle coverage is just HMMWV and Stryker; the other two idle
wheeled sessions are Ford Model T and Maserati. These are small exploratory
subsets, not a new gate-passing benchmark. A source containing multiple models or
vehicles remains one split group and does not create independent sessions.

## Identity findings

The manifest's `vehicle_model` is inherited from the whole source. For **229
windows in three mixed-model recordings**, that field is a combined source label,
not the individual model in the particular interval:

- T-72B3/BMP-3: 37 T-72B3 windows (8 idle, 29 steady-speed); 32 BMP-3 steady-speed windows.
- T-80BVM/MT-LBVMK: 23 T-80BVM acceleration windows; 37 MT-LBVMK windows
  (16 mixed, 20 steady-speed, 1 decelerating).
- Bradley/Abrams: 45 M2 Bradley steady-speed windows; 55 M1 Abrams windows
  (52 steady-speed, 3 accelerating).

The audit copies the worksheet's existing per-interval vehicle names into a
**separate diagnostic inventory**, only when boundaries and conditions match the
active catalog exactly. Catalog notes and original manifest model labels are
retained beside them. This does not mutate labels or authorize finer-class training.
There are 15 session/model groups, not 15 independent sessions. M1 Abrams occurs
in two sessions; unseen-session evaluation is not necessarily unseen-model evaluation.

There are no documented per-interval individual-vehicle IDs, measured RPM values,
or validated engine-type annotations in this audit. These remain null; model names
must not be used to fabricate such measurements.

## State ambiguities and confounds

1. **MT-LBVMK 00:24–00:41 is currently `mixed`, not `idle`.** Both the catalog and
   current worksheet agree. Earlier conversation described a nearby interval as
   idle, so this is a targeted re-review candidate, not permission to relabel it.
   The adjacent 00:41–01:02 interval is approved steady-speed MT-LBVMK movement;
   00:00–00:24 is a different model, T-80BVM.
2. **Maserati 00:15–01:01.5 is revving with unknown physical motion.** It is not
   evidence of tires rolling. Retain `mixed`; do not equate engine acceleration
   with vehicle acceleration.
3. **Stryker 00:05–00:16 has no state label (`unknown`).** Its idle labels describe
   the reviewed scene, not proof that every audible vehicle is stationary.
4. **Ford is idle-only; Abarth is acceleration-only.** Operating state, model and
   recording context are entangled. Ford supplies no tire-motion example.
5. **No admitted idle interval exists for M1 Abrams, M2 Bradley, BMP-3, T-80BVM,
   MPF or StuG.** A moving engine cannot be treated as an isolated stationary reference.
6. Existing conditions are operational labels, not acoustic-source annotations.
   `idle` does not certify engine-only audio; `steady_speed` does not certify
   audible tracks/tires. Dominant vehicle count, audibility of running gear, surface,
   camera changes and vehicle continuity require additional evidence.

Overall the corpus contains 141.3 s idle, 524 s steady-speed, 84.8 s accelerating,
16 s decelerating, 63.5 s mixed and 11 s unknown. There are no wheeled deceleration
intervals. These counts describe existing labels, not fresh motion verification.

## Suggested first gallery excerpts

These are deterministic first-six-second excerpts from the earliest applicable
approved idle and steady-speed intervals for each candidate. They were chosen by
metadata, not predictions or perceived separation quality. Bounds were checked;
they are **proposals, not newly exported/listened-to clips**.

| Source/model | Idle excerpt | Steady-speed excerpt | Role |
|---|---|---|---|
| AMX-30 | 00:41–00:47 | 01:35–01:41 | Initial tracked example |
| T-72B3 | 00:36–00:42 | 00:46–00:52 | Second tracked model, short idle coverage |
| M1151 HMMWV | 00:15–00:21 | 00:37–00:43 | Initial military-wheeled example |
| Stryker convoy | 00:17–00:23 | 00:36–00:42 | Explicit multi-source stress example |

The **decomposition gallery** is a proposed local diagnostic listening page. For
each excerpt it would show original audio and three signal-structure views:
tonal/harmonic, transient/percussive, and residual, with aligned spectrograms,
source timestamps, reviewed state/model, parameters and clear gain handling.
These outputs must not be named engine/track/background ground truth. An engine
can contribute transients, a squeal can be tonal, and the residual can contain
useful vehicle sound. The original must remain available and outputs should
reconstruct it within measured numerical tolerance under the chosen method.

Purpose: let the reviewer judge whether useful cues survive, leak across components,
or are damaged, before training feature-based comparisons. This is not source
separation validation without isolated reference sources. No gallery is built by
this audit, and no architecture or Milestone 7 functionality is introduced.

## Next decisions

- Proceed with AMX-30, T-72B3 and HMMWV as **same-model** diagnostic comparisons;
  include Stryker only with the convoy caveat. They do not require relabeling to
  inspect, but they cannot establish same-individual engine-to-motion matching.
- Ask the operator to clarify MT-LBVMK 00:24–00:41 and, where discernible, whether
  the paired clips show the same individual vehicle with compatible camera position.
  Unknown is acceptable. No immediate wholesale re-review is needed.
- Before state-aware training, add explicit interval identity and separate
  physical-motion / engine-state / acoustic-cue fields in a versioned annotation
  proposal. Do not silently rewrite the existing benchmark or split mixed-model
  recordings into independent sessions. Human motion labels must not silently
  become privileged inputs in a claimed audio-only deployment.
- Preserve the T90M/JLTV reservation and consumed Sherman/PDSounds exclusions.
  Startup stays excluded. No newly found recordings were admitted.

## Artifacts and reproduction

- [Full state/model inventory and all 53 intervals](../runs/state_identity_audit_v1/inventory.md).
- [Machine-readable audit, exact window assignments and provenance](../runs/state_identity_audit_v1/audit.json).
- [Read-only audit CLI](../scripts/audit_state_identity.py).
- [Operator worksheet](audio_segment_review.md).

```bash
.venv/bin/python scripts/audit_state_identity.py --output runs/state_identity_audit_repeat
.venv/bin/python -m pytest tests/test_state_identity_audit.py -q
```

The CLI refuses existing output directories. It records input/code hashes, commit,
dirty status, corpus audit and exclusions. No randomness, model checkpoint, fitted
parameters, new split or performance metrics are involved.

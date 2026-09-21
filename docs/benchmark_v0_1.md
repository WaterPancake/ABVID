# ABVID Benchmark v0.1

Date: 2026-09-20  
Status: refreshed 7/5 development evaluation complete; gates 2-4 failed

## Purpose

ABVID Benchmark v0.1 is the reproducible evaluation artifact for the work completed
through Milestone 6. It keeps three questions separate instead of merging their
numbers into one performance claim:

| Track | Test domain | Role |
|---|---|---|
| Native real | `real -> real nested unseen recording-session pairs` | Primary category-generalization result |
| Controlled corruption | `held-out real events -> controlled corrupted observations` | Noise and microphone robustness |
| Simulated array | `procedural synthetic -> simulated multichannel synthetic` | Array, beamforming, and localization study |

Native-real audio has no assigned SNR. Controlled SNR belongs only to paired
corruptions. Simulated-array localization is not evidence of real-array field
performance.

## Source roles

The source catalog is the authority for development admission. A target belongs to
development only when `admitted_to_corpus: true`. All windows from one original
recording remain in one `recording_session`.

On 2026-09-20, startup intervals were parked at the operator's request. Active
development discovery now excludes AMX-30 9–40 s, Ford Model T 6–14 s and Maserati
0–2.5 s. Audio and review history are retained. The
[preprocessing grid protocol](preprocessing_grid_protocol.md) uses all 785 remaining
two-second windows with no session cap; this is a new dataset version, not a
rewrite of the frozen 593-window baseline, its plots or its audit reel.

| Role | Sources | Permitted use |
|---|---|---|
| Development | 7 tracked / 5 wheeled reviewed sessions | Training, nested validation, and replay-demo examples |
| Consumed locked result | Sherman + PDSounds-194 | Preserve the completed 2026-08-24 result; diagnostics may be labeled post-hoc, never a new locked test |
| Future confirmation | T90M + Fort McCoy JLTV | No training, tuning, exploratory inference, or demo use before the one-shot confirmation |
| Candidate | Downloaded or proposed but not admitted | Audiovisual review only |

The machine-readable role contract is
[`configs/benchmark_v0_1.yaml`](../configs/benchmark_v0_1.yaml). Audit it with:

```bash
uv run python scripts/audit_benchmark.py \
  --config configs/benchmark_v0_1.yaml \
  --targets data/targets \
  --output .artifacts/benchmark_v0_1_audit.json
```

The current role audit passes the minimum session-count gate: tracked has seven
independent development sessions and wheeled has five. The refreshed development
evaluation is now complete and failed gates 2-4; this does not authorize Milestone 7
or use of the reserved confirmation pair.

## Historical evidence boundary

The 77.35% balanced accuracy and 77.42%/77.28% recalls are historical nested
development results on a 4-tracked/3-wheeled corpus. They remain valid only for that
versioned experiment. The minimum held-out recalls were 22.22% and 29.41%.

The frozen Sherman/PDSounds confirmation was evaluated exactly once and achieved
49.69% balanced accuracy. Later experiments that used either source are development
diagnostics, even if current catalog flags exclude the recordings. No old checkpoint
is relabeled as independent from a source it already consumed.

The current 7/5 development result is documented in
[`benchmark_v0_1_native_real_results.md`](benchmark_v0_1_native_real_results.md).
The primary fusion scored 39.70% mean balanced accuracy, 51.98% tracked recall, and
27.42% wheeled recall, with a worst held-out session-context recall of zero.
Historical passes do not transfer to a changed dataset or model.

## Release procedure

Steps 1-5 are complete for the 7/5 baseline and controlled-corruption track;
step 7 remains blocked by the failed development gates. The
[failure diagnosis](benchmark_v0_1_failure_diagnostics.md) explains branch/threshold
behavior, and the [corruption report](benchmark_v0_1_corruption_results.md) covers
30 fixed-model stress conditions. Neither analysis changes the native gate result.

1. Synchronize catalog metadata and rerun the benchmark role audit. The catalog and
   local discovery must match, and every protected source must remain excluded.
2. Prepare a versioned native-real manifest from reviewed intervals. Record its hash,
   configuration, commit, window counts, session counts, licenses, and source roles.
3. Preregister the nested development protocol and the confirmation acceptance rule.
4. Evaluate the existing classical, pretrained-audio, and strongest fusion baselines
   on the same nested session folds. Report per-window metrics, per-session recall,
   worst-session recall, and a session-level uncertainty interval.
5. Run the controlled-corruption track with held-out original background recordings;
   keep its numbers separate from native-real results.
6. Preserve the existing Milestone 4 array experiment as the simulated-array track.
7. Only if development gates 1-4 pass, freeze preprocessing, checkpoint,
   aggregation, and threshold. Complete the JLTV review and evaluate T90M/JLTV once.

## Milestone 7 gate

The benchmark does not authorize Milestone 7 until the same versioned protocol passes
all five gates:

- at least five independent development sessions per class;
- at least 75% nested mean balanced accuracy;
- at least 70% mean recall for both classes;
- at least 50% window recall for every held-out session;
- a passing one-shot result on the reserved confirmation pair.

Benchmark and replay-demo packaging can continue while this gate is open.

# Milestone 6 improvement checkpoint

Date: 2026-08-21

Status: controlled multi-seed diagnostics complete; Milestone 7 readiness not met

Performance claim: **limited to the exact procedural corpus and seven reviewed
native-real recording sessions identified below**

## Experiment identity

| Field | Value |
|---|---|
| Runs | `runs/m6_factorial_2s_session_diagnostic/seed_{7,19,42,73,101}` |
| Aggregate | `runs/m6_factorial_2s_session_diagnostic/aggregate` |
| Git commit | `6bfe7bcaefcdcb431c7b31df98cbabf245154535` |
| Training seeds | 7, 19, 42, 73, 101 |
| Grouped split seed | 42 for every run |
| Combined dataset version | `09ff29d1732fde110a4986052d9c8a19c3d18da3a06a1a76f126a1fbdcf03633` |
| Synthetic corpus | 2,688 fully crossed, two-second paired observations |
| Native-real corpus | 271 two-second windows; 4 tracked and 3 wheeled sessions |
| Model | Small standardized log-Mel CNN; optional projection head |

Every run records its configuration, manifest hashes, grouped split, checkpoint,
metrics, and commit. The aggregate command validates commit, dataset, fixed split,
configuration except training seed, methods, conditions, and evaluation support
before combining runs. Dispersion below is sample standard deviation across the five
training seeds.

## Fixed held-out results

The primary native-real result remains the fixed AMX-30 tracked session and Maserati
GranTurismo wheeled session. Neither appears in training or validation.

| Method | All-corruption synthetic BA | Fixed native-real BA |
|---|---:|---:|
| Standard supervised | 75.14% +/- 2.30% | 35.97% +/- 9.43% |
| Augmentation only | 76.24% +/- 0.76% | **38.79% +/- 9.54%** |
| Paired supervised | **76.65% +/- 1.39%** | 36.75% +/- 5.96% |
| Direct representation invariance | 75.83% +/- 1.72% | 35.66% +/- 4.76% |
| Projected representation invariance | 76.60% +/- 0.96% | 37.05% +/- 6.84% |

The paired and projected objectives do not provide a repeatable real-transfer gain.
All five methods average between 0.7% and 1.3% wheeled recall on the fixed test. The
small synthetic differences are therefore not evidence that Milestone 6 has solved
the sim-to-real problem.

## All-session diagnostic

Milestone 6 models use no native-real audio for fitting or checkpoint selection, so
the evaluator additionally reports every real session. This diagnostic is useful for
failure analysis, but it is **not** a replacement for the fixed held-out result and
must not become a model-selection target. Each session contributes equally within its
class before the two class means are averaged.

| Method | Session-balanced accuracy |
|---|---:|
| Standard supervised | **62.58% +/- 11.44%** |
| Augmentation only | 46.96% +/- 5.16% |
| Paired supervised | 54.32% +/- 9.43% |
| Direct representation invariance | 47.52% +/- 2.08% |
| Projected representation invariance | 57.67% +/- 7.35% |

Source-level behavior is highly heterogeneous. Across seeds, every method averages at
most 1.33% accuracy on the Maserati session, while every method exceeds 91% on the
Ford Model T session and 71% on the Goodwood wheeled session. The real metadata marks
both Ford and Maserati material as containing stationary/startup audio, so the failure
cannot be reduced to a simple `idle == impossible` explanation. Vehicle identity,
recording chain, venue, and operating state remain entangled.

## Interpretation

The current bottleneck is not another consistency-loss coefficient. The strongest
evidence is:

- five objectives have nearly the same fixed synthetic-to-real result;
- the same model can recognize two wheeled sessions and completely miss a third;
- real-only training previously selected a 98.5% validation checkpoint and fell to
  10.2% on unseen sessions, demonstrating session memorization;
- the corpus has only seven independent real sessions and one held-out session per
  class.

Milestone 7 should not start from these results. Category-level performance is not
yet reliable across unseen real sources, and open-set or family labels would add
degrees of freedom before the binary transfer problem is controlled.

## Next improvements

1. Expand to at least five independent, reviewed sessions per class, with multiple
   moving and stationary operating states. Keep every original source in one group.
2. Preserve the fixed real test and add leave-one-session-out reporting on development
   sessions. Use session-balanced sampling and metrics so long recordings do not
   dominate.
3. Add a frozen pretrained environmental-audio encoder plus a linear probe as a
   separate baseline, then compare limited fine-tuning without selecting on the fixed
   real test.
4. Improve procedural source realism, especially pass-by motion/Doppler, transient
   engine behavior, and within-class engine diversity. Calibrate broad ranges from
   development recordings only, not the held-out sessions.
5. Repeat standard, augmentation-only, and the best established invariance baseline
   across the same seeds. Retain an objective only if it improves unseen-session
   performance and both class recalls, not just synthetic embedding similarity.

Three additional public-domain DVIDS wheeled-source discoveries are retained as
`review_required` proposals under `openclaw/vehicle-audio-curator/proposals/`. No
audio was downloaded. Their provider endpoints returned HTTP 403 to metadata access,
and each source still needs content-segment review and explicit operator approval.

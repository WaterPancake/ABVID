# CAST physics prototype results

2026-10-08. The simple physics-based prototype is runnable and its declared
numerical relationships pass verification. It generates source audio from RPM,
torque and cylinder assumptions, then optionally applies distance and delay.
It is **not yet a replacement for CAST**: its fitted mean spectral loss is 2.0901
versus 1.3898 for original CAST on the same 16 synthetic observations.

Open the [audio and plots](../../abvid-artifacts/physics_engine/physics_v0_20261008/index.html),
read the [dataset audit](CAST_physics_data_audit.md), or use the
[reproduction commands](../experiments/physics_engine/README.md).
Original CAST code, frozen settings and historical result files remain intact.

## What was built

Reusable implementation: [`src/abvid/physics`](../src/abvid/physics).
Thin protocol layer: [`experiments/physics_engine`](../experiments/physics_engine).
Canonical numerical configuration: [`engine_v0.json`](../configs/physics/engine_v0.json).

```text
prescribed RPM + cylinder assumption → firing clock → finite-width pulses ─┐
prescribed torque → approximate excitation gain ─────────────────────────┤
seeded broadband excitation ─────────────────────────────────────────────┘
                                  ↓
                  damped exhaust resonance + highpass
                                  ↓
                      relative source pressure
                                  ↓ optional
               prescribed distance + retarded-time delay
                                  ↓
                      relative microphone pressure
```

This is an evenly firing four-stroke pulse/resonator approximation with an
explicit torque-to-excitation heuristic. It does not simulate combustion
chemistry, cylinder pressure, gas flow, exhaust geometry, tires or a cabin.
Absolute pressure and vehicle identity are not calibrated. The full equations,
sampling decisions, controls and omissions are in the [frozen protocol](../experiments/physics_engine/PROTOCOL.md).

The separate eight-second demo sweeps 1000–4000 RPM and 20–200 Nm while a source
travels from x=−60 m to +60 m, passing a microphone 5 m away. Its arrival-time
map creates kinematic Doppler and its pressure gain follows inverse distance.
No Doppler correction is applied twice. Listen to the
[source](../../abvid-artifacts/physics_engine/physics_v0_20261008/demo/source_playback.wav)
and [microphone](../../abvid-artifacts/physics_engine/physics_v0_20261008/demo/passby_playback.wav)
at a common playback gain.

## Fixed comparison

The official Procedural Engine Sounds release supplies synthetic audio plus
RPM/torque controls. The preselected 16 files passed the bounded audit; all
5,935 inventory/metadata records were checked. No existing IDMT, MELAUDIS,
AI4TEN-bank or protected audio was opened. These files belong to one
conservative development ancestry group; this is in-sample reconstruction.

Each arm uses the same centered two seconds, 8 kHz bandwidth and CAST
multi-resolution spectral objective. Physics receives the supplied controls
and an assumed cylinder count. CAST receives the observed audio. Physics fits
four controls using 54 candidates; CAST retains four starts and 300 Adam steps.
This matches observations and measurement, while retaining different information,
capacity and optimizer budgets. The current smooth8 fitter is included as a
second reference; CAST v15 population perturbations are not part of this fit.

| Arm | Mean checking spectral loss ↓ | Median loss ↓ | Mean envelope RMSE ↓ | Total fitting time |
|---|---:|---:|---:|---:|
| Physics defaults | 3.3684 | 3.3502 | 0.1012 | No fitting |
| Physics fitted | 2.0901 | 2.2094 | 0.0975 | 26.39 s |
| Original CAST v0 | **1.3898** | 1.4083 | 0.1258 | 259.56 s |
| Current CAST smooth8 | 1.3971 | **1.3499** | 0.1317 | 262.96 s |

Checking scores average two noise realizations excluded from parameter
selection. The fitted physics model improves on its default on all 16 clips,
by 25.2–63.5% checking loss. It beats each CAST reference on only one clip.
Its lower envelope error is a separate positive diagnostic; the spectral
objective still favors CAST on average. No perceptual listening score was
collected and no classifier was trained.

The observation-conditioned physical parameters are not identified engine
properties. All 16 fits choose the maximum allowed pulse width, 2 ms, and
15 choose the maximum noise gain, 0.12. Four fits retain multiple grid candidates
within 5% of the minimum loss: D/001, D/359, F/359 and H/001. A sole grid winner
also does not establish identifiability. The endpoint choices suggest a model
or search-range limitation; the grid was not expanded after inspecting results.

There were **zero implementation failures** in the 16-clip run. The four
ambiguity records and every boundary choice, candidate loss, CAST start and
history are retained. Do not interpret a numerical pass as physical validation.

Per-timbre descriptive mean losses show why pooled improvement should not be
overinterpreted:

| Author timbre source | Physics fitted | CAST v0 | CAST smooth8 | Selected observations |
|---|---:|---:|---:|---:|
| E1 | 1.1695 | 1.1212 | 1.1953 | 2 |
| E2 | 2.2970 | 1.4969 | 1.5098 | 4 |
| E3 | 2.1915 | 1.4007 | 1.3924 | 10 |

These are correlated synthetic configurations with shared ancestry and pooled
controls. Independent group confidence intervals, held-vehicle performance
and synthetic-to-real generalization cannot be estimated from this pilot.

## Verification and runtime

- **22/22 new tests passed:** firing timing/order, changing RPM, silence,
  torque scaling, deterministic streams, oversampling convergence, inverse
  distance/delay, analytic Doppler, corruption-before-decode, channel/control
  separation, paths, roles and known-fixture fitting.
- **88/88 focused CAST regressions passed**, with four archive tests excluded.
- **Exact replay passed:** 16 targets, 128 checking waveforms with their metrics,
  the pass-by demo, and three complete repeated fits (physics, v0, smooth8).
  The replay/refit pass took 36.09 seconds.
- Both full non-archive suites had only the collection-count canary failing:
  CAST environment **570 passed, 1 failed**; baseline environment **592 passed,
  1 failed**. Observed counts were 571 versus the stored 467 and 593 versus 489.
  These counts include concurrent CAST work in the shared checkout. The
  human-maintained baseline and frozen-file hash checks were preserved.
- Five additional entry-point checks rejected unknown config fields, changed
  grids, changed input-lock hashes, overwriting an existing run and output inside
  the checkout. All 118 HTML resource links resolve; 230 WAV headers are valid.
  The demo and first comparison plots were visually inspected.

One initial **test defect** was corrected after presenting a one-line patch and
the user's instruction to resume. The original assertion assumed that PCM16
quantization preserved exactly zero DC. It now checks that preprocessing
preserves the decoded audio's actual DC within 1e-6; telemetry leakage still
fails. Production preprocessing and existing tests were not altered. The
[reviewable patch](../experiments/physics_engine/test_correction_for_review.patch)
and [verification record](../../abvid-artifacts/physics_engine/physics_v0_20261008/verification.json)
preserve that history.

Recorded workflow time was **591.65 seconds**, including 36.26 seconds for
download/admission, fitting and plots through summary construction; import
startup, replay and final index/hash writing are separate. Peak process RSS
was 657,620,992 bytes (0.612 GiB). The demo's source and path computation took
**0.0471 seconds for eight seconds of audio**, excluding plotting/export. This
is an offline render measurement, not a tested streaming interface.

Hardware was an Apple M3 Pro, CPU execution with one Torch thread. Versions:
Python 3.11.13, NumPy 2.4.6, SciPy 1.17.1, Torch 2.13.0, SoundFile 0.14.0 and
Matplotlib 3.11.1. Timings are from one local run, not an isolated performance
benchmark. No environment or lockfile was updated.

Before merge, a human must review the new tests and regenerate the harness
baseline in accordance with [AGENTS.md](../AGENTS.md). That maintenance item
remains open; the full suite is not described as green. This prototype does not
merge changes or relax the canary.

## Artifacts and reproduction

External run: `artifact:physics_engine/physics_v0_20261008`.

| Artifact | Contents |
|---|---|
| [index.html](../../abvid-artifacts/physics_engine/physics_v0_20261008/index.html) | All audio comparisons, demo and diagnostic plots |
| [summary.json](../../abvid-artifacts/physics_engine/physics_v0_20261008/summary.json) | Losses, descriptive groups, scientific failures and measured runtime |
| [audit.json](../../abvid-artifacts/physics_engine/physics_v0_20261008/audit.json) | Full metadata reconciliation and bounded audio admission |
| [snapshot.json](../../abvid-artifacts/physics_engine/physics_v0_20261008/snapshot.json) | Resolved config, source hashes, commit/dirty state, environment and baseline preservation |
| [replay.json](../../abvid-artifacts/physics_engine/physics_v0_20261008/replay.json) | Exact numerical replay/refit result |
| [artifacts.json](../../abvid-artifacts/physics_engine/physics_v0_20261008/artifacts.json) | SHA-256 manifest of the original 298 run artifacts |
| `clips/<configuration>_<prefix>/` | Original and reconstructed audio, controls, parameters, histories, alternatives, failures and plot |
| `demo/` | Source/component/microphone audio, prescribed states, parameters and plot |

The baseline lock verifies original/current CAST numerical files, historical
reports, the frozen experiment config and `uv.lock` before and after execution.
All implementation changes in this task use new paths. The workspace comparison
also recorded a concurrent change to `CAST_ROADMAP.md`; this task did not edit
or revert it, and its changes do not affect the locked numerical references.

From the checkout, reproduce with a new output directory:

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/abvid-physics-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m abvid.physics run --config repo:configs/physics/engine_v0.json --output artifact:physics_engine/physics_v0_reproduction --download
```

Then run the exact replay command in the [README](../experiments/physics_engine/README.md).
Raw data and generated artifacts remain outside the checkout. Downloaded audio
and its derivatives retain the source's CC BY-NC license conditions.

## Consequence for the next iteration

The engineering bridge from prescribed operating state to testable acoustic
output now exists. CAST remains the stronger spectral reconstruction reference
in this comparison. The present evidence supports checking the systematic
spectral residual and recording coloration before adding a larger source model;
it does not show that a wider parameter grid would identify real physics.

BeamNG can later supply richer operating trajectories, subject to a verified
telemetry/audio adapter. It is not required to continue numerical development,
and its synthesized audio would not replace real physical validation. That next
validation needs synchronized real RPM/audio, known engine configuration and
microphone geometry, with independent recording sessions held out. No such
data was manufactured or claimed in this prototype.

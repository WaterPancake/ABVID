# Simple physics engine development protocol

Frozen before selected audio decode and fitting, 2026-10-08. This additive
prototype answers whether a small state-driven source can run, obey its stated
numerical relationships, and be compared transparently with CAST. It does not
advance CAST classification transfer or replace any historical benchmark.

## Data and admission

Use the official Procedural Engine Sounds release at immutable Hugging Face
revision `e57c44ab64e8528a7b37fc693b553bf4b66440e7`. The full remote inventory
contains 5,935 WAVs. Check every metadata filename, size, duration and sample
rate against that inventory. Decode only the 16 files in `input_lock.json`:
the lexicographically first and midpoint common filename prefixes (`001`,
`359`) in each of eight configurations, selected without inspecting audio.
Download budget is 100 MB; the selected files total 74,767,552 bytes.

Verify LFS SHA-256 or Git blob hashes and exact byte counts before decoding.
Require 48 kHz, PCM16, four channels, finite samples and at least two seconds.
Channels 1–2 are audio; channels 3–4 are RPM times 0.0001 and torque times
0.001. Check decoded control statistics against all corresponding published
per-file statistics within 1e-6. Record clipping and exact control-trace hashes.
No failed file is replaced. Full-corpus audio integrity is outside this audit.

All selected files have one conservative synthetic-development role. The
author describes pooled controls reused across configurations and three timbre
engines, with several microphones on the same engines. Filename prefixes are
candidate links, not session IDs. Preserve `recording_session=unknown` and
`original_control_session=unknown`; retain the whole release as one ancestry
group regardless of whether sampled telemetry hashes match exactly. There is
no train/test split or independent group confidence interval in this prototype.

The release is CC BY-NC 4.0. Preserve LICENSE, CITATION.cff, README and USAGE
alongside external data. This is local noncommercial research; audio and
derivatives are not included in the checkout or published.

Existing IDMT and MELAUDIS metadata/reports may be read to assess available
controls. No IDMT, MELAUDIS, supplied synthetic-bank, military or protected audio
is opened by this workflow. BeamNG documentation is a capability audit, not an
admitted dataset or a running simulator connection.

## Source and path model

The four-stroke firing rate is `RPM * cylinders / 120` events per second.
Integrate that clock to place fractional-sample pulses. Four cylinders are
assumed for A, eight for B–H, following the author's timbre-source descriptions;
these are assumptions, not identified engine specifications of every synthetic
clip. Uniform firing ignores bank offsets, firing-order asymmetry and misfires.

Each pulse excites two causal exponential filters, approximating
`h(t) = t * exp(-t/tau) / tau^2`. A second-order resonator uses
`H(s) = w0^2 / (s^2 + w0*s/Q + w0^2)`, discretized by the bilinear transform.
A 20 Hz highpass removes DC. Seeded white noise passes through the same
resonator. Excitation gain is `0.15 + max(torque_Nm, 0)/400`, a declared
heuristic; negative torque is overrun at the idle floor, not reverse combustion.
Source pressure is uncalibrated and relative. There is no per-component RMS
normalization, absolute sound-pressure prediction, fuel/air/thermal model,
intake geometry, tire contact, cabin transfer or engine identity inference.

Render at 64 kHz and downsample to 16 kHz using polyphase Kaiser-5 filtering.
Prepend 0.25 seconds of constant initial controls to settle causal filters,
then discard it. The prescribed states control the renderer; no learned state
estimator is involved.

A separate demo uses the free-field approximation `p(tr)=s(te)/r(te)` with
`tr=te+r(te)/343`, linear interpolation and zero outside emitted support.
The moving trajectory produces kinematic Doppler through its arrival-time map;
no second Doppler operation is applied. Convective amplitude corrections,
reflections, ground, air absorption and directivity are absent. The two-second
dataset comparison uses an identity path, because its timbre already includes
recording coloration. The demo's geometry is prescribed, never fitted to data.

## Comparison and budgets

Use unchanged CAST preprocessing on audio channels 1–2: arithmetic mean,
48→8→16 kHz, centered two seconds, no padding, DC removal/unit RMS for the loss.
Interpolate separate telemetry at the exact cropped 16 kHz sample times.
Apply an 8 kHz bandwidth route to the physical waveform too. Use the unchanged
CAST multi-resolution STFT objective: FFT sizes 256/512/1024/2048, periodic Hann,
hop N/4, no centering, mean relative magnitude plus mean log magnitude error,
equal resolution weights, epsilon 1e-5. Report per-resolution terms and 50 ms
normalized RMS-envelope error.

| Arm | Information | Optimization |
|---|---|---|
| Physics default | Released RPM/torque; assumed cylinder count | Fixed pulse 0.8 ms, resonance 400 Hz, Q 2, noise gain 0.03 |
| Physics fitted | Same controls plus observed audio | Full 54-point grid: pulse [0.25,0.8,2] ms × resonance [150,400,1000] Hz × Q [0.7,2] × noise [0,0.03,0.12] |
| CAST v0 | Observed audio | Original unmodified renderer/fitter/config, four starts, 300 Adam steps |
| CAST smooth8 | Observed audio | Current parity-checked smooth-noise renderer/fitter, four starts, 300 Adam steps |

Physics uses two fixed fitting realizations and two disjoint checking
realizations, derived from SHA-256 of seed 42, input ID, component, purpose and
index. Select the minimum fitting loss, first candidate on ties; never select
on checking loss. Preserve all 54 losses, fitted parameters and candidates
within 5% of the best fitting loss. Report boundary choices, ambiguity and less
than 5% checking improvement as scientific diagnostics. These do not justify
expanding the grid after seeing results. Defaults are included in the grid.

Inputs, bandwidth and loss are matched. Control access, capacity and optimizer
budgets differ; this is a reconstruction reference, not an equal-compute causal
comparison or an RPM recovery contest. CAST v15 population perturbations are
not involved: the current smooth8 *fitter* is the relevant reference here.
No lower-loss or classification threshold is required for this engineering
prototype to be scientifically complete.

## Verification and artifacts

Analytic tests cover constant/changing RPM firing timing, engine-off silence,
torque scaling, order frequency, deterministic/noise-separated rendering,
oversampling convergence, inverse-distance pressure/delay and analytic Doppler.
Data guards cover corrupted bytes before decode, channel separation/control
alignment, path traversal, fixed selection and role mutation. A small known
fixture verifies fitting improvement, best-state selection and exact replay.
Existing CAST numerical and role checks remain unchanged.

Every run uses a new external artifact directory and retains the config,
input lock, source snapshot, commit/dirty state, dependencies, baseline hashes,
all audio, state traces, fitted alternatives, losses, failures, plots and runtime.
Float audio retains numerical waveforms. Listening exports restore target AC
RMS and apply one common gain across the complete comparison set per clip.
The pass-by source and microphone also share one playback gain.

Replay must reproduce every saved target, both checking waveforms for all four
arms and every metric exactly in the recorded environment, plus the demo.
`--refit-first` repeats complete physics, CAST v0 and smooth8 fits for the first
selected file, requiring equality of all numerical results except wall time.
No audio is accessed from paths absent from the fixed selection.

The repository's human-only harness baseline is not edited by this work.
Legitimate new tests change its collection count and require human review and
regeneration before merge. This is reported separately from functional tests.

## Sources

- Robin Doerfler, [Procedural Engine Sounds Dataset v1.0](https://doi.org/10.5281/zenodo.16883336), 2025.
- [Official pinned dataset card](https://huggingface.co/datasets/rdoerfler/procedural-engine-sounds/blob/e57c44ab64e8528a7b37fc693b553bf4b66440e7/README.md): channels, scaling, release and license.
- [Author project page](https://rdoerfler.github.io/procedural-engine-sounds-page/): three engines, eight capture configurations and pooled controls.
- Doerfler and Wyse, [Analysis-Driven Procedural Generation of an Engine Sound Dataset with Embedded Control Annotations](https://arxiv.org/abs/2603.07584), EUSIPCO 2026, pp. 221–225. The present engine is our simple pulse/resonator model, not a reproduction of their synthesizer.

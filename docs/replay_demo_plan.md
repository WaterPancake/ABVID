# ABVID inspectable replay demo plan

Date: 2026-09-20  
Scope: offline replay of Milestones 1-6; no live-surveillance or Milestone 7 claim

## Visitor experience

The demo tells one complete story: choose a reviewed vehicle event, hear what one
microphone receives, inspect how the model processes overlapping windows, and compare
the prediction with the human-reviewed label.

```text
source and provenance       signal controls          model inspection
---------------------       ---------------          ----------------
recording/session           original/corrupted       waveform + log-Mel
vehicle and condition  ->   SNR/background/mic  ->  window probabilities
reviewed interval           deterministic seed       event aggregation
license and split role                               timing + limitations
```

The first screen should open with a development/demo recording and a known-good
configuration. A second curated example should expose a model failure. Reserved
T90M/JLTV recordings must never appear in the demo catalog.

## UI

Use a local Streamlit application under `demo/`, installed through an optional
`demo` dependency group. It provides a compact Python-only path to audio playback,
plots, controls, caching, and a one-command local launch.

The page has four regions:

1. **Scenario** — reviewed source, recording session, vehicle class/model, operating
   condition, interval, license, attribution, and explicit role badge.
2. **Acoustic controls** — original/corrupted toggle, SNR, background recording,
   microphone response, and deterministic seed. Every change shows the exact
   transformation metadata.
3. **Inside the model** — synchronized playback cursor, waveform, log-Mel image,
   overlapping 2-second windows, audibility state, tracked/wheeled scores, and event
   aggregation. Scores are labeled uncalibrated unless a calibration experiment says
   otherwise.
4. **Evidence** — expected label, final decision or abstention, processing latency,
   model/checkpoint hash, dataset version, test domain, and links to the benchmark
   card. A failure example remains visible rather than being filtered from the demo.

An optional second page visualizes the existing four-microphone simulator: microphone
positions, true and estimated bearing, per-channel waveform, and beamformed result.
It must carry a persistent `SIMULATED ARRAY` label.

## Implementation boundary

Reuse the augmentation engine for deterministic corruption and the existing event
inference windowing/output-head path. Add a demo adapter rather than embedding UI
logic in research modules. The adapter should emit a stable JSON record containing:

- scenario/source ID, session, reviewed interval, split role, and provenance;
- corruption configuration and seed;
- waveform and spectrogram display data;
- window start/end, RMS, audibility state, and class scores;
- aggregation rule, event score, decision/abstention, and processing time;
- model, checkpoint, manifest, and code-version identifiers;
- `is_vehicle_presence_detector: false` and the explicit test domain.

The current gate measures each window relative to the peak of the complete recording.
It is acceptable for offline replay and must be described that way. Live microphone
capture, causal normalization, vehicle-presence detection, false alarms per hour,
real-array localization, and calibrated operational confidence are later work.

## Delivery sequence

### Phase 1 — deterministic replay shell

- Build a demo-only catalog from admitted development sources and reject protected
  source IDs at load time.
- Load one reviewed interval, replay original audio, and render waveform/log-Mel.
- Apply one deterministic corruption configuration and expose its metadata.

Acceptance: the same source/configuration/seed produces byte-identical audio and the
same plot data on repeated CPU runs.

### Phase 2 — inspectable inference

- Load one explicitly versioned development checkpoint.
- Display overlapping windows and event aggregation from the existing inference
  implementation.
- Record wall time, model/hash, and test-domain labels.
- Include one success and one failure scenario.

Acceptance: the displayed event result and saved JSON agree with the CLI result for
every bundled scenario.

### Phase 3 — benchmark and simulated-array views

- Add the controlled-corruption comparison across selected SNR values.
- Add the explicitly simulated array/localization page from checked Milestone 4
  artifacts.
- Link every displayed number to a versioned result artifact.

Acceptance: native-real, controlled-corruption, and simulated-array panels cannot be
mistaken for one another and no reserved source is loaded.

### Phase 4 — portfolio packaging

- Add `uv run` startup, a redistribution-cleared small sample pack, screenshots, and
  a short recorded walkthrough.
- Test clean installation and CPU inference on the demonstrated Mac.
- Document expected startup time, per-event latency, known failures, and unsupported
  claims.

Acceptance: a reviewer can clone, launch, replay a bundled example, inspect its
provenance, and reproduce the saved inference without editing paths.


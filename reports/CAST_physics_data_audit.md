# CAST physics data audit

2026-10-08. The available public RPM data is sufficient to build and debug a
state-driven prototype. It does **not** provide independent validation of real
engine acoustics. Keep CAST as the observation-fitting reference and retain the
existing ABVID datasets in their established roles.

## Candidate decisions

| Candidate | What was checked | RPM and torque readiness | Decision |
|---|---|---|---|
| IDMT V1 | 4,413 admitted car/truck metadata records and provider naming documentation | No explicit RPM or torque annotations; filename road-speed tokens do not establish crankshaft speed | Retain source-domain observation role; no audio opened for this audit |
| MELAUDIS V1 | 8,066 admitted metadata records | No explicit RPM or torque annotations | Retain exposed target-development role; no audio/features opened |
| Released AI4TEN procedural bank | 10,000 admitted metadata records | No per-file RPM/torque or complete source-asset/run provenance in the admitted manifest | Preserve historical baseline; do not invent state labels |
| Released AI4TEN AudioLDM bank | 382 admitted metadata records | No explicit RPM/torque annotations | Preserve historical baseline |
| Procedural Engine Sounds v1.0 | Full remote inventory and all 5,935 metadata rows; 16 decoded WAVs | Separate, time-indexed RPM/torque channels in synthetic audio files | Admit the locked subset for synthetic development and reconstruction only |
| BeamNG | Official telemetry and audio documentation | RPM, load and vehicle-state access exists; a synchronized audio-capture setup was not established here | Candidate future trajectory supplier and synthetic comparator; no integration or dataset admission claimed |

The existing metadata audit verifies the original manifest SHA-256
`ee8b67b002a4528b036e6d2a3ab6f8f6015cd59fffbbc27214ad2786a7d31f96`.
Its 22,861 records have no explicit RPM/engine-speed or torque fields. This is
a statement about the available admitted metadata, not proof that no external
annotations exist. Original null/unknown sessions remain unchanged. See the
[saved metadata audit](../../abvid-artifacts/physics_engine/metadata_audit_20261008_complete.json)
and [prior admission report](dataset_admission_provenance.md).

## Public release audit

Official release: Robin Doerfler,
[Procedural Engine Sounds Dataset](https://doi.org/10.5281/zenodo.16883336),
CC BY-NC 4.0. The
[pinned dataset card](https://huggingface.co/datasets/rdoerfler/procedural-engine-sounds/blob/e57c44ab64e8528a7b37fc693b553bf4b66440e7/README.md)
defines two audio channels followed by RPM and torque channels, scaled by
0.0001 and 0.001 respectively. The implementation keeps telemetry out of the
audio downmix and preserves the release license and citation files externally.

Measured from the pinned inventory and downloaded metadata:

- Revision: `e57c44ab64e8528a7b37fc693b553bf4b66440e7`.
- 5,935 WAVs and 20 metadata/documentation files; 26,277,434,215 bytes total
  (26.28 decimal GB, approximately 24.47 GiB).
- A–D contain 767 files each; E/G/H contain 717 each; F contains **716**.
  The approximate per-set count in the card should not replace this inventory.
- Durations sum to 19.007536 hours. All published numeric fields are finite.
  Metadata ranges are 0–7008.057 RPM and −107.025–718.536 Nm; these extrema
  were not verified by decoding the entire release.
- The predeclared selection is prefixes `001` and `359` from every set:
  16 files, 74,767,552 bytes. It was frozen before opening audio, with no
  replacement based on sound quality or fitting results.

All 16 selected files passed byte-size and SHA-256 verification before decoding,
48 kHz/four-channel/PCM16 checks, finite samples, duration and control-domain
checks. Their decoded statistics agree with the published per-file metadata to
at most 2.274e-13 absolute error. No sampled audio reaches the declared clipping
threshold. The admission/download pass took 36.26 seconds on this machine.
See [audit.json](../../abvid-artifacts/physics_engine/physics_v0_20261008/audit.json)
and the [frozen selection](../experiments/physics_engine/input_lock.json).

PCM16 implies approximately 0.305 RPM and 0.0305 Nm per quantization step under
these scales. Sample-indexed release controls do not establish the precision,
calibration or synchronization accuracy of the original physical sensors.

## Dependence and physical interpretation

The [author's current description](https://rdoerfler.github.io/procedural-engine-sounds-page/)
identifies three timbre engines across eight capture configurations and a
separately pooled collection of operating-state traces. Several configurations
share an engine, and controls can be reused with different timbres. These are
not eight independently recorded vehicles.

The 16 sampled complete control arrays have different exact hashes. Equal
filename prefixes can nevertheless have related or entirely different state
ranges: `001` is not a reliable shared manoeuvre identifier across all sets.
Exact inequality therefore does not establish independence. Every derivative
retains one conservative release-level development group, the published timbre
engine/configuration, its original filename, and unknown original sessions.
No random clip split or independent generalization result is produced.

Four cylinders for A and eight for B–H are prototype assumptions based on the
described timbre sources. The synthetic generation and recording coloration
mean that fitting a pulse width or resonance to these observations does not
identify combustion duration, exhaust dimensions or a dry physical source.
The original paper is Doerfler and Wyse, *Analysis-Driven Procedural Generation
of an Engine Sound Dataset with Embedded Control Annotations*, EUSIPCO 2026,
pp. 221–225; the present prototype is a separate model, not its reproduction.

## BeamNG and the remaining data need

[BeamNGpy's Electrics interface](https://documentation.beamng.com/api/beamngpy/master/beamngpy.html#electrics)
lists RPM, engine load, throttle, gear and wheel speed. These could drive
prescribed operating trajectories. Normalized load must not be silently treated
as torque in Nm; an adapter needs an explicit conversion or a suitable torque
signal, plus simulation timestamps and source-state provenance.

[BeamNG's engine audio documentation](https://docs.beamng.com/modding/vehicle/sections/sounds/engine_audio/)
describes blended sample sets, engine/exhaust layers, load-dependent gains and
filters. Its audio would provide another synthetic reference, not independent
combustion-acoustic ground truth. This audit did not run BeamNG, verify an audio
export path or establish redistribution rights for captured game audio.

The next physical-validation requirement is a small real recording collection
with synchronized RPM and audio, known engine/cylinder configuration, microphone
position and a documented operating-state signal. Separate recording sessions
must remain separate throughout evaluation. RPM alone will not identify every
acoustic mechanism, but it makes a direct timing test possible. Public synthetic
controls let implementation proceed before that collection exists.

## Reproduce

The [prototype command](../experiments/physics_engine/README.md) reproduces the
pinned inventory/metadata checks, bounded audio audit and CAST comparison.
After those inputs are cached, reproduce the broader metadata-only audit with
a new output filename:

```sh
.venv/bin/python -B -m abvid.physics.candidates --output artifact:physics_engine/metadata_audit_reproduction.json
```

No original ABVID dataset, split, environment, artifact or CAST implementation
was changed. Full-release audio admission and independent real-data physical
validation remain outside the evidence established here.

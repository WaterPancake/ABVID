# Dataset readiness audit

Date: 2026-09-28. Protocol assessed: `sim_components_v1.0`.

Subsequent change: the user-authorized [v1.1 amendment](../experiments/AMENDMENT_v1.1.md) corrects sensor selection and removes motorcycle from the core task. This report and its JSON record the original v1.0 findings. The revised six-group car/truck subset still fails the retained independent-group split floors; see the amendment for current readiness.

**All three dataset families are present under `dataset/`, but the frozen three-class experiment is not executable.** IDMT's microphone/channel mapping contradicts the protocol, and its motorcycle recordings cannot satisfy the required independent-group support. Published replay also needs assets absent from the supplied AI4TEN directory. These are data-admission findings, not experimental results.

The audit inspected filenames, provider documentation, archive inventories/checksums, WAV headers and NPY headers. It did not read waveform samples or feature-array values, listen to target recordings, fit models, assign split roles, or modify the frozen protocol. Detailed counts, input-document hashes and limitations are in [dataset_readiness_audit.json](https://github.com/WaterPancake/ABVID/blob/77daa1b339906d35b6fa0a59a166ecf2eac01174/experiments/dataset_readiness_audit.json).

## Inventory and integrity

| Dataset | Supplied contents | Verified in this audit | Remaining uncertainty |
|---|---|---|---|
| IDMT-Traffic | 17,506 WAV files, annotation lists and provider metadata script | Every entry in `idmt_traffic_all.txt` exists; no unlisted WAVs, duplicate list entries or unparsed filenames. All WAV headers readable. | Original ZIP checksum not checked against this expanded directory; per-file hashes and decoded-audio duplicate checks remain outstanding. |
| MELAUDIS | Vehicles and background RAR archives | Both archive MD5s match the release values recorded in DATASET_PROTOCOL §1. Vehicles archive lists 13,668 WAVs without duplicate member paths. | Audio not extracted; headers, event duplication and original-session reconstruction remain unaudited. |
| AI4TEN | 15,000 procedural-labelled WAVs, 601 AudioLDM-labelled WAVs, four feature/scaler arrays and metadata | All WAV headers readable; NPY header dimensions consistent with supplied metadata. | Original release integrity, rendering provenance, source/run grouping and historical feature-row alignment remain unverified. |

MELAUDIS archive checksums:

- `MELAUDIS_Vehicles.rar`: `0bf2cf3ad13436ec8bf060b378508252` — match.
- `MELAUDIS_ BG.rar`: `1b1cfe39b364dd34d32ba5c864561baa` — match. This background archive is not needed for the frozen first phase.

Header observations: IDMT is 48 kHz stereo FLOAT, with 13,904 files containing 96,000 frames and 3,602 containing 96,001 frames. AI4TEN procedural-labelled files are 22,050 Hz mono PCM16, 110,250 frames (5 s); AudioLDM-labelled files are 16 kHz mono PCM16, 81,952 frames (5.122 s). Header readability does not establish complete payload integrity or usable acoustic content.

## IDMT: two blockers for E0 and E1

### The frozen sensor selector needs correction

[DATASET_PROTOCOL §2](../experiments/DATASET_PROTOCOL.md) specifies “CH12/sE8.” The supplied [IDMT README](ARTIFACT_INDEX.md#artifact-c7112509e13e), under “File naming convention,” identifies `SE` as sE8 and explicitly illustrates an `SE_CH34` car recording. Channel number cannot be used as a uniform proxy for microphone type in this release.

Observed non-background counts for the intended three classes:

| Provider label | Car | Truck | Motorcycle |
|---|---:|---:|---:|
| `SE_CH34` | 3,902 | 511 | 5 |
| `SE_CH12` | 0 | 0 | 246 |
| `ME_CH12` | 3,902 | 511 | 179 |

Thus a literal `SE AND CH12` filter retains motorcycles only. This is an error in our frozen specification; the audit has not silently changed it. Selecting all `SE` files is used below solely to diagnose whether correcting that error would make the design feasible. Microphone identity remains a provider label, not an independently measured sensor response.

### Correcting the selector does not fix independent-group support

Applying the protocol's conservative site/calendar-date grouping to `SE` candidate files gives:

| Site/date group | Car | Truck | Motorcycle |
|---|---:|---:|---:|
| Fraunhofer-IDMT / 2019-10-22 | 302 | 10 | 3 |
| Fraunhofer-IDMT / 2019-10-23 | 254 | 14 | 2 |
| Hohenwarte / 2020-08-29 | 0 | 0 | 246 |
| Langewiesener-Strasse / 2019-11-18 | 824 | 146 | 0 |
| Langewiesener-Strasse / 2019-11-19 | 1,365 | 152 | 0 |
| Schleusinger-Allee / 2019-11-12 | 491 | 79 | 0 |
| Schleusinger-Allee / 2019-11-13 | 666 | 110 | 0 |
| **Total** | **3,902** | **511** | **251** |

Under DATASET_PROTOCOL §§4–5, seven groups yield five training, one validation and one test group. This already violates the minimum of two groups in each validation/test partition. Motorcycle support creates a stronger obstruction: 246/251 clips (98.0%) belong to Hohenwarte on one date. Satisfying 200 training motorcycles requires that group, leaving only five motorcycle clips for both validation and test, each of which requires at least 20. No alternative seed fixes this under the frozen grouping rule.

Our interpretation: motorcycle label and acquisition context are heavily confounded in this candidate subset. This does not establish which acoustic cues a classifier would use, and it does not show that IDMT is unusable for every task. It does establish that this particular three-class split is infeasible. Splitting neighbouring clips or paired microphones across roles would not create independent sessions.

The provider [metadata script](ARTIFACT_INDEX.md#artifact-0eb8559945a9), function `import_idmt_traffic_dataset` docstring, calls `speed_kmh` the **site speed limit** and `weather` the dry/wet road condition. Do not promote these to measured vehicle speed or meteorological measurements. The script was read, not executed.

## MELAUDIS: target candidates present, admission incomplete

Filtering archive filenames to `FF` and exactly `1V-Car`, `1V-Truck` or `1V-MC` gives 8,348 candidates: **7,854 car, 256 truck and 238 motorcycle**. These span 22 distinct location-token/date combinations. These are metadata-derived candidate counts, not final admitted events or verified independent sessions.

Before assigning `M_DEV` and `M_LOCK`, reconcile site aliases and same-road acquisition relationships, verify original-session/event provenance and duplicates, inspect integrity metadata, and apply the frozen support floors. Different location tokens do not by themselves prove independent recording sessions. The JSON retains group/class counts for that review. No target split has been selected or acoustically explored.

## AI4TEN: audio present, replay and grouping prerequisites missing

Each `synthetic/pyroadacoustics` class contains 5,000 WAVs. `synthetic/audioldm` contains 201 car, 200 truck and 200 motorcycle WAVs. The extra car file relative to the README's stated 200/class is recorded, not deleted or presumed a duplicate. Directory names identify the release's labels; they do not verify the actual renderer.

The supplied [README](ARTIFACT_INDEX.md#artifact-d6e3ee81f6ab), “Contents,” describes CNN snapshots, configurations and scripts. Those directories are absent from the supplied tree. Therefore E2-P-R/E2-P-S replay cannot run from this tree alone. Recover the checksum-pinned release assets identified in DATASET_PROTOCOL §1 and verify snapshot/scaler/feature identities and row order before any replay. This is a missing-local-assets finding, not a claim that the assets are unavailable from the public release.

The [feature metadata](ARTIFACT_INDEX.md#artifact-07636fef48ca) describes 1,019 historical DATASEC+MAVD evaluation examples. `X_test.npy` has header shape `[1019,216,120,1]`; `y_test.npy` has shape `[1019]`. Their values were not inspected. Shape agreement alone does not establish label/feature alignment or original recording groups, and these historical features are not the raw MELAUDIS target partition.

No generation sidecars establish source-template/run lineage in the supplied synthetic tree. Strict E2-B requires grouped synthetic train/validation selection; indexed filenames alone cannot establish independence. This prerequisite remains unresolved under DATASET_PROTOCOL §5.

## Disposition

- **E0/E1:** blocked under v1.0 by the IDMT selector and independent-group/class support.
- **E2-P replay:** local snapshots/configuration assets and verified feature-row alignment still required.
- **E2-B:** source/run grouping and full corpus admission still required.
- **E3 controlled ablations:** remain design-only; upstream baseline/data-admission requirements have not passed.

The next design step is a versioned protocol amendment based on these metadata findings: correct the sensor rule and resolve the class/session incompatibility through an explicitly revised dataset/task/split design or additional independent recordings. Preserve group separation; do not lower it to random clip splitting. Any amendment should precede modelling and target performance inspection. The original three protocol deliverables remain unchanged, and no experiments have been implemented or run.

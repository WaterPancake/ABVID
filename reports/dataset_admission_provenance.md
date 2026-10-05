# Dataset admission: provenance and H1 readiness

2026-10-02. The requested audit is complete for a **qualified, fixed-head H1 gap study**. Original parent recording IDs, physical-vehicle identities and synthetic generation lineage remain unknown. They are not silently filled in. No model scores were used to select or exclude data.

## Disposition

The previous [integrity report](dataset_admission_integrity.md) remains historical evidence for steps 1–2. The new [audit summary](ARTIFACT_INDEX.md#artifact-8c9a6948fa0b), [executed rules](ARTIFACT_INDEX.md#artifact-063389af6155), [group map](ARTIFACT_INDEX.md#artifact-d2eddf544d99) and [admitted manifest](ARTIFACT_INDEX.md#artifact-76215e0623b7) resolve known links and declare the remaining limitations.

| Corpus | Retained car | Retained truck | H1 role |
|---|---:|---:|---|
| AI4TEN_audioldm | 192 | 190 | Whole released bank, training only |
| AI4TEN_pyroadacoustics | 5,000 | 5,000 | Whole released bank, training only |
| IDMT | 3,902 | 511 | Six source groups, leave-one-group-out |
| MELAUDIS | 7,810 | 256 | Four connected target groups, exposed development |

Largest common budget: **190/class**. IDMT's smallest five-group pool has 2,537 cars and 359 trucks; procedural has 5,000/class; AudioLDM has 192 cars/190 trucks. No synthetic validation is held out because all H1 settings are fixed. Each bank stays in one training-only group. This removes an unsupported synthetic validation claim rather than inventing independence among files.

## What the audit did

All 46,775 indexed file relationships were retained, including excluded classes/sensors. Rehashed and fingerprinted 46,771 integrity-valid, ≥2 s waveforms; the remaining four are short/invalid under the earlier audit. Known same-recording, synchronized-microphone, native exact-equality and original-time-overlap links connect groups before class filtering.

A bounded, class-independent search retrieved aligned center-window neighbors using fixed random projections, gain-normalized hashes and both polarities, within and across datasets. It considered 1,546,076 candidate pairs. All 112 accepted 2 kHz matches also passed ≥.9999 absolute normalized correlation at 8 kHz across the entire center 2 s. No cross-dataset reuse was detected by this search. This does not exclude arbitrary shifted excerpts, codec/time-stretch variants, hidden parent recordings or pretraining overlap. The preliminary v1.3 search and v1.3.1 full-common-band verification are both retained; their admissions agree.

The audit records 5,852 overlapping-time components. Merely overlapping windows are linked, not automatically treated as label errors or as the same physical vehicle. Interval positions follow provider semantics: IDMT centered sample offsets within a candidate recording, MELAUDIS event-clock tokens within a same-location/date axis. Invalid clock tokens were already quarantined without guessed repairs.

Additional exclusions beyond v1.2.1: **11 AudioLDM car/truck excerpts** belong to equivalent-waveform groups carrying conflicting vehicle annotations; **one MELAUDIS car** is a same-label duplicate. The exact groups and retained/excluded file IDs are in [duplicate_components.json](ARTIFACT_INDEX.md#artifact-31242aae68e0) and [additional_exclusions.jsonl](ARTIFACT_INDEX.md#artifact-eafad5bfad22). No class was corrected by guessing. AudioLDM car 0201 remains retained.

## Groups and uncertainty

| IDMT site/date | Car | Truck |
|---|---:|---:|
| Fraunhofer-IDMT 2019-10-22 | 302 | 10 |
| Fraunhofer-IDMT 2019-10-23 | 254 | 14 |
| Langewiesener-Strasse 2019-11-18 | 824 | 146 |
| Langewiesener-Strasse 2019-11-19 | 1,365 | 152 |
| Schleusinger-Allee 2019-11-12 | 491 | 79 |
| Schleusinger-Allee 2019-11-13 | 666 | 110 |

These are six site/date acquisition groups across three locations, not six independent locations or six known physical vehicles. The provider's source-center and site/date metadata are sufficient for conservative grouping, while original recording IDs remain unknown. All synchronized microphone variants remain grouped; only SE_CH34 enters modelling.

MELAUDIS uses whole calendar days across locations, then connected waveform reuse across dates. Numbered street locations are preserved but never promoted to independent sessions. Nine dates join through reuse links (including excluded motorcycle excerpts): 2023-08-01, 08-02, 08-04, 08-05, 09-05, 09-06, 10-11, 10-15 and 11-08. That component contains 6,290 cars/187 trucks. The other three groups are 2024-01-16 (468/32), 2024-01-17 (626/36) and 2024-02-09 (426/1).

Only four conservative target groups remain, and one dominates event count. Group-bootstrap intervals are conditional/descriptive with uncertain coverage. H1 must report per-group outcomes and class support rather than treat 8,066 clips as independent acquisitions. Whole-day grouping may overmerge genuinely distinct recordings; changing that requires original recording evidence, not an attempt to narrow confidence intervals.

## Missing originals and synthetic lineage

The MELAUDIS descriptor documents manual event annotations in Excel, segmentation from parent video audio (Data Annotation and Dataset Building, Figure 4), and mono/stereo filename device annotations (Data Features). The [current Figshare listing](https://api.figshare.com/v2/articles/27115870) contains only the two RAR archives; no parent-video map or event workbook is supplied. The original [descriptor](https://www.nature.com/articles/s41597-025-04689-3) supports the metadata meanings, not our inferred grouping policy. Provider iPhone6/iPhone12 annotations are kept separately from measured WAV channel count.

The local P02 release's `_2_genSyntheticData.py` permits a renderer fallback and derives per-file seeds from Python hash; source code alone cannot recover the actual released runs. No complete per-WAV prompt, source/template, seed, run or backend sidecars were found. [Metadata/code hash inventory](ARTIFACT_INDEX.md#artifact-eed9788a5870) and saved provider file listings document coverage. No new synthesis was run.

Useful optional future requests: original MELAUDIS Excel event annotations, filename-to-parent-video/audio map and recording start/end/device logs; synthetic per-WAV prompt/seed/source-asset/run/backend manifest. The user has supplied the public downloads. No further fetch or author request is needed for this qualified H1; no author was contacted. Missing metadata still blocks source-family generalization or backend-specific causal claims.

## Reproducibility

Run `experiments/reproduction/.venv/bin/python experiments/admission/audit_provenance.py --output <new-directory>` with four audit threads. Runtime for v1.3.1 was 63.18 s on the local M3 Pro. The script, rules, manifests, source file hashes, releases and summary are saved; originals are unchanged. Eighteen admission regression tests cover integrity failures, metadata/channel parsing, conservative unions, original-time overlaps and gain/polarity/resampling duplicate checks.

The [new protocol](../experiments/EXPERIMENTAL_PROTOCOL.md) specifies H1; its exact selections are materialized by `experiments/h1/freeze.py`. All previous audit snapshots, R0 evidence and military evaluation protections remain unchanged. Qualified admission does not certify human-audible label correctness or independent physical-source identity.

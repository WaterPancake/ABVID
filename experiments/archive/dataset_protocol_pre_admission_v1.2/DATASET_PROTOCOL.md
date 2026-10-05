# Dataset protocol: acoustic simulation component study

**Current priority, 2026-09-28:** the custom study below is deferred while we [reproduce published results on their original splits](reproduction/README.md). Its class filters and proposed partition rules do not apply to historical reproduction. See the [current reproduction report](../reports/published_reproduction_status.md) for checkpoint acquisition, inference, and exposure of historical test data. Statements below about pending downloads or metadata-only inspection describe the earlier design freeze, not the current reproduction status.

Version: `sim_components_v1.1`, 2026-09-28. Status: **channel/class amendment recorded; execution blocked by IDMT group support; final manifests not materialized**. See [amendment v1.1](AMENDMENT_v1.1.md) and the preserved [v1.0 archive](archive/sim_components_v1.0/README.md). This document authorizes no execution. The companion [experimental protocol](EXPERIMENTAL_PROTOCOL.md) defines the experiments and decision rules; [experiment matrix](EXPERIMENT_MATRIX.csv) lists their data roles.

## 1. Scope and freeze boundary

The core task is closed-set, single-event classification into **car and truck**, using public civilian recordings. These are broad categories, not vehicle models or physical identities. There is no background output, vehicle-presence claim, congestion classifier, or military-model experiment. Existing ABVID training, demo and reserved recordings are excluded entirely. Existing tracked/wheeled synthetic profiles must not be relabelled as these civilian categories. Motorcycle is excluded from every core training, validation and test role in E0/E1/E2-B/E3, with exclusion reason `class_out_of_scope_v1.1`; raw files are retained. E2-P-R/S remain separate historical three-class snapshot replays, with no new fitting, and are not numerically comparable to the binary core.

Exact releases, eligibility rules, partition algorithms and budgets are specified below. A [metadata audit](../reports/dataset_readiness_audit.md) has established candidate counts and header readability in the supplied `dataset/` tree. File-level SHA256 manifests, final eligible counts, original-session recoverability and some historical evaluation metadata remain **unverified**. They are explicit execution prerequisites, not permission to choose convenient files after viewing scores. A final file-membership freeze cannot honestly be claimed now. The user supplied the audio locally; only metadata, archive checksums and headers have been audited. No model or acoustic analysis has run.

A metadata-only audit may inspect labels, provenance, duration, channel metadata and duplicate identities for every partition. It must not inspect target embeddings, predictions, listening examples, spectra or class-conditioned acoustic summaries. Only the designated evaluator may read final-target waveforms after the complete model/configuration batch has been frozen.

## 2. Release registry

| Registry ID | Exact resource | Role and verified status | Remaining prerequisite |
|---|---|---|---|
| `IDMT_V1` | [IDMT-Traffic 1.0.0, Zenodo 7551553](https://zenodo.org/records/7551553), `IDMT_Traffic.zip`, provider MD5 `7ca2311ca32203aec5074a5fe933e343` | Source real data, present at `dataset/IDMT_Traffic/`. Provider-labelled `SE_CH34` car/truck inventory: 3,902/511, six site/date groups. | Per-file SHA256, original-session/duplicate audit, final eligibility and a feasible group-separated split. Sensor rule corrected below. |
| `MELAUDIS_V1` | [Figshare 27115870 version 1](https://doi.org/10.6084/m9.figshare.27115870.v1), `MELAUDIS_Vehicles.rar`, file ID `49436053`, MD5 `0bf2cf3ad13436ec8bf060b378508252` | Target real data. Version/file metadata and CC BY 4.0 verified through the [Figshare API](https://api.figshare.com/v2/articles/27115870) on 2026-09-27. | Both supplied archive MD5s pass; FF/1V car/truck filename candidates total 7,854/256. Original-session groups, duplicates and final eligible counts remain pending. BG archive is unnecessary for this phase. |
| `P02_RELEASE` | [Zenodo 21264250](https://doi.org/10.5281/zenodo.21264250); `code_and_results.zip`, MD5 `d8d4f6d02de391068e1138e6b279054c`; `data.zip`, MD5 `2599d5d0c91ac941bbeb7ba6b2627516` | Published CNN and synthetic baseline. API access succeeded; code ZIP downloaded to temporary storage and inspected as text, with MD5 verified. Expanded audio/data now supplied at `dataset/AI4TEN/`; the original data ZIP checksum is not verified from this expanded tree. | CNN snapshots/configuration package absent from the supplied tree; source identities, preprocessing provenance, test-row mapping and independence unresolved. |
| `PUB_K` | Historical KVSD derivative: 1,006 filenames in `code_and_results/configs/real_data_split.json` | Historical training reference for published real CNN and scaler. Original source: [Kaggle vehicle-sounds-dataset](https://www.kaggle.com/datasets/janboubiabderrahim/vehicle-sounds-dataset), as linked by release README. | Original download version, derivative-to-original mapping and sessions unknown. Not required to retrain any model in this minimum phase. |
| `PUB_T` | Historical DATASEC + MAVD derivative: 1,019 filenames in that same JSON; release README specifies `data/features/X_test.npy`, `y_test.npy`, `metadata.json` | Published evaluation replay only. [DATASEC descriptor](https://doi.org/10.1038/s41597-025-05991-w); [MAVD release](https://doi.org/10.5281/zenodo.3338727). | Confirm feature-array order maps to the released file list. Raw-source groups and original-version correspondence unknown. |
| `PUB_Y` | `P02_RELEASE` → `data/synthetic/pyroadacoustics/`, 5,000 supplied clips per original class | Binary BEATs bridge uses only car/truck: 10,000 candidate WAVs at `dataset/AI4TEN/data/synthetic/pyroadacoustics/`. Historical E2-P-S retains its original three-class training reference. | Identities, file hashes, generation relationships and duplicates remain unverified. |
| `Y_CORE` | Prospective `sim_components_v1.1` corpus, four cells `S0P0`, `S1P0`, `S0P1`, `S1P1` | Not yet generated. Source recipes and renderer settings are fixed in the experimental protocol. | Future implementation validation, renderer pin verification, source/trajectory manifest and hashes. |

Release MD5 values are acquisition checks, not sufficient provenance. Future manifests must include SHA256 for every source, derivative and checkpoint. These URLs designate resources; successful metadata access does not prove full audio accessibility or raw-source independence.

The IDMT provider lists **CC BY-NC-ND 4.0**, for evaluation; preserve that status and distribute manifests/configurations rather than modified audio in this phase. P02's README states MIT for scripts and CC BY 4.0 for other release material; its statements do not resolve original third-party dataset permissions. [IDMT provider, Dataset Download](https://www.idmt.fraunhofer.de/en/publications/datasets/traffic.html); P02 release `README.md`, License and Datasets.

## 3. Compatibility audit discovered before implementation

These findings update readiness, not the completed literature matrix. Paper statements and released-code observations must remain separate.

| Issue | Established evidence | Frozen response |
|---|---|---|
| Different task definitions | P01 has five outputs including bus/background; P02 has three. [P01 §2.1; P02 §2.1] | Use car/truck only in E0/E1/E2-B/E3; preserve three-class E2-P historical replay. P01's published F1 is motivation, not a numerical replication target. |
| Multiple vehicles and traffic states | MELAUDIS encodes traffic state, multiplicity, category, direction, date/time and street; clips are two seconds. [MELAUDIS descriptor, Data Features and Data Preprocessing](https://www.nature.com/articles/s41597-025-04689-3) | Retain only `FF`, `1V`, one admissible category. Exclude motorcycle/MC, idling, TJN/TJF, mixtures, BG, bicycle, bus and tram. |
| Class-direction confounding | MELAUDIS descriptor's Free Flow discussion treats some distant LR observations differently. | Keep all eligible RL and LR events, with no class-specific direction exception. Report direction and lane support. This is our selection rule, not a claim that all directions have equal quality. |
| Paired microphones are repeated events | IDMT's channel pairs can contain the same passing. [P01 §§2.1,2.5] | Select provider microphone token `SE` (sE8), class token C/T, and require the observed `CH34` stereo pair for these classes. Average the two channels within that WAV. Exclude `ME_CH12`; keep sibling channels in the same provenance group. Unexpected class/sensor/channel combinations are quarantined, never guessed. Evidence: local README File naming convention and audit inventory. |
| Paper versus released generator | P02 §2.2.1 reports 22.05 kHz outputs, range 5–50 m and randomized SNR. Released `_2_genSyntheticData.py` sets `FS_SIM=8000`, upsamples to 22,050, samples distance 3–15 m and uses class-specific speeds; `generate_one_sample` has no explicit additive-background/SNR stage. | Do not infer released audio was generated exactly by this script. Audit waveform bandwidth and generation metadata on training material only. The core deliberately uses a common 8 kHz intermediate rate for real and synthetic inputs. |
| Source impoverishment is not literally absent modulation in the released code | `_2_genSyntheticData.py::make_engine_harmonics` includes 5–15% sinusoidal AM; truck/motorcycle generators include broadband exhaust. | S1 adds specified variability to this baseline. Never describe S0 as having no modulation or no broadband component. This qualifies P02 §3.3's narrative for this release. |
| Silent renderer fallback | `simulate_passby` catches exceptions and returns `None`; `generate_one_sample` then calls a manual fallback. The pinned current upstream tree has no `simulatorScene.py`, although the released wrapper imports it. | Actual renderer for archived audio is unknown. E2-B tests the released corpus, not verified pyroadacoustics physics. E3 uses one explicitly pinned backend and must fail on renderer errors; no silent substitution. |
| Incomplete published trainer | `_4_runExperiments.py` imports `soundclass_v1`, absent from the inspected code ZIP. README's description of `_2_genSyntheticData.py` as AudioLDM also conflicts with its contents. | E2 verifies supplied CNN snapshots; it does not promise exact five-seed retraining. Reconstructing a trainer is outside the minimum phase. |
| Historical scaler and validation | P02 §2.3 and `_3_prepareExperiments.py` fit the scaler to KVSD before any later 15% validation split. Synthetic-only configurations therefore use real statistics. | Label E2-P as a historical, real-statistics-assisted replay. Core BEATs scalers use their own training partition only. No held-out validation examples contribute to core fitted transforms. |
| Historical synthetic selection | `_3_prepareExperiments.py::load_and_extract` sorts paths and takes the first `max_per_class`, rather than random subsampling. | Replayed checkpoints inherit their historical training. E2-B uses the deterministic source-group selection below, with an explicit reproduction-variant label. |
| Historical split IDs are incomplete | Released JSON has training/test paths but no internal validation membership or test-array row IDs. | Replay requires verified row alignment; absent original groups, report event metrics descriptively and mark session CIs unavailable. No invented session IDs. |
| Pretraining and hardware | BEATs iter3+ has prior real-audio training; existing local artifact is a pinned mirror. [P07 §4.3; local BEATs provenance] | Synthetic-only means task-training only. Record pretraining overlap as unknown. Use the same checkpoint in every core arm. |
| Existing project simulator has different labels | `configs/procedural_vehicles.yaml` defines tracked/wheeled profiles. | Do not treat these as car/truck sources or reuse the old classifier head. |

Static code evidence refers to the checksum-pinned P02 ZIP above, `scripts/_2_genSyntheticData.py` functions/constants, `scripts/_3_prepareExperiments.py::load_and_extract/prepare_all`, and `_4_runExperiments.py` import. The generator SHA256 is `01e177684b9204d36b0410ad6bd42a26aeda3f73fadd498aa812c6e7457b7c0e`. No released Python code was imported or run.

## 4. Canonical class map

Core mapping ID is `civilian2_v1`: class order `[car, truck]`, integer IDs `[0,1]`. Historical replay alone uses `civilian3_published_v1`: `[car, truck, motorcycle]`, with verified output-order permutation. A metadata adapter must enumerate actual release spellings before manifests are frozen. Case folding alone is permitted; it must not change semantics.

| Canonical | IDMT semantic label | MELAUDIS vehicle token | P02 and prospective synthetic directory/label |
|---|---|---|---|
| `car` (0) | filename `C` / semantic `car` | `Car` | `car` |
| `truck` (1) | filename `T` / semantic `truck` | `Truck` | `truck` |
| excluded from core | filename `M` / semantic `motorcycle` | `MC` | `motorcycle` |

Unknown labels are excluded and counted. Do not map `commercial vehicle`, van, bus, tractor, scooter, `engine idling`, or mixed tokens to an admitted category without explicit source documentation. No inference of engine fuel, RPM, model, weight or cylinder count from a real label. A speed-limit annotation is not measured vehicle speed.

Motorcycles must not be relabelled as car/truck or retained as an untrained third test class. Unknown/open-set evaluation is outside this amendment.

For E2-P, keep all three original classes and check the saved Keras output order against the archive's label metadata. Any differing integer order must be explicitly permuted before scoring; never assume order from alphabetical sorting.

## 5. Eligibility and provenance units

Real core eligibility: valid decodable PCM; a documented original event; a single admissible class; provider two-second event duration; at least 2.000 s after resampling; nonzero finite waveform. Accept longer clips only by the fixed centre-crop rule in the experimental protocol. Exclude short clips rather than class-dependent padding. Keep clipping, quiet vehicles, wet roads and difficult backgrounds; they are not score-based exclusions. A corrupted file is excluded by the same rule before splitting, with reason recorded.

For MELAUDIS apply the additional `FF & 1V` filter above. This controls annotated multiplicity, not a guarantee that all distant interference is absent. For IDMT use annotated vehicle passings, not background excerpts. Use only provider-labelled sE8 car/truck events with the `SE_CH34` selector above. The observed motorcycle `SE_CH12` anomaly is outside this core task; it is not relabelled or repaired.

Build connected provenance groups **before filtering or cropping**. Join every pair of files sharing an original media recording, documented recording session, duplicate event, synchronized device capture or overlap in source time. Conservatively union all captures with the same dataset/site/calendar-date, across sensors and contiguous original media. Missing date: use the entire documented session; missing session/date: use the whole site. Missing all three: quarantine. A filename or two-second event is never itself sufficient evidence of an independent session.

Across different datasets, match original media IDs/URLs and exact decoded-audio hashes. For near duplicates, a data custodian may compute deterministic fingerprints solely to identify overlap; none of those features enter modelling. Put cross-dataset overlap groups in quarantine on both sides. All synthetic descendants inherit their source/template family and all real derivatives inherit original-session grouping.

These rules measure unseen acquisition groups, not unseen physical vehicles, unless independent vehicle identity is actually documented. Unknown make/model identity stays unknown.

## 6. Deterministic partitions and exact logical manifests

`split_seed=20260927`. All ordering uses the lowercase hex SHA256 of UTF-8 `sim_components_v1.1|20260927|<dataset_id>|<canonical_group_id>`, then the canonical ID to break ties. Never use Python's process-dependent `hash()`.

Construct provenance groups over all raw files first; after class/sensor/eligibility filtering, count and allocate only groups with at least one eligible core event. Excluded-only groups do not count toward support or consume a split slot. For each real dataset, sort those group IDs by that key and use the following assignment, without trying alternative seeds:

- **IDMT:** first `ceil(0.60 G)` groups → `I_TRAIN_POOL`; next `floor(0.20 G)` → `I_VAL`; all remaining → `I_TEST`. Allocation is by groups, not windows. It is within the same dataset but may also hold out entire sites under conservative grouping.
- **MELAUDIS:** first `ceil(0.60 G)` → `M_DEV`; remainder → `M_LOCK`. There is no MELAUDIS training or model-selection validation partition in this minimum phase. `M_DEV` is an exposed diagnostic test domain; `M_LOCK` is the final untouched target.

If a partition lacks a class or fails the support rules below, **stop**. Do not reshuffle until a useful score or convenient class balance appears. Amend the partition design based only on metadata, assign a new protocol version, and freeze it before any modelling. A new metadata-only split is not an experiment result.

Per replicate seed `r ∈ [42,123,456,789,1024]`, define `I_TRAIN_200_r` as exactly 200 unique events/class (400 total) from `I_TRAIN_POOL`. Within each class, cycle through provenance groups in SHA256 order, taking the next event in `SHA256(protocol|r|event_id)` order until 200 are selected. No duplicates or oversampling. Remaining training-pool events are unused for fitting or parameter calibration. The split roles remain fixed across seeds; only within-training sample selection changes.

`I_VAL`, `I_TEST`, `M_DEV`, and `M_LOCK` use **all** eligible events in their assigned groups, retaining native class prevalence. There is one prediction per event, not multiple correlated windows scored independently.

For `PUB_Y`, exclude motorcycle before selecting core events; select whole generation groups into an 80/20 train/validation allocation by the same hash rule, then select exactly 200/class for `PUB_Y_TRAIN_200_r` and 50/class for `PUB_Y_VAL_50_r` (400 training and 100 validation events), round-robin over groups as above. A generation group contains all derivatives of the same dry source realization/run; related output crops never cross roles. If only indexed WAV names exist and source/run grouping cannot be recovered, the strict E2-B arm is not ready. It cannot silently become a random-file split. Shared generic engine-family formulas across splits must be disclosed; they do not represent independent real vehicles.

`Y_CORE` uses only car/truck and has 20 training source templates/class × 10 trajectories = 200 clips/class, and 5 disjoint validation templates/class × 10 trajectories = 50 clips/class, for each seed and each of four factor cells (400 training and 100 validation events/cell/seed). An S0 template and its S1 counterpart share a lineage ID and split role; every P0/P1 rendering stays in that role. Trajectory draws are paired across all classes/cells. Separate source, geometry and waveform-noise RNG streams prevent S1's extra draws from changing P settings. Synthetic validation is a technical diagnostic only; it does not select a classifier.

Logical manifests are frozen names, not existing files:

| Experiment | Train | Validation | Development evaluation | Final evaluation |
|---|---|---|---|---|
| E0 | `I_TRAIN_200_r` | `I_VAL` | `I_TEST` | none additional |
| E1 | E0's exact fitted model; no refit | E0 `I_VAL` | `M_DEV` | `M_LOCK` |
| E2-P-R | historical `PUB_K`, no new fitting | inherited 15% stratified validation; exact IDs unknown | `PUB_T` replay | no new untouched claim |
| E2-P-S | historical 15,000 `PUB_Y` inputs, no new fitting; KVSD scaler | inherited synthetic validation; exact IDs unknown | `PUB_T` replay | no new untouched claim |
| E2-B | `PUB_Y_TRAIN_200_r` | `PUB_Y_VAL_50_r` | `M_DEV` | `M_LOCK` |
| E3-00/10/01/11 | respective `Y_CORE_SiPj_TRAIN_r` | respective `Y_CORE_SiPj_VAL_r` | `M_DEV` | `M_LOCK` |

Minimum support before fitting: IDMT training pool ≥200 events/class in ≥3 groups; IDMT validation/test each ≥20 events/class in ≥2 groups; MELAUDIS development/lock each ≥20 events/class in ≥3 groups and ≥6 groups overall. These are feasibility floors, not a statistical-power guarantee. Report numbers of sites separately. If the grouping collapses IDMT to too few independent sites/sessions, E0 is blocked; random windows cannot rescue it.

### v1.1 metadata feasibility finding

The corrected car/truck subset has six candidate site/date groups; Hohenwarte contains only excluded motorcycles. Under the unchanged 60/20/20 allocation, six groups yield **4 train / 1 validation / 1 test**, failing the two-group validation/test floors. More generally, disjoint support minima of 3 + 2 + 2 require at least seven groups. **Removing motorcycle fixes the class-coverage obstruction, but does not make this split executable.** No new split roles have been assigned, no seeds searched and no support floor relaxed. A later metadata-only split-design amendment or additional independent acquisition groups is required before fitting. Candidate counts may decrease after integrity/provenance admission.

## 7. Required manifest schema and ownership

Each future JSONL row must contain:

```text
dataset_id, release_id, archive_checksum, source_file_sha256, relative_path,
original_media_id, event_id, recording_session, site_id, calendar_date,
device_id, paired_event_id, source_start_s, source_end_s, native_sample_rate_hz,
native_channels, duration_s, raw_label, canonical_class, class_id,
traffic_state, multiplicity, direction, lane, provenance_group_id,
grouping_basis, duplicate_group_id, eligibility, exclusion_reason, split_role,
parent_id, source_template_id, simulation_run_id, source_seed, geometry_seed,
factor_cell, transformation_config_sha256, derived_audio_sha256, licence
```

Use JSON `null` for genuinely unknown metadata and a separate `unknown_reason`; required group identity/class/release/hash fields may not be null in an executable manifest. Historical replay can retain unknown sessions, with inference restrictions explicitly recorded. Store all paths relative to the release root, not machine-specific home directories.

The modelling process can read training/validation manifests and diagnostic evaluation results. A separate evaluator holds M_LOCK labels and raw paths. With one operator, enforce the same separation using immutable manifests and a final-batch script/interface: no model-development path may enumerate M_LOCK waveforms. Metadata admission and automatic integrity checks do not expose acoustic examples to development.

## 8. Data-admission gate

Before any experiment is later implemented or run, materialize a signed/hash-indexed dataset lock containing releases, parsed labels, eligibility counts, grouping evidence, duplicate exclusions, split IDs, per-class/group support and all audio hashes. Verify zero intersections of original groups across roles and zero target access by fitted transforms. For published replay verify snapshot/scaler/feature identities and output-order alignment. If a prerequisite fails, record a blocker; never fill an unknown with a plausible value.

Availability is now better established than in the literature review: API metadata for P02 and MELAUDIS was retrieved. **Raw corpus integrity, source-group independence, and reproducibility of P02's exact generation backend remain unverified.** The channel/class amendment is complete; the IDMT split remains infeasible under the retained support floors, and the other admission checks remain pending. No experiment has run.

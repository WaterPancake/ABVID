# Dataset admission: waveform integrity and class/channel manifest

Completed: 2026-10-02. Protocol: `dataset_admission_v1.2.1`. Scope: the user-authorized admission steps 1–2, following R0. **All 46,775 inventoried WAVs were fully decoded and hashed; 22,873 car/truck files pass these checks.** This is not full training admission. All rows retain `admitted_for_training: false`, `split_role: null` and unresolved final provenance groups.

The authoritative [summary](ARTIFACT_INDEX.md#artifact-e3dd6a1c2104), [passing manifest](ARTIFACT_INDEX.md#artifact-d8cbf9ec37bb) and [all-file manifest](ARTIFACT_INDEX.md#artifact-e447d8f1ef16) record actual measurements and decisions. Counts below are **files**, not independent vehicles, events or sessions. Same-label duplicate resolution remains outstanding.

## Coverage and passing class counts

| Supplied corpus | WAVs fully audited | Passing car | Passing truck | Passing total |
|---|---:|---:|---:|---:|
| IDMT-Traffic, all classes/sensors/backgrounds inventoried | 17,506 | 3,902 | 511 | 4,413 |
| MELAUDIS vehicle archive, all states/classes inventoried | 13,668 | 7,811 | 256 | 8,067 |
| AI4TEN procedural-labelled corpus, all three classes | 15,000 | 5,000 | 5,000 | 10,000 |
| AI4TEN AudioLDM-labelled corpus, all three classes | 601 | 197 | 196 | 393 |
| **Total** | **46,775** | **16,910** | **5,963** | **22,873** |

The remaining **23,902 files** are retained in the [exclusion/quarantine manifest](ARTIFACT_INDEX.md#artifact-96130b86fba5). Most are outside the requested class, sensor or traffic-state scope; they are not all defective recordings. Reasons are recorded separately and can overlap. The MELAUDIS background archive was not decoded in this pass and is not included in these totals.

Every file passed complete decoding, RIFF payload-boundary and declared-frame checks; no nonfinite samples were found. One IDMT file is exactly silent and three MELAUDIS files are shorter than two seconds. No amplitude normalization, resampling, cropping or source-file repair was performed. Near-full-scale samples and quiet signals were not used to discard recordings.

## IDMT: corrected selector passes

All **3,902 car / 511 truck** candidates have provider microphone token `SE`, original channel-pair token `CH34`, and two decoded waveform channels. The future mono operation is the mean of those two columns; their original channel IDs are 3 and 4. `ME_CH12` recordings and motorcycle remain outside the core manifest.

The silent file is an excluded MEMS car recording, `2019-10-22-15-30_Fraunhofer-IDMT_30Kmh_650690_A_D_CR_ME_CH12.wav`; it does not reduce the selected sE8 counts. Candidate microphone-pair links cover all 17,506 files: **8,794 event keys**, of which **8,712 link multiple files**, with no conflicting class annotations within a key. These links derive from provider timestamp/site/speed-limit/sample-position tokens; they do not certify independent vehicles or complete session provenance. [Pair-link manifest](ARTIFACT_INDEX.md#artifact-cd2a5086e9a5).

The six car/truck site/date groups remain unchanged. The prior three/two/two group support minimum is still infeasible with six groups; no new split was assigned.

## MELAUDIS: layout, timestamp and annotation issues

The initial metadata filter produced **7,854 car / 256 truck** files. The final step 1–2 manifest has **7,811 / 256**. The 43 car exclusions are:

- **38 invalid timestamp tokens:** the wider archive has 74 filenames with minute fields 60–62. Their independently readable class, location and channel tokens are preserved. No corrected timestamp was inferred; these files remain quarantined for provenance review.
- **One short event:** `2023-08-04_10-0-0.4-Swanston6_FF_1V-Car_L1_1Lane_RL_mono.wav` is 1.4 seconds. It was not padded. Two additional short recordings are outside the FF single-car/truck task.
- **Four conflicting annotations:** four exact-audio pairs label one copy as `1V-Car` and another as `NoV`. All eight copies are quarantined; four were otherwise core car candidates. No supposedly correct label was chosen.

All **999 files named mono** physically contain two channels. Direct sample comparison found **995 exactly identical channel pairs**, which may pass as redundant mono: averaging leaves their samples unchanged. The remaining **four contain distinct channels** and stay quarantined. All four are motorcycle-labelled and already outside the core task. There are **720 passing core files** with verified redundant mono channels. All other vehicle-archive files also decode to two channels; the label/layout checks now distinguish nominal mono from actual storage layout.

The initial [v1.2 result](ARTIFACT_INDEX.md#artifact-193c0b42e140) conservatively rejected all named-mono/two-channel mismatches and lost readable metadata when rejecting an invalid timestamp. The [diagnostic](https://github.com/WaterPancake/ABVID/blob/77daa1b339906d35b6fa0a59a166ecf2eac01174/experiments/admission/mono_layout_diagnostic_20261002.json) justified the explicit v1.2.1 rule before a fresh full audit. Both runs and their executed scripts/protocol snapshots are preserved. No target model scores informed the change.

MELAUDIS remains exposed development material after R0. This audit does not create an untouched confirmation set or resolve location aliases and session boundaries.

## Synthetic corpora: a real class-label conflict

All procedural-labelled WAVs pass integrity and exact folder/filename class consistency. The car/truck subset contains **5,000/class**. No byte-identical or native decoded-identical groups were found in this corpus; that does not establish source-template independence or identify its actual rendering backend.

All 601 AudioLDM-labelled WAVs decode correctly. However, **three exact-audio groups contain 12 files with conflicting labels**, including identical audio filed as both car and truck and/or motorcycle. Quarantining every member removes **four car and four truck files** from the core candidates, leaving **197 car / 196 truck**. The remaining four conflicting copies are motorcycle-labelled. The proposed 200 distinct training examples/class budget is therefore not currently feasible for this arm, even before reserving any independent validation examples. Do not fill the shortage by repeating examples or changing labels.

The extra `ALDM_car_0201.wav` passes integrity, matches its class filename/folder, and has **no exact byte or decoded-audio duplicate** in the audited inventory. It remains in the candidate manifest. Why the release contains 201 rather than the README's 200 car files is unknown; distinct hashes do not prove independent source content. [Extra-file finding](ARTIFACT_INDEX.md#artifact-580952d300e3).

## Equality checks and remaining boundaries

Across the scan, **25 exact-equality groups cover 59 files**: 22 groups in MELAUDIS and three in AudioLDM. File-byte and canonical decoded-audio group memberships agree. Seven groups have conflicting annotations and are quarantined; same-label equality groups remain recorded for later duplicate/group resolution. No equality group crosses the four corpus labels used in this audit. This native-rate/channel/frame hash is not invariant to resampling, gain, cropping or encoding changes, so it cannot establish absence of cross-dataset reuse or near duplicates. [Equality groups](ARTIFACT_INDEX.md#artifact-b7105df40723); [annotation conflicts](ARTIFACT_INDEX.md#artifact-33c96e8e386b).

Original source files were read only. MELAUDIS was extracted into separate audit work directories from a vehicle archive that again matched its provider MD5; its SHA256 and every audited WAV's file/sample hashes are recorded. Expanded IDMT/AI4TEN hashes identify the local files but do not retroactively verify an absent original ZIP. Provider annotations were not independently confirmed by listening.

## Reproduction and validation

The full v1.2.1 run took **148.83 seconds**, including archive checks, extraction, full decoding/hashing and manifests, with four worker threads on macOS arm64. It used Python 3.11.13, NumPy 1.26.4, SoundFile 0.13.1 and libsndfile 1.2.2; no GPU or model execution was used. Peak RAM was not measured. Code, protocol, input-document, configuration and artifact hashes accompany the summary; Git commit alone is insufficient because the working tree is dirty.

**Thirteen regression tests passed**, covering tolerated WAV truncation, nonfinite/silent/short audio, channel cancellation, metadata-independent sample hashes, exact selectors, invalid timestamps, redundant mono and conflicting labels. The [manifest verifier](https://github.com/WaterPancake/ABVID/blob/77daa1b339906d35b6fa0a59a166ecf2eac01174/experiments/admission/verification_20261002_v1_2_1.json) checks all-file coverage, class/channel rules, conflict quarantine, pair membership, artifact checksums and the absence of training admissions/split roles. Commands and output definitions are in the [admission README](../experiments/admission/README.md).

Remaining admission work is original-session/location and near-duplicate reconciliation, synthetic source/run lineage, final exposure accounting and a supported split/budget amendment. **Steps 1–2 are complete; H1 training has not started.**

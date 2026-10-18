# DCASE 2024 Task 10 metadata admission and seal

Protocol `dcase2024_t10_metadata_v1`, written 2026-10-11 before the admission
run and before any audio is opened. Reusable code:
[`abvid.data.dcase_t10`](../../../src/abvid/data/dcase_t10.py) and
[`abvid.data.remote_zip`](../../../src/abvid/data/remote_zip.py).

## Purpose and role

MELAUDIS is exposed development data and cannot provide independent confirmation
([proposed experiments §2](../../../reports/proposed_experiments.md)). This
protocol admits the DCASE 2024 Task 10 development release
([Zenodo 10700792](https://doi.org/10.5281/zenodo.10700792), Bosch Research / KU Leuven,
CC BY-NC-SA 4.0 on the record page) **from metadata only** and seals it as a
candidate for the unmaterialized `T_confirm` role. No repository experiment has
used this release before. A search of the checkout, the literature matrix, the archived
reports and the archived literature Markdown for the record IDs, "DCASE 2024" and the
companion paper (arXiv 2401.09308) found no match.

## What is read

Only HTTP byte ranges for each archive's end records and central directory, plus
the allowlisted members `locN/meta.json`, `locN/train.csv` and `locN/val.csv`
(regex `loc[1-6]/(meta\.json|train\.csv|val\.csv)`). Audio members (`.flac` and
other audio suffixes) are refused by the reader before any request is made. The
evaluation release ([Zenodo 11209838](https://doi.org/10.5281/zenodo.11209838)) and
`simulation.zip` are inventoried from their directories only. A per-archive
transfer cap of 20 MB applies. Each member's CRC32 is verified; Zenodo's
volume MD5s are recorded but cannot be verified without downloading whole volumes.

## Labels, classes and grouping

- Rows are one-minute segments with `dow, hour, minute` and counts
  `car_left, car_right, cv_left, cv_right`. The source of the labels (coil, radar,
  camera or human) is not disclosed per site. Counts are not event times.
- **cv is not truck.** The provider's "commercial vehicle" class is undefined and
  may include vans. Any comparison uses the task `car` vs `cv` and reports it as
  a proxy for ABVID car/truck, never as the same task. Motorcycles, buses and
  bicycles have no category; their presence in a minute is `unknown`.
- `provenance_group_id = DCASE2024T10|locN` (six groups). Dates and sessions are
  not released; `recording_session = unknown`. Provider train/val splits are
  retained as metadata but have no role here: both are candidate confirmation data.
- Weekday clocks listed more than once indicate several unidentified days; they are
  counted and treated as ambiguous for isolation.

## Candidate event rule (label-only, fixed now)

1. **Primary candidates:** minutes with exactly one counted vehicle
   (`car_total + cv_total = 1`). The class comes from the counts.
2. **Sensitivity candidates:** primary minutes whose previous and next minutes in the
   same site/split are listed, unambiguous and contain zero counted vehicles.
3. **Window placement, to be applied only after authorization:** after the
   comparison's frozen resampling route, take the channel mean of the four capsules,
   compute 50 ms-hop RMS frames, and centre the comparison's fixed observation
   length on the maximum-RMS frame. Clamp the window inside the minute, without padding.
   The rule is class-independent and is not tuned on audio, labels or scores.
4. Vehicle-absent minutes are retained in the manifest for a later detection task
   and are not used for car/cv classification.

## Seal

The admitted metadata and its hashes form the seal record
[`seal.json`](seal.json). The release audio is not downloaded under this protocol.
Opening it requires all of the following, committed by a human:

1. A frozen comparison protocol (for example CAST-5B or an H2 replication) whose
   heads, preprocessing, window rule and analysis code are hashed before access,
   with one primary contrast and the prespecified 3-point margin.
2. An authorization file under `authorizations/` naming that protocol's SHA256.
3. A cross-corpus waveform overlap audit against IDMT, MELAUDIS and the AI4TEN banks
   that outputs match identities only, with no labels or model scores.

One authorized scoring pass consumes the set. Afterwards its role becomes exposed
development data in the ledger; any redesign needs new confirmation data.

## Known limitations

Six sites give few groups, as in IDMT. Single-vehicle minutes may still contain
uncounted sound sources. Label timing is per minute, so window placement can select
a non-vehicle event. The array (four capsules, 24 cm aperture) differs from IDMT and
MELAUDIS sensors. The commercial-vehicle class differs from truck. BEATs pretraining
overlap with this release is `unknown`. These affect interpretation; they are not
reasons to tune on the set.

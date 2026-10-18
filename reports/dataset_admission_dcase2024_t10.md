# DCASE 2024 Task 10: metadata admission and seal

2026-10-11. The DCASE 2024 Task 10 development release (Bosch acoustic traffic
monitoring) is admitted **from metadata only** and sealed as the project's first
candidate for the independent confirmation role `T_confirm`. No audio byte was
requested, and no feature, model score or listening check exists for this release.
It is **not** a car/truck test: its second class is the provider's "commercial
vehicle" (cv).

[Protocol](../experiments/admission/dcase2024_t10/PROTOCOL.md) (written before the
run) · [seal](../experiments/admission/dcase2024_t10/seal.json) ·
[commands](../experiments/admission/dcase2024_t10/README.md) ·
code: [`abvid.data.dcase_t10`](../src/abvid/data/dcase_t10.py),
[`abvid.data.remote_zip`](../src/abvid/data/remote_zip.py),
[`abvid.data.sealing`](../src/abvid/data/sealing.py).

## Why this release

MELAUDIS was scored during R0 and remains exposed development data. Its four
conservative groups hold 256 trucks, 187 of them in one group and 1 in another
([H1](H1_baseline_results.md)). MAVD and DataSEC are partly exposed through the
P02 test material. No checkout file, literature-matrix entry, archived report or
archived literature Markdown mentions this release, its record IDs or its
companion paper (arXiv 2401.09308).

The release also ships per-site synthetic pass-bys made with pyroadacoustics, the
backend vendored here, using each site's geometry. That makes it a candidate
second simulation-to-real pair for H2 work, separate from its role as a target. The
synthetic files are inventoried only and are not admitted for any role.

## What was transferred

| Release | Archives | Archive bytes | Bytes transferred | Audio members listed | Audio bytes requested |
|---|---:|---:|---:|---:|---:|
| Development, [Zenodo 10700792](https://doi.org/10.5281/zenodo.10700792) | 6 sites + `simulation.zip`, 25 volumes | 45.15 GB | 2,078,463 | 16,213 | 0 |
| Evaluation, [Zenodo 11209838](https://doi.org/10.5281/zenodo.11209838) | 6 sites, 18 volumes | 29.42 GB | 1,287,156 | 9,887 | 0 |

Total transfer: 3,365,619 bytes in 56.6 s, plus the two Zenodo record documents.
`engine-sounds.zip` (709 MB of synthetic engine audio) was not opened.
The figures come from `receipt.json`, which logs every byte-range request. Only the
archive end records, central directories and the 18 allowlisted members
`locN/{meta.json,train.csv,val.csv}` were read. Each member passed its archive CRC32.
Zenodo's whole-volume MD5s are recorded but not verified, because that would require
downloading the audio.

- **Evaluation release:** contains only `test/*.flac` and no label table, so it is
  not usable as a labelled target.
- **License:** CC BY-NC-SA 4.0, per the Zenodo record page; the API metadata omits it.
- **Integrity check:** every label row names exactly one listed audio member and
  every audio member has exactly one label row (6 sites × 2 splits).

## Label inventory

Counts of one-minute segments from the development label tables. Single-vehicle
means exactly one counted pass-by in the minute.

| Site | Array height / distance to road (m) | Max speed (km/h) | Minutes | Car pass-bys | cv pass-bys | Single car | Single cv | Weekdays present |
|---|---|---:|---:|---:|---:|---:|---:|---|
| loc1 | 2.7 / 4.0 | 100 | 2,538 | 20,652 | 3,118 | 160 | **96** | 0, 3 |
| loc2 | 3.0 / 0.5 | 50 | 110 | 1,788 | 78 | 1 | 0 | 2, 3 |
| loc3 | 2.0 / 4.0 | 50 | 8,855 | 45,679 | 1,230 | 996 | **93** | 0–6 |
| loc4 | 3.0 / 2.5 | 50 | 31 | 174 | 9 | 1 | 0 | 3 |
| loc5 | 3.0 / 1.0 | 40 | 188 | 298 | 29 | 75 | 4 | 3, 4 |
| loc6 | 3.0 / 8.0 | 90 | 3,291 | 38,202 | 4,133 | 128 | **35** | 0–6 |
| **Total** | | | **15,013** | **106,793** | **8,597** | **1,361** | **228** | |

Among the primary candidates, cv makes up **14.35%** (228/1,589). MELAUDIS has 3.17%
trucks. The candidates are still concentrated: three sites carry 224 of the 228 cv
minutes, loc5 has 4, and loc2 and loc4 have none. A confirmation interval will therefore
rest on about four informative groups with two classes. That is better class support
than MELAUDIS but **not** a large number of independent recording contexts.

Per-split detail, the full segment manifest (`segments.jsonl`) and hashes are stored
outside the checkout at `data:DCASE2024_T10/metadata_v2/`. See the
[summary](ARTIFACT_INDEX.md#artifact-3a180471aab6),
[receipt](ARTIFACT_INDEX.md#artifact-04d853464f38) and
[manifest](ARTIFACT_INDEX.md#artifact-dae994c8ec63).

## Findings that shape later use

1. **Most empty minutes are missing from the label tables.** Only 190 of 15,013 listed
   minutes have zero vehicles, and loc1 has none, while timestamps skip minutes, for
   example loc4 train 09:14 then 09:16. The tables therefore cannot establish that the
   minutes on either side of a single-vehicle minute were empty. The protocol's
   sensitivity rule found **0 isolated single-vehicle minutes**: the rule treats
   unlisted neighbours as unknown, as frozen, and was not relaxed afterwards. A
   label-only isolation sensitivity is not available. Any boundary check must use the
   class-independent window rule after authorization.
2. **Days and sessions are not identifiable.** Labels carry weekday, hour and minute
   only. loc2 val repeats 18 weekday clocks, which means at least two unidentified days.
   Sessions stay `unknown`; grouping is by site.
3. **Splits are not balanced across sites.** For example, loc6 val has one
   single-vehicle minute and train has 162. The provider's train/val split has no role
   here, and both are candidate confirmation data.
4. **Label source is undisclosed** (coil, radar, camera or human, varying by site), so
   label noise may differ by site. The task page defines `car_left` as "passenger
   vehicles going left to right per minute"; direction is not used.
5. **cv is a proxy task.** Commercial vehicle may include vans. Results must be reported
   as car/cv and never merged with car/truck rows.

## Seal

[`seal.json`](../experiments/admission/dcase2024_t10/seal.json) (SHA256
`7ff5f704cea1ad92c14052e00168e1da50fcbcbce84a541f0ba7abafb6aa6f9f`) binds the protocol
(SHA256 `61530ed2…115ff`) to the metadata hash list (`7e1d83a1…e1e1`). The
`verify_seal` check recomputes every metadata hash. `authorize` refuses access unless a
human-committed authorization names this seal, a frozen comparison protocol, an overlap
audit and exactly one scoring pass. No audio download function exists yet. One should
be added only together with that authorization, as part of the frozen confirmation
study.

**Prerequisites before unsealing:**
- a frozen comparison: CAST-5B or an H2 replication with one primary contrast;
- a waveform overlap audit against IDMT, MELAUDIS and the AI4TEN banks that outputs
  match identities only;
- the protocol's class-independent window rule;
- a human authorization file.

One scoring pass consumes the set.

## Limits

This admission establishes label/audio inventory consistency and class support from
metadata. It does not establish:
- the audio quality;
- that single-vehicle minutes are free of other sound sources;
- independence from BEATs pretraining data (`unknown`);
- the absence of overlap with other corpora;
- equivalence between cv and truck.

The earlier attempt directory `metadata_v1` is empty and unused. `metadata_v2` holds
the admitted record.

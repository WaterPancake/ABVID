# P0 six-class metadata feasibility follow-up

2026-10-11 local date. This supplement adds public HTML metadata to the frozen
[candidate ledger](candidate_metadata.json). The original ledger and P0 freeze
remain unchanged. The [new candidate ledger](candidate_metadata_supplement.json)
contains four additional assets, retrieval hashes, caption-only observations,
unknowns and dependence leads.

The combined ledgers now have eleven asset leads spanning six candidate caption
classes: M1 Abrams, Leopard, Bradley, Stryker, M109A6 and HMMWV M1151. This resolves
the lack of a metadata lead for two proposed classes. It does not establish a
six-model roster, eight independent sessions per model or usable vehicle audio.
The verified independent-recording-group count remains zero.

| Additional asset | Metadata evidence | Admission issue |
|---|---|---|
| [M1151 training, 329602](https://www.dvidshub.net/video/329602/humvee-training-b-roll) | Caption names M1151; April 3, 2014; 128th MPAD; 263 seconds. | One caption lead; engine and audible intervals unknown. The M1165A1 brochure does not identify this M1151 engine. |
| [M109A6 firing, 607339](https://www.dvidshub.net/video/607339/arkansas-field-artillery-unit-fires-m109a6-paladins-during-operation-western-strike) | Caption names M109A6 and firing at Camp Guernsey; June 9, 2018; Arkansas NG PAO; 337 seconds. | Gunfire described; contamination and clean engine/background support not auditioned. |
| [Parade package, 967175](https://www.dvidshub.net/video/967175/army-250-parade-celebration-b-roll) | June 14, 2025; XVIII Airborne Corps; 639 seconds. Caption shot log names M109A6, Stryker and Bradley. | Labelled timecodes cannot become blind audio annotations. Event/unit links imply possible dependence; crowd/ceremony audio is unknown. |
| [Staged vehicles, 966208](https://www.dvidshub.net/video/966208/us-army-vehicles-staged-west-potomac-park-waterfront-b-roll) | Caption names M1A2, M2 Bradley and M109A6 arriving for the Army birthday parade; 234 seconds. | Page date June 9 differs from staging text June 10. Transport does not prove running engines. |

The last two captions identify the same Army birthday event and a 1st Cavalry
Division vehicle population. This is a lead for possible repeated physical
vehicles, not proof of a shared shoot or media reuse. Preserve the link and
resolve it before counting independent groups. Different dates and producer
bylines do not establish independent vehicle populations.

The parade caption also names an **M1 heavy wrecker**. This term cannot be
converted into an M1 Abrams label by matching “M1.” Other historical and
out-of-roster vehicles in that package do not expand the pilot roster. Caption
shot-log frame rates and correspondence to audible vehicles remain unknown.

A bounded exact-M1165/M1165A1 search returned publications but no usable
exact-variant video lead. That result does not prove absence from DVIDS. M1151
is a distinct caption-supported lead; it does not inherit M1165A1 engine evidence.

## What a future roster must resolve

The six caption classes mix family and variant granularity. The future protocol
must decide whether L4 identities are families, models or specified variants,
then freeze that choice before collection and evaluation. M109A6 and an
unspecified M109 cannot silently share a confirmed engine label. “HMMWV/JLTV”
cannot be one identity. All recording-specific powerplants remain unknown.

Both metadata ledgers contain only leads. Asset IDs, durations, public-domain
banners and caption diversity do not establish audio suitability, rights for
the intended use, background availability, independent shoots or physical
vehicle identities. No scripts, embedded players or associated media were
loaded. Public HTML snapshots and extracted visible text are retained at
`artifact:military_signature/p0/public-metadata-followup-20261011/`.

A bounded [P1 admission draft](P1_ADMISSION_DRAFT.md) makes the next action
reviewable. It is a proposal for rights/provenance and media admission; it does
not authorize collection or a Tier 0/model experiment.

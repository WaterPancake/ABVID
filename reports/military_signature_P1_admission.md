# Military acoustic signature P1 collection and review preparation

2026-10-11–12 local dates; contract `military_signature_p1_admission_5gb_v1`.
The user authorized more data, retained the 20-minute per-asset limit, then
reduced the starting storage allocation to 5 GB. The separately versioned
[authorization record](../experiments/military_signature/P1_AUTHORIZATION.md)
and [canonical contract](../experiments/military_signature/p1_admission_5gb.json)
record this scope. The original P0 protocol and 55-file freeze remain unchanged.

The collection contains **54 recordings, 8,735.906 seconds (2 h 25 min 35.906 s)**.
The longest recording is 505.706 seconds, below the 1,200-second asset limit.
Data, metadata and seven review packets currently occupy about **4.88 GB** of
the **5,000,000,000-byte** ceiling. The 64-asset first-batch target was not reached
under the storage/reserve limit. Storage is a ceiling, not a fill requirement.

These are collected playable recordings pending human blind review, not an
admitted segment dataset. There are **zero verified independent recording groups**,
zero admitted foreground segments and no experimental role allocation. No
representation, classifier or CAST model was fitted or evaluated by this run.

## Collected caption support

| Candidate caption class | Collected assets containing the caption cue |
|---|---:|
| M1 Abrams family | 14 |
| Bradley family | 11 |
| Stryker family | 11 |
| M109 / Paladin | 10 |
| HMMWV / Humvee | 6 |
| Leopard family | 5 |

Mixed-model captions contribute to more than one row; the rows do not sum to
the 54 unique assets. These counts establish neither audible model identity nor
session independence. The caption classes still mix family and variant granularity.
Engine identities, physical vehicles, foreground/background support and audio
quality remain unknown until the relevant evidence is reviewed.

The metadata names 38 originators. This is candidate diversity, not 38 independent
sessions or outlets. All assets are hosted on DVIDS, so outlet invariance remains
unestablished. Native audio is 48 kHz for 51 recordings, 44.1 kHz for two and
22.05 kHz for one; 51 recordings are stereo and three mono. Retain these recording
and codec differences as possible confounds in any separately authorized pilot.

The metadata-only inventory `metadata_confound_inventory.json` binds all 54
collection records and resolves the format counts by caption class:

| Caption class | 48 kHz | 44.1 kHz | 22.05 kHz | Mono assets |
|---|---:|---:|---:|---:|
| Abrams | 13 | 0 | 1 | 1 |
| Leopard | 5 | 0 | 0 | 0 |
| Bradley | 11 | 0 | 0 | 0 |
| Stryker | 11 | 0 | 0 | 0 |
| M109 | 10 | 0 | 0 | 0 |
| HMMWV | 4 | 2 | 0 | 2 |

All 54 retained audio streams use AAC. The only HLS acquisition is in the Abrams
caption class; the other 53 use explicit page MP4 downloads. There are 51
single-class caption packages and three mixed-class packages. Known date-taken
metadata spans 2005-01-27 through 2026-09-28. These are observed acquisition and
metadata differences, not measured classifier decodability or evidence of an
acoustic mechanism. A later frozen pilot must retain them as confounds and
declare any common preprocessing using training data alone. This inventory does
not assign labels to human-selected intervals or certify independent sessions.

## Provenance, restrictions and failures

The discovery queue contains 109 asset IDs. The run recorded outcomes for 74
unique IDs across 76 attempts: 54 collected IDs and 20 IDs still quarantined.
Both prior and later attempts are retained, including the recovered older HLS
asset. Failure records include rights restrictions, historical-source exclusions,
irrelevant caption hits, rendition byte limits and older-playlist retrieval errors.

Existing archival catalogs were read only as metadata. Eight known DVIDS IDs
and seven historical source-filename tokens were excluded from acquisition.
No historical or protected audio was loaded for fingerprint comparisons, and
their roles and files were unchanged. Asset-level rights decisions were recorded
before payload access. The [DVIDS public-use notice](https://www.dvidshub.net/about/copyright)
was retained alongside source pages; additional restrictions or missing usage
evidence caused quarantine. Metadata-based admission does not guarantee absence
of unmarked third-party content. Media publication remains outside this stage.

Some old public HLS playlists used `application/octet-stream`. The collector's
second version accepts this only for bounded `.m3u8` responses containing an
actual HLS header. It does not treat arbitrary binary responses as playlists.
Earlier errors and both collector versions remain available for review.

One retry reached its output byte bound and was rejected as possibly truncated.
Its failure record, payload SHA256 and byte count were retained before discarding
the 79,072,864-byte incomplete payload. No valid collected rendition was removed.
Other deferred candidates were not replaced by silently trimming overlong media.

## Mechanical verification and completed reuse audit

- All 54 retained source renditions and extracted audio files match their recorded
  SHA256 values. Encoded audio-packet hashes agree between each source and its
  extracted M4A: no audio re-encoding was introduced by extraction.
- A separate cross-asset comparison found no identical whole encoded-audio packet
  sequences among the 54 assets. Current native-file hashes were rechecked against
  the acquisition ledger; `encoded_packet_reuse.json` binds those hashes and
  records the negative result. Distinct whole-stream hashes do not rule out
  partial, edited or transcoded waveform reuse and do not certify independence.
- Every source obeys the 20-minute cap. Caption/source names were absent from
  native audio metadata; all experimental roles remain `unassigned`.
- Native-channel signal integrity was checked for all 54 recordings with no
  resampling or channel downmixing. One retained stream decodes entirely to zeros
  in every channel; 53 contain nonzero decoded samples. No asset has an entirely
  zero mono display while retaining nonzero native-channel audio. The per-asset
  counts and hashes are in `native_signal_integrity.json`. Nonzero samples do not
  establish audible vehicles, usable foreground intervals or source-model labels.
  The zero-stream finding is retained as mechanical evidence; no human rejection
  or accepted interval was fabricated and no collection record was rewritten.
- Seven packets contain 54 caption-hidden review cards. A local server verified
  packet retrieval, native audio byte-range responses and rejection of private
  mapping/traversal paths. Browser audibility and human interval decisions remain
  the reviewer's observations; HTTP checks do not establish them.
- The production native importer/exporter passed a synthetic round trip with
  frames 16–32016 at 16 kHz, exact decoded-sample equality and rejection of a
  changed blind-source identity. No real recording received synthetic annotations.
- Focused regression: 50 checks passed (13 P1 boundary/review/parity checks, 32 P0 checks
  and five existing overlap checks). The CAST harness canary remains red at
  720 collected versus 467 expected; baseline-file and frozen-file checks passed.
  Baseline refresh is human-only, and the new test file needs line-by-line human
  review before merge. No baseline, existing tests, lockfile or frozen configuration
  was changed to make the canary pass.
- The full label-blind waveform reuse audit completed at 2026-10-12 07:10:02 UTC.
  All 54 assets are accounted for: 53 fingerprinted recordings with 4,175 indexed
  windows and one entirely zero stream marked `short_or_silent_unresolved`.
  Its authoritative result is
  `artifact:military_signature/p1_admission/20261011_5gb/reuse_audit.json`.
  One four-second reuse suggestion links assets 897659 and 897660: reference
  start 100.000 seconds, query start 64.384 seconds, 265 landmark votes and
  correlation 0.988650272254228. The frozen original library reproduced the exact
  reported correlation. This agrees with the retained metadata dependence lead;
  it does not establish a verified session or physical-vehicle identity. No-match
  results do not establish independence.

The initial reuse run was intentionally interrupted while incomplete, after 43
recordings had been observed as processed. Its source and interruption record
remain available. A packed-integer vote counter replaces the costly two-column
sort while preserving reference IDs, signed offsets and exact counts. One million
synthetic hit pairs produced identical keys/counts; the controlled positive-match
fixture also produced identical full match rows. The reference library, correlation
calculation, thresholds and full 54-recording population remain unchanged.
The third run's completed result now supplies the full-population evidence;
neither interrupted run is treated as a completed audit.

A subsequent native process profile showed repeated waveform dot products
dominated the remaining runtime. The third implementation nominates possible
winning lags with FFT correlation, then returns scores computed with the original
direct-dot arithmetic. Ill-conditioned cases use the original full lag scan.
Scores were identical in 25 synthetic and 16 newly collected-media cases; all
thresholds and the full population remain unchanged. The second incomplete run,
its source and native stack profile are retained. This is a numerical processing
change, not evidence about an acoustic mechanism or generalization.

A separate workload diagnostic for asset 44 (921247) rebuilt the same first
43-asset reference index without running another full audit. It found 280,765
candidate offsets across 49 reference windows, including 273,110 offsets with
valid query bounds. Twenty-five sampled correlations and their observed runtimes
are retained in `reuse_v3_workload_diagnostic.json`. This large candidate workload
explains why an asset-level progress counter can remain unchanged while the job
is active. The diagnostic did not restart or reduce the full audit, and does not
itself certify completion or session independence.

A small independent dot-product benchmark compared the default BLAS thread
setting with `VECLIB_MAXIMUM_THREADS=1`; both produced identical scalar values
and similar timings. This negative check did not support changing the thread
setting. Additional display-waveform checks found long exact-zero regions in the
two expensive queries. Their constant centered windows explain near-zero lag
scores with many ambiguous candidates; this is a numerical workload observation,
not an acoustic mechanism. The audit's inputs, lag search and criteria remain
unchanged.

An isolated constant-query shortcut was also rejected. Replacing the full lag
scan with one lag for a numerically constant query region failed exact-score
parity on synthetic case 19 (seed 43), after 19 prior cases passed. The original
and shortcut scores differed by 8.673617379884035e-18. The unused source,
`flat_query_correlation_prototype_check.json` and compressed synthetic reproducer
are retained outside the checkout. The attempted check stopped before its planned
new-media cases and timing benchmark; neither is reported as passed. No production
code, thresholds or tests were changed to accept the shortcut. This is numerical
implementation evidence, with no physical or recognition interpretation.

Metadata dependence checks identify the two Bradley packages, 897659 and 897660,
through their shared creator/day VIRIN stem and matching date/originator/producer.
The record `metadata_dependence_leads.json` retains this possible common-shoot
link. It neither assigns session identities nor treats unmatched assets as
independent. Human and audio reuse evidence remain part of admission.

The provisional admission ledger `provisional_admission_groups.json` binds all
54 unchanged collection records and retains these metadata links. Following the
frozen protocol's unknown-session rule, all assets currently share one quarantine
component and cannot enter held-out roles. This component does not assert a known
common shoot or physical vehicle. Its verified independent-group count is zero;
it was not an experimental admission freeze. The completed automated ledger
`admission_groups.json` now incorporates the waveform suggestion, metadata
links and zero-stream flag, while preserving the same quarantine constraint.
All 54 source/native hash pairs, seven review-packet commitments and 11 protected
repository witnesses passed the closing verification. The ledger remains pending
human admission and is not frozen for a fitting/evaluation experiment. No
collection records or prior evidence snapshots were rewritten.

The external run is rooted at
`data:military_signature/p1_admission/20261011_5gb/` and
`artifact:military_signature/p1_admission/20261011_5gb/`.
Key records are `contract.json`, `collection_facts.json`,
`acquisition_verification.json`, `review_packet_receipt.json`, per-asset
`record.json`/rights/FFprobe files, resume/source snapshots and
`synthetic-native-validation/verification.json`.

## Human review and reproducible commands

Open [the local review index](http://127.0.0.1:8766/) in Chrome. Start with packet 1;
record accepted foreground/background intervals or logged rejections for every
recording, then save that packet's annotations. Each packet has a different batch
commitment. Source captions and private mappings are not served by the review UI.
The server requires the external run; it does not copy audio again or contact
external media services during playback.

```sh
PYTHONPATH=src .venv/bin/python -m abvid.signature.admission status
PYTHONPATH=src .venv/bin/python -m abvid.signature.admission_review serve --port 8766
PYTHONPATH=src .venv/bin/python -m abvid.signature.admission_annotations import \
  --packet packet-01 --input /path/to/annotations.json \
  --output artifact:military_signature/p1_admission/20261011_5gb/annotations/packet-01.json
PYTHONPATH=src .venv/bin/python -m abvid.signature.admission_annotations export \
  --input artifact:military_signature/p1_admission/20261011_5gb/annotations/packet-01.json \
  --output artifact:military_signature/p1_admission/20261011_5gb/clips/packet-01
```

Imports bind to the original human JSON, opaque mapping, batch and unchanged
collection. Exports preserve floor-start/ceil-end native frames and retain every
rejection. Exports that cannot fit the remaining storage allowance are explicitly
deferred. The recorded native validations do not substitute for human review of
real material or independently supported shoot/individual provenance.

The next scientific pilot still needs an admitted population, frozen identity
granularity, supported recording groups and a separate fitting/evaluation scope.
The eight-group feasibility target and ten-group proposal minimum remain distinct;
neither can be established from these raw asset counts.

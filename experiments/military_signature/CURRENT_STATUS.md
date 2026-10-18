# Military acoustic signature current status

Updated 2026-10-12 local date. This additive status supersedes the browser-handoff
and powerplant-evidence gaps in the dated [P0_STATUS.md](P0_STATUS.md) where the
new evidence below applies. Frozen v1 artifacts remain unchanged.

All named P0 preparation deliverables have evidence. The user subsequently
authorized P1 collection and preparation with a 5 GB total storage ceiling and
20-minute per-asset limit. Scientific feasibility and model recognition remain
untested; fitting and representation extraction are outside the current scope.

## P1 collection and admission

The [current authorization](P1_AUTHORIZATION.md),
[collection contract](p1_admission_5gb.json) and
[P1 report](../../reports/military_signature_P1_admission.md) supersede the earlier
metadata-only handoff for collection. At the current verified snapshot, 54
recordings total 8,735.906 seconds and occupy 4,879,449,428 bytes across the new
data and artifact roots. The longest recording is 505.706 seconds. The 64-asset
collection target was not reached under the storage/reserve limit.

Seven caption-hidden packets are available at
[the local review index](http://127.0.0.1:8766/). Real-media human review is pending;
the earlier user confirmation concerns only the P0 synthetic packet. No real
foreground segment has been admitted, no independent recording group has been
verified and all experimental roles remain unassigned. Caption asset counts and
the 38 named originators are not independent-session counts.

The full 54-recording waveform reuse audit completed at 2026-10-12 07:10:02 UTC.
Its authoritative output is
`artifact:military_signature/p1_admission/20261011_5gb/reuse_audit.json`.
It accounts for 53 fingerprinted assets, 4,175 indexed windows and one entirely
zero stream with unresolved fingerprint status. One four-second suggestion links
Bradley assets 897659 and 897660 with correlation 0.988650272254228, agreeing with
their metadata dependence lead. The frozen original library reproduced that exact
score. Neither this link nor absent waveform matches establishes independence.
Historical/protected audio remains excluded.

The earlier `provisional_admission_groups.json` snapshot remains preserved.
`admission_groups.json` incorporates the completed waveform audit, metadata links
and zero-stream flag. It binds the 54 unchanged collection records and joins their
unknown sessions in one quarantine component under the frozen protocol. It asserts
no known common shoot or vehicle and is not an experimental admission freeze.
Closing checks verified all source/native hash pairs, all seven review-packet
commitments and 11 protected repository witnesses. Verified independent groups
and admitted foreground segments remain zero.

Native-channel numerical integrity was subsequently checked for all 54 streams.
One decodes entirely to zero in every channel; the other 53 contain nonzero
samples. No entire mono-display cancellation of nonzero native audio was found.
The external `native_signal_integrity.json` retains per-asset hashes and counts.
These observations do not establish vehicle audibility or replace human review.

The latest focused verification passed 50 checks. The CAST test-count canary
remains red at 720 collected versus 467 expected; baseline-file and frozen-file
checks passed. Human-only baseline refresh and line-by-line review of the new
test file remain required before merge. The original 55-file P0 freeze is intact.

## P0 preparation evidence

| Requested deliverable | Evidence | Status and limit |
|---|---|---|
| Frozen draft protocol; phase gates; eight-versus-ten distinction | [PROTOCOL_DRAFT.md](PROTOCOL_DRAFT.md), [canonical JSON](p0_protocol.json), [freeze ledger](p0_freeze.json) | Verified preparation. Eight groups/model is development feasibility; ten is a proposal minimum, not a power calculation. |
| Manifest schema and recording/reuse/known-individual grouping | [manifest_schema.json](manifest_schema.json), [implementation](../../src/abvid/signature/manifest.py) | Verified with artificial cases. Public asset IDs supply no verified session count. |
| Offline trimming and deduplication tooling | [CLI](../../src/abvid/signature/__main__.py), [round trip](ANNOTATION_ROUNDTRIP.md) | Synthetic import/export passed; user confirmed Chrome playback and display. Real-media behavior remains untested. |
| Synthetic validation evidence | `artifact:military_signature/p0/validation-20261011-final/validation.json`; round-trip `verification.json` | Native sample equality, rejected intervals and provenance preserved; 37 focused regression checks passed at the recorded run. |
| Literature, feasibility, confounds and negative findings | [frozen report](../../reports/military_signature_P0.md), [follow-up](../../reports/military_signature_P0_followup.md), [source supplement](source_evidence_supplement.json) | PDF-authoritative selected claims; unsupported variant/engine labels remain unknown. |
| Public metadata inspection | [candidate_metadata.json](candidate_metadata.json) | Seven leads spanning four model captions; zero verified independent recording groups. Rights and audio suitability are unresolved. |
| Harness integrity and handoff | Original report and `artifact:military_signature/p0/verification-20261011.json` | Recorded count canary failed: 707 versus 467; baseline-file and frozen-file checks passed. Human baseline refresh and line-by-line test review remain required before merge. |

The user confirmation receipt is
`artifact:military_signature/p0/browser-roundtrip-20261011T232425Z/browser_confirmation.json`.
It preserves the exact question and answer linked to the supplied annotation
hash. The original annotation is unchanged. The follow-up literature contains
six additional PDFs and eight visually checked pages; it records historical or
variant-specific Bradley, M109A6, Stryker and M1165A1 evidence without assigning
engines to recordings.

The supplements are outside the original 55-file freeze. Their separate hashes
and external-evidence checks are recorded in
`artifact:military_signature/p0/followup-verification-20261011.json`.
The original freeze can still be checked with:

```sh
PYTHONPATH=src .venv/bin/python experiments/military_signature/verify_freeze.py
```

## Next input needed for scientific execution

Current collection and preparation are authorized. The next input is real-media
blind interval review and supported shoot/physical-vehicle provenance. Reuse and
provenance admission must be frozen before allocating experimental roles.

Any subsequent fitting/evaluation decision must separately specify the admitted
population, identity granularity, exact checkpoint and extraction configuration,
fit/compute budget and evaluation roles. Representation extraction, model or CAST
fitting, protected-source evaluation and media publication remain outside the
current scope. Confirmatory model-wide claims additionally need new unexposed
groups, physical-vehicle support and an allocation/power plan. The eight-group
pilot and ten-group model-claim minimum remain distinct and unachieved.
The retired benchmark is not resumed by this work.

No commit or merge was performed. The human-only harness baseline was not changed.

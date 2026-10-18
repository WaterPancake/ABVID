# Military acoustic signature P0 status

P0 is separately authorized for literature, public asset metadata, protocol design,
and implementation using synthetic fixtures. Real media collection or processing,
model fitting, protected-source evaluation, and P1 execution remain unauthorized.
This side project does not resume the retired military benchmark.

## Deliverables

- [x] Frozen draft [protocol](PROTOCOL_DRAFT.md) and explicit prospective phase gates
- [x] [Manifest schema](manifest_schema.json) with conservative session/reuse/known-individual grouping
- [x] Offline caption-hidden trimming and annotation tooling
- [x] Fingerprint and correlation reuse audit
- [x] Synthetic validation and CLI verification: 37 focused tests passed
- [x] PDF-authoritative [source ledger](source_evidence.json), [metadata leads](candidate_metadata.json), and [research report](../../reports/military_signature_P0.md)
- [x] Integrity and human handoff record; existing canary failure preserved

These checkboxes indicate preparation deliverables, not real-media admission or
scientific validation. Native browser playback/layout still requires human review:
Firefox computer-use approval was rejected; DOM event checks used mocked media APIs.
The current CAST canary collects 707 against 467 and fails only the count check;
baseline-file/frozen-file checks pass. The baseline is untouched and human-only.

Final artificial evidence:
`artifact:military_signature/p0/validation-20261011-final/validation.json`.
Freeze check: `PYTHONPATH=src .venv/bin/python experiments/military_signature/verify_freeze.py`.
P1 requires separate authorization, per-asset rights and verified recording groups.

## Initial evidence

Before P0 implementation, the CAST harness canary collected 661 tests against
467 in the existing baseline. The baseline-file and frozen-hash checks passed;
the count check failed. Existing tests, lockfiles, configs, and baseline remain
untouched. Additional unrelated work is present in this shared checkout.

Eight sessions per model is a feasibility target, not a model-recognition claim
gate. The proposal requires at least ten independently verified groups per model
for such a claim, plus the hierarchy, null, and nuisance checks. A DVIDS-only
pilot cannot establish cross-outlet invariance. Producer and camera differences
within DVIDS must still be tracked. Public metadata is not audio-quality or
independence evidence.

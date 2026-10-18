# P0 annotation import and export verification

The supplied `annotation.json` passed the frozen P0 import/export contract on
2026-10-11. All six artificial recordings have an interval or logged rejection.
The seven intervals comprise one foreground, one background and five rejections.

The input matches `/Users/magi/Downloads/annotations.json` semantically and binds
to the final synthetic review batch. Its original bytes are retained outside the
checkout; the supplied file is unchanged. Batch/config/tool-version bindings,
private mapping commitments, review-audio hashes, connected groups and roles all
passed validation.

Two mono clips from `fixture-b` were exported at the native 16,000 Hz rate:

| Kind | Interval | Native frames | Samples |
|---|---|---|---|
| Foreground | 2–6 seconds | 32000–96000, end exclusive | 64000 |
| Background | 0–2 seconds | 0–32000, end exclusive | 32000 |

Each exported waveform matches its raw fixture slice sample for sample. All five
rejection records remain in the manifest and export record. The original 55-file
P0 freeze still verifies; its historical report and protocol were not rewritten.

The detailed evidence, copied input, manifest, reuse audit and native clips are at
`artifact:military_signature/p0/browser-roundtrip-20261011T232425Z/`. The main
record is `verification.json`; clip records are in `native_clips/export.json`.

The user subsequently confirmed the Chrome run: “Yes, playback and display
worked,” in response to a question covering audio playback, waveform,
spectrogram and interval-control display. This completes the requested human
browser check. The exact question, answer and input hash are retained in
`browser_confirmation.json` beside `verification.json`.

The annotation artifact verifies the synthetic import/export workflow; the
separate user confirmation supplies run-level playback/display evidence.
Browser version, observation time and per-recording playback traces remain
unknown. Vehicle labels remain unknown; these artificial signals provide no
model-recognition evidence. Real-media processing, fitting and P1 still require
separate authorization under [PROTOCOL_DRAFT.md](PROTOCOL_DRAFT.md).

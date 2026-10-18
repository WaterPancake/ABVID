# Military acoustic signature: frozen P0 draft

Protocol `military_signature_p0_v1`, 2026-10-11. This document and
`p0_protocol.json` specify preparation and a prospective study. The original
[PLAN.md](PLAN.md) and [PROPOSAL.md](PROPOSAL.md) remain the design history; the
corrections below govern P0. Freeze hashes are in `p0_freeze.json`.

## Authorization and result boundary

P0 permits literature research, public HTML metadata inspection, protocol design,
and local implementation with artificial fixtures. It permits no real media
collection or processing, representation extraction, classifier/CAST fitting,
protected-source evaluation, or P1 execution. The CLI checks the canonical P0
configuration and refuses real-media manifests. This is a declared-provenance
guard, not a cryptographic test of whether a waveform is artificial.

The separately scoped project does not restart the retired military benchmark,
Milestone 7 or OpenClaw. Historical source roles, locks and evidence stay intact.
Reusable implementation is under `src/abvid/signature`. Media, features and
generated validation results use external `data:` and `artifact:` roots resolved
through `abvid.paths`. No large artifact belongs in Git.

P0 outputs establish preparation-tool behavior on artificial chirp/noise signals.
They establish neither vehicle acoustics nor classification performance. No
synthetic CAST source recovery or transfer experiment has been run.

## Evidence and labels

Read each literature Markdown representation before its PDF; verify substantive
claims against the PDF and record paper ID, one-based PDF page, printed page and
section/table. `source_evidence.json` separates author statements, interpretations
and unknowns. Public metadata snapshots have retrieval times and byte hashes in
`candidate_metadata.json`; they are candidates, not admitted recordings.

Captions supply uncertain model/variant labels. Record the exact supporting
caption, source locator and confidence; keep variants unknown when not stated.
Attach labels after blind trimming. A separate curator holds the captions and
private mapping; a reviewer receives only the blind directory. The acoustic
reviewer must not inspect source pages or filenames while selecting intervals.
Sounds can themselves reveal identity, so caption hiding does not prove complete
experimental blinding. The same agent reviewing both roles is a tooling check,
not an independent blinded human review.

L1 is tracked/wheeled; L2 is turbine/diesel/other. Apply these labels only with
evidence for the actual identified model/variant. Do not turn an uncertain family
caption into an exact engine. L3 denotes engine-order structure only where order
ground truth is established; otherwise it remains unknown and no L3 AP is
reported. L4 is model; L5 variant is exploratory and has no claim gate here.

The present evidence verifies the M1/AGT1500 turbine association and the diesel
Leopard 1/2 examples in P04. Exact engines for the candidate variants, and the
proposal's T-72/T-80/T-90/T-80UD table, remain incomplete. Inaccessible primary
sources are recorded as failures. Memory and search snippets do not complete
that table.

## Manifest and admission

`manifest_schema.json` is JSON Schema Draft 2020-12. The runtime additionally
checks interval references, provenance consistency, connected groups and roles.
All fields are explicit; unavailable values use `unknown`.

- Assets retain URL, title, date taken/posted, producing unit, producer, location,
  outlet, per-asset rights note/URL, retrieval time, raw SHA256, native rate,
  channels, duration, recording session and its evidence, verified-independence
  flag, physical vehicle identity and supporting evidence. Preserve conflicting
  dates in the metadata ledger; do not invent a session from a filename or ID.
- Segments retain native-time boundaries, foreground/background/rejected kind,
  role, audible vehicle count, motion, contaminants, reviewer/tool version,
  unknown distance/microphone where unmeasured, and rejection reasons.
- Labels live in a separate table keyed by segment. Reuse links record evidence.
  Model/variant labels never enter the fingerprint algorithm.

Distinct asset IDs, units, posting dates or venues alone do not prove independent
recordings. Session evidence must establish distinct recording events with all
known source links considered. Repeated views of an event, shared raw hashes,
suspected/confirmed waveform reuse and known repeated physical vehicles form
one transitive component. Session IDs are globally scoped, never local counters.
Known same-individual vehicles stay together across roles. Unknown physical
identity stays unknown; it does not certify unseen-individual generalization.

All unknown-session assets are conservatively joined and cannot enter held-out
roles. They are research leads requiring admission, not extra independent groups.
Roles are assigned to whole connected components after the reuse audit; accepted
segments from one component cannot cross development/validation/evaluation.
Quarantined rejection records remain in the manifest and are never training data.

The schema can describe future real assets, but P0 executable validation rejects
them. A future authorized phase requires its own versioned configuration and
admission checks; flipping a P0 flag is deliberately rejected.

## Frozen trimming rules

The offline viewer provides randomized opaque recording IDs, original-rate audio,
waveform and spectrogram, with no caption/title/original filename/session labels.
Its 8 kHz mono spectrogram is a display only: 512-point Hann FFT, 128-sample hop,
0–4 kHz, 80 dB below the recording maximum. Display amplitude is relative. No
denoising, equalization or loudness change is applied to playable/trimmed audio.
Re-encoding to FLOAT WAV removes source tags; it is not a retained raw master.

Record all auditioned exclusion intervals and reasons. Unreviewed time is not
silently admitted. Rejections may overlap accepted intervals as revision records;
accepted intervals may not overlap each other. A reviewed asset must have at
least one interval or a logged rejection before annotations can be saved/imported.

1. Accept foreground only when exactly one vehicle is audibly distinguishable.
   Accept 2–60 seconds. Reject music, narration, gunfire or aircraft contamination;
   reject masking wind, clipping, unclear count, multiple vehicles, uncertain
   audibility, prominent edits and codec damage. Log the applicable reason.
   Mark lesser wind/crowd/AGC artifacts; retain a clean-reference subset for the
   primary analysis. Unknown contamination prevents admission to that subset.
2. Obtain at least one uncontaminated 2–60 second, zero-audible-vehicle background
   interval from the same raw asset, preferably adjacent without a camera edit.
   Log absence explicitly. Another asset in a session is not automatically an
   equivalent channel/noise reference. Backgrounds remain in the foreground's
   group and role. Missing/contaminated backgrounds preclude normalized analyses.
3. Motion may be idle/rev/accelerate/constant/pass_by/brake/unknown. Audio alone
   need not distinguish speed or geometry; do not guess distance, RPM or microphone.
4. Preserve raw hashes and export at the native rate and original channel count.
   Frame bounds are `floor(start_seconds * rate)` and `ceil(end_seconds * rate)`;
   save those integers, clip SHA256 and raw SHA256. This expands by less than one
   sample at each end. The exporter changes no samples before FLOAT serialization.
5. Keep blind files and private mapping in separate directories. Import checks
   manifest/config/batch commitments, mapping integrity and tool version. Output
   paths must be fresh; reruns never overwrite past review or export artifacts.

The current tooling enforces duration, counts, key forbidden contaminants, overlap
and provenance. Audibility, masking, clipping and semantic review remain reviewer
judgments. A synthetic oracle validates mechanics, not reviewer accuracy. Before
P1, an authorized human should complete the supplied synthetic packet in a real
browser and inspect the annotation/import round trip.

## Reuse audit

Reuse `abvid.data.overlap` without changes: mean-channel audio at 8 kHz, landmark
candidate search, then absolute normalized waveform correlation at least 0.9,
with the canonical algorithm settings recorded in `p0_protocol.json`. Index
four-second reference windows every two seconds, at most 600 per asset. Assets
are bounded at 1,200 seconds. Exact raw hashes also join assets.

Matches are conservative grouping suggestions, retained with correlation,
offsets, votes and input hashes. Suspected reuse is merged before splitting even
when it reduces the apparent sample count. Shorter-than-four-second recordings,
silence and no-landmark windows remain unresolved. Boundary overlaps, averaging
opposite stereo channels, codecs, edits, mixtures or time stretching can defeat
the audit; no match certifies independence. Periodic signals may overmerge true
independent events. Record both limitations rather than deleting awkward assets.

## Background and physical identifiability

For `x = Hs + n`, a source-free background observes `n`, not the unknown channel
response `H`. Different `(H,s)` pairs produce the same `Hs`. This is an analyst
identifiability argument, not a claim that the cited papers validated inversion.
Background spectra permit a bounded noise reference or whitening ablation. They
do not remove microphone/propagation effects or turn CAST effective-observation
fits into clean source parameters. Use training-only statistics and freeze the
background operation before target evaluation. Inversion requires independent
calibration or separately justified assumptions in a new protocol.

Source-model dominance over propagation mismatch remains a hypothesis. Report
effective descriptor stability and failures without calling them combustion,
true RPM, track ground truth or recovered physical sources.

## Prospective evaluation contract

Everything in this section is a draft for separately authorized phases. No model
or metric implementation is executed by P0. Freeze checkpoints, layer/pooling,
window length, CAST budget and population before extraction. Use frozen BEATs768,
MFCC26 and declared per-segment CAST observations in Tier 0. UMAP/HDBSCAN plots
are exploratory; primary retrieval uses original normalized descriptors and a
fixed distance, not an embedding selected using vehicle labels. Selection of the
exact checkpoint and extraction conventions is a remaining P1 input.

Vehicle labels are allowed only for evaluation probes/metrics, not contrastive
representation learning or hyperparameter selection. Probe preprocessing fits
training groups only. Reporting phase gates uses development labels and exposes
development results; it does not create an untouched final test. Nuisance labels
may supervise a declared training-only session/outlet adversary in P3.

**Eight versus ten.** The P1 target is six specified models with eight admitted,
verified groups each (48 groups, not 48 videos). It is a feasibility/development
pilot. Use four group folds, each holding two groups per model, with fixed seeds
17,29,43,71,101. If quotas are unmet, report the achieved population and unmet
cells, without weakening the threshold after seeing results. Six models and
eight groups are targets; they are not currently evidenced.

An L4 claim requires at least ten verified groups per model in the claim
population, in addition to label quality, hierarchy/null/nuisance checks and new
physical-vehicle support. This number is an original proposal threshold, not a
power analysis or sufficiency guarantee. A subsequent confirmatory protocol must
reserve new, unexposed groups and document multiple physical vehicles per model
with identities disjoint from training. Unknown physical identity supports only
an unseen-recording claim. Do not relabel the exposed eight-session pilot as
confirmatory evidence by adding two recordings. Final-test allocation and power
analysis require a new pre-evaluation freeze and are unresolved in P0.

**Retrieval.** Exclude the query's entire connected component from the gallery.
Compute AP at L1–L4 only with supported labels and at least one eligible positive
in another component. Average queries within components, then components within
classes, then classes. Report excluded queries and per-class/group support. For
model-level primary retrieval, use single-model groups; mixed-model events remain
linked and are retained as secondary, matched-event diagnostics. Their role as
confound controls must not create extra independent group counts.

**Null and uncertainty.** Permute group-level label blocks 1,999 times, preserving
within-group relations; never permute windows independently. For the primary
single-model groups, shuffle complete model labels among exchangeable groups and
derive their hierarchy labels. Use producer/location strata only where there is
exchangeability and class overlap. If the allocation has no defensible
exchangeable support, the null is unknown and the claim gate fails. Use 2,000
class-stratified group bootstrap replicates, resampling whole components; keep
methods paired in difference estimates. Report the 95% CI and all five seeds.
Seeds/windows are not additional independent experimental units.

**Confounds.** Report balanced accuracy and the chance-normalized score
`N = (BA - 1/K) / (1 - 1/K)` for each K-class probe, including negative values.
K=1 is undefined and remains unknown. Vehicle probes hold out connected groups.
Session probes use a declared blocked-time 70/30 split within known groups with
at least two temporal blocks; these are within-session diagnostics. Producer and
outlet probes must hold out groups with label overlap and use only training-fit
preprocessing. The different session/model probe tasks prevent causal attribution
from their score difference; report both protocols and class counts.

Define descriptive session dominance as `N_session - N_model >= 0.10` with its
paired group-bootstrap lower CI above zero, both before and after the fixed noise
reference ablation. Retain a ratio only as an optional secondary diagnostic with
an explicit denominator; raw accuracies with different chance levels are not a
confound ratio. A nuisance reduction accompanied by lost model retrieval is a
failure, not successful invariance. Single constant-outlet DVIDS yields no outlet
probe and cannot establish outlet invariance. Distinguish distribution outlet,
originator, producing unit and camera (camera usually unknown).

## Measurable phase gates

| Phase | Required evidence before continuation |
|---|---|
| P0 | Canonical config/schema match; focused regression and artificial end-to-end checks pass; source/unknown ledger and freeze check pass. Report any canary/human-review handoff. P1 requires separate collection and execution authorization. |
| P1 | Admission/reuse audit frozen; 6 × 8 group target achieved or explicitly reported as an incomplete feasibility population; baselines, cross-group retrieval, normalized nuisance probes and missing-background counts reported. Stop escalation if session dominance persists after the frozen noise-reference ablation and coarse hierarchy has no above-null evidence. |
| P2 | No fine model claim without L1/L2 macro mAP-minus-null lower 95% CI above zero and permutation p ≤ 0.025 each. Report only the supported population: M1-only turbine support makes L2 a model-associated contrast, not general turbine/diesel invariance. Order ground truth absent means L3 unknown. CAST source/nuisance recovery needs a separately frozen synthetic generator and negative controls; its numerical thresholds remain a P2 design input. |
| P3 | Paired cross-group mAP improvement over Tier 1 has lower 95% CI above zero; session-probe score-change upper CI ≤ 0; no label-based representation/hyperparameter selection. Report every seed and failed hierarchy level. A constant-outlet probe remains unknown. |
| P4 | Separate outlet/media authorization and admission; overlapping model and hierarchy classes across origin outlets plus frozen unseen-outlet groups. Russian models occurring only at new outlets do not identify an outlet effect. Earlier protected assets keep their roles. |

The P2 gate's L2 limit is material: a DVIDS pilot with only M1 turbines cannot
separate turbine characteristics from M1-associated cues. It can document this
conditional contrast and feasibility; broad powerplant or model recognition
requires additional matched support. No phase gate authorizes its successor.

For open-set evaluation, hold one model and all its linked groups out of training;
freeze a rejection threshold using training/validation only, report unknown-model
recall and known-model false rejection. This requires additional group support
and is not a result of the pilot by default.

## Reproduction and handoff

Run from the repo using its existing CAST environment, without changing extras,
lockfiles or the baseline. Choose fresh external output suffixes:

```sh
PYTHONPATH=src .venv/bin/python -m abvid.signature validate-p0 \
  --fixtures data:military_signature/p0_fixtures/my-check \
  --output artifact:military_signature/p0/my-check
.venv/bin/python -m pytest tests/test_signature_p0.py tests/test_overlap_audit.py
PYTHONPATH=src .venv/bin/python experiments/military_signature/verify_freeze.py
```

The validator records fixture seeds/raw hashes, environment, commit, dirty-tree
status, code/config/schema hashes, native-frame comparisons and rejections.
Its blind directory can be opened offline and reviewed; keep `private/` with the
curator. Use the CLI `review`, `import-annotations`, `export`, `dedup`, `group`,
`validate` and `readiness` subcommands for individual steps (`--help` lists args).

Human-only work: inspect all new test lines before merge, refresh the harness
baseline in BOTH environments in the same legitimate suite-change commit, and
review any other protected-file diffs in this shared dirty checkout. An agent
must not refresh the baseline or suppress its failure. Browser playback/layout
requires a human check because Firefox computer-use access was rejected.

Scientific handoff: authorize a bounded P1 collection/execution protocol, identify
its exact checkpoint/extraction choices, resolve asset rights and independence,
and specify the confirmatory allocation before any model-level claim. Current
metadata, artificial tooling checks and historical protected sources do not
satisfy those inputs.

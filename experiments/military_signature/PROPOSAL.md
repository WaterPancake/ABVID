# Side project: unsupervised military-vehicle acoustic signatures (proposal)

Branch `military-signature`, proposed 2026-10-11. **Proposal only. Nothing is
collected, fitted or authorized by this file.** The military tracked/wheeled
program is retired from `main` under `AGENTS.md`. This branch is an explicit,
separately scoped side project: it does not resume Milestone 7, the OpenClaw
workflow or the earlier benchmark, and it does not change their historical
results or the roles of protected sources (including the reviewed T-80BVM/T-90M
segments).

P0 preparation is now separately authorized in the user session. Its corrections,
evidence and remaining gates are in [PROTOCOL_DRAFT.md](PROTOCOL_DRAFT.md) and
[P0_STATUS.md](P0_STATUS.md). The proposal below remains the design history.

## Question

Does real-world audio of particular military vehicle models contain a
**session-invariant structure** that matches the postulated physical signature?
Can an unsupervised method recover it without vehicle labels? CAST supplies the
interpretable parametric description, and the learned structure feeds back into
CAST source priors.

## Why the earlier corpus cannot answer it

The archived military corpora (`real_development_v4`: 325 windows; `real_eval_v2`:
271) each draw every vehicle model from **exactly one recording session**, for example
AMX 30 from one 2015 demonstration, the MPF testbed from one 2023 tankodrome and the
TR-85M1 from one 2014 DVIDS range video. Model and session are perfectly confounded:
any classifier or cluster trained there can be identifying the session. The
central design requirement of this project is therefore **many independent sessions
per model**. Earlier DVIDS collection also hit inaccessible assets (the Humvee
regating), so access failures must be recorded rather than silently replaced.

## Postulated signature as a testable hierarchy

Coarse levels must emerge before fine ones. A method that separates T-72 from T-90
but not turbine from diesel has not learned a vehicle signature.

| Level | Postulated cue | Physical basis |
|---|---|---|
| L1 tracked vs wheeled | track-link/sprocket modulation | modulation rate ≈ speed / track pitch |
| L2 powerplant | gas turbine: tonal whine plus broadband; diesel: firing harmonics tied to RPM | turbine vs reciprocating combustion |
| L3 engine-order structure | harmonic weights normalized by firing frequency | four-stroke firing frequency = RPM/60 × cylinders/2 |
| L4 model | combination of L1–L3 plus exhaust/cooling-fan tonal content | engine, transmission and hull |
| L5 variant | small spectral shifts | engine revisions; often below the noise floor |

Working powerplant table, **from memory; verify against cited sources and record
paper/page before use**:

| Model | Powerplant |
|---|---|
| M1 family | gas turbine (AGT1500) |
| T-80 / T-80B / T-80BVM | gas turbine (GTD-1000 family) |
| T-80UD | diesel (exception within the T-80 line) |
| T-72 / T-90 | V-12 diesel (V-46/V-84/V-92 family) |
| Leopard 2 | V-12 diesel (MTU MB 873) |
| Leopard 1 | V-10 diesel (MTU MB 838) |

Unknown values stay `unknown`.

## Data protocol (DVIDS first)

**Phase 1 uses DVIDS only.** Choose allied models with many independent sessions there
(for example M1, Leopard 2, Bradley, Stryker, M109, HMMWV/JLTV). Russian-built models
are scarce on DVIDS and would mostly come from other outlets, which turns
outlet into a confound. They enter only in Phase 4, as a cross-outlet transfer test.

### Segment manifest (one row per manually trimmed segment)

| Group | Fields |
|---|---|
| Provenance | DVIDS asset ID, URL, title, date, unit, location, outlet, license/usage note, retrieval time, raw SHA256 |
| Grouping | `recording_session` (the shoot; the asset ID unless provenance links several assets), conservative connected group |
| Label | caption-stated model/variant, confidence (`confirmed` / `caption` / `unknown`), number of audible vehicles |
| Condition | motion state (idle / rev / accelerate / constant / pass-by / brake), approximate distance, mic type if inferable |
| Contamination | music, narration, gunfire, aircraft, wind, crowd, AGC pumping |
| Timing | start/end within the raw media, trimming tool version, trimmer ID |

### Trimming rules

1. **Hide labels while trimming.** Trim from audio, waveform and spectrogram with the
   caption and title hidden. Record rejection reasons, so trimming choices cannot
   encode the label.
2. **Cut background segments in every session.** Also trim vehicle-absent segments from
   the same video. They estimate that session's channel and noise floor.
3. **Deduplicate before grouping.** DVIDS republishes the same B-roll in many packages.
   Run an audio fingerprint/cross-correlation duplicate search across all assets and
   merge linked assets into one conservative group, as was done for MELAUDIS.
4. **Never split a session or connected group across roles.**
5. **Prefer two kinds of footage:** videos containing several vehicle types (a
   within-session contrast) and the same model across many units, dates and sites.

### Support target

About 10 or more independent sessions per model before any model-level claim. Coarse
levels (L1/L2) can be tested earlier with fewer models.

Audio, video and derived features stay outside the checkout
(`data:military_signature/...`), resolved through `abvid.paths`.

## Methods (labels used only for evaluation)

Session/outlet IDs are provenance, not vehicle labels. They may be used to enforce
invariance.

**Tier 0: baselines and confound measurement (first, cheap).** Use frozen BEATs768,
MFCC26 and per-segment CAST fits, clustered with UMAP + HDBSCAN, and k-means at fixed
k. The first output is a **confound ratio**: how well session and outlet can be
decoded from each representation, versus vehicle model decoded across held-out
sessions.

**Tier 1: CAST parametric signature.**
- Fit CAST per segment: f0 track, firing-order-normalized harmonic weights, noise
  spectrum and envelope.
- Add a modulation block for track-rate structure.
- Estimate each session's channel and noise from its background segments and divide
  them out.
- Cluster on the source block only.

This tier is interpretable and works with small data.

**Tier 2: nuisance-invariant contrastive learning.**
- **Representation:** a small projection head on frozen BEATs. The data are far too
  small to train an encoder.
- **Positive pairs:** two windows of one segment, made to differ only in nuisance:
  random EQ, codec, AGC simulation, and backgrounds swapped from *other* sessions'
  background banks.
- **CAST resynthesis:** keep a segment's fitted source block and resample its channel
  and noise blocks. This produces positives that differ only in recording coloration.
- **Adversary:** a gradient-reversal head against session and outlet ID.

## Evaluation

1. **Cross-session retrieval.** For each segment, rank all others *excluding its own
   session/group*. Report mAP at L1, L2, L3 and L4 separately.
2. **Confound probes.** Linear session and outlet decodability should fall while
   cross-session model decodability holds. A signature claim needs both.
3. **Null.** Permute labels at the session level, not the segment level.
4. **Open set.** Hold out one model entirely and check whether it forms its own region
   rather than merging into a neighbour.
5. **Synthetic recovery.** Generate CAST observations with fixed source parameters and
   varied nuisance, and require the pipeline to recover the source clusters (mirroring
   the CAST-1 fixtures).
6. **CAST question answered either way.** Are CAST source-block parameters more
   session-invariant than BEATs embeddings? This tests whether CAST separates source
   from recording coloration, even if model recognition fails.
7. **Hierarchy gate.** No L4 claim unless L1 and L2 retrieval are clearly above the
   session-permutation null.

Report group-level uncertainty, every failed or rejected segment, and the distinction
from the roadmap: discriminating particular physical test vehicles is not model/type
recognition across examples and domains.

## Phases and gates

| Phase | Work | Gate to continue |
|---|---|---|
| P0 | protocol, manifest schema, trimming tool, fingerprint deduplication; no learning | schema and tool tests pass; human authorizes collection |
| P1 | DVIDS allied pilot: about 6 models × about 8 sessions; Tier 0 only | confound ratio measured; if session dominates and backgrounds cannot remove it, stop and report |
| P2 | Tier 1 CAST signatures with background normalization | L1/L2 retrieval above session-permutation null |
| P3 | Tier 2 contrastive with CAST-resynthesis positives | beats Tier 1 on cross-session retrieval without higher session decodability |
| P4 | Russian-built models from other outlets: cross-outlet transfer test | separately authorized; protected-source roles unchanged |

Downstream use for CAST: cluster-conditional priors over source parameters become
candidates for the deferred CAST-6 learned estimator, only after P2/P3 evidence.

## Repository placement on this branch

- Reusable code in `src/abvid/signature/`: manifest validation, fingerprint
  deduplication, background-normalized CAST features, retrieval metrics and confound
  probes.
- Thin protocol/configuration layer in `experiments/military_signature/`.
- Tests use synthetic fixtures only. Adding tests changes the harness canary count;
  the baseline refresh is human-only and must be called out in review.
- Do not merge into `main` without a decision on how this branch relates to the
  `AGENTS.md` scope statement.

## Known risks

- **Recording artifacts:** camera AGC and codec damage to envelope and level cues;
  music beds and narration; gunfire and aircraft masking.
- **Labels:** caption misidentification of variants.
- **Coverage and confounds:** scarcity of T-72/T-80/T-90 footage on DVIDS and outlet
  confounding.
- **Licensing:** usage terms differ by asset. Many DVIDS assets are US government
  works, but not all; record terms per asset.
- **Expected outcome:** coarse levels (L1/L2) are plausible. Reliable cross-session
  separation at L4 is the open question this project tests, not an expected result.

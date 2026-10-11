# Military acoustic signature: plan

Branch `military-signature`, 2026-10-11. This is a working outline of an
unsupervised side project that learns the postulated acoustic signature of military
vehicle models (T-72, T-80, T-90, M1, Leopard and others) and feeds it back into CAST.
The full design and its rationale are in [PROPOSAL.md](PROPOSAL.md).

**Status: plan only.** Nothing is collected, fitted or authorized.

## 0. Scope

- [ ] Decide how this branch relates to `AGENTS.md` on `main`, which retires the military
      program and forbids resuming it implicitly. This is a separately authorized side
      project, not Milestone 7, OpenClaw or the earlier benchmark.
- [ ] Earlier protected sources, including the reviewed T-80BVM/T-90M segments, keep
      their current roles.
- [ ] Audio, video and features stay outside the checkout
      (`data:military_signature/...`).
- [ ] Record licensing per asset. DVIDS terms vary, and non-DVIDS outlets have their own.

**Core problem.** In the archived military corpora, every model comes from one
recording session, so model and session are perfectly confounded. The project needs
about 10 or more independent sessions per model before any model-level claim.

## 1. Postulated signature hierarchy

Coarse levels must appear before fine ones:

| Level | Cue |
|---|---|
| L1 tracked vs wheeled | track-link/sprocket modulation proportional to speed |
| L2 powerplant | gas turbine (tonal whine plus broadband) vs diesel (RPM-locked firing harmonics) |
| L3 engine-order structure | harmonic weights normalized by firing frequency |
| L4 model | combination of L1–L3 plus exhaust and fan tones |
| L5 variant | small shifts; often below the noise floor |

- [ ] Verify the powerplant table in PROPOSAL.md against cited sources (paper, page).
      It is currently from memory. Example: M1 and most T-80 variants are turbines;
      T-72, T-90 and Leopard 2 are diesels; T-80UD is a diesel exception.
- **Sanity gate:** separating T-72 from T-90 without separating turbine from diesel
  does not count as learning a signature.

## 2. Data and trimming (DVIDS first)

- [ ] **Segment manifest.** Fields:
  - provenance: asset ID, URL, date, unit, location, outlet, license, raw SHA256;
  - `recording_session`;
  - caption model/variant with confidence;
  - number of audible vehicles;
  - motion state;
  - distance and mic type;
  - contaminants: music, narration, gunfire, aircraft, wind, AGC pumping.
- [ ] **Trimming rules:**
  - Trim with the caption hidden, from audio and spectrogram only, and log rejection
    reasons.
  - Cut **background-only segments** from every session to measure its channel and
    noise floor.
  - Run audio fingerprint deduplication across assets, because DVIDS reposts the same
    footage in many packages. Merge linked assets into one conservative group.
  - Never split a session or group across roles.
  - Prefer multi-vehicle videos and the same model across many units, dates and sites.
- [ ] **Phase 1 uses DVIDS allied models only:** for example M1, Leopard 2, Bradley,
      Stryker, M109, HMMWV/JLTV. Russian-built models come later as a cross-outlet
      test, to avoid an outlet = model confound.

## 3. Methods (vehicle labels used only for evaluation)

- [ ] **Tier 0, baselines:** frozen BEATs768, MFCC26 and per-segment CAST fits,
      clustered with UMAP + HDBSCAN. First output: the **confound ratio**, meaning
      how decodable session and outlet are versus vehicle model across sessions.
- [ ] **Tier 1, CAST parametric signature:**
  - Fit per segment: f0 track, order-normalized harmonics, noise spectrum, envelope,
    and a modulation block.
  - Remove each session's channel and noise using its background segments.
  - Cluster on the source block only.
- [ ] **Tier 2, nuisance-invariant contrastive learning:**
  - A projection head on frozen BEATs.
  - Positive pairs differ only in nuisance: EQ, codec, AGC, and backgrounds swapped
    in from other sessions.
  - CAST resynthesis keeps the source block and resamples the channel and noise.
  - A gradient-reversal adversary works against session and outlet ID.

## 4. Evaluation

- [ ] **Cross-session retrieval:** exclude the query's own session/group; report mAP
      at L1–L4.
- [ ] **Confound probes:** session and outlet decodability should fall while
      cross-session model decodability holds.
- [ ] **Null:** session-level label permutation.
- [ ] **Open set:** hold one model out entirely.
- [ ] **Synthetic recovery:** generate CAST observations with fixed source parameters
      and varied nuisance; the pipeline must recover the source clusters.
- [ ] **CAST question, answered either way:** are CAST source parameters more
      session-invariant than BEATs embeddings?
- [ ] **Hierarchy gate:** no L4 claim unless L1/L2 beat the session-permutation null.
- [ ] Report group-level uncertainty and every rejected segment. Discriminating
      particular physical vehicles is not model recognition.

## 5. Phases and gates

| Phase | Work | Continue only if |
|---|---|---|
| P0 | protocol, manifest schema, trimming tool, fingerprint dedup; no learning | tool/schema tests pass; human authorizes collection |
| P1 | DVIDS allied pilot, about 6 models × 8 sessions; Tier 0 | confound ratio measured; **stop** if session dominates and background normalization cannot remove it |
| P2 | Tier 1 CAST signatures | L1/L2 retrieval above session-permutation null |
| P3 | Tier 2 contrastive with CAST-resynthesis positives | beats Tier 1 cross-session without higher session decodability |
| P4 | Russian-built models from other outlets; cross-outlet test | separately authorized; protected-source roles unchanged |

## 6. Feeding back into CAST

- Report how session-invariant the source blocks are, whatever the recognition result.
- If P2/P3 succeed: cluster-conditional priors over CAST source parameters become
  candidates for the deferred CAST-6 learned estimator.

## 7. Repository placement

- Reusable code: `src/abvid/signature/` (manifest validation, dedup, background-
  normalized CAST features, retrieval metrics, confound probes).
- Protocol/config layer: `experiments/military_signature/`.
- Tests use synthetic fixtures only. New tests change the harness canary count, and
  the baseline refresh is human-only.

## 8. Risks and expectations

- **Recording artifacts:** camera AGC and codec damage; music beds and narration;
  gunfire and aircraft masking.
- **Labels:** captions misidentify variants.
- **Coverage:** T-72/T-80/T-90 footage is scarce on DVIDS and confounded by outlet.
- **Expected outcome:** L1/L2 are plausible; reliable cross-session L4 separation is
  the open question being tested.

## Immediate next step

P0: draft the frozen trimming rule and manifest schema, then a small segment manifest
for a few allied models with many DVIDS sessions. Run Tier 0 alone to measure how
strong the session confound is before building anything larger.

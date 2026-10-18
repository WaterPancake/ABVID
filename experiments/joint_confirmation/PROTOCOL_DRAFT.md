# Joint DCASE confirmation pass: DRAFT, NOT FROZEN

Draft 2026-10-11. This is the single scoring pass over the sealed DCASE 2024 Task 10
development set shared by CAST and H2R ([plan](PLAN.md)). It fixes the access,
feature, scoring and statistics machinery. The study-specific heads come from
[CAST-5A](CAST5A_DRAFT.md) and [H2R](H2R_DRAFT.md). Items marked **DECISION** need
the user. The sealed [admission protocol](../admission/dcase2024_t10/PROTOCOL.md)
governs wherever this draft is silent.

## Sequence

1. Freeze both studies' heads (hash lists), this document and the analysis code.
2. **Human authorization A1 (audit access):** download the development audio and run
   the overlap audit. The audit reads archive member IDs and audio only; it never
   opens `segments.jsonl` or computes model scores.
3. Write the overlap-audit result. Minutes with an accepted match (normalized
   correlation ≥ 0.9) are **excluded before scoring** and listed. This exclusion rule
   is fixed now.
4. **Human authorization A2 (scoring):** a file under
   `../admission/dcase2024_t10/authorizations/` naming the seal SHA256, this frozen
   protocol, the audit result and `scoring_passes: 1`. `abvid.data.sealing.authorize`
   must pass.
5. Run the single pass. Extract features once per representation, score every frozen
   head, write all predictions, then mark the seal `consumed`.

## Population and observation

- Candidate events are minutes with exactly one counted vehicle: 1,361 car and 228 cv.
  Classes are car = 0 and cv = 1, reported as the proxy task **car/cv**.
- Groups are the six sites. Sessions are `unknown`.
- **Observation:**
  1. Take the channel mean of the four capsules.
  2. Apply the comparison's frozen resampling route (H1 `core8_2s`: native → 8 kHz →
     16 kHz).
  3. Compute 50 ms-hop RMS frames.
  4. Cut 2 s centred on the maximum-RMS frame, clamped inside the minute without
     padding.
  5. Apply each study's frozen normalization (shape-only for CAST; the H2 receiver
     normalization does not apply to real inputs).
- **Features:** frozen BEATs768 and MFCC26, computed exactly as in H1 using the cached
  encoder checkpoint hash.

## Metrics and statistics

- **Per-head metrics:**
  - macro-F1 at the fixed argmax rule;
  - balanced accuracy;
  - AUROC;
  - AP against prevalence (14.35%);
  - cv recall at 5% car false-alarm rate (descriptive);
  - per-site recalls.
- **Uncertainty:** paired percentile bootstrap, 10,000 draws.
  - Whole sites are resampled, and draws missing a class are rejected and counted.
  - Study-specific selection or bank dimensions are resampled as in development.
  - Fixed seed.
- **Primary contrast (decided 2026-10-11):** CAST RC − RA is the single primary,
  provided CAST passes 5A. H2R is a pre-declared secondary with no multiplicity
  adjustment and no confirmatory claim.
- **Margins (decided 2026-10-11):** 3 macro-F1 points and 3 AUROC points. The AUROC
  margin is a stated convention, not derived from data.
- **Site support:** the report gives every site row with its support. loc2 and loc4 have
  no cv minutes, so their rows show car-only behaviour. loc5 has 4 cv minutes and is
  flagged.

## Precision expectation (metadata only, optimistic)

A Hanley–McNeil approximation for one arm at AUROC 0.65 with 228/1,361 events gives
an event-level SE of about 2.1 points (±4.1). For a paired difference with arm
correlation 0.8, the SE is about 1.3 points (±2.6).

These figures ignore site clustering. With about four informative sites, the actual
whole-site intervals will be wider, possibly much wider. A 3-point effect may well be
inconclusive. That has to be accepted before the freeze, not discovered afterwards.

## Overlap audit (fixed now)

- **Tool:** [`abvid.data.overlap`](../../src/abvid/data/overlap.py), using the
  settings in its `SETTINGS`, frozen with this document.
- **References:** the admitted IDMT (4,413 car/truck clips), MELAUDIS (8,066) and
  AI4TEN released banks (10,382).
- **Queries:** every DCASE development minute (15,013), identified by archive member.
- **Output:** match rows only. A match leads to exclusion as in step 3, and the full
  list is reported.

## After the pass

The DCASE set becomes exposed development data, and the ledger and target panel are
updated. Any redesign needs new confirmation data, such as the planned roadside
collection.

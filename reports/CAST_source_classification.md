# CAST-5A: classification utility within IDMT

Completed 2026-10-12 under the frozen protocol
[`cast5a_utility_v1`](../experiments/cast_utility/PROTOCOL.md) (SHA256 `b19e78b4…6b64`).
Domain: held-out IDMT connected groups, all previously exposed in H1, so this is
**development evidence**. The task is car/truck with frozen BEATs768 (primary) and
MFCC26.

**Primary result: CAST does not add practical classification value beyond
conventional augmentation.** BEATs RC − RA macro-F1 is **+0.47 points, 95% CI
[−0.97, +2.08]**. The upper bound is below the declared 3-point margin, so a practical
gain is **ruled out**. The lower bound is below zero, so no gain of any size is
established. AUROC agrees: −1.78 [−4.75, +0.70]. CAST-only training does carry class
information the unfitted renderer lacks (AUROC 60.6 vs 47.5), but adding CAST samples to
the real parents changes nothing measurable, and neither does conventional
augmentation.

## What was run

Every arm used the same design:
- **Roles:** 30 roles, one per IDMT fold (6) and H1 training selection (5), each with
  190 car + 190 truck real parents.
- **Input profile:** shape-only: the H1 route, then DC removal and unit RMS.
- **Heads:** 7 arms × 2 representations × 30 roles = **420 heads**. Each is H1's fixed
  StandardScaler + C = 1 logistic regression with the argmax rule.
- **Paired bootstrap:** 10,000 draws over folds and selections, seed 20261013, plus a
  site-block sensitivity check.

| Arm | Rows per class | Content |
|---|---:|---|
| R | 190 | real parents |
| RR | 380 | real parents twice |
| RA | 380 | real + one conventional augmentation per parent (3 random peaking EQs ±6 dB, IDMT training-group background at 10–30 dB SNR) |
| RU | 380 | real + unfitted same-renderer prior |
| RC | 380 | real + 190 CAST v15 samples from the role's own population |
| U | 190 | unfitted prior only |
| C | 190 | CAST v15 samples only |

**Parents:**
- 3,245 parents were needed: 1,749 from the CAST-4A run, read through its validated
  loader, and 1,496 fitted here with the unchanged chain.
- **All 3,245 complete, 0 failures.** All 30 role populations and banks are complete.

**Unfitted prior.** As the protocol states, the car and truck spacing ranges overlap
almost completely, so RU and U carry little class information. U behaves accordingly,
near chance.

## Results (percentages; brackets are 95% paired bootstrap intervals)

### BEATs768: arm means over 30 heads

| Arm | Macro-F1 | BA | AUROC | AP | Truck recall | Car recall |
|---|---:|---:|---:|---:|---:|---:|
| R | 52.24 [47.79, 56.91] | 65.23 [61.12, 69.04] | 71.31 [65.18, 76.45] | 30.09 [18.11, 41.85] | 65.57 [55.95, 74.13] | 64.89 [60.85, 68.86] |
| RR | 52.11 [47.70, 56.72] | 65.15 [61.25, 68.81] | 71.05 [64.82, 76.23] | 29.72 [17.87, 41.29] | 65.60 [56.62, 73.88] | 64.70 [60.57, 68.80] |
| RA | 52.26 [47.16, 57.37] | 65.34 [60.57, 69.44] | 71.54 [64.89, 76.89] | 30.73 [18.76, 42.83] | 66.17 [57.02, 74.26] | 64.51 [59.88, 69.34] |
| RU | 51.60 [47.72, 55.87] | 65.43 [62.82, 68.31] | 70.78 [67.17, 74.61] | 27.02 [16.37, 37.83] | 66.96 [60.07, 73.40] | 63.90 [59.51, 68.18] |
| RC | 52.73 [48.44, 56.99] | 64.63 [59.35, 68.65] | 69.76 [60.81, 75.87] | 29.08 [17.40, 40.35] | 62.77 [50.48, 73.12] | 66.49 [62.02, 70.77] |
| U | 34.45 [20.91, 47.24] | 48.88 [47.25, 50.08] | 47.54 [42.54, 51.64] | 12.15 [8.85, 15.05] | 45.34 [11.76, 78.99] | 52.43 [18.86, 86.36] |
| C | 48.50 [41.32, 54.07] | 54.10 [51.45, 57.27] | 60.55 [54.82, 65.70] | 18.46 [11.54, 25.42] | 29.56 [17.26, 47.23] | 78.64 [60.26, 90.39] |

### BEATs768: paired contrasts (points)

| Contrast | Macro-F1 | BA | AUROC | AP | Truck recall | Site-block macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| **RC − RA (primary)** | **+0.47 [−0.97, 2.08]** | −0.72 [−2.76, 1.13] | −1.78 [−4.75, 0.70] | −1.64 [−4.41, 1.12] | −3.40 [−8.84, 0.32] | +0.47 [−0.72, 2.06] |
| RC − RR | +0.61 [−0.31, 1.60] | −0.53 [−3.67, 1.66] | −1.29 [−5.05, 1.40] | −0.64 [−2.82, 1.52] | −2.84 [−10.03, 1.80] | +0.61 [−0.19, 1.61] |
| RC − R | +0.49 [−0.50, 1.52] | −0.60 [−3.84, 1.78] | −1.56 [−5.42, 1.19] | −1.01 [−3.18, 1.14] | −2.80 [−10.17, 2.03] | +0.49 [−0.37, 1.55] |
| RC − RU | +1.13 [−0.32, 2.27] | −0.81 [−6.74, 2.52] | −1.03 [−8.63, 3.57] | +2.06 [−0.61, 4.53] | −4.20 [−16.71, 2.67] | +1.13 [0.07, 2.30] |
| C − U | +14.05 [−1.15, 29.37] | +5.21 [2.38, 8.47] | **+13.01 [5.52, 18.33]** | +6.31 [0.69, 11.48] | −15.78 [−54.23, 22.82] | +14.05 [−1.67, 29.85] |
| RA − R | +0.02 [−1.09, 1.27] | +0.11 [−2.97, 2.71] | +0.22 [−1.51, 1.31] | +0.64 [−0.59, 1.88] | +0.60 [−5.00, 4.84] | +0.02 [−0.77, 0.86] |

### MFCC26: arm means and key contrasts

| Arm | Macro-F1 | AUROC | Truck recall |
|---|---:|---:|---:|
| R | 55.55 [51.00, 59.81] | 76.65 [71.95, 80.45] | 66.73 [54.20, 78.37] |
| RA | 54.95 [51.51, 58.33] | 76.71 [73.76, 79.37] | 69.06 [59.77, 77.79] |
| RC | 55.40 [50.28, 60.36] | 75.86 [70.55, 80.04] | 64.12 [48.16, 76.73] |
| U | 37.41 [25.33, 47.79] | 51.54 [41.99, 59.87] | 38.47 [9.53, 67.25] |
| C | 31.51 [22.09, 40.51] | 52.98 [45.59, 59.15] | 69.12 [53.50, 84.63] |

MFCC RC − RA: macro-F1 +0.45 [−1.80, 2.65] and AUROC −0.85 [−3.77, 1.01]. MFCC C − U
AUROC: +1.43 [−8.80, 11.44]. Under MFCC, CAST-only training is near chance.

### Decision flags (BEATs768 RC − RA, as frozen)

- Practical gain supported: **no**.
- Practical gain ruled out (upper bound < 3): **yes**.
- Any gain (lower bound > 0): **no**.
- Truck-recall regression beyond the 5-point limit: no (−3.40 points).
- AUROC lower bound > 3: no. AUROC upper bound < 3: **yes**.

### Per-fold BEATs768 macro-F1 (mean over five selections)

| Held group | Held events (trucks) | R | RA | RC | C |
|---|---:|---:|---:|---:|---:|
| connected_4001f06f… | 570 (79) | 60.60 | 61.59 | 60.26 | 51.08 |
| connected_654b5df5… | 776 (110) | 54.83 | 54.85 | 55.15 | 36.07 |
| connected_667018ac… | 268 (**14**) | 48.04 | 49.48 | 49.48 | 45.41 |
| connected_72766641… | 1,517 (152) | 49.83 | 49.56 | 50.88 | 57.08 |
| connected_b00d69dd… | 312 (**10**) | 43.69 | 41.71 | 44.35 | 47.47 |
| connected_c6148cf0… | 970 (146) | 56.45 | 56.34 | 56.22 | 53.88 |

RC and RA stay within about 2 points of R in every fold. The two Fraunhofer groups have
only 10 and 14 trucks.

## Interpretation

1. **CAST is not useful as augmentation here.** At 190 real parents per class, with a
   frozen encoder and a linear head, adding 190 CAST samples per class leaves every
   metric within about ±2 points of real-only training (RC − R, RC − RR). The AUROC and
   truck-recall point estimates lean slightly negative.
2. **Conventional augmentation is equally ineffective** (RA − R ≈ 0 on every metric).
   So this setting is insensitive to either kind of extra variation; the frozen
   representation plus linear head may be the limit.
3. **The fitting captures real class information.** CAST-only BEATs heads rank held
   real trucks at 60.6% AUROC, against 47.5% for the unfitted renderer (+13.0
   [5.5, 18.3]). That is still far below real-only training (71.3%). The class signal
   survives the effective-observation model but is weaker than the real recordings it
   came from. It is visible to BEATs and not to MFCC.
4. **Coverage did not become utility.** CAST v15 met its descriptor-coverage objective in
   CAST-4. This result is the mismatch the roadmap told us to record as substantive:
   broader acoustic coverage does not add classification value at this budget and with
   this representation.
5. **Precision:** the paired RC − RA interval is about 3.0 points wide, narrower than the
   3.6–5.1-point planning scenarios, because the paired arms are strongly correlated.
   That is why a practical gain could be ruled out rather than left inconclusive.

**Roadmap decision:** "If RA matches CAST, the fitter has not demonstrated added value."
That condition is met. The precondition for CAST-5B ("A gain beyond RA and RR justifies a
frozen cross-dataset check") is **not** met. Under the joint confirmation plan, CAST
cannot be the primary of the DCASE pass; the plan's consequences are recorded there.

## Limits

- The held IDMT groups were exposed in H1. There are six groups at three sites, and two
  groups have 10 and 14 trucks.
- The result is conditional on: frozen BEATs768/MFCC26, the linear head, the 190-per-class
  budget, the shape-only profile, 2-second observations and the v15 width.
- It does not test: larger synthetic budgets, encoder fine-tuning, other
  representations, cross-dataset transfer, or CAST variants other than v15.
- The unfitted prior's class information is weak by construction, so RC − RU is not a
  test against a strong manual generator.
- RA uses one specific recipe; other augmentation recipes are untested.

## Verification and provenance

- **Heads:** all 420 converged (the fixed fit raises on non-convergence; maximum lbfgs
  iterations was 331). They were saved and hashed before any held prediction.
  - Heads manifest SHA256: `9315e73b…db61`, checked against its summary at scoring time.
  - Every model was re-hashed before use.
- **Arm consistency:** the R arm reproduces the H1 within-IDMT level under the new
  profile: BEATs AUROC 71.31 vs 71.49, macro-F1 52.24 vs 53.08.
- **Determinism:** after a mid-run restart (below), role `utility_s42_0`'s population
  fingerprint `4931ff0d…` was reproduced exactly, and its sample bundle was re-verified
  byte-identical.
- **Incident 1, performance only.** The first RC attempt rebuilt the pinned CAST-4A
  context for every parent because of an eager `dict.setdefault` default. It was
  stopped after one role, fixed, given a regression test, and rerun. The context
  content was identical either way, so there was no numerical effect.
- **Incident 2, branch change.** During the run the working tree was switched from
  `main` (`2a1d40e`) to `learned-audio-signature` (`1937c89`) outside this session.
  That branch differs from `2a1d40e` only by two Markdown files under
  `experiments/military_signature/`, so all executed code was identical. Stages record
  commit `1937c89` with a dirty tree of 77 paths.

**Hashes:**
- Design identity: `bceaf26b983c0ef948fc492ea25045d77d29c5e0c23cd57a60c48f63f88832fc`.
- Features: real `6e1a4b84…`, RA `da036817…`, RU `0d3dd623…`, RC `27e2a4a6…`.
- Results: `ce2df9b3…2bf3` ([index](ARTIFACT_INDEX.md#artifact-aa9ec94d5703)).
- Predictions: `86cd3803…5203`.

**Environments:**
- CAST readiness: torch 2.13, numpy 2.4.6.
- Baseline: torch 2.11, numpy 1.26.4, scikit-learn 1.4.2.
- macOS arm64, CPU only.

**Wall time:**
- Parent fits: about 5 h, with 4 workers.
- RC banks: 7 min.
- Base features: 60 min (shared CPU).
- RC features: 21 min.
- Heads: 21 s.
- Scoring: 23 s.

Tables come from `experiments/cast_utility/report_tables.py`; commands are in the
[README](../experiments/cast_utility/README.md).

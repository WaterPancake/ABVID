# CAST-5A: classification utility within IDMT

Protocol `cast5a_utility_v1`, frozen 2026-10-11 before any CAST-5A fitting,
generation, feature extraction or scoring. It implements the
[CAST-5A stage](../../CAST_ROADMAP.md#cast-5a-test-classification-utility-within-idmt)
with the defaults the user accepted on 2026-10-11 ([draft](../joint_confirmation/CAST5A_DRAFT.md)).
Held IDMT groups were exposed in H1, so every result is **development evidence**.
CAST-4A pinned code is reused unchanged; new code lives in `src/abvid/cast/utility_*.py`.

## Question and primary contrast

Does CAST add classification value beyond conventional augmentation of the same real
parents? **Primary:** frozen BEATs768, macro-F1 of RC − RA, averaged over 30 heads per
arm, with a paired bootstrap. The practical margin is **3 points**.
- A lower 95% bound above 3 **supports** the gain.
- An upper bound below 3 **rules it out**.
- Otherwise the result is **inconclusive**.

A separate claim of "any gain" uses zero as the threshold. RC is not called better if
its mean truck recall is more than 5 points below RA's.

AUROC is co-reported, with the same 3-point margin, as a descriptive decision. The user
accepted that a true 3-point gain will probably be inconclusive
([precision planning](../../reports/CAST5A_precision_planning.md)).

## Roles

- **Partitions:** 30 role partitions, `utility_s{seed}_f{fold}`. For each of the six
  H1 folds and each selection seed in `[42, 123, 456, 789, 1024]`:
  - training = H1 lock `folds[fold].train_ids[str(seed)]` in original order (190 car +
    190 truck);
  - held = `folds[fold].test_ids`;
  - held group = `folds[fold].held_out_group`.

  Roles are built with the unchanged `grouped_design.partition` and `roles.build_roles`
  on the frozen CAST-4A source rows, so the 12 ancestry links are enforced.
- **Parents:** the 3,245-parent union is prepared once with the unchanged four-start
  smooth8 + v7–v11 chain and numerical contract.
  - Parents completed in the CAST-4A run are read through the 4A context's validated
    loader.
  - The other 1,496 are fitted in a CAST-5A context with the same numerical contract.
  - Technical failures are terminal. A role with a failed parent cannot be built and is
    reported as failed; nothing is replaced.

## Common input profile (shape-only)

- **Real audio:** the unchanged H1 observation route (channel mean, native → 8 kHz →
  16 kHz, centred 2 s, Kaiser β = 5), then subtract the mean and divide by RMS.
  Observations with RMS below 1e-7 are rejected and counted.
- **Generated and augmented audio:** the same DC removal and unit RMS, at 16 kHz and
  32,000 samples.

## Arms

| Arm | Rows per class per head | Definition |
|---|---:|---|
| R | 190 | role's real parents |
| RR | 380 | the same parents twice |
| RA | 190 + 190 | real + one RA copy per parent (below) |
| RU | 190 + 190 | real + RU bank of the role's selection seed (below) |
| RC | 190 + 190 | real + 190 CAST samples from the role's own population (below) |
| U | 190 | RU bank only |
| C | 190 | CAST samples only |

That is 7 arms × 2 representations × 30 roles = **420 heads**. Each head uses the
unchanged H1 classifier: StandardScaler + L2 logistic regression with C = 1, lbfgs,
tol 1e-6, max_iter 5,000 and the fixed argmax rule, fitted on its rows only. Car = 0,
truck = 1. Features are frozen BEATs768 (H1 encoder hash) and MFCC26 (H1 config).

**CAST samples (RC, C).**
- One `GroupedPopulation` (unchanged CAST-4A class) per role, built from its 380
  parents and frozen before drawing.
- The arm draws the v15 (T = 1.1) joint sampler from the frozen CAST-4A schedule entries
  for that partition, ordered by seed `[42, 123, 456, 789, 1024]` then index. The first
  190 per class are used: seeds 42/123/456 in full, plus seed 789 indices 0–39.
- These sampling streams are fold-independent, as in CAST-4A. The sampled parameters
  differ by role because each population differs.

**RU, the unfitted same renderer.** One bank per selection seed (190 per class), shared
by that seed's six roles; it learns no statistics. For each sample:
- **Seeds:** PCG64, seeded by `hash_seed(["cast5a_ru_v1", class, seed, index])`, with
  the renderer input ID `cast5a_ru_v1:{class}:{seed}:{index:04d}`.
- **Spacing:** centre `c ~ U(class range)`, with each of 3 knots `c·exp(U(−0.1, 0.1))`
  clipped to the fitter bounds [10, 400] Hz. Spacing is assumed to equal the engine
  firing frequency (a manual assumption).
  - **Car, 26.7–100 Hz:** 800–3,000 r/min (P11 Göksu, §II) with a four-cylinder
    four-stroke engine (P02 AI4TEN, synthetic-car model).
  - **Truck, 25–110 Hz:** 500–2,200 r/min, from an industry article on heavy-duty
    diesel engine speeds (Transport Engineer, "Engine management: idle talk", not
    peer-reviewed), with a six-cylinder four-stroke engine (P02 synthetic-truck model).
  - The two ranges overlap almost completely, so **RU carries little class information**
    and works close to a class-independent control. RC − RU is interpreted accordingly.
- **Other parameters:** harmonic weights ~ Dirichlet(1⁸); harmonic fraction ~ U(0, 1);
  noise weights ~ Dirichlet(1¹⁶). The envelope has 5 anchor values ~ U(0.1, 3.0) at
  equal spacing, linearly interpolated to the 41 controls inside the fitter bounds.
- **Rendering:** the unchanged spectrum16 renderer with sampling streams.

**RA, conventional augmentation.** One copy per parent per role, seeded by
`hash_seed(["cast5a_ra_v1", partition, file_id])`:
1. Start from the shape-only real observation.
2. Apply three cascaded RBJ peaking filters at 16 kHz: f0 log-uniform 100–3,000 Hz,
   Q ~ U(0.7, 2), gain ~ U(−6, 6) dB.
3. Rescale to unit RMS.
4. Add one IDMT `*_SE_CH34-BG.wav` background excerpt at SNR ~ U(10, 30) dB:
   - The background file is chosen uniformly from those whose (site, date) maps to one
     of the role's training groups. The mapping comes from the admitted car/truck rows;
     unmapped background files are excluded.
   - The excerpt goes through the same route, then a 2 s window with a uniform start
     and unit RMS.
5. Remove the mean and normalize to unit RMS.

Background ancestry is recorded, and no background from a held group is ever used.

## Scoring and uncertainty

- **What is scored:** each head on all held recordings of its fold (H1 `test_ids`,
  shape-only profile).
- **Per-head metrics:**
  - macro-F1 and balanced accuracy at the argmax rule;
  - AUROC;
  - average precision against prevalence;
  - per-class recall and precision;
  - confusion matrix.
- **Arm score:** the mean over the 30 heads (six folds × five selections, equal weight).
- **Uncertainty:** 10,000 paired draws (PCG64 seed 20261013) resampling folds and
  selections independently, shared by all arms and representations.
- **Site sensitivity:** resample the three sites, each carrying both of its date
  folds, together with the selections.
- **Groups with few trucks:** the two IDMT groups with 10 and 14 trucks are flagged,
  and per-fold results are reported.

**Supporting contrasts:** RC − RR, RC − R, RC − RU, C − U and RA − R, for both
representations.

## Missing data and stages

Each stage is create-only and hash-recorded under `artifact:cast/utility/`:
1. design;
2. parents (CAST environment);
3. arm banks (CAST environment);
4. features, heads and scores (baseline environment).

A failed role is reported and keeps its slots as missing. The primary interval is
computed only if all 30 roles of both compared arms are complete; there are no
subset-only intervals. No parameter, recipe or threshold is changed after held access.

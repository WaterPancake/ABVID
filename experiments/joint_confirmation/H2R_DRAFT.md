# H2 replication (H2R): DRAFT, NOT FROZEN

Draft 2026-10-11. The goal is to rerun H2's source × path question with the filter
latency removed, then carry it to the joint DCASE confirmation pass.

## What the corrected development evidence says

[`h2r_development.py`](h2r_development.py) assembles the latency-free 2×2 from
existing hashed predictions on exposed MELAUDIS, with no rendering or fitting. Its
results are in `artifact:joint_confirmation/h2r_corrected_assembly_20261011_v1/results.json`.
The macro-F1 values reproduce the published H2 numbers (F00 3.47, F10 3.29) and the
latency-control mean (A1L0 3.68).

| BEATs768 cell | Macro-F1 | AUROC |
|---|---:|---:|
| S0, direct path (H2 F00) | 3.47 | 50.68 [48.03, 60.23] |
| S1, direct path (H2 F10) | 3.29 | 46.55 [43.06, 54.53] |
| S0, corrected path (ground + centred air) | 3.88 | 58.13 [51.09, 62.78] |
| S1, corrected path | 3.47 | 51.03 [43.30, 55.51] |

| BEATs768 contrast | Macro-F1 | AUROC |
|---|---:|---:|
| Path marginal | +0.30 [−0.02, 1.21] | +5.96 [−6.51, 9.52] |
| Source marginal | −0.29 [−0.60, 0.08] | −5.61 [−10.04, −3.71] |
| D = source − path | −0.59 [−1.69, −0.02] | −11.58 [−15.84, −0.45] |

**What this shows:**
- With latency removed, the H2 path benefit essentially disappears. Every BEATs cell
  stays at the always-truck macro-F1 level and close to chance AUROC.
- The S1 source change lowers BEATs ranking.
- MFCC ranks at 65–71% AUROC in every cell, and the corrected path *lowers* its AUROC
  by 5.1 points.

**Consequence:** H2's source-versus-path question cannot be ranked at this
performance floor, because no corrected cell is a usable classifier. Scoring these
heads on DCASE would mostly measure how they fail.

## Renderer v2

[`abvid.simulation.propagation_v2`](../../src/abvid/simulation/propagation_v2.py)
applies the air FIR centred by default and requires any latency to be declared; its
render record states the latency. Tests in `tests/test_propagation_v2.py`:
- `air="causal"` reproduces v1 exactly.
- The centred mode, plus a declared 5/10-sample shift, equals causal for static
  geometry.
- Default relative arrival follows geometry only.
- The centred filters are zero-phase and positive.
- The centred mode reproduces an archived latency-control A1L0 render bit for bit
  (archive-marked test).

## Options for H2R (decided 2026-10-11: option 1)

1. **Secondary only, no new training (recommended now).** Freeze the four corrected
   cells' existing hashed heads as a pre-declared secondary in the joint batch: 4 cells
   × 5 banks = 20 heads per representation, 40 in total. Cost: none. Value: a direct
   out-of-domain check of the corrected development finding.
2. **Redesign before confirmation.** Add a path factor that is physically motivated
   rather than accidental. One example is receiver height and range sampled from the
   DCASE sites' declared geometry (metadata only), which varies the ground-interference
   comb. Develop it on exposed MELAUDIS and freeze it only if some arm reaches a
   usable development level. This needs renderer v2 corpus generation (about 9,600
   observations) and new heads.
3. **Drop H2R from the joint batch.** Spend the batch on CAST alone.

Whichever is chosen must be fixed before the joint freeze. Option 2's geometry draw
would use DCASE site metadata, which must be declared as target-informed simulation.

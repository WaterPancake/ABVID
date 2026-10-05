# Fixed-vehicle idle/moving A/B diagnostic — completed 2026-09-21

**Domain: real -> real within-session cross-state diagnostic.** The operator
explicitly allowed this one exception to session-separated evaluation. The source
intervals and exact waveforms do not overlap across train/test, but recording
sessions deliberately do. These are not benchmark, independent-validation,
confirmation or Milestone 7 results.

## Outcome

The existing frozen BEATs features support strong separation of the selected
recording/model pairs across idle and steady-speed audio. They are not inherently
incapable of retaining distinguishing information across these states. However,
these experiments cannot attribute that information specifically to engines,
tracks or tires, or separate vehicle identity from recording context.

No encoder was fine-tuned and no sound separation was applied. Three pretrained
feature representations were compared using the same fixed C=1 class-balanced
logistic head, training-only weighted standardization, and threshold 0.5. All
18 preregistered pair/direction/representation fits are reported below.

## Balanced accuracy

Here moving means **steady_speed only**. It excludes accelerating, decelerating,
mixed and unknown intervals. A constant decision gives 50% balanced accuracy.

| Fixed pair | Train -> test | BEATs 768 | PANNs full 2048 | PANNs proxy 35 |
|---|---|---:|---:|---:|
| AMX-30 / HMMWV | Idle -> moving | 100.00% | 65.61% | 69.93% |
| AMX-30 / HMMWV | Moving -> idle | 95.83% | 63.87% | 71.49% |
| AMX-30 / T-72B3 | Idle -> moving | 100.00% | 84.38% | 70.31% |
| AMX-30 / T-72B3 | Moving -> idle | 100.00% | 100.00% | 75.00% |
| HMMWV / Stryker convoy | Idle -> moving | 80.50% | 71.49% | 69.46% |
| HMMWV / Stryker convoy | Moving -> idle | 91.67% | 68.75% | 62.80% |

The stronger BEATs performance here does not supersede its previous 57.05%
unseen-session development result: that result tested a different, substantially
harder question. Do not compare the percentages as if this were an improvement
on the same benchmark or proof that state conditioning caused a gain.

## Per-vehicle failures remain important

BEATs confusion counts (rows true, columns predicted, in the pair order shown):

| Pair | Train -> test | Confusion matrix | Recall first / second vehicle |
|---|---|---|---|
| AMX-30 / HMMWV | Idle -> moving | `[[64, 0], [0, 45]]` | 100% / 100% |
| AMX-30 / HMMWV | Moving -> idle | `[[35, 0], [2, 22]]` | 100% / 91.67% |
| AMX-30 / T-72B3 | Idle -> moving | `[[64, 0], [0, 29]]` | 100% / 100% |
| AMX-30 / T-72B3 | Moving -> idle | `[[35, 0], [0, 8]]` | 100% / 100% |
| HMMWV / Stryker convoy | Idle -> moving | `[[28, 17], [1, 81]]` | 62.22% / 98.78% |
| HMMWV / Stryker convoy | Moving -> idle | `[[20, 4], [0, 28]]` | 83.33% / 100% |

In particular, the 80.50% wheeled-pair score hides **17 of 45 moving HMMWV
windows predicted as Stryker**. The reverse direction is better but is not a
matched-sample causal comparison: the training sets, test sets and sample counts
change with direction.

All BEATs/full-PANNs training balanced accuracies were 100%. Semantic-PANNs
training scores ranged from 97.56% to 100%. Training fit is resubstitution only;
it is not independent evidence of recognition.

## What this establishes, and what it does not

- **Supported within these recordings:** a small linear classifier on frozen
  BEATs features can distinguish both cross-class and same-class fixed pairs
  across the reviewed state change. Generic embeddings need not be discarded
  before investigating the data and representation structure further.
- **Still unresolved:** actual vehicle identity versus shared microphone,
  location, source processing, background and other recording characteristics.
  Every fixed model is represented by only one session in this probe.
- **Not established:** an engine-only signature, isolated track/tire sound,
  state-invariant recognition in new recordings, unseen-model generalization,
  physical individual-vehicle continuity, vehicle presence detection or an
  independent performance ceiling.
- Stryker is convoy audio. T-72B3's eight overlapping idle windows come from
  just nine approved seconds. The 315 distinct parent windows used across these
  tasks are not 315 independent examples, and reuse across pairs is not new data.
- AudioSet pretraining/source overlap is unknown. The BEATs checkpoint's public
  mirror integrity is verified, but equivalence to an official checksum was not
  independently verified; this is unchanged from the preceding comparison.

The next decisive validation would use additional independent sessions of these
same vehicle models, with idle and moving coverage, and keep those sessions out
of head fitting. The planned decomposition gallery remains a separate diagnostic:
this result does not by itself show whether separating components helps.

## Verification and artifacts

- Frozen [protocol](state_pair_ab_protocol.md) and [configuration](../configs/state_pair_ab_v1.yaml).
- [Complete 18-fit table with train/test counts and both recalls](../runs/state_pair_ab_v1/results.md).
- [Metrics and verification](../runs/state_pair_ab_v1/results.json).
- [Exact split records and shared sessions](../runs/state_pair_ab_v1/splits.json).
- [Saved float64 heads](../runs/state_pair_ab_v1/models.json) and
  [per-window predictions](../runs/state_pair_ab_v1/predictions.json).
- [Experiment provenance](../runs/state_pair_ab_v1/experiment.json) and
  [nine artifact hashes](../runs/state_pair_ab_v1/artifact_sha256.json).

All 785 parent windows passed the fail-closed development loader. All 18 models
passed train/test replay (36 checks); maximum runner replay error was zero.
An independent disk-readback check replayed the serialized heads using elementwise
logit sums and recomputed confusion matrices and balanced accuracies with sklearn;
all matched. Nine artifact hashes and eleven current input/code hashes matched.
All six splits were checked for timestamp overlap and exact waveform duplication
across train/test: none. The dedicated command refuses to run without the explicit
diagnostic flag; normal benchmark split guards were not modified.

Full regression suite: **141 tests passed**, including five new diagnostic tests;
new-code lint and `git diff --check` passed. No source/catalog/sidecar, protected
recording, historical benchmark or benchmark checkpoint was changed.

Fitting and in-process replay took 0.093 s; the run including corpus/cache/checkpoint
hash verification took 0.909 s using cached features and one BLAS thread. These
times exclude encoder inference and are not streaming latency measurements. The
host reported macOS/arm64; exact CPU-model query was blocked by the sandbox. That
metadata query stopped the initial attempt before any fitting or results; it was
replaced with an explicit unknown-CPU note before the successful run. No model,
data, split or hyperparameter was changed in response to scores.

Reproduce without overwriting the completed diagnostic:

```bash
MPLCONFIGDIR=/private/tmp/abvid-mpl-state-audit .venv/bin/python scripts/run_state_pair_diagnostic.py \
  --allow-within-session-diagnostic --output runs/state_pair_ab_repeat
```

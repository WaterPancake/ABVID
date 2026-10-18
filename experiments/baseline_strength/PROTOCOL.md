# B1: baseline strength within IDMT

Protocol `baseline_strength_v1`, frozen 2026-10-12 before feature extraction or
fitting. This is development evidence on previously exposed IDMT groups, with no
MELAUDIS, DCASE or synthetic access.

## Question

How much can frozen-representation choices improve held-out-group car/truck
discrimination within IDMT? The choices are encoder or layer, bandwidth, training
budget and regularization. The H1 baseline is weak and overfits: BEATs reaches
training AUROC 100 but held-out AUROC 71.5
([H1 diagnostics](../../reports/H1_source_diagnostics.md)). Simulator components cannot
be told apart at that level.

## Data and folds

- **Data:** all 4,413 admitted IDMT sE8 CH34 car/truck excerpts (frozen CAST-4A source
  rows), with labels car = 0 and truck = 1.
- **Folds:** the six H1 outer folds (held = H1 `test_ids`).
- **Budget `b190`:** H1 lock `train_ids[seed]`, seeds `[42, 123, 456, 789, 1024]`.
- **Budget `ball`:** every recording of the fold's five training groups, with
  `class_weight = "balanced"` and a single fit.

## Grid (96 configurations, all reported)

| Factor | Levels |
|---|---|
| Encoder | `beats_l12` (= H1 BEATs768), `beats_l6`, `beats_avg` (mean of the 12 layer means), `cnn14` (PANNs CNN14 2048-d embedding, official checkpoint SHA256 `0dc499e4…4b31`), `mfcc26` (H1 config), `beats_l12+mfcc26` (concatenation) |
| Bandwidth route | `core8`: the unchanged H1 route (channel mean, native → 8 kHz → 16 kHz, centred 2 s, Kaiser β = 5). `full`: channel mean, native → 16 kHz directly, centred 2 s. CNN14 takes the route's 16 kHz output resampled ×2 to 32 kHz |
| Budget | `b190`, `ball` |
| L2 strength | C ∈ {0.001, 0.01, 0.1, 1} |

Every head is a train-only StandardScaler plus lbfgs logistic regression (tol 1e-6,
max_iter 5,000, fixed argmax rule). Encoders are frozen.

## Outcomes

- **Per fit:** held-group AUROC (primary), balanced accuracy, macro-F1 and AP.
- **Configuration score:** the mean over the six folds of the seed-mean (`b190`) or of
  the single fit (`ball`).

**Reference R0:** `beats_l12`, `core8`, `b190`, C = 1 (H1).

**Primary estimate: nested selection.** For each outer fold, every configuration gets
an inner leave-one-group-out score over the five training groups:
- train on four groups (`b190`: the fold's seed-42 selection restricted to those groups;
  `ball`: all their recordings);
- take AUROC on the fifth;
- average the five.

The configuration with the highest inner score is selected; ties go to the earlier
entry in grid order. The procedure's outer score for that fold is the selected
configuration's outer score. Outer held groups are never used for selection.

**Primary contrast:** the procedure minus R0, paired by fold. 95% percentile bootstrap
over folds, 10,000 draws, PCG64 seed 20261014, plus a site-block sensitivity. The
practical reference is 3 AUROC points.

**Secondary:**
- the full 96-configuration outer table;
- descriptive marginal effects by factor;
- the best fixed configuration by outer AUROC, explicitly flagged as **post hoc and
  optimistic**.

## Boundaries

Feature caches, fits and results are create-only under
`artifact:baseline_strength/b1_20261012_v1/`. If any fit fails to converge, the fit is
recorded as a failure and nothing is replaced. The result says what to carry forward
into a later frozen comparison. It is not an untouched estimate.

# Semantic unseen-session evaluation

Date: 2026-08-21

Status: strongest real-session category baseline so far; Milestone 7 gate not met

Performance claim: **real-only development evaluation on the exact seven-session
corpus identified below; not an independent final test**

## Experiment identity

| Field | Value |
|---|---|
| Run | `runs/m6_semantic_nested_sessions` |
| Git commit | `a23773dda5944413e21ba4c875f1b32aa4ff7da6` |
| Dataset version | `84fb1ef0e9f44df7f8d5c48400ea3f9b3bac3de2b3bdf5ae63004b073883c1dd` |
| Test domain | `real -> real nested unseen recording-session pairs` |
| Native-real corpus | 271 two-second windows; 4 tracked and 3 wheeled sessions |
| External encoder | Frozen PANNs Cnn14 pretrained on AudioSet |
| PANNs checkpoint SHA-256 | `0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31` |
| Classifier | Session-balanced logistic regression |
| Features | Fixed 35-label vehicle/engine subset of the 527 AudioSet outputs |
| Outer folds | Every tracked-session x wheeled-session pair: 4 x 3 = 12 |
| Inner selection | Pair-wise held-out sessions among the remaining data |
| Regularization candidates | `C = 0.001, 0.01, 0.1, 1, 10` |
| Random seed | 42 |

The semantic feature list was fixed from the AudioSet ontology before this formal
evaluation. It includes vehicle, road/rail motion, engine, idling/revving, rattle,
and clatter outputs. It excludes unrelated low-variance speech and music outputs
that acquired very large coefficients in the unrestricted 527-output probe.

For every outer fold, one complete tracked and one complete wheeled recording session
are unavailable to both fitting and hyperparameter selection. Candidate
regularization is ranked by mean balanced accuracy across inner held-out session
pairs. The final fold model is then refit on every remaining session and evaluated on
the untouched outer pair. Training gives equal total mass to each class, equal mass
to every session within a class, and equal mass to windows within a session.

## Results

| Metric across 12 outer pairs | Mean | Sample SD | Median | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| Balanced accuracy | **70.72%** | 19.21% | 74.27% | 40.63% | 94.17% |
| Macro F1 | 65.56% | 24.76% | 72.52% | 27.50% | 91.83% |
| Tracked recall | **89.37%** | 10.69% | 93.32% | 66.67% | 100.00% |
| Wheeled recall | **52.07%** | 37.90% | 64.29% | 0.00% | 91.67% |

Both held-out sessions receive the correct recording-level prediction in 7 of 12
outer folds (58.33%). Fold dispersion is descriptive, not a confidence interval:
outer pairs are correlated because the same session participates in multiple pairs.

Performance is source-dependent:

- Maserati wheeled recall is 78.33% to 91.67% across its four outer contexts;
- Goodwood wheeled recall is 57.14% to 85.71%;
- Ford Model T wheeled recall is 0% to 5.88% and is the remaining catastrophic
  unseen-session failure;
- tracked recall is at least 66.67% in every outer pair.

The restricted semantic representation therefore improves category-level evidence
and avoids the earlier fixed-test-only interpretation. It does not establish a
reliable wheeled category: a model that recognizes race-car and revving sessions but
nearly always rejects an unseen startup/idle session is still learning an incomplete
category boundary.

## Working Milestone 7 gate

To prevent later experiments from redefining “reliable,” the next corpus/model must
meet all of these development gates before hierarchical or open-set work begins:

1. at least five independent, reviewed recording sessions per class;
2. nested unseen-session mean balanced accuracy of at least 75%;
3. mean recall of at least 70% for both tracked and wheeled classes;
4. no held-out recording session with below-50% majority/window recall;
5. confirmation on a newly admitted, locked test pair that was not used for model or
   hyperparameter development.

The current run fails items 1, 2, 3, 4, and has no fresh confirmation for item 5.
It must not be used to authorize Milestone 7.

## Next action and acquisition status

The dominant need is a wheeled startup/idle or heavy-vehicle session acoustically
different from Goodwood and Maserati, plus additional moving tracked material. The
catalog already records prior operator approval for T-18, Vanwall, cobblestone-car,
and night-traffic sources, but earlier collection attempts returned Wikimedia HTTP
429. A compliant one-source retry of `target-wheeled-car-cobblestone-pass` on
2026-08-21 also returned HTTP 429 and is recorded in
`data/collection_manifest.jsonl`. No alternate download route was used.

Once the provider cooldown clears—or separately approved non-Wikimedia candidates
become collectible—the new sessions should be reviewed, added to a new corpus
version, and assigned so at least one tracked/wheeled pair remains untouched until
the model and gate are frozen.

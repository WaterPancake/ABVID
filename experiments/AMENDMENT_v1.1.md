# Protocol amendment v1.1: car/truck and corrected IDMT sensor selection

Date: 2026-09-28. Active protocol: `sim_components_v1.1`.

The user requested correcting channel selection and leaving motorcycle out of training for now. This amendment defines a closed-set **car versus truck** core task. It excludes motorcycle from core validation and testing as well: retaining an untrained third class would require a different, open-set experiment. Raw dataset files remain unchanged. No experiment is implemented or run.

The updated specifications are [EXPERIMENTAL_PROTOCOL.md](EXPERIMENTAL_PROTOCOL.md), [DATASET_PROTOCOL.md](DATASET_PROTOCOL.md) and [EXPERIMENT_MATRIX.csv](EXPERIMENT_MATRIX.csv). The previous three documents are preserved byte-for-byte in [archive/sim_components_v1.0](archive/sim_components_v1.0/README.md), with SHA256 checksums. The literature review and original [metadata audit JSON](dataset_readiness_audit.json) remain historical evidence, not current split manifests.

## Changes

| Component | v1.1 rule |
|---|---|
| IDMT labels | Admit only annotated vehicle passings with filename class `C` or `T`; map to car=0, truck=1. |
| IDMT microphone | Select provider token `SE`, documented as sE8. Require `CH34` for these two classes; average the two channels contained in each WAV. Exclude `ME_CH12`. Unexpected combinations are quarantined for metadata review. |
| Core classes | `civilian2_v1`, ordered `[car, truck]`, for E0, E1, E2-B and all E3 cells. |
| Motorcycle | Exclude from every core train/validation/development-test/final-test role, with reason `class_out_of_scope_v1.1`. Retain raw files and provenance relationships. Do not relabel as truck/car or add an unknown class. |
| MELAUDIS | Retain `FF` and exactly `1V-Car` or `1V-Truck`; `1V-MC` is excluded. Other eligibility and leakage controls remain unchanged. |
| Published synthetic bridge | Use car/truck only from the supplied procedural-labelled corpus. Source/template/run grouping remains required. |
| Training budget | Keep 200 events/class: 400 total per core fit. Synthetic validation remains 50/class: 100 total. E3 has 500 events/cell/seed, 2,500/cell over five seeds and 10,000 across four cells. |
| Model profile | `CORE8_BEATS_LR1_CT`: same frozen BEATs encoder and preprocessing; train-only scaler and binary L2 logistic regression, C=1, with probabilities ordered car/truck. |
| Metrics | Core macro-F1 and balanced accuracy cover two classes. Brier is the event mean of the sum of squared errors across both class probabilities. All confidence-interval and support rules remain unchanged. |
| Published CNN replay | E2-P-R/S remain frozen historical **three-class** replays under `civilian3_published_v1`. No new training, output deletion or test-row removal. Their scores are not subtracted from binary-core scores. |

Evidence for channel selection is the supplied [IDMT README, File naming convention](../dataset/IDMT_Traffic/readme.md), the provider [metadata script docstring](../dataset/IDMT_Traffic/annotation/import_idmt_traffic_dataset.py), and the [readiness audit](../reports/dataset_readiness_audit.md). Channel identifiers denote original stereo pairs; microphone identity comes from the separate `SE` token. The two waveform columns in an `SE_CH34` file represent original channels 3 and 4, not a reason to select a different file.

## Metadata check and remaining split constraint

A fresh filename-only check of `idmt_traffic_all.txt` confirms **3,902 car and 511 truck files**, all labelled `SE_CH34`, across the following six site/date groups:

| Site/date | Car | Truck |
|---|---:|---:|
| Fraunhofer-IDMT / 2019-10-22 | 302 | 10 |
| Fraunhofer-IDMT / 2019-10-23 | 254 | 14 |
| Langewiesener-Strasse / 2019-11-18 | 824 | 146 |
| Langewiesener-Strasse / 2019-11-19 | 1,365 | 152 |
| Schleusinger-Allee / 2019-11-12 | 491 | 79 |
| Schleusinger-Allee / 2019-11-13 | 666 | 110 |

These are candidate file counts, not final admitted independent events. Both classes now occur in every candidate group. Hohenwarte contains only excluded motorcycles and cannot count as support for the binary task. Construct provenance groups before filtering, then allocate only groups containing eligible core events.

The original split rule is retained: `ceil(0.60 G)` train, `floor(0.20 G)` validation, remainder test. At G=6 this gives **4/1/1 groups**. The retained minimum support requires at least three training groups and two groups in each validation and test partition: at least seven disjoint groups overall. Therefore **the channel/class correction is complete, but E0/E1 execution remains blocked by the split design**. No seed selection or random clip split can create the missing independent groups. No split roles were assigned during this amendment.

A further metadata-based split-design amendment or additional independent recording groups is needed before training. The present amendment does not weaken the support floors. MELAUDIS has 7,854 car and 256 truck filename candidates under the revised filter; its session relationships, duplicates and final split eligibility still require admission. AI4TEN's supplied procedural-labelled inventory contains 5,000 clips per retained class; historical CNN assets and synthetic source/run provenance remain pending as previously documented.

## Unchanged controls and interpretation

Keep two-second windows, 8 kHz intermediate bandwidth, 16 kHz encoder input, the pinned encoder/checkpoint, fixed C=1, no target hyperparameter tuning, no augmentation outside the specified simulator factors, and the same source/path intervention bounds. Replicate seeds remain `[42,123,456,789,1024]`, split seed `20260927`, bootstrap seed `314159`. The protocol ID in deterministic hashes changes to v1.1; any eventual memberships will belong to v1.1, not v1.0. No earlier memberships were materialized.

Source/propagation hypotheses are now conditional on car/truck discrimination. Results cannot establish motorcycle transfer or be treated as a reproduction of published three-class accuracy. Runtime ranges remain conservative planning estimates, not measured binary-task timings. Existing military benchmarks and protected recordings are unaffected.

Validation of this amendment covers filename counts, selector consistency, CSV structure/class maps and dependent budgets, agreement between the two protocol documents and the matrix, and unchanged v1.0 archive checksums. It does not constitute dataset admission, audio preprocessing or an experimental result.

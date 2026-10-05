> Repository layout updated 2026-10-05. Original experiment paths below refer
> to the preserved archive; see [structure](../docs/STRUCTURE.md) and the current experiment registry.
> Scientific settings, dataset roles and completed results are unchanged.

# H1 experimental protocol: measure the gaps

Protocol `h1_common_budget_v1.3`, 2026-10-02. **Execution update:** the minimum BEATs/MFCC H1 run is complete and verified; see [results](../reports/H1_baseline_results.md). The [pre-fit snapshot](../reports/ARTIFACT_INDEX.md#artifact-0e3678c5ed62) preserves the freeze unchanged. The user authorized completion of admission, the largest common budget and H1 execution. This explicitly replaces the infeasible numerical design in `sim_components_v1.1`. [Previous documents and checksums](../reports/ARTIFACT_INDEX.md#artifact-2d685a84b22a) preserve that design. Historical R0 results are unchanged.

**Research question:** which simulation components determine transfer to real vehicle audio using pretrained representations? **Working hypothesis:** source mismatch contributes more than propagation mismatch. H1 measures the gaps; it cannot attribute them to either mechanism. The separately frozen source × propagation experiment E4–E5 is now [complete as H2](../reports/H2_source_path_results.md); its [own protocol](h2_simulation/notes/PROTOCOL.md) and execution locks govern that run. This document retains the H1 contract. No new simulator, encoder training, military benchmark or reserved military recording was part of H1.

## 1. Frozen design and budget

Ordered classes are `car=0`, `truck=1`. Motorcycle is excluded from all new roles. Six leave-one-connected-group-out IDMT folds use five source groups as the eligible training pool and the sixth for evaluation. No random window split.

Five training-selection seeds: `[42,123,456,789,1024]`. Within each class, order files using PCG64 independently per group, then select round-robin across groups without replacement. Seed streams are the first 16 SHA256 hex digits of `h1_common_budget_v1.3|seed|stream`, interpreted as an unsigned integer. Stream names and exact IDs are saved by `h1/freeze.py`. Labels and metadata determine selection; features and predictions do not.

**Largest common budget:** N is the minimum eligible class count across every IDMT five-group training pool and both released synthetic banks. N resolves to **190/class**, 380 excerpts per single-origin fit. This counts unique retained files/excerpts, not independent vehicles or generation templates. Mixed arms use N real + N synthetic per class, with a repeated-real control for the doubled row count. Do not silently replace matched N with all available real or procedural data.

Validation is **none**. Encoder, preprocessing, head, C, stopping tolerance and decision threshold are fixed, with no hyperparameter search or early stopping. Synthetic banks remain training-only because source/template identities are unknown. A later trainable model needs a separately frozen source-grouped validation design.

The former 200 training/class, 50 synthetic validation/class, ≥20 evaluation/class, and 3/2/2 group minima were local design choices, not requirements of a paper, provider or statistical theorem. The old 60/20/20 allocation did not satisfy its own group minima with six groups. This explicit amendment replaces it, retaining all eligible held-out observations, including small truck groups, and reporting the uncertainty.

## 2. Exact data, grouping and leakage controls

The authoritative admission is [provenance v1.3.1](../reports/ARTIFACT_INDEX.md#artifact-8c9a6948fa0b), following integrity v1.2.1. Release identities, class mapping, unknown metadata and exclusions are specified in [DATASET_PROTOCOL.md](DATASET_PROTOCOL.md).

| Corpus | Cars | Trucks | Role and grouping |
|---|---:|---:|---|
| IDMT SE_CH34 | 3,902 | 511 | Six connected site/date groups. Source training or held-out evaluation by fold. |
| MELAUDIS FF, exact 1V | 7,810 | 256 | Four conservative connected day groups. Development evaluation only. |
| Released procedural | 5,000 | 5,000 | One training-only bank group; per-file generation lineage/backend unknown. |
| Released AudioLDM | 192 | 190 | One training-only bank group after conflict quarantine; generation lineage unknown. |

Grouping precedes class/sensor filtering. Links include original-time overlap, synchronized microphones, exact duplicates and verified near-identical waveform excerpts, including links through excluded classes. Original session IDs and physical vehicles remain unknown where not supplied; connected groups are conservative proxies. MELAUDIS day-level grouping deliberately merges numbered street positions rather than declaring their independence.

MELAUDIS was exposed by R0. It is not a newly untouched final test, and no `M_LOCK` or `T_confirm` is materialized. Keep its natural class prior. A new independent corpus would be needed for untouched confirmation.

Fit every scaler/head only on that arm's training features. No target labels, scores, embeddings, normalization statistics or nearest neighbors may inform fitting, hyperparameters, prompts, calibration or sample selection. Metadata/integrity/reuse checks are allowed before the freeze. Caching a fixed per-example frozen representation is allowed, without cross-example fitted statistics. Each derivative inherits its parent group. Known file/hash/duplicate/group intersections between training and evaluation must be empty. Prediction errors may not be used to exclude files.

## 3. Common observation and model

| Setting | Fixed value |
|---|---|
| Class mapping | IDMT C→car, T→truck; MELAUDIS FF/1V-Car→car, FF/1V-Truck→truck; exact synthetic folder/filename car/truck. Exclude other categories, mixtures, congestion and invalid/conflicting annotations. |
| Channels | IDMT exact SE_CH34: mean both stored columns, exclude ME_CH12. MELAUDIS mean actual channels, accepting only verified redundant copies for mono-labelled stereo. Synthetic mono. |
| Resampling | Float64 downmix; SciPy resample_poly native→8,000→16,000 Hz, reduced integer factors, Kaiser beta=5, constant-zero boundaries. |
| Segmentation | Center crop after both resamples to 32,000 samples: start floor((length−32000)/2); float32 encoder input. One 2 s window/file, no padding or repeated windows. Short/nonfinite input fails. |
| Other preprocessing | No amplitude normalization, denoising, EQ or target-statistic adjustment. Common intermediate rate limits information to approximately 0–4 kHz for every arm. |
| Primary representation | Frozen official BEATs_iter3_plus_AS2M, final patch-token mean, 768D, evaluation mode and no gradients; fixed official frontend. |
| Checkpoint | .artifacts/models/beats/BEATs_iter3_plus_AS2M.pt; SHA256 d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34. |
| Encoder code | Revision 732d834db70ee0fc3886b4bcbcfb4ce7fb829be2, per-file hashes verified by existing beats_adapter.py. No task-trained released head/scaler reused. |
| Classical representation | MFCC26: first 13 coefficients including coefficient 0; mean and population SD per coefficient. Librosa 0.11.0, same 16 kHz waveform, Hann FFT/window=400, hop=160, center=True, constant padding, power=2, 40 Slaney-normalized mel filters, 0–8 kHz, htk=False; dB ref=1/top_db=80; orthonormal type-II DCT. |
| Scaler/classifier | Train-only StandardScaler then binary L2 LogisticRegression, C=1, lbfgs, tol=1e-6, max_iter=5000, intercept=True, class_weight=None. Report/fail nonconvergence. |
| Frozen/trainable | Encoder and MFCC transform frozen; fit only training scaler, logistic weights/intercept. |
| Decision | argmax in [car,truck] order; ties resolve to car. No threshold tuning. |
| Augmentation | None in primary arms. Mixed arms add released synthetic files. Repeated-real control repeats identical training features twice; no new independent audio. |

Synthetic-only means task-training on synthetic examples. BEATs/AudioLDM pretraining used real data; unknown pretraining overlap is a limitation. The common-task bridge is not a literal replication of P02's original three-class CNN task.

## 4. Experiment matrix and exact logical manifests

All IDs resolve in `h1/frozen/h1_common_budget_v1.3/lock.json` and its hash-pinned admitted manifest. I_train(f,r) is N/class from five source groups; I_test(f) is the entire held-out group. M_dev is all 8,066 target excerpts. Y(r)/A(r) are N/class from the respective whole released bank. Validation and final confirmation are null for every new row.

| Experiment/arm | Train | Evaluation | Fits per representation |
|---|---|---|---:|
| R0 | Historical references; no new fit | Original paper arrays/splits/classes | 17/18 reference matrices matched; not retraining. |
| E0 / real | I_train(f,r), 190/class | I_test(f) | 30 |
| E1 / real | Exact E0 scaler/head, no refit | M_dev | 0 new |
| E2-Y / procedural | Y(r), 190/class | M_dev | 5 |
| E2-A / AudioLDM | A(r), 190/class | M_dev | 5 |
| E2-RR / repeated real | I_train(f,r) twice, 380/class rows, 190/class unique | M_dev | 30 |
| E2-RY / mixed procedural | Same I_train(f,r) + Y(r), 380/class | M_dev | 30 |
| E2-RA / mixed AudioLDM | Same I_train(f,r) + A(r), 380/class | M_dev | 30 |
| E3 / MFCC control | Repeat all selections above using MFCC26 | Identical source and target manifests | 130 |

Total 130 fits per representation, 260 fixed lightweight heads. All 190 AudioLDM trucks recur; seeds vary only 190-of-192 cars. These are not five independent generation runs. Synthetic fits are not redundantly refitted six times to inflate replication.

RR controls doubled sample likelihood/effective regularization at fixed C. Mixing changes unique audio count and origin, so compare mixed against both real and RR. Conventional corruption augmentation is a later control before attributing any mixing benefit to physics.

Explicitly deferred from the broader proposal: official frozen AST, newly trained CNN, conventional augmentation, both-synthetic mixtures and full-real sensitivity. AST needs an independently pinned pretrained backbone; the task-trained replay checkpoint cannot substitute. CNN needs a source-group validation/training recipe. These are not prerequisites to the minimum H1 gap measurement. BEATs remains primary regardless of MFCC ranking. E4–E9 are not implemented by this authorization.

## 5. Estimands, uncertainty and falsification

Primary Q: binary macro-F1, both labels included, zero_division=0. Also report balanced accuracy, accuracy, per-class precision/recall/F1/support, confusion matrices, positive-truck Brier score and log loss. Include per-fold/per-target-group results and worst observed class recall. Missing-class recall is null. Pooled source out-of-fold scores are a separate descriptive summary. Constant-car/truck references expose class imbalance.

Primary source score is mean of six held-out-group macro-F1s, averaged over five selections. Primary target score for real/mixed models is mean of six fold-specific target macro-F1s, averaged over selections: no ensembling. Synthetic target score averages the five actual fits.

```
Delta_domain = mean_f,r Q(I_train(f,r) → I_test(f))
             − mean_f,r Q(I_train(f,r) → M_dev)
Delta_sim2real(Y/A) = mean_f,r Q(I_train(f,r) → M_dev)
                    − mean_r Q(Y(r)/A(r) → M_dev)
Gain_mix(Y/A) = mean_f,r Q(I_train(f,r)+Y(r)/A(r) → M_dev)
              − mean_f,r Q(I_train(f,r) → M_dev)
```

Report mixed-minus-RR and paired BEATs-minus-MFCC target scores too. Delta_domain changes class priors, fleet and recording population as well as acoustics; it is descriptive, not causal. A smaller gap caused by worse source performance is not improved transfer.

95% percentile intervals: 10,000 draws with PCG64 seed 314159. Resample six source-group/fold IDs, four whole target groups and five selection IDs with replacement. Share each draw across all arms/representations. Source score uses held-out confusion matrices for sampled folds. For target, sum sampled target-group confusion matrices separately per model, compute each score, then average sampled folds/seeds. Synthetic models average only sampled seeds. Count/reject any draw missing a target class. Do not bootstrap windows independently or count overlapping training folds as independent training corpora. These are conditional fitted-model/group-resampling intervals, not full retraining or population-generalization intervals. Four uneven target groups and nearly exhausted AudioLDM support sharply limit uncertainty interpretation. Show per-group outcomes alongside intervals.

Expected: positive domain and synthetic-training gaps, without assuming either. Prespecified practical margin: .05 macro-F1. An upper 95% bound below .05 rules out a ≥.05 gap under this conditional protocol; upper bound ≤0 contradicts a positive gap. Wide intervals are inconclusive. Weak source classification weakens gap interpretation. Augmentation/representation superiority requires its own paired target estimate. H1 cannot falsify source-versus-propagation dominance; that requires the later controlled factorial. Negative or negligible gaps remain valid findings.

## 6. One canonical configuration format

[config.json](https://github.com/WaterPancake/ABVID/blob/77daa1b339906d35b6fa0a59a166ecf2eac01174/experiments/h1/config.json) is the single machine-readable JSON schema used by all H1 rows. It records protocol_id, ordered classes, admission manifest/hash, largest-common budget, seeds, split/validation/target exposure/final-test roles, preprocessing, representations, pinned encoder, MFCC settings, classifier, arms, augmentation, bootstrap and hardware. `freeze.py` adds exact selected file IDs, fold membership, manifest/config/document hashes, package versions, Git HEAD and dirty status to lock.json. Each fitted-model record adds representation, arm, fold, selection seed, train/test IDs, fitted scaler/head and prediction hashes. No unresolved placeholder is accepted in an executed lock. No implicit parameter search is allowed.

Preserve executed source snapshots and hashes because this workspace has uncommitted work. Verify source and model hashes, feature ordering and group separation. Existing result directories are never overwritten. Save all predictions, not just a best run. Model selection and target exclusions remain prohibited after target scoring.

## 7. Hardware, runtime and expected artifacts

Host: Apple M3 Pro, 18 GiB RAM, CPU only; four Torch threads, one BLAS thread, batch size 8. No GPU/cloud/API/new dataset request is needed. Incremental disk approximately 1–2 GB, without derived WAV copies or audio redistribution.

Pre-run estimates: provenance audit 1–5 min; BEATs extraction over approximately 12–15k distinct selected/evaluation excerpts 10–25 min; MFCC 1–5 min; 260 fits/predictions 2–10 min; uncertainty/reporting 1–5 min. These are estimates; actual timings must be recorded. Never reduce evaluation data for runtime.

Expected artifacts: immutable admission and configuration locks; exclusions/reuse/overlap evidence; source/target feature caches and ordered IDs; 260 fitted scaler/head records; all prediction probabilities; source/target and per-group metrics; constant controls; paired uncertainty; exact hashes, code snapshots and environment; `reports/dataset_admission_provenance.md` and `reports/H1_baseline_results.md`. The experiment matrix links to this scope and actual run artifacts.

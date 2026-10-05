# Published-result reproduction status

Updated: 2026-09-29 (initial replay: 2026-09-28). Active task: validate the papers using their original partitions and class definitions before choosing our own training splits.

**Verdict: partial reproduction, with the main BEATs replay discrepancy resolved.** Nine of P01's ten model/domain confusion matrices now reproduce exactly: CNN, AST, BEATs and DANN on both domains, and ArcFace on MELAUDIS. ArcFace on IDMT CH34 differs by one net truck count. All eight P02 snapshots still match archived full-test run-3 matrices. **No training runs have been reproduced.** Saved-table arithmetic, checkpoint inference and independent retraining remain separate claims.

The [R0 result ledger](ARTIFACT_INDEX.md#artifact-4317fa516d8f) records each tested result, unresolved result and missing prerequisite. The September 28 outputs are preserved; their immediate-seed BEATs sampling is superseded, not evidence that the released BEATs/DANN checkpoints are irreproducible.

The scope is the two AI4TEN studies associated with the supplied civilian datasets: **P01**, *Fine-Tuning Pre-Trained Audio Transformers (BEATs, AST) for Cross-Dataset Acoustic Vehicle Classification with Domain Adaptation*, [release 22203993](https://zenodo.org/records/22203993); and **P02**, *Synthetic-to-Real Transfer for Acoustic Vehicle Classification Using Physics-Based and AI-Generated Training Data*, [release 21264250](https://zenodo.org/records/21264250). This is not a reproduction claim for the entire literature collection.

Markdown was read first. The original PDFs were checked for P01 Table 3 (PDF p.10) and P02 Tables 3–4 (PDF p.8); P02 Table 5 is on PDF p.11. Paper claims, observations of released artifacts, and our interpretations are distinguished below. All macro-F1 scores are on a 0–1 scale.

## Original data roles retained

| Study | Historical training and validation | Evaluation used here | Classes and selection |
|---|---|---|---|
| P01 | Corrected IDMT MEMS/CH12 pool, 8,718 clips; released corrected code takes 15% for validation by seeded permutation. No fitting here. | Supplied IDMT sE8/CH34 features: 8,788 rows. Supplied MELAUDIS features: 14,663 rows. CNN evaluated all rows, then scored the original seed-42 balanced subset. Transformer inference evaluated 250 reconstructed candidate waveforms per domain, with model-specific sampling state recovered from the released evaluator. | `car, truck, motorcycle, bus, background`; 50/class/domain, NumPy MT19937 seed 42 reset separately for each domain. BEATs-family inference consumes RNG draws before sampling; see the correction below. |
| P02 | Real KVSD derivative: 1,006 listed training clips; different configurations add or substitute released AudioLDM/procedural samples. JSON specifies validation fraction 0.15, but exact membership/implementation is not recoverable without the missing trainer. No fitting here. | All 1,019 supplied DATASEC+MAVD test-feature rows, in their supplied order: 703 car, 75 truck, 241 motorcycle. | `car, truck, motorcycle`. Exact historical balanced subsets of 75/class were not reconstructed or replaced. |

Sources: P01 §§2.1–2.5, release `revisionReRunProcessAll_p2.py`, notebook setup/evaluation cells, `baseline_results.json`; P02 §§2.1–2.4, `configs/real_data_split.json`, per-experiment result JSONs. [P01 release](https://zenodo.org/records/22203993); [P02 release](https://zenodo.org/records/21264250).

P01's historical split is a microphone transfer evaluation, with paired acquisition events possible across microphones. It is not IDMT's separate provider EUSIPCO-2021 train/test partition, and neither has been substituted for the other. P01 MELAUDIS selection includes exact single-vehicle category labels across traffic states plus background; it is not the future study's free-flow-only car/truck filter. The corrected P01 interpretation assigns sE8 motorcycle recordings to CH34 even when their raw filenames contain CH12; original files remain unchanged. [P01 §§2.1, 2.5; release correction script and local filename inventory.]

The future binary-study decision to omit motorcycle remains in `sim_components_v1.1`. It does not apply to historical reproduction: removing a published output class would change the task and invalidate direct comparison.

## P01 checkpoint replay

The CNN uses the released corrected architecture, checkpoint and standardized MFCC + delta + delta-delta arrays without refitting a scaler. Only the model class was loaded from the script; its top-level training/data-movement code was not executed. The release README identifies the checkpoint as seed 42. The evaluator's original balanced sampling procedure was applied without searching for a matching seed.

| Model | MELAUDIS: replay / archived | IDMT CH34: replay / archived | Confusion-matrix agreement |
|---|---:|---:|---|
| CNN, seed 42 | 0.257882 / 0.2579 | 0.476909 / 0.4769 | Exact on both domains, using supplied features. |
| AST | 0.426926 / 0.4269 | 0.622028 / 0.6220 | Exact on both domains, using reconstructed waveforms and a compatibility library version. |
| BEATs | 0.462992 / 0.4630 | 0.753668 / 0.7537 | Exact on both domains after restoring published RNG consumption. |
| BEATs + DANN | 0.306197 / 0.3062 | 0.801498 / 0.8015 | Exact on both domains after restoring published RNG consumption. |
| BEATs + ArcFace | 0.499971 / 0.5000 | 0.738561 / 0.7330 | Exact on MELAUDIS; one net truck-count shift on CH34 remains unresolved. |

Archived comparisons come from released `baseline_results.json` (seed 42), `ast_results.json`, `beats_results.json`, and `da_results_full.json`. P01 Table 3, PDF p.10, presents the CNN five-run aggregate, not the single CNN snapshot's score. Recomputing the archived five CNN confusion matrices per domain recovers the table's **0.26 ± 0.03** and **0.52 ± 0.04** after rounding. These are archived run standard deviations, not confidence intervals from independent sessions or new training runs.

Evidence: [CNN replay](ARTIFACT_INDEX.md#artifact-aa088aacb864), [corrected BEATs-family replay](ARTIFACT_INDEX.md#artifact-3dc99b7bd228), [AST compatibility replay](ARTIFACT_INDEX.md#artifact-1fec9e33080e).

### R0 correction: inference changes the balanced subset

The earlier replay made a local evaluation error: it drew the seed-42 balanced subset immediately after seeding. The released notebook instead calls `np.random.seed(42)`, runs the whole domain through the model in batches of 16, and **then** samples 50 examples/class. In the pinned BEATs `backbone.py`, lines 135–136, `np.random.random()` executes once per layer before the `self.training` condition. Evaluation therefore consumes random numbers even though no layer is dropped.

An actual evaluation forward verified exactly **12 NumPy uniform draws**. The original full MELAUDIS loader has `ceil(14663/16) = 917` batches, hence **11,004 draws** before selection; CH34 has `ceil(8788/16) = 550` batches, hence **6,600 draws**. Advancing the original MT19937 state by these code-derived counts reconstructs the selection without evaluating thousands of unselected waveforms. The selected manifest was saved before any predictions were compared. No seed, subset or score search was used. [Corrected replay script](https://github.com/WaterPancake/ABVID/blob/77daa1b339906d35b6fa0a59a166ecf2eac01174/experiments/reproduction/replay_p01_beats_rng.py); [sampling manifest](ARTIFACT_INDEX.md#artifact-6ac0b0e9c6df); [pinned upstream source](https://github.com/microsoft/unilm/blob/732d834db70ee0fc3886b4bcbcfb4ce7fb829be2/beats/backbone.py#L135).

**Consequence for the paper's comparison:** equal seed values do not imply equal balanced event subsets across CNN/AST and BEATs-family evaluations. The BEATs-family subset also depends on evaluation batch size. This is a released-code observation, not evidence of intentional selection. Historical reproduction preserves it; a future common-task comparison should freeze shared event IDs independently of inference.

The corrected replay recovers five of the six previously mismatched BEATs-family matrices exactly. ArcFace CH34 has replay truck row `[10,19,20,0,1]`, versus archived `[11,18,20,0,1]`; all other class rows match. Its F1 differs by **0.005561**, not merely rounding. This is one **net count shift**, not proof of exactly one identifiable individual prediction error, because author per-event predictions are unavailable.

The lowest-margin truck candidate has truck/car probabilities approximately 0.40694/0.40611. A fixed diagnostic tested that waveform alone, in its selected-subset batch and in its original full-dataset batch. All predicted truck; the maximum probability difference was approximately 1.31e-6. These checks do not resolve the mismatch or establish whether raw-input differences, another numerical environment or another cause explains it. [Boundary diagnostic](ARTIFACT_INDEX.md#artifact-9be7e71c53d4).

For the new 500 selected candidates, **196/500** pass the unchanged `rtol=1e-4, atol=1e-4` MFCC check; maximum standardized-feature difference is **0.001799807**. This remains qualified waveform reconstruction. The older 206/500 check below concerns the earlier, different subset and is retained as historical evidence. No tolerance was relaxed. [New input validation](ARTIFACT_INDEX.md#artifact-b84170d219a6).

### Input reconstruction and compatibility qualifications

P01's released `data.zip` contains only `DATASET_SOURCES.txt`. The preprocessing input CSVs and the processed waveform archive used by the notebook are not supplied. Raw candidates were reconstructed from local IDMT annotations and MELAUDIS archive filenames. Their class counts and class-label sequence match the supplied feature arrays. Windows `Path` ordering matches the supplied MFCC row order substantially better than POSIX ordering; mixed-case MELAUDIS background names matter. This was identified by comparison with supplied features, not by optimizing model scores.

For 500 selected rows, preprocessing follows the released procedure: resample to 22,050 Hz mono, peak-normalize to 0.95, write PCM16, then compute first-two-second MFCCs and derivatives and apply the supplied training scaler. **206/500** rows pass the initial `rtol=1e-4, atol=1e-4` comparison; **294 fail it**. Maximum absolute standardized-feature difference is **0.001387626**. The tolerance was not relaxed and no rows were excluded to improve agreement. Small residuals are consistent with numerical/environment differences, but their cause has not been established. This is partial input validation, not bit-identical reconstruction. Initial POSIX-order results are retained separately. [Raw-row validation](ARTIFACT_INDEX.md#artifact-69144cffe3ad); [candidate manifest](ARTIFACT_INDEX.md#artifact-b7923adc3dc0).

Transformers receive the reconstructed PCM16 samples, decoded to float32, resampled by torchaudio to 16 kHz, padded/truncated to the first 32,000 samples. The supplied notebook model declarations are used with strict state-dictionary loading, evaluation mode and no gradients. SoundFile decoding replaces torchaudio's optional codec dependency; the PCM16 samples are preserved. No augmentation or model selection is applied. The original waveform/row manifest remains necessary to establish exact input identity and investigate the residual ArcFace discrepancy; matching aggregate matrices for AST does not prove raw-event identity for every row. The corrected BEATs-family replay uses the notebook's Colab/POSIX path order for sampling and a separate Windows-order filename lookup when checking the supplied MFCC rows. These two row-index spaces are explicitly recorded.

AST fails strict loading under the release's stated `transformers==5.8.1`: checkpoint keys use `backbone.layers...q_proj`, whereas that implementation expects `backbone.encoder.layer...query`. **Transformers 5.17.0 loads the unmodified checkpoint strictly and reproduces both matrices.** This is explicitly a compatibility replay, not the exact declared environment. The failed 5.8.1 run and its log are retained; no checkpoint key renaming or weight conversion was used for AST. Model configuration and preprocessing configuration are pinned to Hugging Face revision `f826b80d28226b62986cc218e5cec390b1096902` and hashed in the replay output.

The BEATs implementation is pinned to upstream commit `732d834db70ee0fc3886b4bcbcfb4ce7fb829be2`, with source-file hashes checked before use. The original pretrained checkpoint SHA256 is `d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34`. The release notebook itself clones upstream without a commit pin. The live upstream history checked on September 29 lists no BEATs.py changes after its initial release and only an April 2023 in-place-operation change in backbone.py. The identified large discrepancy was evaluator RNG handling, not evidence of a different backbone implementation. No implementation or preprocessing search was conducted against target scores.

### Released implementation versus method claims

- **Frozen BEATs evidence:** all 250 backbone state tensors in each released BEATs, DANN and ArcFace checkpoint are byte-identical to the pinned original pretrained checkpoint. The notebook's `extract_features` wraps backbone computation in `torch.no_grad()`, including the purported unfrozen BEATs phase. Our interpretation: the released artifacts support learned heads on an unchanged BEATs backbone; they do not substantiate encoder fine-tuning. This training-method finding is separate from the now-resolved evaluation RNG discrepancy. [Notebook BEATs class/training cells; tensor comparisons in replay JSON.]
- **AST code:** the released AST forward method also wraps backbone computation in `torch.no_grad()`. Exact checkpoint inference does not verify the paper's fine-tuning mechanism or training trajectory. [Notebook AST class/training cell.]
- **Target exposure:** DANN training constructs its unlabelled target iterator from `test_mel_loader`. Its historical MELAUDIS result is transductive adaptation to evaluated target waveforms, not target-data-free transfer. This historical behavior was documented, not changed or re-executed. [Notebook domain-adaptation cell.]
- **Validation membership:** the corrected CNN script uses a random permutation for its 15% validation subset, rather than an explicitly stratified operation. Preserve this distinction from the paper's stratification description when attempting retraining. [P01 §2.5; `revisionReRunProcessAll_p2.py`, validation split block.]

These are release-audit findings. They should not be generalized into claims about unpublished runs or the authors' intent.

## P02 checkpoint replay

All eight unmodified released snapshots initially failed Keras loading because HDF5 group names contain literal Windows backslashes. A documented compatibility conversion writes separate portable checkpoints with normalized group-path separators. It preserves the configuration and **every tensor's shape, dtype and bytes**, including optimizer state. Original archives and checkpoints remain untouched. Per-tensor hashes are recorded in [the replay result](ARTIFACT_INDEX.md#artifact-6b26cb4a0110).

| Released checkpoint configuration | Full-test macro-F1 replay | Exact archived match | Also matches JSON's `best_run`? | Paper Table 4 balanced five-run F1 |
|---|---:|---|---|---:|
| Real only | 0.130711 | run 3, seed 789 | No | 0.25 ± 0.06 |
| Real + SpecAugment | 0.362721 | run 3, seed 789 | No | 0.32 ± 0.05 |
| AudioLDM only | 0.392043 | run 3, seed 789 | Yes | 0.22 ± 0.11 |
| Procedural/pyroad only | 0.275077 | run 3, seed 789 | No | 0.19 ± 0.03 |
| Both synthetic sources | 0.331645 | run 3, seed 789 | No | 0.35 ± 0.02 |
| Real + AudioLDM | 0.166833 | run 3, seed 789 | No | 0.27 ± 0.03 |
| Real + procedural/pyroad | 0.315095 | run 3, seed 789 | Yes | 0.36 ± 0.02 |
| Real + both synthetic sources | 0.389986 | run 3, seed 789 | Yes | 0.39 ± 0.03 |

**The last column is a different evaluation population and a five-run aggregate. It must not be compared directly with the single-checkpoint full-test column as a reproduction error or improvement.** P02 Table 4, PDF p.8, uses balanced subsamples of 75/class; our inference replay uses all 1,019 supplied test rows. Each exact match refers to the full confusion matrix, not merely a rounded score. Run matching was an after-the-fact audit of the fixed released checkpoints against all archived runs, not checkpoint selection.

The release README describes best-run models, but five of eight supplied checkpoints do not match the respective JSON `best_run` confusion matrix. All eight match `run_3`/seed 789. This discrepancy is retained explicitly rather than relabelling the artifacts as independently verified best models.

For all eight configurations, recomputing macro-F1 from the five stored balanced confusion matrices and then computing the mean and population standard deviation reproduces the printed Table 4 values. The same arithmetic check reproduces the four matched-count Table 5 rows (PDF p.11). **These are saved-result checks, not new predictions from those five runs.** The original `soundclass_v1` module imported by the trainer is missing, and exact balanced-subset indices are absent. They were not inferred by searching seeds or matching target results. Only one checkpoint per Table 4 configuration is supplied.

Supplied test-feature labels agree with the class sequence of the 1,019 released test filenames. That does not independently verify raw-recording-to-feature-row identity. Original KVSD derivative audio, complete source correspondence and the missing trainer are still needed for end-to-end reproduction. P02 ratio sweeps, congestion results and synthetic-source generation have not been independently replayed. Their available saved arithmetic is audited below.

## P02 extended audit and current release limits

The public P01/P02 Zenodo file checksums were rechecked on September 29 and are unchanged. Complete inventories of both code ZIPs are retained in the R0 ledger. P02's wrapper imports `soundclass_v1` at `_4_runExperiments.py:73`, but that module is absent from its 45 archive members and the supplied local dataset. A targeted public search did not locate an alternative source. That search does not prove the module is unavailable everywhere; the current published release remains insufficient for exact training and balanced-selection reconstruction. No author was contacted and no missing sampling rule was invented. [P02 release](https://zenodo.org/records/21264250).

After checking Markdown, the original PDF pages 13 and 17 were visually inspected. The extended saved-result audit finds:

- **Table 6, PDF p.13:** all 12 ratio-sweep macro-F1 mean/SD entries round correctly from stored confusion matrices: six class-balanced-real conditions and six original-imbalanced-real conditions. This does not reproduce their training or predictions.
- **Table 7, PDF p.17:** all four F1 and accuracy entries round correctly from stored matrices. Density-1 saved confidence is **0.8546429**, which rounds to **0.85**, whereas the paper prints **0.86**. Confidence cannot be independently recomputed without the original per-window probabilities. The saved density-1 result has **146 evaluated windows**, despite the caption's 150-windows-per-density statement; the other densities have 150.
- **Figure 5, PDF p.13:** the plotted counts sum to **1,019**, matching the independently replayed real-plus-both full-test matrix. Its caption calls this the balanced test set, but it is not the 225-example, 75/class subset. This matters when comparing figure scores with Table 4.

The first audit's 14 table-row checks plus these 16 extended rows give **30 archived arithmetic rows**. Macro-F1/accuracy table comparisons pass where audited; the confidence-rounding and window-count discrepancies remain explicit. Spectral analysis and congestion waveform generation are not independently reproduced from raw audio.

## Reproducibility artifacts and execution

The current [R0 ledger](ARTIFACT_INDEX.md#artifact-4317fa516d8f) verifies **18 current prediction artifacts** (10 P01 model/domain comparisons and eight P02 full-test configurations). **17/18 matrices match** their respective archived references; ArcFace CH34 is retained as a mismatch. It also records release inventories, hashes, 30 archived arithmetic rows, method findings and unexecuted results. The earlier six BEATs-family prediction artifacts remain untouched for comparison.

The initial [machine-readable audit](ARTIFACT_INDEX.md#artifact-580bea086017) verifies seven downloaded files against provider MD5 values, verifies hashes and confusion matrices for **18 prediction artifacts**, and records **14 archived table-row arithmetic checks**. All 14 round to the printed values. Replay JSONs contain input/checkpoint/script hashes, versions, prediction hashes, confusion matrices and elapsed times. Logs preserve loader failures. Download metadata and original packages are stored separately under `experiments/reproduction/releases/`; no raw dataset or author artifact was overwritten.

Commands below assume the checked release assets, local datasets and reconstructed processed clips are present. The preparation scripts and manifests describe reconstruction; the commands do not silently download missing data or start training.

```bash
# Supplied-checkpoint inference, original classes and fixed test arrays.
experiments/reproduction/.venv/bin/python experiments/reproduction/replay_p02.py --normalize-hdf5-paths
experiments/reproduction/.venv/bin/python experiments/reproduction/replay_p01_cnn.py

# Corrected BEATs-family replay into fresh output/work directories.
experiments/reproduction/.venv/bin/python experiments/reproduction/replay_p01_beats_rng.py --output experiments/reproduction/results/R0_recheck --work-dir experiments/reproduction/work/R0_recheck
experiments/reproduction/.venv/bin/python experiments/reproduction/replay_p01_transformers.py --models AST --output experiments/reproduction/results/P01_AST_compatibility

# No inference: verify saved predictions, checksums, and archived arithmetic.
experiments/reproduction/.venv/bin/python experiments/reproduction/audit_r0.py
```

Environment snapshots: [initial replay lock](https://github.com/WaterPancake/ABVID/blob/77daa1b339906d35b6fa0a59a166ecf2eac01174/experiments/reproduction/requirements-replay-lock.txt), [AST compatibility lock](https://github.com/WaterPancake/ABVID/blob/77daa1b339906d35b6fa0a59a166ecf2eac01174/experiments/reproduction/requirements-ast-compatibility-lock.txt). Use a separate `uv venv` and `uv pip sync --python <venv>/bin/python <lock>` to reconstruct the relevant environment. The existing reproduction environment currently has the AST compatibility version. The recorded CNN replay used the project's PyTorch 2.13.0/NumPy 2.4.6/scikit-learn 1.9.0 environment; P02 used TensorFlow/Keras 2.15.0, NumPy 1.26.4 and scikit-learn 1.4.2. The initial BEATs replay used PyTorch/torchaudio 2.11.0 and Transformers 5.8.1; AST and the corrected September 29 BEATs-family replay used the same isolated numerical stack with Transformers 5.17.0. These are compatibility environments, not a claim to reproduce the authors' complete Colab stack. Executed replay scripts are retained alongside their results because the current scripts subsequently gained selection/output arguments.

Observed runtime on macOS 26.6.2 arm64, CPU inference with four threads: P02 eight snapshots **20.46 s**; P01 CNN two full test arrays **24.47 s**; BEATs/DANN/ArcFace each two balanced sets **13.8–14.3 s**; AST two balanced sets **124.09 s**; selected raw-row validation **17.66 s**. The corrected September 29 BEATs-family replay took **13.5–13.7 s per model** for the two selected sets; new selected-row preprocessing took **10.05 s**, excluding archive extraction. These script timings exclude downloads and generally exclude interpreter imports/input preparation. No GPU was required. Peak memory and exact chip/RAM were not measured; no training-runtime estimate is established by these inference timings.

The working tree contains pre-existing changes; the recorded Git commit alone is not a complete code snapshot. Use the per-script and per-input hashes, version locks and preserved source releases with the result files.

## R0 disposition and what remains before claiming full reproduction

1. Investigate the remaining ArcFace CH34 net-count difference using original per-event predictions/processed inputs if recoverable. BEATs and DANN matrix discrepancies are resolved by the published RNG semantics. Preserve all prior outputs and the residual mismatch; do not tune transformations to maximize agreement.
2. Recover P02's exact balanced-test selections and `soundclass_v1`, and clarify the mismatch between supplied checkpoint labels and `best_run`. The current release is sufficient for full-test checkpoint replay, not exact balanced-subset or full training reproduction.
3. P01 training code and public raw audio are available, but exact processed training inputs/row identity have not been verified; its training trajectories are **not run**, rather than declared universally impossible. P02 exact training is **not executable from the current release** because the trainer is absent and raw derivative identities are incomplete. Reconstructed/corrected trainers must be labelled variants. No independent training-seed variance has been reproduced.
4. The current-release R0 audit now has a bounded result-by-result disposition: exact/compatibility checkpoint matches, one residual P01 mismatch, saved-arithmetic checks and explicit unavailable/unexecuted results. This closes the present checkpoint-audit pass, **not** full paper reproduction or authorization to start E0–E9. Our future training partitions and simulator ablations remain deferred.

Historical test features and selected waveforms have now been inspected for reproduction. These exposed examples cannot later be described as untouched confirmation data. Reproduction findings do not change the original military benchmark's reserved-source roles, and those recordings were not used here.

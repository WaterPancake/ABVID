# First experimental phase: simulation components and real transfer

Protocol ID: `sim_components_v1.0`  
Design freeze: 2026-09-27  
Status: **documents only; no experiment implemented or executed**  
Companions: [dataset protocol](DATASET_PROTOCOL.md), [experiment matrix](EXPERIMENT_MATRIX.csv).

## 1. Question, estimand and limits

**Primary question:** Which components of acoustic simulation most strongly determine synthetic-to-real transfer performance for vehicle classification using pretrained audio representations?

**Working hypothesis:** Source-model mismatch contributes more to the synthetic-to-real performance gap than propagation-model mismatch.

This is not assumed true. The minimum first phase tests a narrower, observable implication: **under matched budgets, does a specified source enrichment improve real-target macro-F1 more than adding ground reflection and atmospheric absorption to an already moving direct-path renderer?** A four-cell factorial identifies intervention effects within this simulator, task, bandwidth and encoder. It cannot identify a universal fraction of the gap caused by each physical layer, prove either simulator is physically correct, or rank untested sensor/environment interventions.

The source intervention is deliberately called **enrichment**, not verified fidelity improvement. Parameters are research design choices, not fitted physical measurements. No outcome can establish that all source mismatch matters more than all propagation mismatch. A source gain under these controls supports the operational hypothesis; an opposing propagation gain contradicts it in this setting.

The completed review motivates this design: source/state entanglement and missing factorial controls [research gaps §§1–3](../reports/research_gaps.md); coupled source/ground effects [physical model, taxonomy](../reports/physical_signal_model.md); unknown vehicle-specific BEATs invariances [representation taxonomy](../reports/representation_taxonomy.md); unresolved source dominance [domain-shift taxonomy §Does source mismatch dominate propagation mismatch?](../reports/domain_shift_taxonomy.md). Paper IDs resolve in the unchanged [literature matrix](../literature/literature_matrix.csv). All numeric settings below are protocol decisions unless explicitly attributed.

## 2. Minimum scope and dependencies

The new experiment IDs replace the numbering in the broader [proposed roadmap](../reports/proposed_experiments.md) for this phase only.

| Stage | Experiment rows | Question answered | Required before proceeding |
|---|---|---|---|
| E0 | `E0` | Does a fixed pretrained representation classify the three categories on independent groups within IDMT? | Dataset admission and usable within-domain baseline. |
| E1 | `E1`, reusing E0 fits | How large is the IDMT-to-MELAUDIS gap? | Report gap on development evaluation, with uncertainty and per-class results. |
| E2 | `E2-P-R`, `E2-P-S`, `E2-B` | Can supplied published CNN baselines be replayed, and does the published synthetic corpus leave a gap with our unchanged BEATs protocol? | Published replay or explicit reproduction blocker; BEATs bridge evaluated before new simulator modifications. |
| E3 | `E3-00`, `E3-10`, `E3-01`, `E3-11` | What are the separate source and propagation gains and their interaction? | Complete frozen four-cell design, paired random draws and successful technical controls. |

There are **nine matrix rows, six new training configurations × five paired replicates = 30 logistic fits**, two fixed published CNN snapshot evaluations, and E1 reuses E0's five models. No AST comparison, encoder fine-tuning, domain adaptation, AudioLDM generation, real-plus-synthetic augmentation sweep, sensor/environment sweep or new loss belongs to this minimum phase. Their omission keeps the physical attribution test interpretable. The optional broader experiments in the literature roadmap are deferred, not automatically authorized.

Use `car`, `truck`, `motorcycle` throughout, with the exact mapping and exclusions in [DATASET_PROTOCOL §§4–6](DATASET_PROTOCOL.md). None of these experiments advances military family/model recognition or ABVID's Milestone 7 gate.

## 3. Data roles and target protection

All logical datasets refer to the release-pinned manifests defined in DATASET_PROTOCOL. Membership is determined before audio feature extraction. `I_TRAIN_200_r` has exactly 200 unique events/class; `I_VAL` and `I_TEST` are disjoint acquisition groups. `M_DEV` and `M_LOCK` are disjoint MELAUDIS groups. There is **no target-domain model-selection validation set** and no labelled or unlabelled target training in the core.

- E0 fitting and scaler estimation use `I_TRAIN_200_r` only. `I_VAL` is a source diagnostic; the fixed classifier has no hyperparameter search or early stopping.
- E1 uses the exact E0 checkpoints and transforms. It reports `I_TEST` versus `M_DEV` before simulator work. E2-B then reports real-trained versus synthetic-trained performance on the same `M_DEV` events.
- Development target scores answer the preregistered questions but cannot change C, source ranges, renderer choices, filtering, checkpoint, class map, sampling, exclusions or training schedule in this version. If a result motivates redesign, end this version and document the change before a new study.
- `M_LOCK` waveforms, labels, embeddings and acoustic summaries are inaccessible to development. No target-feature standardization, batch-statistic update, calibration, prompt tuning, nearest-neighbour selection or unlabelled adaptation is allowed.
- After all six core configurations and five replicates have passed source-side checks, freeze their checkpoints, transforms, manifests and configuration hashes. Then evaluate **all predeclared arms in one final batch** on M_LOCK. Do not choose only development winners. The same target may score every fixed arm; single use means a single frozen comparison batch with no feedback to development.
- A target-side bug invalidates the batch. Preserve it and document the fault; do not repeatedly repair and rescore while retaining an untouched-test claim.

E2-P's historical DATASEC/MAVD test features are a separate **published replay set**, not this project's final lock. Its results may already be known. Confirm no original-media overlap with IDMT/MELAUDIS before using its training corpus. The two domains' scores are never subtracted.

## 4. Common core processing and model: profile `CORE8_BEATS_LR1`

Applies without variation to E0, E1, E2-B and every E3 cell.

| Item | Frozen value |
|---|---|
| Channel rule | IDMT: arithmetic mean of verified CH1/CH2 sE8 pair, excluding CH34. MELAUDIS: arithmetic mean of the channels in each eligible event; mono unchanged. Synthetic: mono. Sibling channel events share one provenance group. |
| Decode | Floating point PCM in nominal [-1,1], retain native rate until resampling; reject decode errors/nonfinite/all-zero signals before split freeze. |
| Common bandwidth | Resample every waveform to 8,000 Hz, then to 16,000 Hz using `scipy.signal.resample_poly`, default Kaiser beta 5.0, explicit constant-zero boundary padding, integer ratios reduced by GCD. Identity-rate stages are skipped. This fixes the approximately 0–4 kHz information bandwidth; 16 kHz is the encoder input rate. |
| Rationale | Released P02 generation code uses an 8 kHz internal rate. Upsampling does not restore lost information. Bandwidth matching prevents it becoming an uncontrolled source/representation difference. Results are restricted to this bandwidth; a native-16-kHz study would be a later protocol. |
| Segmentation | Exactly one 2.000 s / 32,000-sample window/event. Centre crop after resampling: start `floor((N−32000)/2)`. Real short clips are excluded before splitting; no repeat/zero padding. Published 5 s synthetic clips use their centre 2 s; E3 centres on closest approach at receiver arrival time. No overlapping windows. |
| External preprocessing | No denoising, silence trimming, peak/RMS normalization, EQ, pre-emphasis outside BEATs, automatic audibility gate or class-dependent processing. |
| Encoder | `BEATs_iter3_plus_AS2M.pt`, 768-dimensional final-token mean, masked to valid tokens. Checkpoint SHA256 `d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34`; official code commit `732d834db70ee0fc3886b4bcbcfb4ce7fb829be2`. |
| Artifact provenance | Same artifact specified in [existing BEATs config](../configs/benchmark_beats_v1.yaml): mirror revision `5b53b0404df452a3a607d7e67687227730e5bad1`. Equivalence to an official checksum remains unverified, as already reported locally. No old ABVID classifier/scaler is reused. |
| Internal features | Use pinned upstream BEATs preprocessing unchanged: 16 kHz, 128 filterbanks, 25 ms frames, 10 ms hop, upstream fixed normalization. Record resolved upstream constants/code hashes. No dataset-fitted Mel normalization. [P07 §§3,4.2] |
| Frozen components | Whole encoder in evaluation mode, gradients disabled, no dropout or batch-statistic updates; no generator training. |
| Trainable components | Train-only `StandardScaler` on the 600 training embedding vectors, then multinomial L2 logistic regression, intercept enabled, `C=1.0`, `solver=lbfgs`, `tol=1e-6`, `max_iter=5000`, no class weighting because training is balanced. Float64 scaler/head; float32 encoder. |
| Selection | No C grid, early stopping, checkpoint selection, feature selection or calibration fit. All runs use the converged fixed objective. A convergence failure is a technical failure, not permission to tune on target. |
| Decision | Softmax argmax in fixed class order `[car,truck,motorcycle]`; ties choose lowest integer ID. One event is one score. No threshold tuning, rejection class or voting. |
| Augmentation | None beyond the predeclared synthetic source/propagation factors. No SpecAugment, random crop, pitch/time stretch, added real noise or sensor randomization. |

BEATs is pretrained on real audio; synthetic-only refers to task training, not absence of prior real-data knowledge. The local encoder's earlier use does not contaminate new training if its pretrained weights remain unchanged and no old fitted head/transform is reused. Pretraining overlap with public source media is unknown and must be stated.

## 5. E0 and E1: real-data baselines

**E0 train:** `IDMT_V1 / I_TRAIN_200_r`, 200/class. **Validation:** all `I_VAL`. **Test:** all `I_TEST`. Profile `CORE8_BEATS_LR1`; no augmentation; five replicate seeds below. Store fitted train-only scaler/head and predictions on both validation/test. There is no claim of a random within-dataset split: all acquisition groups are disjoint.

**E1 train/model:** the exact E0 fits, with no refit. **Validation:** the same I_VAL source diagnostic. **Development test:** all `M_DEV`. **Final test:** all `M_LOCK`, in the final batch. This is a controlled real cross-dataset baseline, not exact reproduction of P01's five-class, fine-tuned transformer result.

For a fixed metric `Q` and replicate-mean scores, report:

\[
\Delta_{domain,dev}=Q(I_{train}\to I_{test})-Q(I_{train}\to M_{dev}),
\]

and replace M_DEV with M_LOCK only at final scoring. This changes test populations as well as domains; it is a descriptive gap, not the causal effect of geography. A target-trained reference would help separate target difficulty, but costs target training groups and is deferred. Do not call the real-IDMT reference a target-domain ceiling.

**Falsification:** the expected positive domain loss is contradicted if its 95% interval lies at or below zero. A practical loss ≥0.05 is ruled out when the upper bound is <0.05; that does not necessarily refute a smaller loss. An interval crossing zero is inconclusive. For E0, failure to beat the constant-majority training-prior predictor in balanced accuracy with a positive paired interval prevents treating it as an established useful baseline. Report macro-F1 for the constant predictor too; do not assume every chance macro-F1 equals 1/3.

**Gate:** finish and report E0/E1 development estimates before E2/E3. Low performance prompts a data/provenance/error audit without using M_LOCK. Do not improve the baseline until it matches P01's unrelated headline result.

## 6. E2: published baseline and pretrained bridge

### E2-P-R and E2-P-S: verify supplied published CNN snapshots

These are **inference-only historical replays**, not new training or exact reproduction of five-seed mean scores.

- E2-P-R historical training: `PUB_K`, KVSD 1,006 files (204 car, 265 truck, 537 motorcycle), with training class weights. Snapshot: `models/01_real_only.keras`.
- E2-P-S historical training: 15,000 released procedural examples (5,000/class); snapshot: `models/04_pyroad_only.keras`. Its scaler was fitted on real KVSD. Call it *synthetic task-training with real-source statistics*.
- Historical validation for both: P02 §2.4 says 15% stratified training holdout; exact membership and parent-group separation are unknown. No new validation selection occurs. The inspected archive lacks the imported trainer `soundclass_v1`, so exact retraining is not an admission assumption.
- Test: `PUB_T`, the released 1,019 DATASEC/MAVD feature rows, with verified labels/order and frozen file list. Score the full set, and a secondary fixed 75/class subset selected with NumPy `default_rng(42)` without replacement after canonical event-ID sorting. This subset is our deterministic replay subset unless original subset indices are recovered; never claim exact original subsampling from a seed guess.
- Preprocessing profile `P02_SNAPSHOT`: use supplied standardized features whose metadata verifies 22,050 Hz mono, first 5 s or right-zero-padding to 5 s, 40 MFCCs plus first/second deltas, FFT 2048, hop 512, 128 Mel bands, input 216×120×1. P02's released extraction uses librosa defaults for omitted options. No recomputation is necessary for replay. Verify that features already include the appropriate KVSD scaler; do not standardize twice.
- Model: saved three-block CNN with 32/64/128 3×3 filters, BN/ReLU/pool/dropout, global pooling, 256/128 dense layers and 3-way softmax. All weights and stored BN statistics frozen. No augmentation. Verify architecture/output order from saved configuration before loading; do not execute custom deserialized code.
- Historical training seeds: 42,123,456,789,1024. Each supplied file represents one selected snapshot; its exact selected training seed is unknown until matched to release metadata. Do not report five repetitions of one snapshot as five training seeds or recreate the article's mean±SD from these two checkpoints.

Paper references: P02 §§2.1–2.5, Tables 1–4, PDF pp.3–8; release `_3_prepareExperiments.py::extract_mfcc/load_and_extract/prepare_all`; `README.md`, Models and Features. The reviewed main-table means 0.25 real and 0.19 procedural concern a balanced subset and multiple training runs, not necessarily these best snapshots.

**Falsification / failure criterion:** if supplied checkpoint predictions fail to match corresponding archived single-run full-set confusion counts exactly, after verified row/order alignment, the numerical replay is not verified. If comparable single-run predictions/metrics are absent, record verification as unavailable. Differing scores on our fixed balanced subset are not by themselves falsification of the paper. If the replay synthetic-minus-real comparison has the opposite sign, report it; selected-snapshot differences do not establish the sign across training runs.

**Uncertainty:** when raw acquisition groups can be reconstructed, use a paired group bootstrap for the snapshot difference, conditional on these fixed weights. Otherwise report metrics with `session_CI=unknown`; no window bootstrap masquerading as session uncertainty. Historical internal validation dependence remains a limitation even if full-set predictions match.

### E2-B: published synthetic corpus with the fixed pretrained representation

**Train:** `PUB_Y_TRAIN_200_r`, exactly 200/class. **Validation:** `PUB_Y_VAL_50_r`, exactly 50/class in distinct generation groups. **Development/final tests:** M_DEV / M_LOCK. Use `CORE8_BEATS_LR1`, five paired replicates, no extra augmentation. Own synthetic-training scaler only; no KVSD or IDMT statistics. The published corpus is held fixed; this experiment does not certify which renderer generated it.

This is the necessary bridge between the historical CNN comparison and the new BEATs ablations. It is a **published-corpus reproduction variant** because model, duration, bandwidth matching, training budget and target differ from P02. Its scores are comparable to E1, not numerically interchangeable with E2-P.

\[
\Delta_{sim2real,dev}=Q(I_{train,200}\to M_{dev})-Q(PUB\_Y_{train,200}\to M_{dev}).
\]

Report this as the gap relative to a matched-count **IDMT-trained real reference**, not a target-trained oracle. Use the same metric and exact target events for both terms; repeat on M_LOCK only in the final batch.

**Falsification:** an upper 95% paired interval ≤0 contradicts an expected positive gap; upper bound <0.05 rules out a practically large ≥0.05 gap. If class support/provenance fails, E2-B is blocked, not silently converted to random-file validation. If baseline gaps are inconclusive, report that before ablations; do not portray an uncertain gap as established. The four-cell causal study may proceed only after the baseline is auditable; it need not force a positive gap to exist, and if no gap exists its claim becomes intervention effects rather than closing a demonstrated deficit.

## 7. E3: the minimum crossed simulator ablation

| Row | Source | Propagation | Primary role |
|---|---|---|---|
| `E3-00` | S0 | P0 | New controlled-renderer baseline. |
| `E3-10` | S1 | P0 | Source change with direct-path rendering fixed. |
| `E3-01` | S0 | P1 | Propagation change with source fixed. |
| `E3-11` | S1 | P1 | Joint change and interaction. |

Each row trains on its 600-event `Y_CORE` training manifest and has a disjoint 150-event synthetic validation manifest; evaluates all M_DEV events then M_LOCK. Use `CORE8_BEATS_LR1`, same five seeds and no additional augmentation. The same latent source IDs, geometry, noise realizations, template count, bandwidth and training budget are paired across cells. E3-00 is required because the P02 corpus's actual backend is unknown; it cannot substitute for this controlled baseline.

### Source factors, with numerical bounds

S0 reuses the **dry-source functions only** from checksum-pinned P02 `_2_genSyntheticData.py`: `make_engine_harmonics`, `make_tire_noise`, `make_exhaust_noise`, `generate_car_source`, `generate_truck_source`, `generate_motorcycle_source`. Do not use `generate_one_sample`, its silent renderer fallback, output normalization, process-dependent Python hashes, or class-specific path sampler. Generate float64 dry sources at 8 kHz and keep the published component normalization inside these functions identical across S0/S1.

S0 parameters, as observed in released code (not inferred from the paper):

| Class | Engine | Harmonics / rolloff | Other components |
|---|---|---|---|
| car | Four-stroke, 4 cylinders; RPM U[1500,4000]; firing RPM/30 | 8; 4 dB/order | Tire centre U[900,1100] Hz, width U[600,1000]; published RPM-linked engine/tire weights. |
| truck | Four-stroke, 6 cylinders; RPM U[1000,2200]; firing RPM/20 | 12; 2.5 dB/order | Tire centre U[700,900], width U[800,1200]; engine/tire/exhaust weights U[.35,.45]/U[.20,.30]/U[.20,.30]. |
| motorcycle | Four-stroke, cylinder count uniformly {1,2}; RPM U[3000,9000] | 10; 3 dB/order | Tire centre U[1000,1300], width U[400,600]; engine/exhaust/tire weights U[.35,.45]/U[.35,.45]/U[.05,.15]. |

All S0 engine components retain published sinusoidal AM depth U[.05,.15] at firing/2. Exhaust retains the released low-pass broadband plus harmonic mixture. These templates are approximations, not calibrated examples of actual vehicle makes. Published tire weighting can depend on RPM rather than measured road speed; hold that simplification fixed in the first test and report it.

S1 uses the same base RPM, cylinder choice, noise draws, phases, component filters and count, and adds exactly these three changes:

1. **Harmonic envelope diversity:** per source template, add one U[−1,+1] dB/order offset to the class's engine rolloff. All resulting rolloffs remain positive. No new harmonics or extra bandwidth.
2. **Slow source fluctuation:** template draws `a=U[0,.02]`, `f=U[.5,2] Hz`, `phi=U[0,2π]`. Engine/exhaust harmonic instantaneous frequency is `f0(t)=f0_base*(1+a*sin(2π f t+phi))`. Integrate frequency into phase; do not resample the entire source, which would alter tire noise and introduce a path change. Keep the existing faster engine AM at its base firing/2 rate.
3. **Component balance diversity:** multiply each existing engine/tire/exhaust component by an independently drawn gain U[−3,+3] dB, before their unchanged final mixing normalization. An absent component remains absent.

These three parameters are class-independent distributions. They form one bundled source factor, so a positive effect cannot be assigned to any individual source subcomponent. Their magnitudes are frozen illustrative priors, not a claim of measured real-vehicle ranges. Configuration, fuel, source directivity, genuine load/gear physics and additional engine architectures remain untreated. The same number of source templates is used in both levels; more samples is not the source intervention. Source and propagation intervention magnitudes have no common physical scale: the observed ranking is conditional on these bounds, not an intrinsic ranking of the two mechanisms. Do not expand one factor's range after seeing its score.

Each replicate has 20 independent training templates/class and 5 validation templates/class. Template parameters stay fixed across ten trajectories. Waveform noise and phase realizations can vary by run, but parent template identity remains grouped. Synthetic template holdout is a diagnostic, not a claim of unseen real-model recognition.

### Propagation factors and shared scene

Use [pyroadacoustics](https://github.com/steDamiano/pyroadacoustics/tree/642f95ea83417b662906712c5902d7b7dd665092), commit `642f95ea83417b662906712c5902d7b7dd665092`, its `Environment` API and one common interpolation setting `Sinc`. Pin source code and dependency versions in the future execution lock. Both levels use its moving-source delay lines and spherical spreading:

- **P0:** direct path only; `include_reflection=false`, `include_air_absorption=false`.
- **P1:** same trajectory/backend/interpolation, `include_reflection=true`, `include_air_absorption=true`. This adds the ground image path and atmospheric filtering; Doppler is already present in P0 and is not the tested factor.

For both: 8 kHz simulation; omnidirectional point source and one omnidirectional microphone; microphone at `[0,0,1.2]` m; source height `0.5` m for every category; closest lateral distance U[5,50] m; constant road speed U[30,90] km/h; direction equiprobable ±1. Speed and range draws are the same across categories, source levels and propagation levels. RPM is sampled independently of speed as a declared gear/load simplification; no claim of a physically calibrated drivetrain.

Render at least 10 s with closest approach at source time 5 s, trajectory `[d, direction*v*(t−5), .5]`, plus adequate source lead-in and receiver tail for maximum propagation delay. Keep the receiver's 2 s window centred at `5 + sqrt(d²+(.5−1.2)²)/c` seconds, where `c` is the pinned renderer's speed of sound. Both paths use that same window. No time stretching to fit a fixed road length, no interior zero padding and no duration correlated with class/speed. Lead-in/tail rules must be validated before generation; failure blocks the corpus rather than changing window selection.

Fixed environment: 20 °C, pressure 1 atm, relative humidity 50%, ground parameter `road_material=20000` in the pinned upstream API (asphalt flow-resistance parameter; preserve upstream units). Fixed sensor: ideal response, no sensor noise/nonlinearity/codec. Background: none; SNR is `not_applicable_no_added_background`, never a made-up finite dB value. This tests source/path at a fixed environment, not random-environment robustness.

Source output before rendering is matched to RMS 0.1 over the common central source interval. Receiver scalar level is then matched to −26 dBFS RMS over the 2 s window in every E3 cell. This deliberately conditions out scalar loudness to focus on spectral/temporal changes. It preserves each window's internal pass envelope but removes absolute attenuation; **this protocol cannot estimate distance-induced loudness/SNR contributions**. Real and published-corpus core preprocessing remains unnormalized; E3 matching is a controlled corpus-construction rule shared by all four cells. If any paired cell exceeds peak 0.98 after scaling, reject/regenerate the entire tuple using the next deterministic source-run sub-seed, log the rejection and retain the same geometry. No clipping or one-cell rejection.

No real background or recorded/AudioLDM source is used: their pre-existing channel traces would prevent a dry-source-only comparison. This phase does not require unavailable anechoic recordings.

### Technical controls before any target scoring

Required future checks: unchanged source hashes across P0/P1; unchanged path metadata across S0/S1; exact pairing, counts and disjoint template groups; equal sample rates/window lengths; common receiver RMS within 0.1 dB; no nonfinite/clipped/padded core windows; deterministic regeneration; a stationary/constant-tone path sanity check for propagation delay/Doppler; confirmed ground/absorption toggles without fallback. Log harmonic/modulation changes from synthetic inputs as intervention checks, never as proof of real transfer.

These checks are future acceptance criteria, not tests written or run now. A failed physical check blocks interpreting classifier scores. E3 identifies a change in the specified simulator only if source and path interventions actually change the intended mechanisms.

## 8. Metrics, uncertainty and hypothesis decisions

**Primary:** event-level macro-F1 over all three classes, native eligible target prevalence. **Secondary:** balanced accuracy, accuracy, per-class precision/recall/F1, confusion counts, per-acquisition-group recall, worst group-class recall, per-site scores, multiclass Brier score and top-label ECE with ten fixed equal-width bins. Use zero for precision/F1 of a never-predicted class, but a test split missing a true class fails admission. Worst group-class recall uses cells with at least five true events; list smaller cells separately. Report raw event/group counts alongside every score.

No balanced test resampling in the core. Average the metric over the five fitted replicates, not the probabilities across models: this is repeatability assessment, not an ensemble. Report every replicate and sample SD as well. The model trained on real data and each synthetic arm have the same 600-event budget, but different numbers of physical sources; report that distinction.

**95% confidence intervals:** 10,000 paired hierarchical bootstrap replicates, seed `314159`. Resample the five replicate IDs with replacement, and resample complete target provenance groups with replacement. Use identical sampled replicate IDs/groups for every arm and contrast. Recalculate per-model event metrics on each sampled group multiset, then average over sampled replicates. For the source-minus-target domain gap, resample IDMT test and MELAUDIS groups separately, retaining paired training replicate IDs; do not pair unrelated recordings. Report percentile 2.5/97.5 bounds.

Discard a bootstrap draw missing a true class and redraw; log the invalid-draw rate. If it exceeds 10%, or support floors fail, mark confirmatory intervals unestimable and the conclusion inconclusive. Do not substitute an event bootstrap. This interval is conditional on the fixed real training split and selected sites, not uncertainty over all possible fleets/geographies. Five repeats do not make 600 synthetic crops into independent real sessions. No power guarantee is made for a three-point effect; report interval width rather than post-hoc observed power.

Let `Qij` be mean target macro-F1 for source level i / propagation level j:

\[
\delta_S=\tfrac12[(Q_{10}-Q_{00})+(Q_{11}-Q_{01})],\quad
\delta_P=\tfrac12[(Q_{01}-Q_{00})+(Q_{11}-Q_{10})].
\]

\[
D=\delta_S-\delta_P=Q_{10}-Q_{01},\qquad
I=Q_{11}-Q_{10}-Q_{01}+Q_{00}.
\]

Also report both conditional source gains, both conditional path gains, and changes relative to E1 and E2-B. E1−Q00 is the controlled baseline gap; E1−Q11 is the residual gap. Improvements may exceed or reverse a real-source reference gap; do not truncate negative gaps to zero or call them a physical ceiling.

**Primary confirmatory contrast:** D on M_LOCK, with the practical margin `m=0.03` macro-F1. All others are secondary descriptive contrasts with unadjusted intervals, explicitly labelled; do not select a secondary p-value as the primary result.

| Outcome | Interpretation |
|---|---|
| Lower 95% bound of D >0 and lower bound of δS >0 | Evidence for a beneficial source intervention larger than the tested propagation intervention. |
| Lower bound of D >0.03 and lower bound of δS >0 | Evidence for practically meaningful source dominance under these settings. |
| Upper bound of D ≤0 | Falsifies directional source dominance for these interventions; strictly negative bounds support propagation dominance. |
| Upper bound of D <0.03 but D may be positive | Falsifies the ≥0.03 practical claim only; does not refute every positive source advantage. |
| Upper bound of δS ≤0 | Source enrichment did not beneficially close the gap; D being positive because propagation is harmful cannot count as the intended support. |
| Intervals cross the relevant thresholds | Inconclusive. Do not accept the hypothesis merely because a point estimate is positive. |
| Conditional effects reverse signs across the other factor | Report interaction and conditional effects; an unconditional ordering is not a stable mechanism ranking. |

Treat `|I|≥0.03` as a preregistered practically relevant interaction magnitude, report its interval without turning it into a second primary discovery claim. A null S1 effect can reflect a poor or insufficient source intervention; it does not prove real sources do not matter. Conversely, propagation's small incremental gain over moving direct-path rendering does not prove full propagation is unimportant.

## 9. Randomness, compute and runtime budget

All core experiments use replicate seeds `[42,123,456,789,1024]`; split seed `20260927`; bootstrap seed `314159`; historical replay subset seed `42`. Derive independent RNG seeds using the first 16 hex digits of `SHA256(protocol_id|replicate_seed|stream_name|class|template_id|run_id)`, interpreted as an unsigned 64-bit integer, with NumPy PCG64. Geometry stream omits class and factor cell and uses a class-independent template slot number; source-base stream omits factor cell; source-enrichment uses its own stream. Prefix template slots with `train` or `validation` so split roles also have independent draws. Retain resolved seeds in manifests. Logistic LBFGS is deterministic: replicate variation comes from selected real/published training events or independently generated paired synthetic banks, not a falsely claimed stochastic optimizer.

Core reference hardware: Apple M3 Pro CPU, ≥18 GB unified memory, four Torch threads, one BLAS thread, embedding batch size 8. CPU execution is sufficient. Optional accelerator for feature extraction: ≥8 GB VRAM, float32, same numerical settings; use one device type throughout a paired experiment and record differences. No A100 is required. CNN replay: TensorFlow-compatible CPU with ≥16 GB RAM; GPU optional. Reserve 50 GB disk for releases/derived data/checkpoints, subject to actual archive sizes. Do not download or reserve resources during this document-only task.

Runtime estimates below are **planning ranges, not measurements**. Existing local BEATs work reported 18.99 s for 785 two-second windows on M3 Pro, but that does not benchmark the new data loader, renderer or TensorFlow replay. [Local runtime evidence](../docs/beats_comparison_results.md), Reproducibility and limitations. Allow for resampling, model load, I/O and bootstrap overhead.

| Rows | Work size | Estimated elapsed time on reference CPU |
|---|---|---|
| E0 | IDMT core embedding cache once; five 600-event fits; source diagnostics and CIs | 5–30 min after data admission. |
| E1 | M_DEV embedding cache and five E0 heads; M_LOCK encoded only in final batch | 5–30 min development; 5–30 min final batch shared with all core arms. |
| E2-P-R/S | Two fixed CNNs ×1,019 feature rows, replay checks | 2–20 min per snapshot; 15–60 min including first framework load/installation, not model training. |
| E2-B | Five 600-event training subsets plus validation, deduplicated cache; five fits | 10–45 min. |
| Each E3 cell | Five ×750 rendered events =3,750; five heads | Rendering 1–12 h/cell provisional; embedding/fits/CIs 10–45 min/cell. Path simulation dominates uncertainty. |

Four E3 cells total 15,000 rendered examples; dry-source and geometry products are reusable where identical. Allow roughly 4–48 h serial rendering plus 1–4 h other work, excluding downloads and human provenance audit. These are deliberately broad engineering estimates. Before full generation, a future 30-tuple synthetic-only timing check may refine resource estimates without changing source/path parameters or looking at real targets. Record measured seconds/render and extrapolation, not a promise of the current range. Installation/provenance review time remains unknown.

## 10. Canonical experiment configuration format

All experiments use **one YAML schema**, `abvid.sim_components/1.0`, with mode enum `[fit_evaluate, evaluate_existing, replay]`. There are no alternate ad hoc command-line settings. The CSV is a human-readable index; the resolved YAML is the execution authority. This example is a document specification, not a newly created runnable configuration:

```yaml
schema_version: abvid.sim_components/1.0
protocol_id: sim_components_v1.0
experiment_id: E3-10
mode: fit_evaluate
status: design_frozen_pending_data_lock
domain: synthetic_to_real
replicate_seed: 42
provenance:
  git_commit: null
  dirty_patch_sha256: null
  dependency_lock_sha256: null
  protocol_sha256: null
  dataset_lock_sha256: null
datasets:
  train: {registry: Y_CORE, manifest_id: Y_CORE_S1P0_TRAIN_42, sha256: null}
  validation: {registry: Y_CORE, manifest_id: Y_CORE_S1P0_VAL_42, sha256: null}
  development_test: {registry: MELAUDIS_V1, manifest_id: M_DEV, sha256: null}
  final_test: {registry: MELAUDIS_V1, manifest_id: M_LOCK, sha256: null}
  target_access: evaluation_only
  group_key: provenance_group_id
  examples_per_class: 200
labels:
  ordered_classes: [car, truck, motorcycle]
  mapping_id: civilian3_v1
  mapping_sha256: null
preprocessing:
  profile: CORE8_BEATS_LR1
  intermediate_sample_rate_hz: 8000
  sample_rate_hz: 16000
  resampler: scipy_resample_poly_kaiser5
  window_seconds: 2.0
  crop: centre
  short_clip_policy: exclude
  external_normalization: none
representation:
  name: BEATs_iter3_plus_AS2M
  checkpoint_sha256: d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34
  code_commit: 732d834db70ee0fc3886b4bcbcfb4ce7fb829be2
  frozen: true
  pooling: final_layer_valid_token_mean
  dimension: 768
classifier:
  name: multinomial_logistic_regression
  scaler: training_only_standard_scaler
  C: 1.0
  solver: lbfgs
  max_iter: 5000
  tol: 0.000001
  fit_intercept: true
  class_weight: null
  calibration: none
training:
  selection: fixed_converged_objective
  validation_role: diagnostic_only
  trainable_components: [embedding_scaler, classifier]
  augmentation: none
simulation:
  source_level: S1
  propagation_level: P0
  source_recipe: P02_release_dry_plus_S1_v1
  source_recipe_sha256: null
  renderer_commit: 642f95ea83417b662906712c5902d7b7dd665092
  scene_profile: paired_constant_scene_v1
  resolved_parameters_sha256: null
  render_fallback: forbidden
randomness:
  split_seed: 20260927
  replicate_seeds: [42, 123, 456, 789, 1024]
  bootstrap_seed: 314159
  seed_derivation: sha256_first64_pcg64
evaluation:
  unit: original_event
  primary_metric: macro_f1
  secondary_metrics: [balanced_accuracy, accuracy, per_class_prf, confusion, group_recall, brier, ece10]
  decision: argmax
  confidence: {method: paired_group_and_replicate_bootstrap, draws: 10000, level: 0.95}
  primary_contrast: source_gain_minus_propagation_gain
  practical_margin: 0.03
  final_batch_id: null
resources:
  device: cpu
  encoder_dtype: float32
  head_dtype: float64
  batch_size: 8
  torch_threads: 4
  blas_threads: 1
  estimated_hours: [1.2, 12.75]
outputs:
  run_id: sim_components_v1.0/E3-10/seed_42
  artifact_index_sha256: null
```

All manifest/checkpoint/config references must resolve to immutable versions at execution time. `null` is allowed in this **design example** only for genuinely unmaterialized metadata. The future runner must refuse `status: executable` with missing required hashes, unresolved source/renderer parameters, unknown group IDs, an unfrozen target batch, duplicate roles or an unregistered field. YAML booleans/numbers/lists remain typed. Resolve any profile into explicit settings and record both the profile ID and resolved content/hash; library defaults cannot silently drift.

For E0/E1 use `domain: real_to_real`, `simulation: null`, and the declared IDMT manifests. E1 has `mode: evaluate_existing` and a required parent E0 checkpoint/scaler hash, no new fitting. For E2-P use `mode: replay`, `profile: P02_SNAPSHOT`, saved CNN/scaler/feature hashes, `simulation: null`, `trainable_components: []`, historical training references, `validation_role: historical_unknown_membership`, and `final_test: null`; the two historical model identifiers remain distinct. For E2-B use PUB_Y manifests and `simulation: null` with recorded archived-source provenance. The same schema supports all three modes; unknown historical fields remain explicit and prevent claims they cannot support.

Immutable run identity is experiment ID + replicate ID + resolved-config SHA256, with input-manifest and code/dependency hashes included. A dataset/parameter/seed/preprocessing change creates a new run and, if it changes scientific design, a new protocol version. No mutable `latest` checkpoint or implicit environment-dependent setting.

## 11. Expected artifacts and completion criteria

Every core row must produce, in a future run directory:

- Resolved configuration, protocol/version/commit, dirty patch if any, dependency lock, hardware/device/thread metadata and timing.
- Dataset/release/class-map/split manifests; licence/provenance audit; group intersections/duplicate exclusions; counts and source/checkpoint/code hashes.
- Training-only scaler and classifier states; encoder provenance; feature-cache index keyed by input hash plus processing/checkpoint hash.
- Event predictions with class probabilities, true label, event/session/site IDs and split role; validation predictions stored separately.
- Metrics JSON, per-class and per-group CSVs, confusion matrices, all replicate scores, bootstrap specification/draw index, CI results and explicit inconclusive flags.
- Data-access log, final-batch lock and immutable artifact SHA256 index. Include negative results and failed runs.

E2-P additionally needs archive/model/scaler/test-array identities, output-order mapping, historical validation unknowns, replay comparisons and selected-snapshot limitations. E3 additionally needs paired source/trajectory/parameter manifests, source/render waveforms or reproducible hashes, render backend logs, rejection counts, intervention checks and the four-cell effects/interaction report. Do not publish third-party modified audio merely because the manifest is publishable.

**Current completion:** this protocol, dataset protocol and CSV specify the first phase. They do not establish that data admission, historical replay, rendering, training, test scoring or any hypothesis test has passed. Execution readiness remains conditional on the concrete gates in DATASET_PROTOCOL. Stop here; implementation requires a subsequent task.

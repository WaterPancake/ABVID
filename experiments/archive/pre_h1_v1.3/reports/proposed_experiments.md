# Proposed experiments: reproduction, domain gaps and simulation components

Revision: 2026-09-29. **Planning document only; no experiments are implemented or run by this revision.** This refines the [ABVID research roadmap](../ABVID_ROADMAP.md), using the completed literature analysis, the channel/class amendment and the saved reproduction results. Paper IDs resolve through the [literature inventory](literature_review.md#source-inventory). Literature-derived motivations are cited; hypotheses, numerical margins and future design choices below are our proposals.

**Primary question:** Which components of acoustic simulation most strongly determine synthetic-to-real transfer performance for vehicle classification using pretrained audio representations?

**Working hypothesis:** source-model mismatch contributes more than propagation-model mismatch. This is not established by the papers or by the checkpoint replays. The first controlled test estimates the effects of specified source and path interventions; it cannot identify a universal percentage of the domain gap attributable to each physical mechanism.

The immediate task remains **reproduction on the papers' own splits**. Our training partitions are deferred. The subsequent sequence is baseline measurement, a source × propagation experiment, then conditional environment/sensor, real-to-sim-to-real and representation-learning studies. It does not advance military family recognition or bypass the gates in [AGENTS.md](../AGENTS.md).

## 1. What changes from the earlier proposal

- Preserve the roadmap's **E0–E9 labels**. Put historical reproduction, labelled **R0**, first in execution order; it supplies the historical part of E2 even though E2 appears later in the conceptual ladder.
- Retain **car/truck** for future core training, validation and evaluation, as requested. Historical P01 reproduction retains five classes and P02 retains three, including motorcycle. Scores from these tasks are not interchangeable.
- Replace the obsolete assumption that the AI4TEN releases have not been retrieved with the actual partial-reproduction status. Separate saved-metric arithmetic, checkpoint inference and independent retraining.
- Make the initial source/propagation test a four-cell factorial. A recorded or generated waveform rendered through the same path is a useful source-*asset* comparison, but is not automatically a clean source-physics intervention.
- Add the roadmap's missing **E8: real → sim → real** test. Keep it separate from E9 representation adaptation because it changes the data distribution rather than the learned invariance objective.
- Assign each manipulated mechanism to one experimental factor. Ground/air transfer belongs to propagation; additive background belongs to environment; acquisition response belongs to sensor.
- Treat already evaluated MELAUDIS material as **exposed development data**. A new split name cannot restore an untouched final test.

This revision replaces the ordering and hypotheses in the previous version of this file. It **does not silently amend** the frozen numerical settings or matrix in `sim_components_v1.1`. Before future training, those protocol files need a versioned reconciliation with this proposal, after R0 and the deferred split decision.

| Earlier document identifier | Identifier in this proposal | Meaning |
|---|---|---|
| Old proposed Experiment 0 | E0, E1, E3 | Within-domain, cross-domain and representation controls separated. |
| Old proposed Experiment 1 | R0 and E2 | Literal paper reproduction separated from a common-task synthetic-data comparison. |
| Old proposed Experiment 2 / frozen protocol E3-00/10/01/11 | E4–E5, cells F00/F10/F01/F11 | Joint source × propagation experiment. Old result IDs must retain their protocol namespace. |
| Old proposed Experiment 3 | E6–E7 | Environment and sensor effects separated. |
| No prior counterpart | E8 | Estimate simulation distributions from source-domain real data. |
| Old proposed Experiment 4 | E9 | Matched augmentation, consistency and adaptation comparison. |

References for the earlier freeze: [experimental protocol](../experiments/EXPERIMENTAL_PROTOCOL.md), [dataset protocol](../experiments/DATASET_PROTOCOL.md), [experiment matrix](../experiments/EXPERIMENT_MATRIX.csv), [v1.1 amendment](../experiments/AMENDMENT_v1.1.md).

## 2. Evidence and available data at this revision

| Resource or result | Established locally | Consequence for this plan |
|---|---|---|
| P01 CNN and AST | Archived confusion matrices reproduced on both reported domains; AST required a documented compatible library version and reconstructed waveforms. | Valid checkpoint-replay references, not proof of independently reproduced training or source-free evaluation. |
| P01 BEATs, DANN, ArcFace | R0 update, September 29: restoring evaluation RNG consumption reproduces BEATs/DANN on both domains and ArcFace on MELAUDIS; ArcFace CH34 retains one net-count difference. Released BEATs backbones still match original pretrained weights; notebook gradient handling does not substantiate encoder fine-tuning. | The large earlier inference discrepancy was our sampling error. Preserve the remaining mismatch and distinguish successful checkpoint inference from unverified fine-tuning/training. |
| P02's eight checkpoints | Full-test matrices match archived run 3, seed 789. Five do not match their JSON's `best_run`. Balanced-table arithmetic checks pass. | Full-test checkpoint replay has succeeded with a weight-preserving compatibility fix. Exact balanced selections and five-seed retraining remain unresolved. |
| IDMT-Traffic | Local `dataset/IDMT_Traffic/`. Car/truck candidates: 3,902/511, provider `SE_CH34`, six site/date groups. | Use sE8 sensor token `SE`, then verify CH34 for these classes and average its stereo channels. Original acquisition groups, paired microphones and derivatives stay together. Counts are candidates, not a frozen split. |
| MELAUDIS | Local vehicle/background archives. Future `FF` + exact `1V-Car`/`1V-Truck` filename candidates: 7,854/256. Historical P01 test features were evaluated. | Suitable for a declared development-domain gap study after admission; no currently verified untouched confirmation partition. Session aliases, original events and duplicates still need audit. |
| P02 synthetic corpora | Local `dataset/AI4TEN/`: 5,000 procedural-labelled clips per original class; AudioLDM 201 car, 200 truck, 200 motorcycle. | Reuse released audio before generating more. Audit source lineage and the extra car file; do not silently delete it or assume it is a duplicate. |
| P02 real training/evaluation derivatives | Historical split lists and evaluation features available. Complete raw-source identity and missing `soundclass_v1` trainer unresolved. | Missing inputs limit literal retraining. They do not require substituting a new split and calling it the published result. |
| MAVD/DataSEC | Represented in P02's compiled evaluation material; their independently usable raw partitions are not established by that fact. | Optional later source/confirmation resources only after separate provenance, overlap and reuse audits. Never repurpose historical test assets as training sources unnoticed. |
| BVP/ACIDS and private military data | Not required by the civilian plan. | Optional external validation if later obtained; no access request or unavailable dataset is a prerequisite here. |

The first three rows were checked against saved result JSONs, including the September 29 R0 correction. Detailed evidence and limitations: [reproduction report](published_reproduction_status.md), [current R0 ledger](../experiments/reproduction/results/R0_20260929/R0_result_ledger.json). Dataset counts and channel rules: [v1.1 amendment](../experiments/AMENDMENT_v1.1.md) and [metadata audit](dataset_readiness_audit.md). Older review/audit statements about unavailable local checkpoints are historical, superseded by the reproduction report. Dataset licences remain in the dataset protocol; presence on disk does not establish redistribution permission.

The six IDMT candidate groups cannot satisfy the old requirement for at least three training, two validation and two test groups simultaneously. **Do not resolve that by randomly splitting clips.** After R0, decide between a versioned grouped-resampling design supported by these data or additional independent groups. No split allocation, reduced support floor or claim of statistical power is made in this revision.

## 3. Dependency ladder and scope

| Stage | Specific question | Required inputs / predecessor | Output needed before proceeding |
|---|---|---|---|
| **R0 — active** | Which published results can actually be reproduced on their original tasks and partitions? | Supplied releases, original paper/code definitions. | A result-by-result reconciliation, including unresolved or unavailable results, compatibility changes and retraining status. |
| **E0** | Does the common-task pipeline classify held-out IDMT acquisition groups? | R0 disposition; separately versioned future data/protocol freeze. | Within-domain predictions, data checks, per-class/group results and uncertainty. |
| **E1** | How much does that same fitted system degrade on MELAUDIS? | E0; admitted development target. | Comparable within/cross-domain gap. No refit on the target. |
| **E2** | How large is the synthetic-to-real gap, and does synthetic augmentation help? | R0 reference and E0–E1; admitted released synthetic corpora. | Matched-budget real/synthetic/mixed baselines on one common target. |
| **E3** | Do the gap and useful training-data contrasts depend on the representation? | Reuse E0–E2 manifests; audited encoder implementations. | CNN/classical versus frozen BEATs/AST comparison; fixed primary analysis before simulator changes. |
| **E4–E5 — first new physical test** | Does source enrichment help more than a specified propagation enrichment? | E0–E3 interpretable; auditable dry sources and one working renderer. | Four-cell factorial, marginal gains and interaction, including negative results. |
| **E6** | Does background/SNR diversity address residual transfer failures? | Fixed E4–E5 reference bank, independent background originals. | Native-real results plus controlled-SNR classification curves. |
| **E7** | Does sensor diversity add benefit and survive an independent-event device test? | E6 environment controls, paired-sensor provenance. | Separate sensor effect and environment × sensor interaction. |
| **E8** | Does estimating simulation distributions from real source data outperform manual randomization? | Identified tunable factors and source-only calibration data. | Equal-budget manual versus estimated-distribution comparison. |
| **E9** | Does representation learning add value beyond the same simulated/augmented inputs? | Fixed input distribution and remaining representation-related errors. | Matched objective comparison, with target access explicitly separated. |

E3 is a reuse/comparison stage, not a reason to train a large architecture sweep. E4 and E5 are **one crossed design**, not two sequential searches that each choose their best settings. E6–E9 are conditional extensions, not part of the minimum source-versus-propagation answer. Negative or inconclusive results count as completed experiments; progression depends on interpretability and remaining questions, not on obtaining a desired positive score.

The initial scientific deliverable is **R0 plus E0–E5**: a bounded reproduction result, measured gaps and a falsifiable source/path comparison. Ranking all four components additionally requires E6–E7. E8–E9 test remedies after the physical comparison; they are not prerequisites for reporting a negative physical result.

## 4. Shared design for future experiments

### Tasks, logical partitions and exposure

Use ordered core classes `[car, truck]`. IDMT maps documented C/T labels, MELAUDIS maps exact `1V-Car`/`1V-Truck` under `FF`, and the synthetic release maps exact car/truck labels. Motorcycle, bus, background and mixed/ambiguous events are outside this closed-set core; preserve and count exclusions. `commercial vehicle`, van, tractor, fuel type and model identity must not be inferred from broad labels. Historical R0 uses the authors' exact ontologies and preprocessing instead. [P01/P02 §2.1; v1.1 amendment.]

Future symbols denote **unmaterialized roles**, not exact file lists: `I_train/I_val/I_test` for IDMT; `M_dev` for declared MELAUDIS development evaluation; `T_confirm` for independently verified future confirmation; `Y_train/Y_val` for synthetic source families. `M_fit` and `M_adapt`, if used later, must be independent target training/adaptation groups disjoint from evaluation groups. Their admission is not assumed.

Group before filtering/cropping by original media/session, repeated acquisition context, synchronized microphones and duplicate-event links. Every derivative and crop inherits its parent's group. A site/date grouping is a conservative proxy, not proof of different vehicles. Hold out original background recordings as well as target recordings. Never infer road speed from IDMT's posted speed limit, or weather measurements from a dry/wet-road label. [Physical model; IDMT provider metadata semantics recorded in the readiness audit.]

The full P01 MELAUDIS feature set has already been scored. Treat its covered groups as exposed, even if some raw WAVs were never listened to. Check cross-dataset overlap before using any released test material as an asset, background, calibration example or adaptation input. Until independently unexposed groups are verified, all new target findings are **development results**. A completed development study is useful, but does not satisfy the roadmap's untouched-confirmation requirement.

Reserve a separate confirmation batch for each frozen study, or defer confirmation until all planned comparisons for one batch are fixed. Do not repeatedly inspect the same final target at each ladder step. After evaluation, a confirmation set is consumed; subsequent redesign needs new confirmation data. No currently unavailable corpus is required to begin the development work, and BVP is never mandatory.

### Preprocessing, representation and budgets

Carry forward the candidate `CORE8_BEATS_LR1_CT` control from the frozen protocol: mono; one centre-cropped 2 s event window; common 8 kHz intermediate bandwidth then 16 kHz encoder input; frozen official BEATs; train-only embedding `StandardScaler`; L2 logistic regression with `C=1`, `lbfgs`, tolerance `1e-6`, maximum 5,000 iterations. Classes remain balanced in training. This is a proposed future control, not a repaired version of P01's released head. Pin encoder code/checkpoint, front-end, pooling and numerical environment before use.

The 8 kHz intermediate rate controls the limited bandwidth of the procedural baseline; it restricts conclusions to approximately 0–4 kHz information. A native/common-16-kHz comparison is a **separate bandwidth ablation**, not a change made in just one source cell. AST uses its pinned native feature extractor on the same observation waveform; do not feed a PCEN transform into a pretrained encoder that expects a different front-end.

Initial task-training budget: 200 distinct events/class, 400 total, subject to the deferred provenance/support audit. Use the same five replicate seeds `[42, 123, 456, 789, 1024]`; separate data-selection, source, path, environment and sensor RNG streams. Frozen logistic fits are deterministic: these seeds vary admitted training examples or synthetic banks, not fictitious stochastic optimization. Report actual training seeds separately for CNNs or later trainable projections.

For a synthetic versus real comparison, match event count, class prior, duration, bandwidth and head. For augmentation, compare `400 real + 400 synthetic` with the same real data receiving an equal presentation budget through conventional augmentation or resampling; also retain the 400-real baseline. Unique real-event counts do not increase in the controls. Avoid an immediate ratio sweep. Keep fitting transforms within each arm's training role; P02's historical real-fitted scaler is an R0 condition, not a strict synthetic-only control.

Recorded assets, unlabelled real calibration and generator/encoder pretraining are real-data access. “Synthetic-only” means **task-training on synthetic examples**, not absence of real audio in pretraining. A recorded-source hybrid is `real-derived simulated → real`; it must not be represented as zero-real-data synthesis.

### Metrics, uncertainty and decisions

Primary metric `Q`: macro-F1 over the full fixed **event** manifest, computed identically for all compared arms. Report balanced accuracy, per-class precision/recall/F1, confusion matrices, per-group results and worst-group recall. Report test class counts and priors; do not assume macro-F1's chance value is `1/K`. Historical balanced subsamples remain separate R0 outputs, not replacements for common-task scoring. If future windows are added, aggregate within event before scoring.

Use five replicate results and 95% paired intervals over complete independent acquisition groups and replicate IDs; proposed bootstrap: 10,000 draws, seed `314159`. For common-target contrasts, use identical resampled groups across arms. For the source-test minus target-test gap, resample the two domain group sets separately. For future grouped cross-validation, retain fold membership and the pairing of out-of-fold predictions; overlapping folds are not independent sample units. The estimator must be finalized with the deferred split design. Report class-missing draws and insufficient group support; do not manufacture confidence by bootstrapping thousands of windows from a few sessions.

Proposed practical margins: **0.05** for a baseline gap and **0.03** for an intervention gain, on the 0–1 F1 scale. These are design decisions, not published effect thresholds. Lower interval bound above the margin supports the practical claim; upper bound below it rules out that effect size; an interval crossing it is inconclusive. A claim of *any* improvement/dominance instead uses zero. Failure to exceed 0.03 does not establish equivalence or reverse the sign. Report effect sizes and interval widths, not only pass/fail.

Declare one primary contrast per hypothesis. Additional component removals, source families, representations and SNR strata are secondary; report them all and use a prespecified multiplicity adjustment when making a family of confirmatory claims. No choice of seeds, classes, thresholds, subset, pooling or checkpoints may be optimized against `T_confirm`. If development scores motivate a revised design, version it and retain the original result.

Calibration is secondary: multiclass-form Brier score and fixed 10-bin equal-width ECE, without target calibration fitting. Embedding distances, domain-probe accuracy, spectral coverage, FAD and listening quality are diagnostics, not substitutes for real-target classification. [P06 §5; P07 §3; P14 §§3–4; representation taxonomy.]

### One owner for each simulator change

| Factor | Included mechanisms | Explicit boundary |
|---|---|---|
| Source | Engine/exhaust harmonic envelope, source-state dynamics, component balance, rolling excitation; emission directivity if later supported. | RPM is not vehicle speed. Surface-induced tire excitation is a source effect. |
| Propagation | Travel time, spreading, moving-source Doppler, trajectory/range, ground reflection, atmospheric transfer. | Physical weather/ground parameters can affect other layers too; change only the named path effect in a controlled arm. |
| Environment | Added background and interfering-source identity, temporal structure and SNR. | Ground reflection/air absorption are not added a second time as “environment.” Overlapping vehicles require a separate task/label definition. |
| Sensor | Frequency response, gain, bandwidth, self-noise, supported nonlinear/codec effects. | Physical microphone response is unknown unless measured; random EQ is a proxy, not a calibrated device model. |

This operational ownership supports attribution within the simulator. It does not deny real physical interactions. [P04 §§1.2, 3–6; P05 §3.4; P13 chapter 3 §2.3; physical and domain-shift taxonomies.]

## 5. R0 — Finish the paper-reproduction disposition first

R0 is an engineering/scientific validation prerequisite, not one of the five new hypotheses. Keep original classes, partitions, saved feature ordering, published sampling and compatibility changes explicit. Preserve published quirks; any corrected-method result gets a distinct label.

The September 29 current-release audit resolves the main BEATs-family sampling discrepancy and documents a bounded R0 disposition. Remaining work is the ArcFace CH34 residual, P02 balanced-selection/trainer omissions, and checkpoint identity mismatch. Reconstruct exact original inputs where evidence permits; do not search preprocessing choices until the target number matches. Full retraining, if original inputs and training code can be recovered, uses the authors' original seeds and split definitions. An unavailable dependency remains `unknown`/not reproduced; a locally reconstructed trainer is a reproduction variant, not the missing original. Checkpoint-audit completion is not full training reproduction or authorization to start E0.

**Completion criterion:** a bounded result ledger listing `arithmetic verified`, `checkpoint replay exact`, `compatibility replay`, `retraining reproduced`, `mismatch unresolved`, or `not executable`, with evidence per reported result. “Not executable from the available release” is an admissible disposition, not a pass. Missing artifacts must not indefinitely require unavailable data or be hidden by a new split. Report that disposition before a separate future-study freeze.

No new paper-score claims are introduced here. Current evidence remains in [published_reproduction_status.md](published_reproduction_status.md) and [reproduction/README.md](../experiments/reproduction/README.md).

## 6. H1 — Establish the gaps before explaining them (E0–E3)

**Research question / hypothesis.** Does a meaningful cross-domain loss persist under a common car/truck task, and does synthetic task-training leave a meaningful deficit relative to real training on the same target? Treat these as separate H1a/H1b contrasts. Synthetic augmentation and representation advantages are secondary measured questions, not assumptions required for progression.

**Motivation from literature.** P01 compares same-event sensor transfer with MELAUDIS transfer; that is not an independent-session source/target gap. P02's reported synthetic deficit depends on source counts, normalization and test sampling. The release audit further separates checkpoint behavior from claimed training. [P01 §§2.4–2.5, Table 3; P02 §§2.3–2.4, Tables 4–5; reproduction report.]

**Independent variables:** evaluation domain; task-training data origin; established representation. **Dependent variables:** domain gap, synthetic-to-real gap, augmentation gain, per-class/group failure and calibration.

**Datasets required:** admitted IDMT, MELAUDIS development data and the local released procedural/AudioLDM corpora. Original P01/P02 artifacts remain R0 references on separate tasks. No US8K fallback is needed now that vehicle releases are locally available; a general-audio reproduction would not resolve missing vehicle split provenance.

| Experiment | Train / validation | Evaluation | Representation/model and baseline |
|---|---|---|---|
| E0 | `I_train`; `I_val`, with group roles fixed before fitting. | `I_test`. | Primary frozen BEATs + fixed logistic head; lightweight MFCC + logistic comparator and constant/prior-based descriptive controls. No augmentation. |
| E1 | Reuse the exact E0 model/scaler; no target validation fitting. | Same model on `I_test` and `M_dev`. | Identical preprocessing and class map to E0; no adaptation. |
| E2 | Matched-budget real, released procedural, released AudioLDM and real+each; train-only transforms. `Y_val` groups audit synthetic behavior, not real-target selection. | Same `M_dev` manifest for every arm. | Same primary head; real-only, real resampling and conventional augmentation controls. R0 supplies the published CNN reference separately. Both-synthetic/real+both are secondary after single-source contrasts. |
| E3 | Reuse E0/E2 data selections; appropriate source validation only. | Reuse identical source and target evaluation manifests. | Frozen BEATs + LR versus frozen AST + LR, MFCC + LR, and the released CNN architecture trained anew on the core task as a separately labelled baseline. No encoder fine-tuning or architecture search. |

**Experimental control:** hold classes, budgets, event windows and evaluation populations fixed. Use the same fitted model in the within/cross-domain comparison. AST/CNN historical replay success does not replace the need to validate their new binary-task pipelines. If the independently pinned BEATs pipeline fails verification, resolve it or preregister an AST primary before target evaluation; do not choose the primary encoder by target ranking.

**Ablations:** strict synthetic-only scaling versus the separately labelled historical real-statistics-assisted condition; equal presentation-budget augmentation; frozen representation versus the same small classifier on MFCC. Native-bandwidth and target-trained references are later sensitivity analyses, not conditions for starting the gap measurement. A target-trained reference requires independently admitted `M_fit/M_val`, equal training budgets and evaluation on the same disjoint `M_dev`; otherwise its gap is `not estimable`.

**Evaluation metrics and estimands:** with fixed metric `Q`, report

```text
Delta_domain = Q(I_train -> I_test) - Q(I_train -> M_dev)
Delta_sim2real(Y) = Q(I_train -> M_dev) - Q(Y_train -> M_dev)
Gain_aug(Y) = Q(I_train + Y_train -> M_dev) - Q(I_train -> M_dev)
```

Use paired training replicates and group uncertainty as above. `Delta_domain` changes the test population as well as the domain; it is descriptive, not a causal geography effect. If the target-trained reference becomes feasible, report `Q(M_fit -> M_dev) - Q(I_train -> M_dev)` separately. A smaller gap caused by worse within-domain performance is not improved transfer: require and report absolute target gains.

**Expected result:** nontrivial real-to-real and synthetic-to-real gaps, with representation- and class-dependent effects; augmentation may help, fail or hurt. **Falsification criterion:** an upper 95% bound below 0.05 rules out the proposed practically important gap for H1a or H1b separately; an upper bound at or below zero contradicts a positive gap. Wide intervals are inconclusive. No pretrained-advantage or augmentation claim is made without its own paired estimate.

**Likely confounding variables:** event/channel leakage; class priors and fleet differences; sample and unique-source counts; repeated synthetic templates; bandwidth; preprocessing statistics; encoder/generator pretraining overlap; target exposure; unknown physical-vehicle identities.

**Dependency gate:** data/software controls pass and gap estimates are interpretable before E4–E5. A clearly harmful simulator baseline can justify diagnosis even when augmentation gives no gain. Do not require positive augmentation or proof of source dominance before testing it. If the gap is precisely negligible, change the question; if uncertainty is dominated by too few groups, improve evidence before a larger simulator sweep.

**Expected artifacts:** fixed manifests and role/exposure ledger; feature-cache/checkpoint hashes; matched per-event predictions; source/target class/group tables; gap and augmentation contrasts with intervals; R0-to-new-task reconciliation. No pooled headline across two-, three- and five-class tasks.

## 7. H2 — Compare source and propagation contributions (E4–E5)

**Research question / hypothesis.** Under the same source-family count, observation bandwidth and pretrained representation, does source enrichment improve real-target F1 more than a specified propagation enrichment? The directional working hypothesis is `delta_S > delta_P`; the practical version is `delta_S - delta_P >= 0.03`, with `delta_S > 0` also required to claim a useful source improvement.

**Motivation from literature.** P02 argues that source variability is missing but does not isolate its effect; its released generator already contains AM and broadband components, so “adding modulation where none existed” is not an accurate baseline description. Harmonic balance/state matter in P04/P05, while station/terrain effects in P09/P13 prevent assuming path mismatch is negligible. [P02 §3.3 and released generator audit; P04 §5.3; P05 §§3.4–5; P09 §V; P13 chapter 3 §2.3.]

**Independent variables:** source level S0/S1 crossed with propagation level P0/P1. **Dependent variables:** common-target F1, marginal source/path effects, interaction, class/group recall. **Datasets required:** the admitted IDMT/MELAUDIS baseline roles and a new, versioned dry-source/rendered corpus. The P02 procedural-labelled archive is a transfer baseline, not a verified factor-cell input: its actual rendering backend and lineage remain uncertain.

**Representation/model and baseline:** unchanged primary frozen encoder + LR from E3; F00 is the controlled-renderer reference. Repeat the *complete* four-cell comparison with frozen AST as a prespecified representation sensitivity check when resources permit. A result from BEATs alone is conditional on BEATs, not all pretrained audio representations.

| Cell | Source | Propagation | Question |
|---|---|---|---|
| F00 | S0, audited simple procedural dry source. | P0, moving direct path with spreading, delay and Doppler. | Controlled baseline. |
| F10 | S1, same base draws plus harmonic-envelope diversity, slow source fluctuations and component-balance diversity. | P0 unchanged. | Source change with path fixed. |
| F01 | S0 unchanged. | P1, same trajectory/backend plus ground reflection and atmospheric filtering. | Path change with source fixed. |
| F11 | S1. | P1. | Joint change and interaction. |

This carries forward the narrowly specified v1.1 source/path contrast; its numerical priors are candidate settings for the next freeze, not measurements of real vehicles. Doppler is present in both P levels, so this first comparison does **not** estimate Doppler's benefit. Follow-up E5 mechanism removals can test Doppler, reflection and absorption separately after the four cells are reported. Source directivity, actual load/gearing, wet-road excitation and additional engine architectures are not silently included in S1.

**Experimental control:** pair base source IDs and trajectories across all cells; use equal template counts and 200 rendered training events/class, with distinct template families in diagnostic validation. The earlier candidate is 20 templates/class × 10 trajectories, not 200 independent physical vehicles. Class-independent path draws prevent range/speed from encoding the label. Use separate RNG streams so extra S1 draws cannot change P. Fix environmental background and sensor response; prohibit silent renderer fallbacks. Match source and receiver scalar RMS in this first contrast and retain the same windows. This conditions out absolute distance attenuation/SNR effects; it cannot rank the entire contribution of propagation in naturally varying recordings.

Before target scoring, verify deterministic generation, dry-source hashes invariant across P, geometry invariant across S, durations/rates, factor toggles, no clipping, and an analytic stationary-tone/moving-path sanity check. Interventions must actually change the intended acoustic mechanisms. Log all generated failures and apply one paired rejection rule across cells.

**Ablations:** required four cells; optional S1 component removals only after the bundled result; a later non-RMS-normalized path comparison with fixed additive noise to restore range-induced level/SNR effects. Freeze all follow-up contrasts before their evaluation. More templates or wider parameter ranges are separate diversity/dose interventions, not a free increase for the favoured factor.

**Evaluation metrics:** for common-target scores `Qij`, report

```text
delta_S = ((Q10 - Q00) + (Q11 - Q01)) / 2
delta_P = ((Q01 - Q00) + (Q11 - Q10)) / 2
interaction = Q11 - Q10 - Q01 + Q00
primary contrast = delta_S - delta_P
```

Report both conditional gains for each factor as well as these marginal averages. Use paired groups/replicates and the same target population. Spectral/harmonic/modulation changes are manipulation checks; they do not establish real transfer. No “percentage of total gap explained” is inferred from these four cells.

**Expected result:** S1 may help more than P1, but improvements, reversal and interaction are all plausible. **Falsification criterion:** an upper bound below 0.03 rules out the practical margin; an upper bound at or below zero contradicts directional source dominance in this setting. If both interventions harm transfer, a less harmful source intervention is not evidence that source enrichment works. Strong conditional reversals prevent a single context-free ranking, even if an average contrast is positive.

**Likely confounding variables:** arbitrarily unequal intervention strengths; unrealistic state combinations; class-specific source priors; common bandwidth removing useful propagation effects; RMS normalization; hidden backend substitution; source/trajectory correlation; representation dependence and too few independent target groups.

### E4 asset-source extension: recorded/generated sources through the same path

This is a conditional extension of H2, not required for its minimum test. Compare analytic dry sources, released generated assets and admitted recorded real assets through the **same H**. Include each recorded/generated asset with **no additional H** and with conventional corruption, so extra physical rendering can be compared with simply using the asset.

Roadside recordings already contain motion, path, environment and sensor effects; AudioLDM outputs can also contain scene effects. Applying H again may create double Doppler or double coloration. Therefore this tests the utility of a source-asset strategy, not isolated dry-source realism. Real assets come only from `I_train` or independently admitted source material with compatible use terms, never evaluation/adaptation/confirmation groups. Hold original assets and every derivative together; match distinct asset-family counts and report prior real-data access. Do not require an unavailable anechoic corpus. Learn a new source model only if these existing-asset comparisons justify it.

**Dependency gate and artifacts:** a four-cell result table, paired latent/render manifests, renderer checks and effect/interaction intervals are the first physical result. An unsupported hypothesis is reported rather than repaired through target-driven source tuning. Preserve the E3 baseline as a reference when later choosing a useful input bank; do not require E4/E5 to succeed before studying another plausible component.

## 8. H3 — Test environment and sensor randomization separately (E6–E7)

**Research question / hypothesis.** With source/path held fixed, do additive-environment diversity and sensor-response diversity produce useful transfer gains, separately or jointly? H3a concerns environment and H3b sensor; practical gain threshold is 0.03. The combined result alone cannot identify which factor helped.

**Motivation from literature.** P13's terrain shifts mix mechanisms, P01's same-event microphone comparison does not establish independent-event device robustness, and P16 shows device adaptation can matter in another acoustic task. PCEN is a candidate frontend motivated by P03, not established proof of vehicle-domain invariance. [P13 chapter 3 §2.3; P01 §§2.5, 3.3; P16 §§2–4; P03 §3.2.]

**Independent variables:** added-background diversity off/on × approximate sensor-response diversity off/on. **Dependent variables:** native-real transfer, controlled-SNR degradation, per-device/class/group recall and interaction. **Datasets required:** the fixed earlier source/path bank, admitted training backgrounds, held-out original backgrounds, and IDMT paired sensors on wholly held-out events. MELAUDIS remains a bundled development-domain test, not a labelled weather or calibrated-sensor experiment.

**Representation/model and baseline:** fixed primary encoder/head and source/path bank; no extra corruption versus background-only, sensor-only and joint training. Compare joint randomization with an equal-count conventional gain/noise/SpecAugment control. Source/path cannot be reoptimized between cells. A PCEN versus log-Mel frontend comparison is a secondary fixed-CNN comparison with matched architecture/training, not a modification of the expected BEATs/AST input.

**Experimental control:** backgrounds must be independent of class and original source recordings must be disjoint across training and evaluation. Keep sensor filters/noise independent of labels and background identity. E6 changes only additive background/SNR; E7 changes response/bandwidth/gain/self-noise. Ground/atmosphere transfer stays at its earlier fixed setting. Use class-independent plausible response families, explicitly approximate when measurements are unavailable.

For sensor transfer, all microphone views of the test event/session must be absent from training. Score held-out events through each actual microphone and report the paired prediction change alongside absolute scores. Training on one channel and testing its sibling from the same training event is only a familiarity diagnostic, not the primary sensor-generalization result. If two-device metadata cannot support unseen-event/unseen-device claims simultaneously, state which axis is held out.

**Ablations:** the four environment/sensor cells; background identity versus SNR range; response filtering versus sensor self-noise; optional PCEN/log-Mel. Exclude overlapping-vehicle mixtures from the minimum single-label test: adding a second vehicle is not automatically label-preserving background augmentation.

**Evaluation metrics:** full native-real metrics plus classification curves at `30, 20, 10, 5, 0, -5, -10 dB`. Define controlled SNR as the target-component power divided by the added-noise power at the declared observation point; report post-sensor achieved SNR as well as requested pre-sensor SNR. Existing noise inside the target clip remains unknown, so these are incremental-corruption curves, not calibrated total field SNR. Never renormalize each evaluation cell in a way that erases the tested gain/noise effect. No detection probability or false-alarm claim is made without a separately admitted vehicle-absent task.

**Expected result:** some randomization helps low-SNR/device robustness, with possible clean-audio tradeoffs. **Falsification criterion:** upper interval below 0.03 rules out the respective practical benefit; a reliably negative gain contradicts improvement. For the proposed robust-benefit claim, additionally require non-inferiority of each class recall and worst-group recall with a 0.05 margin: their paired lower bounds must exceed -0.05. This margin is a design choice; insufficient precision leaves robustness inconclusive. Missing device or background independence makes the corresponding claim untestable, not positive.

**Likely confounding variables:** residual noise in “clean” clips; background/class shortcuts; filter-induced SNR changes; differing sensor event populations; proxy responses unlike actual devices; same-session familiarity; label-changing confusers.

**Dependency gate and artifacts:** fixed background/sensor manifests, achieved-SNR checks, native and corrupted results separated, paired device results and interaction estimates. Advance to E8 only for factors that can be calibrated from available source data; retain null/harmful results in the component comparison.

## 9. H4 — Real → sim → real distribution calibration (E8)

**Research question / hypothesis.** Do simulator distributions estimated from real source-training recordings improve target transfer by at least 0.03 F1 over the same simulator with hand-set randomization, at equal real-data access and synthetic budgets?

**Motivation from literature.** Physical signatures vary with state and acquisition, while P02/P14 show that adding synthetic examples alone does not guarantee transfer. The roadmap proposes matching task-relevant variation instead of waveform realism. **This distribution-calibration intervention is new proposed work, not a demonstrated result of those papers.** [P04 §§4–5; P02 §§3.1–3.3; P14 §§3–4; roadmap RQ5.]

**Independent variable:** manual parameter priors versus priors/statistical observation distributions estimated from `I_train` only. **Dependent variables:** target F1 gain, source-held-out descriptor coverage, low-SNR/device behavior and class/group failures.

**Datasets required:** admitted `I_train` recordings/metadata and the existing simulator bank; independent `I_val` for calibration-method checks; unchanged `M_dev` for evaluation. No target waveforms, features or labels enter calibration in the primary arm. Both arms receive the same source-real access budget. If a later target-calibrated version is proposed, it requires a distinct admitted `M_adapt` and a target-access-matched control.

**Representation/model and baseline:** same frozen encoder/head and simulator structure as the preceding reference. Primary baseline is manual randomization at identical rendering/template counts. Include unchanged real-only and conventional real augmentation results from E2 as context; do not interpret real-calibrated synthesis as zero-real-data training.

**Experimental control:** freeze the descriptor set and one calibration procedure before evaluating `M_dev`. Start with source-training distributions that can actually be estimated—e.g. observed spectral/modulation statistics or documented sensor/condition metadata. Do not turn speed limits into speed distributions, spectral peaks into known RPM, or a single-channel spectral envelope into separately identified source and path parameters. When physical parameters are unidentifiable, call the result *observation-statistics calibration* and retain `unknown` for the physical values. Use constrained priors and source-validation checks; no optimization against target F1 or target embedding distance.

**Ablations:** manual prior; estimated distribution; estimated marginal distributions with dependency structure removed if joint fitting is justified by sample support. Calibrate source and nuisance families separately before a joint fit; keep all other factors constant. Defer training a generative source model until simpler calibrated distributions and recorded-asset controls demonstrate a need. An estimated point value versus a distribution is optional, not part of the first minimum comparison.

**Evaluation metrics:** paired target macro-F1 gain and common robustness metrics; held-out source descriptor agreement as secondary diagnostic. Matching descriptors is not itself a transfer success. Report effective calibration groups/counts, fitted parameters and uncertainty/identifiability limits.

**Expected result:** constrained source-data calibration may improve coverage, but can also overfit the source environment and hurt the target. **Falsification criterion:** upper interval below 0.03 rules out the practical gain; a confidently nonpositive gain contradicts improvement. Better source descriptor fit without a target gain falsifies the claim that this calibration method improves transfer in the tested setting.

**Likely confounding variables:** extra real-data information; class-condition imbalance; jointly unidentifiable source/path/sensor effects; calibration overfitting; estimated priors too narrow; selection using target diagnostics; different synthetic entropy rather than better physical validity.

**Dependency gate and artifacts:** calibration-data manifest, descriptor/procedure specification, manual/estimated distributions, source-held-out checks, paired generated banks and target effect interval. The evidence must identify what was estimated and what remains unknown before interpreting a real→sim→real benefit.

## 10. H5 — Does representation learning add to the same data? (E9)

**Research question / hypothesis.** On a fixed source/simulation/augmentation distribution, does an established class-preserving consistency objective improve independent real transfer by at least 0.03 over augmentation-only training? Domain adaptation is a separate target-access question, not part of the same source-only effect.

**Motivation from literature.** P07 motivates pretrained semantic representations, P03 considers similarity learning, and P16 combines domain alignment with label preservation. P01's negative DANN result and released gradient-handling problems motivate matched implementation controls rather than adopting its reported fine-tuning comparison as established. [P07 §3; P03 §3.6; P16 §2; P01 §3.4; reproduction report.]

**Independent variables:** objective with fixed architecture/data—classification only, classification plus established same-event consistency, and source-supervised ArcFace; separately DANN with a fixed unlabelled target budget. **Dependent variables:** target F1 gain, class/group recall, calibration, paired-view stability and class separation.

**Datasets required:** the fixed admitted corpus and metadata-linked transformed views. Initial positive pairs are the same source event under label-preserving observation changes; two cars are not the same physical identity merely because both are labelled car. DANN additionally requires disjoint `M_adapt`; if it cannot be admitted, omit that arm rather than adapting to evaluated events.

**Representation/model and baseline:** identical pretrained backbone, projection and classification capacity across the objective comparison. Begin with the backbone frozen; train a matched projection/head in both classification-only and consistency arms. The augmentation-only control consumes both views and the same number of optimizer updates. ArcFace is a source-supervised margin objective, not inherently target-domain adaptation. Later backbone fine-tuning changes trainable capacity equally across arms and requires a gradient/weight-change check.

**Experimental control:** identical waveform pairs, labels, batches, update count, selected epoch rule and real-data access. Choose objective weights/early stopping on source-group validation using a preregistered small search or fixed settings. For DANN, match source labels and `M_adapt` access; use a zero-domain-loss control with the same target passes and normalization behavior. Report any fitted target statistics. No confirmation-domain audio is accessible to training.

**Ablations:** consistency weight zero versus the fixed positive value; class loss retained; source-only ArcFace versus matched classification; target exposure off/on with domain loss off/on. Domain probes and label-loss removal are diagnostics, not new contenders selected on the target. Fine-tuning, hard state positives and multiple adaptation methods are deferred until this limited comparison is interpretable.

**Evaluation metrics:** paired macro-F1 gain and common class/group/calibration metrics; within-source embedding distances and held-out domain probes as diagnostics. Domain confusion or low embedding variance is useful only if class discrimination survives. Report collapse and negative transfer.

**Expected result:** a possible incremental gain from label-preserving consistency; DANN may help or hurt under class-conditional mismatch. **Falsification criterion:** upper interval below 0.03 rules out the practical consistency benefit; confidently nonpositive target gain contradicts improvement. Better invariance probes accompanied by worse classification do not support H5. A DANN benefit supports the separate access-matched adaptation claim, not source-only robustness.

**Likely confounding variables:** accidentally frozen/unfrozen modules; unequal projection capacity or view counts; class-changing positive pairs; target class priors; hidden label/statistic access; optimizer budget and differing checkpoint selection.

**Dependency gate and artifacts:** identical-input objective comparison, trainable-parameter/gradient audit, objective and target-access records, embeddings/checkpoints and paired predictions. One final confirmation batch may follow a complete freeze if independently untouched data exist; otherwise report development evidence with that limitation.

## 11. Interpretation, compute and records

A component can improve transfer without being the largest physical error in the original simulator. Intervention magnitudes have no shared physical unit. Report **conditional intervention gains**, tested ranges and interactions, not an intrinsic universal ranking. E4–E5 answer source versus a limited propagation enrichment; E6–E7 extend coverage. Effects measured after changing the reference bank are not directly comparable to earlier effects: for a later four-component ranking, rerun the prespecified relevant cells on one common bank, model and target before comparing gains.

An elaborate propagation model that gives no benefit is a useful negative result, especially when level normalization or encoder bandwidth explains the scope. A recorded-source hybrid that helps demonstrates an asset strategy, not recovery of a dry engine source. A smaller domain probe score is not sufficient evidence of generalization. These distinctions follow the [research gaps](research_gaps.md), [physical signal model](physical_signal_model.md), [domain-shift taxonomy](domain_shift_taxonomy.md) and [representation taxonomy](representation_taxonomy.md).

Use local CPU for indexing, replay, simulation checks and cached-feature heads. Benchmark a fixed pilot before estimating full runtime; record device, precision, peak memory/VRAM, throughput, load time and end-to-end time. Frozen embedding extraction may move to a GPU if measurements justify it; do not provision GPU infrastructure or implement a new source generator for this document revision. Cache by input, preprocessing and encoder hashes. P01/P02 replay timings are available in the reproduction report but do not predict training or a full simulator sweep. Arrays, localization, continuous detection, Jetson deployment and fine-grained/open-set recognition remain separate later work.

Reuse the **one canonical YAML structure** in the [experimental protocol, §10](../experiments/EXPERIMENTAL_PROTOCOL.md#10-canonical-experiment-configuration-format) when the next version is frozen. Extend it consistently rather than creating a different format per stage. Every run must identify protocol-qualified experiment ID, hypothesis/primary contrast, original versus variant status, commit plus dirty-tree/source hashes, dataset releases/licences, exact manifests/roles and exposure history, class map, group IDs, preprocessing/window/rate, checkpoint/code versions, trainable modules, augmentation/factor levels and physical units, real-data access, all RNG streams, budgets, metrics/CI method, compute and artifact paths. Unknown values remain explicit; unknown required membership or lineage blocks the associated claim.

Append the roadmap's structured entry to `RESEARCH_LOG.md` when an experiment is actually run. Store numerical results and per-event predictions in machine-readable files and link them from the report. No log entry should imply training occurred during this planning revision.

Before the next executable protocol freeze, resolve exactly these planning dependencies: R0 disposition; feasible IDMT grouping and admitted target-development roles; exposed versus independently untouched evaluation groups; synthetic template lineage; primary/secondary encoder implementation checks; fixed factor priors and contrasts; and matching versions of the protocol, dataset document and matrix. That is the handoff from this refined proposal to future execution.

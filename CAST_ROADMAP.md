# CAST research roadmap

Updated 2026-10-06. CAST fits an effective model of recorded vehicle acoustics,
then samples its fitted population to generate additional observations. Its
research objective is to determine whether that calibration improves vehicle
classification across recording groups and datasets, beyond ordinary augmentation
of the same real recordings.

The next priority is **validate the current population more broadly, then test
its classification utility under matched controls**. Increasing renderer size or
training a neural generator is conditional on evidence of a specific limitation.

This is the CAST branch of [ABVID_ROADMAP.md](ABVID_ROADMAP.md), particularly
RQ5 and E8, with interfaces to H1 baselines and H2 component studies. It retains
the CAST-0 through CAST-6 numbering from the [original plan](experiments/cast_coverage/PLAN.md).
Completed protocols and results remain historical references. Proposed stages
below require their own frozen contracts; this roadmap does not authorize a
transfer run or access to additional audio.

## Current evidence

| Stage | Status | Established result and remaining limit |
|---|---|---|
| CAST-0 access and interfaces | Complete | Source allowlists, ancestry, configuration and replay records exist. |
| CAST-1 numerical fitting | Complete | Declared synthetic fixtures and ambiguity checks pass. This verifies the implemented renderer and fitter. |
| CAST-2 real pilot | Complete | All 50 IDMT fits completed and improved checking loss; 43/50 retained disagreement between near-equivalent starts. Physical parameters are not identified. |
| CAST-3 fitted population | Implemented through v15 | The current population uses 380 training observations, 190/class, from five conservative groups. Generated examples retain all contributing ancestry. |
| CAST-4 acoustic coverage | Development objective met | v15 passes the declared mean coverage, control-distance and spread criteria on one repeatedly exposed IDMT outer group. Independent confirmation remains open. |
| CAST-5 classification transfer | Unrun | CAST has no established classification gain. |
| CAST-6 learned parameter estimator | Deferred | Consider only after demonstrating a useful fitting task and a measured runtime bottleneck. |

Sources: [pilot](reports/CAST_pilot.md), [original coverage failure](reports/CAST_generalization.md),
and [coverage improvement record](reports/CAST_improvement.md). The pilot's
original stopping statement records its historical authorization; later separately
authorized coverage work is documented in the subsequent reports.

The v15 reference fixes width T=1.1 for noise-spectrum, harmonic/noise-mixture
and envelope variation. It leaves harmonic spacing and weights unchanged.

| Outer development measure | Car | Truck |
|---|---:|---:|
| Mean marginal coverage | 82.019% | 81.286% |
| Scaled marginal Wasserstein-1 distance | 0.719913 | 0.673616 |
| W1 reduction against matched independent marginals | 12.295% | 17.207% |
| W1 reduction against matched prototype | 20.946% | 24.048% |

These are means over five fixed synthesis seeds on 570 real observations in
one outer group. Two car seeds remain below 80% coverage. Class-balanced W1 is
4.460% worse than the narrower v14 population: widening improved coverage at a
cost in distance. The completed v15 study passed 306 historical tests and full
source/outer replay. The later [repository parity checks](reports/REFACTOR.md)
are a separate validation of code relocation, not another scientific experiment.

The current W1 averages one-dimensional feature-distribution distances, scaled
using training-real variability, within four equally weighted descriptor families.
Coverage averages the fraction of real feature values inside generated 5th–95th
percentile intervals. Neither measures the full joint distribution or recognition
accuracy. The label “joint W1” in historical reports names the sampler, not a
multivariate transport metric. Preserve these definitions and historical thresholds.

## Relationship to ABVID

| ABVID component | CAST dependency or contribution |
|---|---|
| Admission and R0 | Inherit admitted labels, original-source grouping, licenses and checkpoint provenance. Keep original-paper class tasks separate from car/truck. |
| H1 | Reuse group manifests, matched training selections, frozen BEATs768, MFCC26 and the fixed linear head. H1 provides reference behavior, not proof of CAST utility. |
| [P0 offline prototype](ABVID_ROADMAP.md#p0-offline-classification-prototype) | Deliver the existing H1 classifier independently of CAST. CAST supplies candidate training data for a later model version, not a runtime dependency. |
| H2 and E4–E5 | Share validated signal-processing components and attribution checks. CAST outputs contain recording coloration and cannot silently replace a dry-source factor. |
| E6–E7 | Later test background and sensor variation separately, once the minimal CAST comparison is interpretable. |
| E8 and H4 | CAST supplies a concrete real-to-simulation-to-real calibration method and its matched manual-prior comparison. |
| E9 | Representation adaptation is later work, after measuring CAST with a fixed representation. |

The [H2 latency control](reports/H2_air_latency_control.md) found that filter
delay reproduced the earlier implemented-path benefit, while attenuation added
little in that comparison. Physical source-model dominance remains a hypothesis.
CAST must keep numerical effects, observation matching and physical mechanisms
separate when interpreting future results.

The baseline CLI/model bundle and a future inspectable replay interface can
progress alongside CAST-4A/4B. Packaging that prototype does not authorize CAST
transfer evaluation. A CAST-trained replacement must first be evaluated under
the separately frozen CAST-5 comparison, retaining the original real-data model
as its reference. Descriptor coverage alone is insufficient to claim a better
prototype classifier.

## Research questions and decisions

1. Does the fixed v15 population cover different recording groups, or mainly the
   development context that informed its design?
2. Does marginal coverage preserve feature relationships and useful class
   differences, or simply produce wider distributions?
3. Does CAST improve classification beyond matched real-data repetition,
   conventional augmentation and an unfitted same-renderer prior?
4. If useful, which fitted blocks contribute, and how do they interact with
   additional path, environment and sensor transformations?
5. Can useful fitting be accelerated without losing reconstruction quality,
   coverage or explicit uncertainty?

An experiment is complete when its protocol, artifacts and audit are complete,
including a negative result. Advancement and positive scientific claims depend
on the evidence; a failed hypothesis is not an implementation defect.

## CAST-4A broaden grouped coverage evaluation

**Question:** Does the frozen v15 method meet its existing descriptor criteria
across the admitted IDMT groups?

Start with the six existing conservative connected-group holdouts, using the
exact H1 seed-42 training selection of 190 observations/class in each fold.
Retain the five synthesis seeds and 50 generated observations/class/seed/arm
from the coverage protocol. This is a bounded six-fold comparison, not a new
width search. Fit and construct each fold's bank using its training IDs only;
recompute population centers, group effects, prototypes and feature scales
without the held group. Reuse a saved per-recording computation only if its
complete numerical settings and ancestry match the new training role.

Compare fixed v15 T=1.1 with v14 T=1, the matched prototype and the matched
independent-marginal controls. Match random streams and sample counts. Preserve
every failed fit, seed and fold, including boundary projections and alternative
fits. If a bank cannot be formed, record a failed fold rather than replacing
its examples or relaxing grouping.

Report each class and fold, the worst observed fold, equal-group summaries,
and the coverage/W1/spread trade-off. Assess the existing criteria explicitly:
80% mean marginal coverage, at least 2.5% W1 reduction against each matched
control, and family spread ratios in [0.5,2]. A claim that the criterion holds
across groups requires every evaluated group/class to pass on its fixed-seed
mean; pooled success alone is insufficient. Keep the original single-group
result unchanged regardless of this outcome.

Add a separately labeled whole-location holdout sensitivity analysis after a
metadata-only budget check. All dates from a location must stay together.
If 190/class is infeasible, freeze a new common budget for every sensitivity
arm and compare within that analysis. Do not borrow held-site examples. The
six groups cover only three locations; neither grouping scheme establishes
independent physical vehicle identities.

**Deliverable:** `reports/CAST_grouped_coverage.md`, fold manifests, fixed
controls, all metrics, failure records, runtime and verified replay commands.
Use paired whole-group uncertainty where estimable and show all group results;
synthesis-seed variation is a separate quantity. Previously inspected IDMT
groups remain development evidence even with correct per-fold fitting.

**Decision:** Passing supports broader source-domain coverage. Failure directs
one bounded revision at the dominant source-side error, or a documented decision
to test the unchanged method's utility as a limited falsification experiment.
It does not justify repeated tuning against the same outer group.

## CAST-4B test feature relationships and representation coverage

**Question:** What does the marginal score miss, and does its improvement align
with features used by the classifier?

Keep the CAST-4A banks fixed. Add one declared multivariate discrepancy:
maximum mean discrepancy (MMD) with a fixed Gaussian kernel on the combined,
training-scaled descriptor vector. MMD compares distributions through kernel
feature means; the method is described by Gretton et al., *A Kernel Two-Sample
Test*, §2, pp. 725–727 ([primary paper](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf)).
Its application here is a proposed diagnostic, not a published CAST result.
Freeze the estimator, family weighting and bandwidth rule using training data
before scoring held groups. Do not import independent-sample significance claims
into these correlated recording groups.

Verify the metric on synthetic controls with matched marginals but different
dependencies, collapsed samples and excessive spread. Also inspect class-conditional
covariance, per-family residuals and nearest-parent distances against a
group-disjoint real-to-real reference. These are diagnostics of dependence and
replay, not measures of physical realism or independent vehicle diversity.

With separately declared source-only feature access, repeat fixed distribution
diagnostics in frozen BEATs768 and MFCC26 representations. Use source-only caches;
do not open mixed-domain caches. Class separation and within-class coverage must
both be visible. A pooled real/synthetic match can hide class collapse. Keep
embeddings out of the fitter's objective in this comparison, so the check remains
an evaluation of the current method.

**Deliverable:** `reports/CAST_representation_coverage.md`, metric definitions,
sensitivity fixtures, fixed-sample comparisons and all class/group outcomes.

**Decision:** Retain marginal W1 for continuity; add the joint diagnostic without
retroactively redefining v15 success. No universal MMD pass threshold is assumed.
If wider marginal coverage worsens dependence or class separation, retain that
finding and target a specific mechanism. Lower feature distance alone is still
insufficient to claim classification utility. Conditional held-audio fitting,
if later needed to diagnose renderer capacity, gets a separate protocol and its
fits never enter a generative-coverage bank.

## CAST-5A test classification utility within IDMT

**Question:** Does CAST create useful training variation beyond the same real
parents and ordinary augmentation?

This is the first CAST classifier experiment and needs separate execution authorization
and a frozen protocol. Use six grouped IDMT folds and the five H1 training-selection
seeds [42,123,456,789,1024], with 190 unique real parents/class per fold/selection.
Rebuild all learned population statistics within each training role. Selection
seeds and synthesis seeds have distinct recorded purposes; neither creates new
independent recordings. Any new method or classifier selection stays inside
source-training groups, with those real-data costs counted.

Use frozen BEATs768 as primary and MFCC26 as comparator. Retain train-only
StandardScaler and L2 logistic regression with C=1, the H1 convergence settings,
and its fixed argmax rule. Do not choose an encoder or threshold by target results.

For the first comparison, propose a common **shape-only input profile**: H1's
`core8_2s` route followed by the same DC removal and unit-RMS normalization for
real, augmented and generated audio. Freeze the silence rule for every arm.
This aligns the classifier comparison with CAST's fitted shape model and excludes
absolute level from the question. It is a new preprocessing profile: recompute
all controls and do not substitute historical H1 scores. A later level-restoration
ablation must derive generated levels from source-training data and apply its
declared rule consistently, without target RMS matching.

| Arm | Classifier rows per class | Purpose |
|---|---:|---|
| R real | 190 | Matched real-only reference |
| RR repeated real | 380 from the same 190 parents | Controls row count and the effect of repetition at fixed regularization |
| RA real plus conventional augmentation | 190 + 190 | Practical augmentation comparator |
| RU real plus unfitted renderer | 190 + 190 | Same-renderer manual-prior comparator |
| RC real plus CAST | 190 + 190 | Primary candidate for incremental augmentation value |
| U unfitted renderer only | 190 | Manual-prior generated-data comparator |
| C CAST only | 190 | Utility of real-calibrated generated observations without original-waveform rows |

The proposed matrix has 420 fixed heads: six folds × five selections × seven
arms × two representations. Reuse identical generated/augmented feature banks
between single-origin and mixed arms. Measure source fitting and feature-extraction
cost before the full run; target scoring later reuses these heads without refitting.

Freeze a small conventional spectral-filtering/EQ and additive-noise recipe from
source-side evidence. Scalar gain alone disappears under the proposed shape
normalization and cannot serve as the augmentation comparator. Any background
asset needs its own original-recording ancestry and role. No augmentation
descendant crosses its parent's split. Define the unfitted prior's class information,
ranges and real-data access explicitly. A class-independent random prior is a
negative control; beating it cannot establish superiority over a meaningful
class-conditioned manual generator. Both U and CAST use the same renderer.

All arms share the same allowed real-data pool; report how much each actually
uses. Count recordings used to fit priors, set manual ranges, validate methods
or estimate preprocessing, even if their original waveforms never train the head.
“CAST only” means only generated task-training waveforms, not zero real-data access.
Record BEATs pretraining separately. Freeze synthetic counts and fitting budgets.

The primary contrast is **RC minus RA**. RC minus RR/R, RC minus RU and C minus U
are required supporting contrasts. Use the ABVID proposal's **3 macro-F1 point**
practical margin from the [shared experimental design](reports/proposed_experiments.md)
as the proposed utility target; it is unrelated to the historical
2.5% relative W1 criterion. A lower paired 95% interval above 3 points supports
that practical gain, an upper bound below 3 rules out that effect size, and a
crossing interval is inconclusive. Use zero separately for a claim of any gain.

Also report balanced accuracy, both recalls, confusion matrices, calibration,
group and worst-group outcomes, constant predictors and runtime. Predeclare
class-recall regression limits and any confirmatory multiplicity handling in
the executable protocol. An aggregate improvement with zero-recall cases does
not establish reliable recognition. Paired group/selection uncertainty must
retain overlapping-fold dependencies and disclose limited group support.

**Deliverable:** `reports/CAST_source_classification.md`, every scaler/head,
prediction, ancestry manifest, feature/checkpoint hash and verification receipt.

**Decision:** A gain beyond RA and RR justifies a frozen cross-dataset check.
If RA matches CAST, the fitter has not demonstrated added value. If coverage
improves but classification does not, record that mismatch as a substantive
result and inspect source-side class information before increasing complexity.

## CAST-5B test transfer to exposed development data

Freeze the selected procedure, all source-fitted heads, preprocessing, augmentation
and evaluation code before opening target features. Apply the unchanged heads and
matched arms to the admitted MELAUDIS development population. No target-fitted
normalization, threshold, prior width, example selection or checkpoint tuning is permitted
inside this source-only transfer experiment.

Retain RC minus RA as primary and report the same supporting contrasts and
3-point practical margin. Pair arms on the same target groups and training
selections. MELAUDIS is highly imbalanced and has only four conservative groups;
report per-class and per-group support, class-missing uncertainty draws, constant
predictors and conditional interval limitations. Retain H1's mean of per-model
metrics; averaging model parameters or ensembling fold-specific predictions would
be a different, separately declared evaluation.

Name the domain accurately: real-calibrated generated observations to real target
audio, or real plus real-calibrated generated observations to real target audio.
The [H1 result](reports/H1_baseline_results.md) already exposed MELAUDIS; a new
CAST freeze cannot turn it into untouched confirmation. If results motivate a
revision, retain the failed version and call the next test further development.
Any target-informed fitting requires a different adaptation protocol with
group-disjoint adaptation and evaluation roles.

**Deliverable:** `reports/CAST_transfer_development.md`, matched effects,
class/group failures, locked heads, full predictions, uncertainty and replay.

**Decision:** A development gain supports proceeding to independent confirmation
and controlled component attribution. A null or adverse result completes this
experiment honestly and limits the claim to reconstruction or coverage.

## CAST-5C obtain independent confirmation

Begin admission planning in parallel with the development stages. The concrete
missing input is an admitted manifest of **previously unexposed original recording
contexts containing the required classes**, with an auditable exposure history.
It cannot be manufactured by selecting new windows from an existing recording.
Previously inspected IDMT/MELAUDIS groups and historical protected recordings
are not available as fresh confirmation by relabeling their roles.

Choose the number of independent groups through a preregistered precision or
power calculation based on group variability, rather than a target clip count.
Collect multiple sites/devices when those shifts form part of the claim; retain
vehicle identity as unknown unless independently established. Admit technical
integrity, annotations, licensing and provenance before reserving the final role.

Freeze one primary CAST procedure and matched comparators before evaluation.
Separate model development from the confirmation lock and evaluate the declared
comparison once. Use the same practical-gain question and group-level uncertainty;
a wide interval remains inconclusive. A new-domain claim must identify the actual
domain sampled and cannot automatically become unseen-vehicle-model recognition.

**Deliverable:** `reports/CAST_confirmation.md` and a locked confirmation package,
or an explicit missing-data record if admission is incomplete. Development work
can finish without claiming confirmation.

## Controlled components after utility is understood

Use one new, bounded H2-linked protocol to compare fitted spectral texture,
harmonic structure, mixture and temporal envelope contributions. Change one
declared block at a time with matched donors, streams, renderer capacity and
budgets, and include a declared interaction where needed. Parameter coordinates
are effective observation controls; removing a block does not identify a physical
engine, tire, background or propagation mechanism. Compare v14 and v15 explicitly
if asking whether their coverage/W1 trade-off predicts utility.

For CAST plus propagation, retain a **no-additional-path** reference. Existing
recording coloration can otherwise be applied twice. Validate direct/reflected
relative delay, phase, gain, boundaries and normalization before a physical-path
comparison, using a new renderer version that preserves the historical H2 results.
Additional path/environment/sensor transformations initially test robustness of
observations. Clean-source recovery requires independently measured or controlled
source/path data; it cannot be inferred from successful classification.

Add SNR/background and sensor-response studies only with fixed source populations,
independent background-recording roles and known perturbations. Save component
waveforms and report source-by-transformation interactions. New renderer capacity
is justified by a persistent, identified source-side residual, not a target-score
sweep. Arrays, localization, open-set recognition and military model recognition
remain outside this CAST roadmap.

## CAST-6 accelerate a useful fitter

Profile the accepted fitting pipeline before introducing a learned estimator.
The original pilot took 813.51 seconds for 50 clips, with a median numerical fit
of 15.85 seconds/clip; these historical timings do not predict the current full
pipeline's cost. Measure fitting, sampling, feature extraction and audit separately.

If fitting is the relevant bottleneck, compare a small spectrogram-to-parameter
estimator, optionally followed by a fixed number of refinement steps, against
the original optimizer on group-disjoint data. Keep renderer, objective, input
profile and prior fixed initially. Compare runtime and memory at matched
reconstruction/coverage quality, including difficult and ambiguous fits.
Multiple parameter vectors can describe similar observations, so parameter MSE
against one arbitrary fitted winner is insufficient as the quality criterion.

**Deliverable:** `reports/CAST_amortized_fitting.md`, a quality/runtime curve,
failure analysis, frozen estimator and deterministic evaluation. Adopt it only
if the measured speed benefit meets a predeclared quality tolerance. The estimator
still requires an input recording; unconditional generation continues to require
an evaluated parameter distribution.

## Execution and preservation

Reusable work belongs in `src/abvid/cast`, using the existing representation,
learning, evaluation and provenance packages. New experiments are thin protocols
and configurations; no sibling-script imports or new parallel implementation tree.
Keep the [historical plan](experiments/cast_coverage/PLAN.md), locks, environments,
datasets, results and unsuccessful attempts unchanged.

Validate canonical JSON and resolve all storage through `abvid.paths`. The
current [configuration schema](experiments/CONFIGURATION.md) describes frozen
replay; extend and test it explicitly if a new execution mode is needed. Do not
rewrite a historical wrapper's hashes to accommodate new code. Audio, fitted
banks, embeddings, checkpoints and full runs stay outside the checkout. Respect
source licenses; local reconstruction artifacts are not automatically redistributable.

Each new stage must freeze exact input roles, hashes, all numerical settings,
seed purposes, sample budgets, metrics, practical margins, failures and stopping
rules before the relevant evaluation access. Save commit and dirty-state/source
snapshot, environment, checkpoint, resolved configuration, predictions or losses,
complete ancestry, runtime/memory and a replay audit. Record hardware and precision
with timings. Publish all declared outcomes, including negative ones.

For implementation changes, use focused synthetic numerical, determinism,
provenance and parity tests. Do not rerun a scientific experiment to validate a
documentation or packaging change. Historical replay commands and their original
environment requirements remain in [CAST replay instructions](experiments/cast_coverage/REPLAY.md).

The existing wrapper can be inspected without launching a study:

```sh
.venv/bin/python -m abvid.cli validate configs/experiments/cast_coverage.json --check-files
.venv/bin/python -m abvid.cli replay configs/experiments/cast_coverage.json
```

The second command prints the historical command; it does not execute it. Its
current help default is not a reproduction of the full v15 workflow.

## Immediate work order

1. Prepare the CAST-4A source-only protocol and metadata inventory, retaining
   fixed v15 and the existing acceptance criteria. Check whole-site feasibility
   before choosing its sensitivity-analysis budget.
2. Define and verify the CAST-4B joint metric on synthetic dependence/dispersion
   fixtures, then evaluate the same frozen banks after source-only authorization.
3. Freeze the CAST-5A matched arm table, common input profile, manual prior,
   conventional augmentation and statistical decisions. Resolve every executable
   configuration choice before requesting or performing the classifier experiment.
4. Plan admission of genuinely unexposed recording groups alongside this work.
   Defer cross-dataset scoring, component expansion and amortized fitting until
   their stated evidence and access conditions are satisfied.

The first useful scientific milestone is a paired answer to whether CAST's
broader acoustic coverage produces added classification value over conventional
augmentation. Independent confirmation and physical interpretation require their
own evidence beyond that result.

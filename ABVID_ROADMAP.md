> Repository layout updated 2026-10-05. Original experiment paths below refer
> to the preserved archive; see [structure](docs/STRUCTURE.md) and the current experiment registry.
> Scientific settings, dataset roles and completed results are unchanged.

# Passive Acoustic Vehicle Recognition --- Research Roadmap

**Focus:** deep learning, domain robustness, and sim-to-real transfer
for passive acoustic vehicle recognition. Military tracked/wheeled
literature motivates the physical problem; public civilian datasets
provide the first reproducible experimental testbed.

## 1. Core thesis

Acoustic vehicle classification itself is not the main unanswered
question. The harder problem is **generalization under domain shift**.
The observed signal changes with vehicle source state, speed, distance,
direction, propagation, ground, weather, background noise, microphone
response, and recording protocol.

Model the observation conceptually as:

`source -> propagation -> environment -> sensor -> representation -> classifier`

or

`x(t) = M[H(s(t; c, z_s); z_p) + n(t; z_e)]`

where `c` is vehicle class/identity, `z_s` is source state, `H` is
propagation, `z_p` is propagation/geometry, `n` is environmental sound,
and `M` is sensor response.

The project should determine which components cause the
synthetic-to-real gap and which forms of simulation/randomization
actually improve held-out real-world performance.

The [CAST research roadmap](CAST_ROADMAP.md) defines the connected branch for
effective-observation fitting, source coverage and a later separately frozen
classification-transfer test. It supplies the detailed path toward RQ5/E8;
its coverage results do not establish transfer or clean-source recovery.

## 2. Research questions

### RQ1 --- Synthetic-to-real

Can physically meaningful acoustic simulation generate enough
task-relevant diversity for deep audio representations to generalize to
real vehicle recordings?

### RQ2 --- Source vs propagation mismatch

Which contributes more to the sim-to-real gap: source-model mismatch,
propagation mismatch, environment mismatch, or sensor mismatch?

**Working hypothesis, not an assumption:** source-model mismatch may
dominate propagation-model mismatch.

### RQ3 --- Representation robustness

How much domain shift can be absorbed by representations such as
PCEN/log-Mel, learned CNN features, BEATs, AST, or domain-adapted
embeddings?

### RQ4 --- Useful domain randomization

Which nuisance variables measurably improve transfer when randomized:
source state, speed, Doppler, distance, geometry, ground, atmosphere,
background sound, overlapping sources, or microphone response?

### RQ5 --- Real -\> Sim -\> Real

Does estimating nuisance/source distributions from real recordings
outperform broad hand-designed domain randomization?

The objective is task-relevant distribution matching, not necessarily
perceptually perfect waveform synthesis.

## 3. Lessons carried over from the acoustic-UAS work

The UAS literature adds methodology that should become part of the
vehicle project.

-   **Low-SNR evaluation:** use SNR curves rather than only aggregate
    accuracy/F1.
-   **PCEN and domain-aware representations:** worth comparing against
    ordinary log-Mel features.
-   **Microphone heterogeneity:** sensor domain should be explicitly
    tested.
-   **Session leakage:** never split adjacent clips from the same
    continuous recording between train/test.
-   **Grouped evaluation:** prefer session/pass-by, temporal, location,
    sensor, and cross-dataset holdouts.
-   **Learned nuisance distributions:** compare manual randomization
    with distributions estimated/learned from real audio.
-   **Arrays are later work:** GCC-PHAT, SRP-PHAT, beamforming and
    tracking are relevant extensions, but localization should not
    contaminate the initial classification study.

EchoHawk is especially useful as an evaluation warning: session-grouped
splits can expose optimistic results caused by clip-level leakage.

## 4. Military acoustic anchor: BVP

The Bochum Verification Project / WTD 91 experiments are the key
historical anchor.

Altmann, Linev & Weiss (2002) report: - five tracked and five wheeled
military vehicles; - seven speeds; - both directions; - four lanes; -
microphone plus three-axis geophone sensing; - harmonic-based
vehicle-type recognition.

The underlying recordings are not known to be publicly downloadable.

**Access contact:** PD Dr. Juergen Altmann, TU Dortmund University ---
`juergen.altmann@tu-dortmund.de`.

Request raw/minimally processed microphone data plus vehicle type,
speed, lane/geometry, direction, sensor metadata, and reuse conditions.

The project must **not depend on BVP access**. If obtained, use it as an
external validation dataset.

When reading old military papers, distinguish: 1. tracked-vs-wheeled
recognition; 2. discrimination among particular physical test vehicles;
3. genuine model/type recognition across different examples and domains.

Do not treat (2) as proof of (3).

## 5. Seed literature and reading order

### Phase A --- Physical signatures / military acoustics

1.  **Altmann, Linev & Weiss (2002)** --- *Acoustic-seismic detection
    and classification of military vehicles---developing tools for
    disarmament and peace-keeping*\
    https://doi.org/10.1016/S0003-682X(02)00021-X

2.  **Altmann (2004)** --- *Acoustic and seismic signals of heavy
    military vehicles for co-operative verification*\
    https://doi.org/10.1016/j.jsv.2003.05.002

3.  **Wu & Mendel (2003)** --- *Classifier designs for binary
    classifications of ground vehicles*\
    https://doi.org/10.1117/12.484909

4.  **Wu & Mendel (2007)** --- *Classification of Battlefield Ground
    Vehicles Using Acoustic Features and Fuzzy Logic Rule-Based
    Classifiers*\
    https://doi.org/10.1109/TFUZZ.2006.889760

5.  **Wu & Mendel (2010)** --- *Classification of Battlefield Ground
    Vehicles Based on the Acoustic Emissions*\
    https://link.springer.com/book/10.1007/978-3-642-14084-6

6.  **William & Hoffman (2011)** --- *Classification of Military Ground
    Vehicles Using Time Domain Harmonics' Amplitudes*\
    https://digitalcommons.unl.edu/electricalengineeringfacpub/194/

7.  **Guo, Nixon & Damarla (2011)** --- *Improving acoustic vehicle
    classification by information fusion*\
    https://eprints.soton.ac.uk/272713/

**Output:** `reports/physical_signal_model.md`

### Phase B --- Invariance / representations

8.  **Goksu (2018)** --- *Engine Speed-Independent Acoustic Signature
    for Vehicles*\
    https://doi.org/10.1177/0020294018769080

9.  **Wieczorkowska et al. (2018)** --- *Spectral features for audio
    based vehicle and engine classification*\
    https://link.springer.com/article/10.1007/s10844-017-0459-2

10. **Becker et al. (2020)** --- *Audio Feature Extraction for Vehicle
    Engine Noise Classification*\
    https://rc.signalprocessingsociety.org/conferences/icassp-2020/spsicassp20vid0334

11. **Gong et al. (2021)** --- *AST: Audio Spectrogram Transformer*\
    https://arxiv.org/abs/2104.01778

12. **Chen et al. (2022)** --- *BEATs: Audio Pre-Training with Acoustic
    Tokenizers*\
    https://arxiv.org/abs/2212.09058

**Output:** `reports/representation_taxonomy.md`

### Phase C --- Domain shift / synthetic-to-real

13. **Gharib et al. (2018)** --- *Unsupervised Adversarial Domain
    Adaptation for Acoustic Scene Classification*\
    https://arxiv.org/abs/1808.05777

14. **Ronchini, Comanducci & Antonacci (2024)** --- *Synthetic Training
    Set Generation Using Text-to-Audio Models for Environmental Sound
    Classification*\
    https://arxiv.org/abs/2403.17864

15. **Khan, Ryzhikov & Kolehmainen (2026)** --- *AI4TEN:
    Synthetic-to-Real Transfer for Acoustic Vehicle Classification Using
    Physics-Based and AI-Generated Training Data*\
    https://doi.org/10.3390/app16147234

16. **Khan (2026)** --- *AI4TEN: Fine-Tuning Pre-Trained Audio
    Transformers (BEATs, AST) for Cross-Dataset Acoustic Vehicle
    Classification with Domain Adaptation*\
    https://doi.org/10.3390/acoustics8030063

17. **EchoHawk (2026)** --- *A Reproducible Acoustic Pipeline for Drone
    Detection, Classification, and Direction-Finding, with a Cautionary
    Study of Session-Level Data Leakage*\
    https://arxiv.org/abs/2606.29589

**Outputs:** `reports/domain_shift_taxonomy.md`,
`reports/research_gaps.md`, `reports/proposed_experiments.md`

## 6. Dataset strategy

Start with public civilian data so experiments are reproducible.

-   **IDMT-Traffic:** controlled real source domain; useful
    microphone/location variation.
-   **MELAUDIS:** harder real target domain; useful for messy
    environmental/domain shift.
-   **MAVD-Traffic:** audio + video; useful model for future
    camera-labeled roadside collection.
-   **DataSEC:** comparatively isolated acoustic events; useful
    source-audio material.
-   **AI4TEN resources:** use released synthetic/real data, code and
    models to reproduce the modern baseline.

Keep dataset identities separate. Do not make a giant pooled dataset
immediately. Dataset identity is itself an experimental variable.

BVP remains an access-request/external-validation track.

## 7. Codex literature setup

Use both PDF and Markdown:

-   PDF = authoritative source;
-   Markdown = search/analysis representation.

Suggested tree:

``` text
acoustic-sim2real/
├── AGENTS.md
├── literature/
│   ├── pdf/
│   ├── markdown/
│   ├── metadata/
│   ├── bibliography.bib
│   └── literature_matrix.csv
├── datasets/
├── src/
│   ├── data/
│   ├── simulation/
│   ├── features/
│   ├── models/
│   └── evaluation/
├── experiments/
├── reports/
└── RESEARCH_LOG.md
```

Codex rules: 1. read Markdown first; 2. PDF remains authoritative; 3.
inspect PDF for ambiguous tables/equations/figures/numbers; 4. record
paper + page/section provenance; 5. use `unknown` instead of guessing.

The literature matrix should track task, classes, dataset/public status,
sensor, environment,
source/speed/distance/direction/weather/noise/sensor variation,
representation, model, augmentation, simulation, adaptation, train/test
domain, metrics, cross-domain evaluation, limitations, code/data
availability, and source page/section.

## 8. Experimental rules

Before novel work, reproduce baselines.

Hard requirements: - no session/pass-by leakage; - group continuous
recordings; - keep final target domains untouched; - no hyperparameter
tuning against the final cross-domain test set; - multiple seeds; -
balanced F1 + per-class metrics; - confidence intervals where
feasible; - SNR-dependent performance; - explicit
sensor/location/session composition; - versioned experiment configs and
manifests.

## 9. Experimental ladder

**Progress — 2026-10-04:** qualified civilian data admission and minimum
H1 are complete: E0, E1, released-bank E2 and the MFCC comparison in E3.
The [R0 checkpoint audit](reports/published_reproduction_status.md)
retains unresolved items; it does not establish full training reproduction.
See [H1 results](reports/H1_baseline_results.md) and the current
[experimental proposal](reports/proposed_experiments.md). The core task
remains car/truck; motorcycle is excluded. AST/new CNN comparisons remain
unrun extensions. The minimum E4–E5 four-cell H2 comparison is now
[complete and verified](reports/H2_source_path_results.md): 9,600 observations
and 40 fixed heads. BEATs source-minus-path D is −34.90 F1 points
(conditional 95% CI −43.56 to −29.17), contradicting source dominance for
these specified interventions. All cells retain zero-recall group/class cases;
this is exposed-development evidence, not independent confirmation. The first
[ground/air and collapse follow-up](reports/H2_ground_air_collapse.md) is also
verified: neither branch alone recovers the gain; the BEATs interaction is +31.80
F1 points, and gain matching does not resolve the direct-path truck bias. The
air branch also changes relative FIR latency. The [completed attenuation-versus-latency
control](reports/H2_air_latency_control.md) (2026-10-05) finds 35.43% BEATs F1
with latency only, 35.46% with both, and 3.68% with attenuation only. The prior
gain is therefore an implemented ground/filter-delay effect, not isolated
physical-absorption evidence. E6–E9 remain proposed.

The [completed IDMT-only diagnostics](reports/H1_source_diagnostics.md)
found modest regularization/bandwidth benefits, little gain from increasing
190 to 359 examples/class, a truck-recall cost from F1-selected thresholds,
and substantial whole-location holdout losses. That pass did not evaluate
the target. The subsequent [regularization transfer check](reports/H1_regularization_transfer.md)
is also complete: BEATs improves target balanced accuracy and truck recall,
but its +0.63-point F1 interval crosses zero; worst-group failures persist.
H1 is unchanged, and MELAUDIS remains exposed development data.
Location mixes source and recording conditions, so those diagnostics alone did
not isolate source-versus-propagation effects. The later H2 result above tests
the declared interventions, not an intrinsic division of all domain mismatch.
The earlier military program is retired from the active checkout; its
historical results and protected-source roles remain unchanged under
[AGENTS.md](AGENTS.md).

**CAST progress — 2026-10-06:** the [v15 coverage study](reports/CAST_improvement.md)
met its declared descriptor-level development objective, with 82.019% car and
81.286% truck mean marginal coverage on one repeatedly exposed IDMT group.
Classification transfer remains untested. Follow the [CAST roadmap](CAST_ROADMAP.md)
for broader grouped coverage checks, feature-dependence diagnostics and the
proposed matched classification experiment; new transfer access requires its own
authorization and frozen protocol.

### E0 --- Within-domain sanity check

`IDMT_train -> IDMT_test`

Verify preprocessing, labels and model pipeline.

### E1 --- Real-to-real domain gap

`IDMT -> MELAUDIS`

Measure:

`Delta_domain = performance_within - performance_cross_domain`

### E2 --- Reproduce AI4TEN

Reproduce real-only, physics-synthetic-only, generated-synthetic-only,
real+physics, and real+generated baselines before modifying the
simulator.

### E3 --- Pretrained encoder baseline

Compare a simple CNN/classifier with frozen BEATs and AST. Cache
embeddings.

### P0: Offline classification prototype

**Placement:** package the completed minimum H1 baseline immediately after
E0–E3, alongside continued H2 and CAST research. The unrun AST/CNN extensions
and a positive synthetic-data result are not prerequisites for this delivery.
P0 is a packaging milestone, not a new scientific experiment.

**Current status — 2026-10-06:** a local baseline CLI and model bundle package
the existing H1 real-data BEATs arm, with parity checks against its archived
predictions. The implementation and local report are separate from this roadmap
update. An inspectable replay interface remains a proposed follow-up.

The first prototype accepts a supplied vehicle event and returns car/truck
classification using the frozen H1 preprocessing, BEATs encoder and a declared
linear head. Preserve the original grouped evaluation, model identity, input
hashes and known failure cases; do not select a demonstration head by performance.
Motorcycles remain excluded. This closed-set prototype does not establish
vehicle-presence detection, calibrated confidence or independent generalization.

The next delivery checkpoint is an offline replay interface with audio playback,
waveform/spectrogram, the analyzed interval, predictions and model provenance.
Include known successes and failures. Longer-recording predictions over time need
declared window, hop and timestamp rules; the centered-event benchmark does not
validate those additional behaviors. Keep large models and audio outside the
checkout, and verify packaging against the fixed baseline without new tuning.

CAST belongs in the training-data pipeline and is not a runtime dependency of
P0. A CAST-trained prototype version requires the separately frozen, authorized
classification-utility comparison in the [CAST roadmap](CAST_ROADMAP.md).
Live vehicle detection and deployment are later milestones requiring a new task
protocol, vehicle-absent examples, causal inference and deployment evaluation.

### E4 --- Source-model ablation

Hold propagation fixed and compare: - simple analytic source; -
generated source; - recorded real source; - learned source model, if
justified.

Critical comparison:

`recorded source -> same H -> classifier`

versus

`synthetic source -> same H -> classifier`

This compares source assets under the specified path. Recorded assets may
retain their original propagation, environment and sensor imprint, so it
does not automatically isolate source physics.

### E5 --- Propagation ablation

With source fixed, progressively vary: - spreading/distance; -
Doppler; - direction/geometry; - ground; - atmospheric effects.

Measure paired changes on the same declared development target. Reserve
untouched confirmation for a fully frozen comparison; the already exposed
MELAUDIS material cannot serve as untouched confirmation.

### E6 --- Environment / SNR

Add controlled real/synthetic background sound, SNR randomization and
overlapping confusers. Plot F1/detection performance versus SNR.

### E7 --- Sensor domain

Randomize/model microphone response, gain, bandwidth and noise floor;
perform held-out-microphone tests where possible.

### E8 --- Real -\> Sim -\> Real

Compare hand-selected randomization ranges with distributions estimated
or learned from real recordings.

### E9 --- Domain adaptation

Only after understanding the physical factors, add DANN/ArcFace or
comparable domain-adaptation methods and test whether they complement
simulation.

## 10. Minimum publishable scientific story

A useful paper is not "we trained another vehicle classifier."

A stronger story is:

1.  quantify real-to-real and synthetic-to-real domain gaps;
2.  reproduce an existing synthetic-data baseline;
3.  decompose the simulator into source, propagation, environment and
    sensor components;
4.  ablate those components under strict grouped/cross-dataset
    evaluation;
5.  identify which mismatches dominate;
6.  test whether learned/recorded source models + physics propagation
    outperform simpler synthetic pipelines;
7.  test whether pretrained/domain-adapted representations complement
    simulation.

A negative result is still useful if it establishes that a physically
elaborate component does **not** improve transfer.

## 11. Compute plan

### CPU/local first

Use local hardware for: - literature processing; - dataset indexing; -
preprocessing; - dataset statistics; - most physics-simulation
development; - plotting and analysis.

### GPU when E0/E1 are ready

Use RunPod or equivalent for: - BEATs/AST embedding extraction; -
fine-tuning; - AudioLDM generation; - large ablation sweeps.

A 16--24 GB GPU is enough for much of the initial work. Cache frozen
encoder embeddings and generated audio rather than recomputing them.

### Jetson later

Use the Jetson Orin Nano for a later edge-inference study: latency,
model compression and deployment constraints---not as the primary
training machine.

Every experiment should log GPU/device, precision, peak VRAM, wall time,
throughput and seeds.

## 12. Research log

For every run, append to `RESEARCH_LOG.md`:

``` text
Experiment:
Date:
Hypothesis:
Config:
Dataset split:
Seed(s):
Result:
Interpretation:
Failure modes:
Decision:
Next experiment:
```

Avoid directories full of `final_v3_REAL` nonsense.

## 13. Immediate next actions

The [P0 delivery track](#p0-offline-classification-prototype) can proceed alongside
the research actions below: preserve the local baseline CLI/bundle and its parity
record, then scope the inspectable offline replay interface. CAST classification
gains and further simulator development are not prerequisites for that interface.

1.  Preserve the completed R0/H1, source-only diagnostic and regularization
    transfer results as versioned references; retain MFCC alongside BEATs.
2.  Close the bounded regularization check without automatically replacing
    H1 or starting another target-driven C/threshold sweep. Carry its partial
    benefit and unresolved worst-group failures into the physical comparison.
3.  Keep bandwidth and training budgets matched across comparison arms.
    Report class recall, balanced accuracy, ranking and worst-group results
    alongside macro-F1; a threshold-only F1 gain is not better representation
    learning. The larger source-only budget does not replace H1's 190/class.
4.  Use the [frozen H2 source × propagation design](experiments/h2_simulation/REPLAY.md):
    four paired cells, 190 training events/class, five seeds, fixed C=1 and
    frozen BEATs/MFCC. Source, geometry and target-role manifests are checked;
    no IDMT validation-label selection enters the synthetic arms.
5.  Preserve the [completed H2 result](reports/H2_source_path_results.md) and
    [independent audit](reports/ARTIFACT_INDEX.md#artifact-70e853936c31).
    The repaired renderer passed full replay, and all 40 heads were locked before
    scoring. The declared path intervention helped BEATs more than the declared
    source intervention; it did not establish reliable recognition or real-road
    calibration. Preserve the [completed ground/air follow-up](reports/H2_ground_air_collapse.md):
    its benefit is joint, and scalar gain alone does not explain the collapse.
    The [attenuation/latency control](reports/H2_air_latency_control.md) is complete:
    latency reproduces the gain; attenuation adds only 0.03 F1 points with latency
    present. Preserve all results. A later physical-path comparison must first
    declare and validate relative filter delay in a new renderer version;
    the broad physical source-versus-propagation hypothesis remains open.
    Independent confirmation still requires
    verified unexposed recordings; E6–E9 remain later conditional extensions.
6.  Favor independent recording contexts over more excerpts from the same
    groups. Keep BVP access optional; do not make it a project dependency.
7.  Advance the [CAST branch](CAST_ROADMAP.md#immediate-work-order) through fixed
    grouped coverage and joint-feature diagnostics before a separately frozen
    utility comparison. Preserve v15 as the reference and compare CAST against
    ordinary augmentation and matched real-data budgets, without treating
    descriptor coverage as a classification result.

## 14. Decision gate

This gate governs further scientific investment in simulation; it does not block
packaging the completed baseline as the P0 offline prototype.

After E0--E3, stop and ask:

-   Is the cross-domain gap large and reproducible?
-   Does synthetic augmentation measurably help?
-   Does the result survive grouped/session-safe evaluation?
-   Do pretrained representations materially change the gap?
-   Is there evidence that source quality is a bottleneck?

If **yes**, proceed to controlled simulator ablations.

If **no**, revise the research question before spending time building a
sophisticated simulator.

------------------------------------------------------------------------

## Working north star

> **Determine which aspects of an acoustic vehicle simulator must be
> realistic for synthetic training data to improve generalization to
> genuinely held-out real recordings.**

Everything else---military historical data, foundation models, PCEN,
domain adaptation, arrays, edge deployment---is supporting machinery
around that question.

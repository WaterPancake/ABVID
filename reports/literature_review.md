# Passive acoustic vehicle recognition: structured literature review

Reviewed 2026-09-27. This is a literature analysis; no models were implemented, trained or evaluated.

## Answer to the research question

**Physically meaningful simulation is supported as a way to supply useful variation, but the collection does not establish that it supplies enough diversity for deep representations to generalize reliably to real vehicles.** The strongest direct simulation study reports gains from combining real, procedural and generated audio, while substantial cross-dataset error remains. The transformer study demonstrates useful real-to-real transfer, but never tests a physics-simulated training corpus with those transformers. Combining the two findings into a demonstrated physics-to-BEATs transfer result would exceed the evidence. [Review synthesis: P02, §§2.2–2.5, 3.1–3.3; P01, §§2–4]

The hypothesis that **source-model mismatch dominates propagation-model mismatch remains unresolved**. The procedural sources have acknowledged missing acoustic detail; this motivates improving source models. However, no reviewed experiment independently crosses source fidelity and propagation fidelity on a common real test set. Terrain, viewpoint, device and annotation effects also have direct evidence. [P02, §3.3; P04, §5.3; P13, chapter 3, §2.3; P09, §V; P16, §§3–4]

## Scope and method

The requested `literature/papers/` path was absent. The supplied collection is under `ABVID Lit Review/`, with **16 matching Markdown/PDF pairs**. All are represented in [literature_matrix.csv](../literature/literature_matrix.csv). The book is one supplied document; its acoustic vehicle chapter (Wu–Mendel, chapter 3, printed pp.55–77; PDF pp.59–81) is the analysis unit. Its unrelated chapters are outside the vehicle-acoustics question. The optional standalone AST paper was not present; AST use is examined through P01, without claiming a separate review of Gong et al.

The prescribed core reading order was followed. Phase 1 produced [physical_signal_model.md](physical_signal_model.md) before Phase 2; Phase 2 produced [representation_taxonomy.md](representation_taxonomy.md) before Phase 3. The additional P08/P10 classifier papers were read after the Phase 2 core papers, and P06 AudioLDM after the Phase 3 core papers. Phase 3 produced [domain_shift_taxonomy.md](domain_shift_taxonomy.md) before this cross-phase synthesis.

Markdown was read first. The original PDFs resolved ambiguous tables, equations and figures. Citations such as `[P09, §V, Table V]` identify the supplied paper and its section/table; `PDF p.` is one-based, not a zero-based Markdown image number. **Author** means a reported method, result or interpretation. **Review** means our evidence assessment or proposal. Unknown means not established; it does not mean absent in reality.

The CSV has one row per supplied document and 37 columns. Bibliographic identifiers and paths resolve through `metadata_reference`; all substantive extraction cells include their paper ID and section/page reference. Physical-variable cells specify **controlled**, **randomized**, **measured**, explicitly **ignored/excluded**, or **unknown** where justified. “Varied” is retained when variation is described without a randomization scheme. An invariance objective and its empirical validation are recorded separately. No unreported variable is silently classified as ignored. Multiple evaluation columns can apply to the same paper because some papers run materially different protocols.

## What constitutes a class?

| Evidence level | What the label means | Examples and boundaries |
|---|---|---|
| Broad vehicle category | Mobility, size/function, or ordinary traffic category shared by many vehicles. | Four heavy/light tracked/wheeled categories in P13/P08; three binary category problems in P10; civilian road categories in P15; car/truck/motorcycle in P02; plus bus/background in P01. [P13, chapter 3, §3; P08, §§III–V; P10, §1; P15, §§2,5–6; P02/P01, §2.1] |
| Engine attribute | A property such as fuel type rather than an exact vehicle identity. | Diesel/petrol in P03; diesel/gasoline and car/van dyno tasks in P15. These do not identify an engine model. [P03, §4.3; P15, §5.3] |
| Individual recorded vehicle or nominal type label | Named cars or anonymous vehicle kinds represented by recorded exemplars. | P09 explicitly has one physical vehicle per class; P11 records one of each named car. P05's labels are conditional on recoverable engine signatures. P12's physical exemplars per nominal type are unknown. [P09, §V footnote 5; P11, §II; P05, §§4.3–5; P12, §6] |
| Genuine model/type recognition | Same model/type recognized across independent physical exemplars, with acquisition/source groups held out. | **Not established in this collection's vehicle experiments.** This is our evidential criterion, not a redefinition of the authors' labels. [Review assessment: P05, §4.4; P09, §V; P11, §II; P12, §6] |
| General sound/scene semantics | Event, scene, word or emotion labels; sometimes vehicle-related words occur. | P07 benchmarks audio events and speech tasks; P16 classifies scenes; P14 classifies environmental events. P06 is audio generation, and P04 is physical characterization. Their results are relevant support, not vehicle-model recognition scores. [P07, §4.1; P16, §2; P14, §2.2; P06, §5; P04, §§2–7] |

## Cross-phase findings

**The signature is state-dependent.** Engine-cycle harmonics and their relative amplitudes, exhaust configuration, running gear and broadband components all offer potential discriminative information. RPM and translation speed are distinct, coupled variables; an invariant harmonic frequency index does not remove changes in harmonic balance. Ground affects both excitation and propagation. These facts argue for a structured source/path/environment/sensor model, with interactions, rather than a single generic corruption variable. [P04, §§1.2,5–6; P05, §§3.4,5; P13, chapter 3, §2.3]

**Representations make different tradeoffs.** Harmonic features are interpretable but omit much broadband information. Guo's selected bins add complementary content while retaining speed/road sensitivity. TDHA retains event energy and strongest frequency; wavelet entropy is empirically stable only over the sampled sweeps. Mod-PCEN represents modulation after adaptive normalization, and BEATs learns contextual semantic features. None of these transformations provides a universal physical-factor invariance guarantee. [P12, §§3–4,7; P09, §§III–IV; P11, §§II–V; P03, §3; P07, §§3,4.5]

**Evaluation independence changes the claim.** Same-pass vectors, random half-second events, separate runs, paired sensors and new datasets are different tests. In P09, random-event accuracy exceeds separate-run/station accuracy. P10's sponsor-scored blind runs are useful independent scoring evidence, but their conditions are insufficiently specified for a cross-domain label. In P13/P08/P10, the development “testing” values also influence optimization. These are review concerns; they are kept separate from author-acknowledged limitations. [P09, §V; P10, §§3–4; P08, §VI; P13, chapter 3, §5]

**Synthetic quantity, fidelity and coverage are different variables.** P02's main procedural/AudioLDM comparison has a 25-fold synthetic sample-volume difference. Its matched-volume experiment narrows that distinction, and its spectral analysis finds closer AudioLDM averages without clearly superior classification. P14 likewise finds augmentation useful but synthetic replacement limited, and increasing volume does not ensure monotonically better performance. A few spectral means or a t-SNE plot cannot establish support over the class-conditional real distribution. [P02, Tables 4–6, §3.3; P14, §§3–4; review interpretation]

**Pretraining and adaptation address only part of the gap.** P01's pretrained models outperform its small CNN on a real cross-dataset problem, but pretraining data, capacity and representation all change. BEATs iter3+ includes supervised-teacher information according to P07, so its advantage over AST cannot isolate self-supervision. DANN reduces P01 cross-dataset F1 despite access to unlabelled target audio; ArcFace improves the reported score without target audio. P16 provides positive device-adaptation evidence using a different adversarial design and task. [P01, §§2.2–2.4,3.1,3.4; P07, §4.3; P16, §§2–4]

## Quantitative anchors, with the test domain attached

These are **author-reported results**, checked against the indicated PDF tables. They are not new replications or a common leaderboard.

| Paper and domain | Comparison | Outcome | Constraint on interpretation |
|---|---|---|---|
| P09, real → real BVP | Random events / separate runs / separate station | 90.38% / 82.77% / 82.10% event accuracy | Same nine physical vehicles; voting results are separate. [Tables IV–V, PDF pp.9–10] |
| P12, real → real ARL, separate runs | Harmonic / modified decision fusion | 73.44% / 84.24% mean accuracy | New physical exemplars per nominal type unknown. [Table 2, PDF p.12] |
| P16, real A → real B/C scenes | Before / after adaptation, Kaggle CNN | 20.28% / 31.67% target accuracy | Unlabelled target-device training audio available. [Table 1, PDF p.4] |
| P14, real or generated → real US8K | Real / best listed synthetic-only / best listed augmentation, CNN | 64.68% / 46.04% / 69.64% accuracy | Different generator configurations; folds wording ambiguous; environmental classes. [Tables 1–2, PDF pp.2–3] |
| P02, real/synthetic/mixed → real DATASEC+MAVD | Real / procedural / AudioLDM / both synthetic / real + both | 0.25 / 0.19 / 0.22 / 0.35 / 0.39 macro-F1 on balanced subsample | Unequal main training volumes; real-fitted scaler for all arms. [Table 4, PDF p.8] |
| P02, matched 200/class → same real target | Procedural / AudioLDM, alone; with real | 0.18 / 0.20; 0.29 / 0.24 | Unbalanced real training in this comparison; no source × propagation factorial. [Table 5, PDF p.11] |
| P01, real IDMT → real MELAUDIS | CNN / AST / BEATs / DANN / ArcFace | 0.26 / 0.43 / 0.46 / 0.31 / 0.50 balanced-subsample macro-F1 | Transformers trained once; DANN target-audio access; frozen adaptation versus full fine-tuning. [Table 3, PDF p.10; §§2.3–2.4] |

P01 and P02 differ in label count, data, duration, preprocessing and evaluation sampling. Their scores cannot estimate a common `Δ_sim2real`. P02's within-KVSD validation accuracy also cannot be subtracted from target macro-F1 to estimate `Δ_domain`. The roadmap specifies matched measurements instead. [Review interpretation: P01, §2; P02, §§2,3.1]

## Evidence for the proposed hybrid architecture

```text
recorded / learned source
        -> physics-based propagation
        -> randomized environment
        -> randomized sensor response
        -> pretrained audio representation
        -> classifier / optional adaptation
```

| Component | Existing support | What remains new experimental work |
|---|---|---|
| Recorded/learned source | Real recordings contain class information; generated audio offers semantically relevant textures. [P09, §V; P14, §3; P06, §§3–5] | Isolate a source from its original channel/background, verify physical state control, and test source diversity at fixed propagation. A roadside recording is already an observation; rendering it again risks applying propagation twice. [Review inference: P04, §§1.2,5; P06, Appendix I] |
| Physical propagation | Measured range/direction/ground effects; implemented moving-source propagation in P02. [P04, §§3–6; P02, §2.2.1] | Quantify recognition benefit of each path mechanism using an unchanged source bank and common target. |
| Random environment | P02 randomizes SNR; general synthetic augmentation has transfer benefits. [P02, §§2.2,3.1; P14, §3] | Independent background-source holdout, plausible weather/ground variation, and factorial tests of source–environment coupling. |
| Random sensor response | Device mismatch is demonstrably consequential and can be adapted. [P16, §§3–4; P01, §3.3] | Verify calibrated response/noise/codec interventions on paired sensors from events excluded from training. No reviewed paper validates the complete randomized-sensor stage. |
| Pretrained representation | BEATs general-audio results and P01 real cross-dataset improvements. [P07, §4; P01, §3.1] | Test the same frozen/pretrained checkpoint on procedural or hybrid training, disentangling representation benefit from training-set changes. |
| Classifier/adaptation | Feature fusion, source-label-preserving adversarial adaptation and metric learning each have conditional support. [P12, §§5–6; P16, §§2–4; P01, §3.4] | Matched head/backbone and target-data budgets; protect class information while aligning domains; quantify negative transfer. |

The connected end-to-end hybrid is therefore a **research proposal**, not an architecture validated by this collection. The relevant problem chains appear in [research_gaps.md](research_gaps.md), followed by five dependency-ranked hypotheses in [proposed_experiments.md](proposed_experiments.md).

## Availability and reproducibility boundaries

Older BVP/ACIDS collections, Göksu's cars and the Wieczorkowska recordings have **unknown public raw-data availability from their papers**. No availability was inferred from an open-access article or a named institutional collection. P03 provides a public dataset/code repository, whose README and dataset listing were accessible during this review. [P05, §2; P13, chapter 3, §2.1; P11, §II; P15, §2; P03, §1 footnote; [Becker repository](https://github.com/LucaBeckerICA/VehicleEngineNoiseClassification)]

IDMT's primary release is publicly listed, including a downloadable archive; MELAUDIS has a public Figshare listing and primary data descriptor. The descriptor documents filenames with date/time, street, traffic status, vehicle multiplicity, lane/direction and device clues, useful for grouping. These release records are **external availability/provenance checks**, not additions to the 16-paper matrix. [IDMT release](https://zenodo.org/records/7551553), [MELAUDIS release](https://figshare.com/articles/dataset/_b_MELAUDIS_The_First_Acoustic_ITS_Dataset_in_Urban_Environment_b_/27115870), [MELAUDIS Data Features section](https://www.nature.com/articles/s41597-025-04689-3)

P01/P02 declare public Zenodo archives, but their DOI and record URLs could not be retrieved through the browsing tool. Their current accessibility is **unknown**, not “unavailable.” Large audio archives and model checkpoints were not downloaded or executed. Exact replication therefore remains a future artifact audit, and the roadmap includes a public-data fallback rather than depending on private military recordings. [P01/P02, Data Availability Statements]

PDF corrections and unresolved source inconsistencies are documented in the three phase reports and each CSV row. Material examples include lost micro-units, an inserted unrelated equation paragraph, malformed table columns, a balanced-test caption inconsistent with figure counts, and checkpoint-supervision/target-access qualifications. Corrections were made to the analysis, not to the supplied source files.

Final structural validation confirmed one-to-one coverage of the actual 16 PDFs and 16 Markdown representations, 37 CSV columns with 464 cited substantive extraction cells, all six requested reports, all twelve required specification fields in each of five experiments, and 47 working local file links. These checks establish artifact completeness and traceability, not independent replication of the papers' experimental results.

## Source inventory

| ID | Citation / document title | Reading phase | Local sources |
|---|---|---|---|
| P05 | Altmann; Linev; Weiß (2002). Acoustic–seismic detection and classification of military vehicles—developing tools for disarmament and peace-keeping. | 1 | [Markdown](ARTIFACT_INDEX.md#artifact-6d2ff2e60a2b) · [PDF](ARTIFACT_INDEX.md#artifact-261348853b34) |
| P04 | Altmann (2004). Acoustic and seismic signals of heavy military vehicles for co-operative verification. | 1 | [Markdown](ARTIFACT_INDEX.md#artifact-92685c00cb11) · [PDF](ARTIFACT_INDEX.md#artifact-5c54fe090b9c) |
| P13 | Wu; Mendel (chapter 3) (2010). Innovations in Defence Support Systems – 1 — chapter 3: Classification of Battlefield Ground Vehicles Based on the Acoustic Emissions. | 1 | [Markdown](ARTIFACT_INDEX.md#artifact-1e2450832279) · [PDF](ARTIFACT_INDEX.md#artifact-7016555030d9) |
| P11 | Göksu (2018). Engine Speed–Independent Acoustic Signature for Vehicles. | 1 | [Markdown](ARTIFACT_INDEX.md#artifact-d62cdfbe8edf) · [PDF](ARTIFACT_INDEX.md#artifact-3900059517f4) |
| P12 | Guo; Nixon; Damarla (2011). Improving acoustic vehicle classification by information fusion. | 2 | [Markdown](ARTIFACT_INDEX.md#artifact-adf9600f854f) · [PDF](ARTIFACT_INDEX.md#artifact-bd4aa2dcc5da) |
| P09 | William; Hoffman (2011). Classification of Military Ground Vehicles Using Time Domain Harmonics’ Amplitudes. | 2 | [Markdown](ARTIFACT_INDEX.md#artifact-dc4b3fab561a) · [PDF](ARTIFACT_INDEX.md#artifact-527ccf3b5720) |
| P15 | Wieczorkowska; Kubera; Słowik; Skrzypiec (2018). Spectral features for audio based vehicle and engine classification. | 2 | [Markdown](ARTIFACT_INDEX.md#artifact-6c2d7bdf7ba1) · [PDF](ARTIFACT_INDEX.md#artifact-d1793f0911de) |
| P03 | Becker; Nelus; Gauer; Rudolph; Martin (2020). AUDIO FEATURE EXTRACTION FOR VEHICLE ENGINE NOISE CLASSIFICATION. | 2 | [Markdown](ARTIFACT_INDEX.md#artifact-6f3fc6ee09c2) · [PDF](ARTIFACT_INDEX.md#artifact-b903542a71ba) |
| P07 | Chen et al. (2022). BEATs: Audio Pre-Training with Acoustic Tokenizers. | 2 | [Markdown](ARTIFACT_INDEX.md#artifact-62d230be5fb3) · [PDF](ARTIFACT_INDEX.md#artifact-41b932595cb5) |
| P08 | Wu; Mendel (2007). Classification of Battlefield Ground Vehicles Using Acoustic Features and Fuzzy Logic Rule-Based Classifiers. | 2 supporting | [Markdown](ARTIFACT_INDEX.md#artifact-4cdb68133719) · [PDF](ARTIFACT_INDEX.md#artifact-21c90d31f605) |
| P10 | Wu; Mendel (2003). Classifier designs for binary classifications of ground vehicles. | 2 supporting | [Markdown](ARTIFACT_INDEX.md#artifact-85784501609e) · [PDF](ARTIFACT_INDEX.md#artifact-e1087aa6fdf6) |
| P16 | Gharib; Drossos; Çakır; Serdyuk; Virtanen (2018). Unsupervised adversarial domain adaptation for acoustic scene classification. | 3 | [Markdown](ARTIFACT_INDEX.md#artifact-e92f52a563a9) · [PDF](ARTIFACT_INDEX.md#artifact-16c92eb3f34a) |
| P14 | Ronchini; Comanducci; Antonacci (2024). SYNTHETIC TRAINING SET GENERATION USING TEXT-TO-AUDIO MODELS FOR ENVIRONMENTAL SOUND CLASSIFICATION. | 3 | [Markdown](ARTIFACT_INDEX.md#artifact-58cbf5c98533) · [PDF](ARTIFACT_INDEX.md#artifact-094b08d78e57) |
| P02 | Khan; Ryzhikov; Kolehmainen (2026). AI4TEN: Synthetic-to-Real Transfer for Acoustic Vehicle Classification Using Physics-Based and AI-Generated Training Data. | 3 | [Markdown](ARTIFACT_INDEX.md#artifact-22791dac9303) · [PDF](ARTIFACT_INDEX.md#artifact-8cc165b71310) |
| P01 | Khan (2026). AI4TEN: Fine-Tuning Pre-Trained Audio Transformers (BEATs, AST) for Cross-Dataset Acoustic Vehicle Classification with Domain Adaptation. | 3 | [Markdown](ARTIFACT_INDEX.md#artifact-e31435da0c8b) · [PDF](ARTIFACT_INDEX.md#artifact-3c4a61aa71d3) |
| P06 | Liu et al. (2023). AudioLDM: Text-to-Audio Generation with Latent Diffusion Models. | 3 supporting | [Markdown](ARTIFACT_INDEX.md#artifact-44f5ad23ea70) · [PDF](ARTIFACT_INDEX.md#artifact-1f6a15ff20c3) |

# Recurring research gaps

This synthesis follows all three reading phases. Each chain below is a **review interpretation** of cited author evidence, not a claim that the papers experimentally measured every causal link. The endpoint is a question that the [sequential experiments](proposed_experiments.md) can address. IDs resolve through [the source inventory](literature_review.md#source-inventory).

## 1. Vehicle identity is entangled with operating state and exemplar

**Physical cause:** engine/exhaust configuration, RPM, load and gearing affect sound; vehicles of different models can share engines. **Signal effect:** harmonic positions and relative strengths change, and broadband contributions can change with speed. **Representation effect:** a classifier can identify the particular recorded engine/state, or its acquisition context, while appearing to identify a vehicle model. **Domain shift:** a new physical vehicle, engine state or fleet composition changes the learned class-conditional distribution. [P04, §§5.1–5.3; P05, §§3.4,5]

**Existing mitigation:** harmonic-order normalization, repeated RPM sweeps, uncertainty modelling and held-out runs. **Remaining gap:** these do not establish within-model variability across independent physical exemplars. P09 explicitly has one exemplar per class; P11 has one of each named car; P12's exemplar count is unknown. The required new evidence is a hierarchy of held-out event, session, physical vehicle and model evaluations. Public civilian category datasets can test category generalization first; they cannot supply missing make/model/exemplar labels by inference. [P09, §V footnote 5; P11, §II; P12, §6; P13, chapter 3, §5]

## 2. Synthetic variation may cover paths while undersampling sources

**Physical cause:** many speed/range combinations can be generated from one narrow engine template per class. **Signal effect:** stationary harmonic patterns recur, while source jitter, amplitude modulation, turbulence and configuration variation are missing. **Representation effect:** a learner may rely on generator-specific regularities instead of the diversity within real classes. **Domain shift:** real vehicles fall outside the source distribution even when the renderer contains physically valid Doppler, attenuation and reflection. [P02, §§2.2.1,3.3]

**Existing mitigation:** more simulations, text-to-audio textures, mixtures of synthetic sources and real-data anchors. **Remaining gap:** neither volume sweeps nor averaged spectral similarity isolate source diversity from propagation diversity, bandwidth, balancing or training count. P02's matched-volume results limit the claim that realism alone wins; P14's gains also saturate or vary by generator/class. A source × propagation factorial is needed, with environmental/sensor factors fixed initially. This motivates the source-dominance hypothesis, but does not confirm it. [P02, Tables 4–6, §§3.1–3.3; P14, §§3–4]

## 3. Propagation and sensor effects remain mixed with environmental change

**Physical cause:** changing position, ground, weather and sensor response changes multiple transfer mechanisms at once. **Signal effect:** frequency-dependent coloration, Doppler, multipath, lower SNR and channel bandwidth affect different components unequally. **Representation effect:** scalar normalization cannot preserve every harmonic ratio or spectral feature; semantic embeddings may retain domain cues. **Domain shift:** new-station or new-dataset degradation cannot be assigned to one physical layer without controlled interventions. [P04, §§1.2,3–6; P05, §3.4; P13, chapter 3, §2.3]

**Existing mitigation:** station transfer, terrain-conditioned fuzzy rules, PCEN, device adversarial adaptation and paired microphone tests. **Remaining gap:** same-event cross-microphone testing is easier in a different way from new-event cross-dataset testing. P01's comparison does not isolate the dominance of environment, and P16 shows that device mismatch alone can strongly impair a scene classifier. Held-out-event paired sensors and fixed-source path interventions are needed before ranking source, path and sensor errors. [P09, §V; P03, §3.2; P16, §§3–4; P01, §§2.5,3.3]

## 4. Invariance can remove the information needed for classification

**Physical cause:** the same frequency bands carry vehicle information and nuisance effects; class priors/fleets differ between domains. **Signal effect:** source and channel changes overlap rather than occupy cleanly separable bands. **Representation effect:** global alignment or aggressive normalization can suppress useful class separation as well as domain cues. **Domain shift:** a visually aligned embedding may still misclassify target examples, especially minority classes. [Review chain grounded in P05, §3.4; P12, §§3–4,7; P01, §§2.3,3.4]

**Existing mitigation:** source-label-preserving adversarial learning, class-conditioned similarity/metric learning, pretrained semantic tokens and complementary feature fusion. **Remaining gap:** the studies do not compare waveform diversity and representation consistency using the same encoder/head, source bank, target exposure and sample budget. P16's additional label loss matters; P01 DANN shows negative transfer, but its frozen projection versus full fine-tuning comparison is not a clean loss-only ablation. Domain confusion must be judged together with held-out class performance and calibration. [P16, §2; P01, §§2.2–2.4,3.4; P03, §3.6; P07, §3; P12, §5]

## 5. Recording context and split dependence can masquerade as acoustic robustness

**Physical cause:** many windows, microphone channels or repeated passes share a source vehicle, background, location and recording chain. **Signal effect:** persistent context is repeated across nominal samples. **Representation effect:** the learner can retain event/session-specific cues; frequent classes can dominate a decision rule. **Domain shift:** performance falls when those shared contexts disappear, while random-window evaluation may appear strong. [P05, §4.4; P09, §V; P15, §§2,5–6]

**Existing mitigation:** whole-run splits, segment-grouped folds, leave-one-vehicle-out dyno evaluation and cross-dataset tests. **Remaining gap:** provenance-clean grouping and untouched model selection are inconsistent. ACIDS development “testing” participates in optimization; P01 transformer error bars largely reflect evaluation sampling rather than retraining; original-source duplication across compiled web datasets is not excluded by dataset names alone. The first experimental objective must be a versioned, group-aware estimate of the gap, with source-held-out uncertainty, rather than another within-dataset architecture comparison. [P10, §§3–4; P08, §VI; P13, chapter 3, §5; P01, §2.4; P02, §2.1]

## 6. Background and overlapping vehicles change the task itself

**Physical cause:** multiple moving vehicles and unrelated sound sources occur simultaneously. **Signal effect:** masking and mixtures of harmonics/broadband energy obscure a single source's signature. **Representation effect:** an embedding can represent several events, but a single-label head must collapse them to one answer. **Domain shift:** vehicle category, vehicle presence, traffic scene and dominant-source classification become conflated. [P04, §§4.2,5.2; P15, §§2,5; P02, §§2.5,3.4]

**Existing mitigation:** audible-event selection, multilabel formulations, background classes, temporal voting and synthetic congestion mixtures. **Remaining gap:** mixtures with a greatest-overlap label and partial training-source reuse do not establish performance on independent real congestion. Mean softmax confidence is not a calibration guarantee. Begin with audited single-vehicle categories; reserve independently grouped overlapping/vehicle-absent scenes for later presence, multilabel and false-alarm work. [P05, §4.3; P15, §5; P09, §IV; P01, §2.1; P02, §§3.4,4]

## 7. Realistic generation scores are weak substitutes for transfer evidence

**Physical cause:** generated recordings can contain plausible timbre and scene effects without reproducing physical state or class diversity. **Signal effect:** good average spectral matches coexist with narrow within-class distributions or incorrect semantic events. **Representation effect:** generation metrics measure agreement in a particular embedding or feature distribution, not the target classifier's decision boundary. **Domain shift:** improved FD/FAD, listening quality or t-SNE overlap may fail to improve independent real recognition. [P06, §5, Appendix F; P02, §3.3; P14, §4]

**Existing mitigation:** human relevance ratings, several spectral metrics, real-target classification and multiple generator comparisons. **Remaining gap:** calibrated physical-parameter fidelity, source diversity and downstream transfer need to be measured separately. Pretrained generators/encoders also contain prior real-data knowledge; “synthetic-only training” must identify whether it means classifier task-training only, and whether normalization or adaptation uses real audio. [P06, §§3,5; P07, §§3–4; P02, §§2.2–2.3; P01, §2.3]

## Dependency implied by these gaps

First establish dataset identities, group independence, shared labels and comparable metrics (gap 5). Then reproduce real cross-domain and synthetic-to-real baselines (gaps 2 and 7). Only then vary source and propagation separately (gaps 1–3), test environment/sensor diversity (gap 3), and compare representation objectives under equal data access (gap 4). Polyphony and fine-grained model identity require additional task-specific evidence and should not be inferred from success on the initial civilian categories (gaps 1 and 6).

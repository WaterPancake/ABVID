> Historical protocol/command reference. Commands retain their original paths
> and run from `ABVID_ARCHIVE_ROOT`, not from the reorganized code checkout.
> See the [experiment registry](../../README.md).

# H2: paired source × propagation design freeze

2026-10-04 · `h2_source_path_v1` · freeze `source_path_20261004_v1` · roadmap E4–E5.

**Execution update — H2 complete.** The immutable snapshot preserves the original metadata-only status. Repairs and pre-target checks passed under [execution r1](../../../reports/ARTIFACT_INDEX.md#artifact-570c30d67290); complete corpus replay, feature extraction, all 40 fixed heads and full target scoring then passed the [independent final audit](../../../reports/ARTIFACT_INDEX.md#artifact-70e853936c31). See the [results](../../../reports/H2_source_path_results.md) and [implementation amendments](IMPLEMENTATION_AMENDMENTS.md) for numerical corrections, the passivity/RMS clarifications and measured resources. The design below remains the original scientific contract; future-tense gate language records what execution was required to establish. Historical H1/R0 results and military milestone gates are unchanged.

## Question and scope

Does the specified source enrichment improve **synthetic → real car/truck classification** more than the specified path enrichment with a frozen pretrained audio representation? Source dominance is a hypothesis. This estimates conditional intervention gains; it cannot assign an intrinsic percentage of the real domain gap to source or propagation errors.

The design carries forward [proposed experiments §7](../../../reports/proposed_experiments.md#7-h2--compare-source-and-propagation-contributions-e4e5) and the numerical contrast in [archived protocol, source/path sections](https://github.com/WaterPancake/ABVID/blob/77daa1b339906d35b6fa0a59a166ecf2eac01174/experiments/archive/pre_h1_v1.3/EXPERIMENTAL_PROTOCOL.md). Numerical priors are our experimental choices, not measured distributions of real vehicles. The archived split, 200-event budget and M_LOCK do **not** carry over. The four cells below are all required, regardless of score.

| Cell | Source | Path | Interpretation |
|---|---|---|---|
| F00 | S0: simple procedural source | P0: moving direct path | Controlled reference |
| F10 | S1: same base with source enrichment | P0 | Source intervention at P0 |
| F01 | S0 | P1: direct + ground + air filtering | Path intervention at S0 |
| F11 | S1 | P1 | Joint change and interaction |

Doppler is common to P0/P1. Ground reflection and air absorption are bundled. S1 also bundles three mechanisms. Neither factor identifies its subcomponents separately. No background, sensor randomization, real-source assets, AudioLDM, AST, fine tuning, adaptation or augmentation enters this minimum experiment.

## Dataset roles, counts and pairing

Canonical configuration is [config.json](https://github.com/WaterPancake/ABVID/blob/77daa1b339906d35b6fa0a59a166ecf2eac01174/experiments/h2/config.json). Every cell uses its own newly generated training and synthetic validation observations, with exactly the same latent families/geometry as the other cells. Class mapping is `car → 0`, `truck → 1` throughout; motorcycles and every other label are excluded. This is broad category recognition, not vehicle-model recognition.

| Role | Per class per seed per cell | Use |
|---|---:|---|
| Synthetic training | 19 templates × 10 trajectories = 190 events | Fit that cell's scaler and head |
| Synthetic validation | 5 separate templates × 10 trajectories = 50 events | Report unseen-template diagnostics only; no model/parameter selection |
| Real development evaluation | Same 7,810 cars + 256 trucks in H1 MELAUDIS | Transfer metrics after the execution lock |
| Untouched final test | None available in this protocol | Independent confirmation remains unestablished |

N=190 is an explicit event-budget match to completed H1, not a physical minimum or the maximum possible simulated corpus. Nineteen templates result from retaining ten trajectory observations/template under this budget; five validation templates retain a separate diagnostic role. These choices can be changed only in a new protocol version before target scoring. They do not imply 19 independent real vehicles or establish sufficient diversity. The earlier 20-template/200-event proposal was not a paper requirement.

Replicate seeds are `[42,123,456,789,1024]`. Across five banks there are 240 base templates, 1,200 geometry slots shared across the two labels, 2,400 base events and 9,600 four-cell observations (7,600 training; 2,000 validation). There will be 480 S-level dry waveforms, since each template's exact waveform is reused across its ten trajectories and both P levels. Forty scaler/heads are planned: four cells × five banks × two representations. Replicates are different sampled simulated banks, not new target sessions.

Every template and all derivatives belong wholly to train or validation. No windows from one template cross roles. Within each seed/role/template-slot/trajectory-slot, **both classes and all four cells share the geometry exactly**. RPM is independent of road speed. Base phases, noises and source parameters are identical across S0/S1 except the declared S1 transformations; dry source IDs are identical across P. Geometry seeds exclude class and S/P. Source streams include class and exclude S/P. Each stream is PCG64, seeded by the first 16 SHA256 hex digits of the compact JSON key array specified in config. No Python `hash()`, shared global RNG or renderer fallback is allowed.

`templates.jsonl` contains source parameters and component stream seeds; `geometries.jsonl` contains actual path values and crop indices; `events.jsonl` joins them; `render_plan.jsonl` defines four paired jobs/event; `fit_plan.jsonl` defines exact train/validation job IDs and the common head; `target_manifest.jsonl` copies exact H1 target IDs, paths, hashes, labels and conservative group IDs. These are plans, with waveform/checkpoint hashes null until execution. A template is a controlled latent source family, not a known physical identity. `timing_slots.jsonl` preselects 30 validation-only geometry slots for a later synthetic timing/check pass.

MELAUDIS uses the exact H1 connected-group population, in original frozen ID order. The four group supports (car/truck) are 626/36, 468/32, 6,290/187 and 426/1. These are conservative provenance components; original recording-session boundaries are unknown. The largest component dominates event-weighted metrics. Prior H1 and regularization work already exposed this target; a new lock cannot restore a blind test. No target labels, scores, feature moments or predictions may choose priors, normalization, head C, threshold, stopping point or samples. The inherited high-level design itself is informed by prior development work and is not an independent preregistration against unseen data.

IDMT contributes no examples or validation labels to H2 training/selection. H1 real/procedural/AudioLDM scores remain contextual baselines, not factorial cells: their lineage and normalization differ. Reuse the H1 target feature cache only after checking IDs, waveform/preprocessing/encoder hashes. Fit no transforms on it. Synthetic val does not become training after scoring. Provider licences/provenance qualifications remain as admitted in H1; no new dataset, user download or access request is required for this scope.

## Source factors

The dry-function reference is the released P02 script, SHA256 `01e177684b9204d36b0410ad6bd42a26aeda3f73fadd498aa812c6e7457b7c0e`; the snapshot and exact locations are in [the audit](BACKEND_AUDIT.md). Future implementation will port the five dry functions into an explicit-parameter adapter. It will not execute the release wrapper, its silent fallback, import-time logging or process-dependent seeds. Externalized RNG and S1 changes mean this is a new controlled corpus, not bitwise reproduction of the released bank.

Generate 80,000 float64 samples at 8 kHz (10 s). All uniform ranges are continuous, half-open as NumPy implements them. Engine firing frequency is RPM × cylinders / 120 for four-stroke engines. Each harmonic has an independent uniform phase [0,2π); amplitude at order h is `10^(-rolloff*(h-1)/20)`. The source uses engine AM `1 + depth*sin(2π*(base_firing/2)*t)`, depth U[.05,.15]. Components and final mixture retain the released peak normalization before the common RMS step.

| Class | Engine | Tire band | Base mixture |
|---|---|---|---|
| Car | RPM U[1500,4000], 4 cylinders; 8 harmonics, 4 dB/order | centre U[900,1100] Hz; width U[600,1000] Hz | `u=(rpm-1500)/2500`; engine `.3+.4*(1-u)`, tire `.3+.4*u`; no exhaust |
| Truck | RPM U[1000,2200], 6 cylinders; 12 harmonics, 2.5 dB/order | centre U[700,900] Hz; width U[800,1200] Hz | independent engine U[.35,.45], tire U[.20,.30], exhaust U[.20,.30] |

Tire: unit Gaussian noise → fourth-order Butterworth bandpass (`butter` normalized by Nyquist, `lfilter`); low=max(20,centre−width/2), high=min(Nyquist−1,centre+width/2); peak normalization. Exhaust: unit Gaussian noise → third-order Butterworth lowpass at min(2000,Nyquist−1), plus five firing harmonics with 1/h amplitudes and independent phases. Normalize broadband and harmonic components separately, mix .6/.4, normalize again. Preserve filter startup and discard it through the central observation timing, rather than substituting zero-phase filters.

S1 changes exactly three things, fixed per template:

1. Add U[−1,+1] dB/order to engine harmonic rolloff. Harmonic counts and all component filter bands remain fixed.
2. Set engine/exhaust instantaneous firing `f0(t)=f0_base*(1+a*sin(2π*f*t+phi))`, with a U[0,.02], f U[.5,2] Hz, phi U[0,2π]. For sample n, phase increment is `2π/8000 * sum(f0[k], k=0..n-1)`; each order h multiplies that increment and retains its original harmonic phase. This left-sample integration is part of the design. Faster engine AM stays at base_firing/2. Tire noise and broadband exhaust are not resampled/modulated.
3. Multiply each present component by an independent U[−3,+3] dB gain before the unchanged mixture peak normalization. Do not add exhaust to cars.

Normalize each resulting dry waveform by one scalar to RMS .1 over source samples [32000,48000). Log pre/post RMS and gain; reject nonfinite or RMS≤1e−12. Do not normalize each period or time frame. Identical fixed noise/phase draws across all ten paths are a deliberate tightening of the old proposal's optional per-run variation: path changes cannot silently introduce new source realizations.

Engine configuration, weights and RPM ranges make the category partly a hand-defined source prior. Neither source level is calibrated to real engine makes, load, fuel, gearing, exhaust transfer functions or speed-dependent tire excitation. S1 means *specified enrichment*, not verified greater realism. Harmonic/FM/balance effects must be checked in components before evaluating transfer; normalizing aggregate amplitude can mask some intended changes.

## Path, observation and amplitude controls

One repaired, explicitly versioned backend must serve both P levels, with the pinned upstream code retained as reference. The [repair contract and blockers](BACKEND_AUDIT.md) are part of this design. Source/path changes must not be implemented through different backends.

Both levels: 8 kHz, Sinc interpolation, omnidirectional point source at z=.5 m; one omnidirectional mic at [0,0,1.2] m; lateral d U[5,50] m; speed U[30,90] km/h; equiprobable direction ±1. Exact sample geometry is `[d, direction*(speed/3.6)*(n/8000-5), .5]`, n=0..79,999. The adapter must override/check the upstream endpoint interpolation against this grid; do not accept its slight endpoint-grid speed discrepancy. A full length-matched signal is supplied; no automatic looping/default siren.

P0 includes moving delay and spherical direct spreading, no reflection or air absorption. P1 enables both extras at fixed 20°C, 1 atm, 50% relative humidity and the inherited integer ground parameter 20,000. Its physical units/calibration are **unknown in the audited API documentation**; do not label it a measured pavement. Mic response is identity; no background, sensor noise, wind, clipping, compression or codec. SNR is not applicable to these noiseless scenes. Environmental weather is fixed even though its atmospheric absorption is toggled as a path factor.

Let `c=331.3*sqrt(1+20/273.15)`. Crop a 16,000-sample receiver window starting at `floor(8000*(5+sqrt(d²+.7²)/c-1)+.5)`. Use this same crop for both paths; do not recenter on output peaks or compensate one level's filters. Entire source/render is ten seconds, so the retained window is well inside generated support: maximum full-scene image-path delay is <.40 s and crop spans approximately 4.015–6.146 s. This supplies several seconds of startup/tail margin; the future renderer must verify it with impulses/delays. Metadata arithmetic alone does not validate acoustic timing.

Match receiver RMS over that crop to `10^(-26/20)` by a single scalar, preserve its temporal envelope, then resample that 8 kHz crop to 16 kHz with H1 `scipy.resample_poly` Kaiser beta5, constant extension. Final input is 32,000 float32 samples. Record raw full-render and normalized-crop hashes/RMS/peaks. Synthesis/master renders should retain float64 arrays; listening WAVs are float32 derivatives, not the canonical numerical master. No integer quantization is introduced into training.

For the real target, retain H1's arithmetic mean of stored channels, native→8→16 kHz resampling, center 2 s, no gain normalization and no padding. Generated input already has its deliberate receiver RMS control; this differs from both native real gain and the historical released synthetic bank. Thus H2 is internally controlled but its F00 score is not a reconstruction of H1's procedural baseline. Scalar source/receiver normalization removes absolute attenuation and distance-induced SNR as tested benefits; frequency response, delay, interference and pass-by shape remain. This design cannot settle the entire importance of propagation.

**Failure rule:** nonfinite, silent, shape/rate mismatch, incomplete signal, backend substitution or peak >.98 after 16 kHz resampling invalidates the full paired geometry slot (both classes, four cells) and stops the bank. There are zero redraw retries in v1. Never clip, drop only one cell, choose replacement seeds or silently change gains. A pre-target versioned amendment may change an infeasible rule, followed by a complete matched regeneration. This is stricter than the archived candidate redraw rule and avoids conditioning on cell-specific rejection. Save every failed artifact and reason.

## Representation, classifier and common rule

Primary: exact frozen H1 BEATs768, official code commit `732d834db70ee0fc3886b4bcbcfb4ce7fb829be2`, checkpoint SHA256 `d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34`. This is the already locally hash-verified mirror download; an independently published official checkpoint checksum remains unknown. Run eval/inference mode, no gradient, masked mean of valid output tokens. No layer selection, fine tuning or target adaptation. Inference RNG fixed at42; extraction threads4/batch8/CPU as H1.

Comparator: H1 MFCC26 = 13 coefficient means + population standard deviations; 16 kHz, FFT400/Hann400, hop160, center=true/constant padding, 40 Slaney Mel bands 0–8000 Hz, power2, power-to-dB reference1/top80, orthonormal DCT-II. The actual observation bandwidth is ≤4 kHz. Encoder and MFCC definitions/settings are in config and pinned extraction/adapter evidence.

For **each** cell/seed/representation fit a fresh StandardScaler (mean/variance from its 380 training rows only) and L2 logistic regression: C=1, lbfgs, tolerance1e−6, max_iter5000, intercept=true, class_weight=null, float64 head inputs. Assign that replicate seed to the estimator. Probability `p(truck)>.5` predicts truck; exact ties predict car. No calibrator, model ensembling, early stopping, validation fit, head reuse from H1 or IDMT-selected C. Synthetic validation is diagnostic only. If convergence fails, save the failure and stop before target; do not choose another head from target results. The October 4 regularization sensitivity remains a separate result, not an automatic head promotion.

## Metrics, uncertainty and decisions

For each representation and each seed report all four cells on the whole target and every conservative group: confusion matrix; macro-F1 (primary); BA; accuracy; both classes' precision/recall/F1; truck AUROC/AP; Brier/log loss and ten-bin truck-probability ECE. Undefined AUROC/AP for a one-class subset is null; absent-class recall is null in group summaries, never silently treated as successful recall. Report support, group-average recall, worst group/class recall and the worst individual seed/group/class case. Include always-car/always-truck references. Do not present improved prevalence-sensitive F1 without class recalls. Report synthetic validation with its disjoint templates and domain label separately.

Point Qij is each seed's full-target macro-F1, followed by an arithmetic average over the five seeds, **not** F1 from pooled predictions or an ensemble. Within each paired seed:

```text
delta_S = ((Q10-Q00) + (Q11-Q01))/2
delta_P = ((Q01-Q00) + (Q11-Q10))/2
interaction = Q11-Q10-Q01+Q00
D = delta_S-delta_P = Q10-Q01
```

Report the two conditional source gains, two conditional path gains, both marginals and interaction. Algebraically D only depends on the two off-diagonal cells, but all four cells are essential for benefit and interaction interpretation. The sole primary contrast is BEATs macro-F1 D; MFCC and other metrics are prespecified secondary/descriptive contrasts, not alternate chances to select a positive conclusion.

Use 10,000 bootstrap draws, seed314159 PCG64. Each draw samples four target groups with replacement and five bank indices with replacement, independently. Use the same sampled groups/banks for every cell and representation. Within a bank, sum the selected groups' confusion matrices (repeat a group with its multiplicity), compute each Q, then average over selected banks and calculate contrasts. Preserve each group's full event population: no window bootstrap, target balancing or treating seeds as independent target recordings. Take 2.5/97.5 percentiles, NumPy linear quantiles. For ranking/calibration intervals use the identical group/seed indices and weighted whole-event probabilities. Save draw indices and bootstrap samples. Four uneven, incompletely reconstructed groups provide weak conditional uncertainty; this is not population-valid proof or independent confirmation.

Practical source dominance is supported only if the lower95% bound of D is ≥.03 **and** that of delta_S is >0. An upper D bound <.03 falsifies the ≥3-point practical margin; ≤0 contradicts directional source dominance for these interventions. An upper delta_S bound ≤0 contradicts a useful source improvement. Other intervals are inconclusive. If conditional source/path gains reverse across the other factor, report the interaction and avoid a universal ordering. If both interventions hurt, a less harmful source change is not useful source enrichment. Neither the sign nor size is guaranteed. Report any class-recall loss beside the primary result; reliable recognition is not established by D alone. Do not widen priors, change C or rerun selectively in response to target scores.

## Execution gates and expected artifacts

1. **Completed here:** immutable metadata/config/evidence hashes; exact pairing and role checks; source/geometry draw determinism; H1 target-population match; protected historical artifact hashes. No model/acoustic tests are claimed.
2. **Next, before bulk generation:** implement the dry-source adapter and audited renderer repairs. Preserve the reference; save patch diff, implementation/code/dependency hashes and compatibility results. Pin all installed dependencies in an execution environment lock. Run analytic and manipulation checks below. Do not install or execute the snapshot merely because it is present.
3. **Before fitting/target access:** render all four cells, verify every count/hash/role and gate, then freeze the generated corpus and exact executable code, features, training selections and evaluation command. No null execution hashes may remain. Use synthetic-only validation/timing diagnostics to find implementation faults, not target performance to choose a simulator.
4. **Then:** fit all40 fixed heads, save scalers/checkpoints, convergence logs and synthetic validation predictions. Verify train-only scaler moments. Freeze these before the first H2 target score. Score the complete matched set once, save raw probabilities, per-group metrics, contrasts/intervals and verification report. No partial-cell winner selection.

Required future tests: deterministic waveform rerender and seeded metadata reconstruction; identical dry hashes across P/trajectories and base draws across S; unchanged class-independent geometry; source component harmonic amplitude ratios/instantaneous frequency within1% of declared laws on central samples; unchanged tire noise/filter realization; finite nonzero fixed length/rate; source and receiver RMS relative error≤1e−6 before float32; float32 receiver RMS relative error≤1e−5 and peak≤.98; no accidental default source, looping or fallback. Check actual combined-source spectral/modulation differences and retain component outputs for interpretation.

Physical checks: direct static impulses at5/20/50m with delay within2 samples after declared interpolation latency and direct amplitude within1% of1/r; stationary tones125/500/1500/3000Hz within1% frequency, ≤.5dB magnitude error against the implemented path transfer; angle-dependent complex reflection table checked against direct formula at0/30/60/85degrees; unity-reflection test path uses image distance1/(a+b), not1/(a*b); air coefficients/finite FIR response agree with the pinned formula, passive gains do not exceed1 by>.5dB; approaching/receding tone tracks checked against an independent retarded-time calculation within1% frequency over the retained crop at range5/50m and speed30/90km/h. Measure and disclose any filter latency; no cell-specific output alignment. Check all range/speed corners and the30 prespecified paired slots, including ground finite outputs at grazing angles. A numerical tolerance is an acceptance choice, not field validation. Failure blocks execution or requires a new disclosed pre-target revision.

Expected later artifacts: execution lock + environment lock; source masters and parameters; paired full renders/crops and generation failures; observation/feature caches; 40 fitted pipelines and exact fit IDs; synthetic-validation and target probabilities; per-class/group/seed tables; bootstrap draw files; manipulation/physics plots and report; measured runtime/peak memory; final four-cell interpretation including null/negative results. None is represented as completed by this freeze.

## Hardware and runtime

Use the existing Apple M3 Pro/18GiB CPU, four Torch threads, one BLAS thread, batch8; GPU/cloud is unnecessary for frozen embeddings and small heads. The original ~5GiB storage estimate was too small: budget roughly 9–11GB for full renders, dry/component sources, observations, features and logs. The admitted renderer uses four bounded worker processes with one BLAS thread each; the measured validation run used about 359MB peak parent RSS and 503MB peak child RSS (these are separate maxima, not a measured simultaneous total). Avoid duplicating dry sources per trajectory.

Renderer runtime is **unmeasured and potentially dominant**: 9,600 ×80,000 =768million sample updates in the reference loop. For scheduling only, a hypothetical5–50s per ten-second render means13–133CPU-hours; this is a planning range, not a benchmark or guaranteed bound. The first30 validation geometry slots mean240 paired class/cell renders: measure their actual per-P timings and extrapolate byP before committing to the bank. Do not change the bank budget based on target scores. H1's14,630-window extraction took383s on this machine; roughly4–10min for9,600 new windows is an extrapolation, with target features reused. Reserve1–5min for40 heads/scoring/bootstrap and separate verification, also unmeasured. Record actual timing, memory and any resource amendment before target scoring.

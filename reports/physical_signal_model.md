# Physical signal model

Phase 1 of the ABVID literature analysis, 2026-09-27. This intermediate model was written before the representation and domain-shift phases. Paper IDs refer to the inventory in [literature_review.md](literature_review.md#source-inventory). Section numbers refer to the source, and **PDF p.** means a one-based page in the supplied PDF, not a Markdown image index. **Author evidence** describes the source; **review interpretation** identifies our deductions or proposed abstractions.

## What the observations can identify

**Author evidence.** Engine-cycle harmonics, their relative strengths, and track-related periodicity contain discriminative information in the measured heavy vehicles. Engine and track fundamentals are different quantities: the engine cycle depends on crankshaft rate and stroke cycle, whereas track repetition depends approximately on translation speed divided by track-element length. Gear ratios couple the two without making RPM and road speed interchangeable. [P04, §§5.1–5.3, 7]

**Author evidence.** Altmann et al. explicitly restrict their harmonic classifier to engine signatures, including exhaust configuration: different vehicle types sharing an engine may be grouped together. Their eight recognized labels name the recorded vehicles/types, but two additional vehicles did not yield usable harmonic series. Training and testing use vectors from the same passes and sensors. This does not test transfer to another physical vehicle of the same model. [P05, §§3.5, 4.3–4.4, 5; Table 3, PDF p.20]

**Author evidence.** Wu and Mendel's 2010 chapter predicts four broad categories: heavy tracked, light tracked, heavy wheeled, and light wheeled. Nine underlying vehicle kinds contribute data; they are not nine output classes in that experiment. Göksu predicts four individually recorded cars, one named car per model, across repeated RPM sweeps. [P13, chapter 3, §§2.1, 3, 5; P11, §II]

**Review interpretation.** We must separate (1) broad category recognition, (2) recognition among recorded exemplars, and (3) model/type recognition validated on independent physical exemplars. A model name as a label establishes a nominal task, not the third level of evidence. None of these Phase 1 classification protocols establishes that third level. [P05, §4.4; P13, chapter 3, §5; P11, §II]

## Approximate decomposition

**Review proposal, not an equation fitted in these papers:**

```text
mechanical source and state
    -> direction-dependent radiation and propagation
    -> environment and interfering sources
    -> sensor and acquisition chain
    -> observed waveform
```

\[
x_m(t)=F_m\{s(t),q(t),v(t),r_m(t),\theta_m(t),g,e,m\}.
\]

Here `q` is source state, `v` is vehicle speed, `r` is source–sensor distance, `theta` is emission/observation direction, `g` is geometry, `e` is environment, and `m` is the sensor chain. A useful local approximation is a sum over source components and propagation paths, followed by the sensor response:

\[
x_m(t)\approx M_m\left[\sum_k\int h_{mk}(t,\tau;g,e)\,
s_k(t-\tau;q,v,\theta)\,d\tau+b_m(t)\right]+\eta_m(t).
\]

This is an accounting model, not evidence that the factors can be independently identified from one microphone. Time-varying paths allow motion; multiple components allow engine/exhaust and running-gear contributions. Terrain participates in both source excitation and propagation, and weather can affect both propagation and background noise. A single fixed convolution or scalar attenuation cannot represent every effect described in the papers. [Review interpretation grounded in P04, §§1.2, 3–6; P05, §§3.4, 4.1–4.3]

## Proposed physical taxonomy

| Layer and variable | Author evidence: effect on the signal | Experimental treatment and evidence boundary |
|---|---|---|
| Source: engine cycle, cylinders, firing sequence | Engine-cycle harmonic series is usually prominent in heavy-vehicle audio. The largest harmonic is often related to cylinder count, but exceptions occur. | Engine specifications documented; RPM inferred from harmonic series in the BVP analysis. Cylinder count cannot be recovered reliably from the largest peak alone. [P04, §§2, 5.1, 5.3; Fig.10, PDF p.20] |
| Source: exhaust and intake configuration | Exhausts, followed by intakes, are described as major heavy-vehicle acoustic sources; outlet arrangement and relative propagation paths affect the harmonic mixture. | Vehicle/exhaust specifications and both travel directions are observed. This is not a controlled replacement of exhausts on an otherwise identical vehicle. [P04, §§1.2, 5.3; P05, Table 2, §5] |
| Source: RPM, throttle/load, gearing | RPM shifts harmonic frequencies; relative harmonic amplitudes can also change with RPM. Gear choice couples engine rate to translation speed. | Göksu varies RPM from 800 to 3000 r/min. BVP directs driving speeds and examines RPM/gear relationships; some tractors run loaded and unloaded. Throttle trajectories and independent load measurements are unknown for the classifiers. [P11, §II; P04, §§2, 4.1, 5.1; P05, §5] |
| Source: tracks, tires, drivetrain and auxiliaries | Track sprocket engagement contributes airborne sound; track elements and road wheels strongly excite ground vibration. Tire/road interaction, suspension, engine auxiliaries and drivetrain are also discussed. | Track length, speed and spectra allow some assignments; not every line is assigned. The unusually clear seismic track signal must not be treated as equivalent airborne evidence. Independent acoustic signatures of every drivetrain/auxiliary component are unknown. [P04, §§1.2, 3, 5.1–5.2, 7] |
| Source: speed, acceleration and movement | Speed changes running-gear periodicity and sound level; RPM can vary within a nominally constant-speed pass. | Directed passes at several nominal speeds; light-beam motion measurements in later BVP experiments. ACIDS runs are approximately constant speed, 5–40 km/h. An independently crossed acceleration/RPM experiment is unknown. [P04, §2, §§4.1, 5.1; P05, §2, §4.3; P13, chapter 3, §§2.1, 2.3] |
| Propagation: distance and SNR | Pass maxima decline approximately as inverse distance in the reported acoustic measurements. Distance also changes source-to-background ratio. | Microphone positions and lanes are known in BVP. The inverse-distance fit concerns pass maxima, not an exact instantaneous law. Harmonic-power normalization removes scalar gain but not frequency-dependent propagation or the SNR change. [P04, §4.1; P05, §3.4 and footnote 4; P13, chapter 3, §2.3] |
| Propagation: direction and source orientation | Relative harmonic powers and acoustic amplitude depend on observation side/angle; rear exhaust radiation helps explain maxima after closest approach. | BVP measures both directions and multiple sensor positions. Near-maximum selection partly restricts observation angle in P05; it is not an angle-invariant validation. [P04, §§3, 5.3; P05, §§2, 4.2, 4.4] |
| Propagation: geometry, motion and multipath | Reflection, diffraction, ground interaction and multiple paths modify the observed mixture. A changing Doppler shift can broaden lines during an analysis interval. | Geometry and sensor coordinates are specified in BVP; no independent ablation of every propagation mechanism. Nearby microphones often measure delayed similar waveforms, with exceptions for multiple sources. [P04, §§1.2, 3, 6.1; P05, §3.4, footnote 4] |
| Environment: terrain and ground | Ground affects excitation and propagation. The same vehicle's ACIDS feature distributions differ between terrains; in some dimensions the change is comparable to between-vehicle differences. | BVP includes different roads/soils and a deliberate bump experiment, whose strongest reported effects are seismic. ACIDS includes desert, normal and two arctic conditions; terrain is represented in training, not held out wholesale in the chapter. [P04, §2, §6.2; P13, chapter 3, §§2.1, 2.3, 5] |
| Environment: weather | Wind shear and temperature structure can refract sound; wind/rain and associated activity affect recorded background. | Meteorological instrumentation is present in P05, with background observations. A controlled weather factorial or held-out-weather recognition result is unknown. [P04, §§1.2, 4.2; P05, §§2, 4.1] |
| Environment: other vehicles and noise | Louder distant vehicles can mask quieter nearby ones; competing harmonic series can be confused. | Background measured and discussed; P05 selects sufficiently audible pass segments and sometimes requires operator correction. Noise is not uniformly randomized at prescribed SNRs in these studies. [P04, §§4.2, 5.2; P05, §§3.2, 4.1, 4.3] |
| Sensor: microphone, response, gain and acquisition | Bandwidth, filtering and instrument noise affect the available signature. P05 attributes much of its acoustic noise floor to the power supply. | Sensor models, sampling and filtering documented in BVP. P11 fixes one microphone/recorder arrangement, 1 m to the engine's left, at 44.1 kHz; microphone make/response is unknown. Cross-sensor classification is not demonstrated by these experiments. [P04, §2, Table 2; P05, §§2, 4.1; P11, §II] |

## What is stable, and under which transformation?

**Author evidence.** Relative harmonic powers indexed by harmonic order are designed to remove scalar amplitude changes and uniform frequency scaling, and power spectra remove absolute time shift. The authors explicitly retain sensitivity to emission angle, changes in the engine's harmonic balance, frequency-dependent propagation, and low SNR. The fundamental may be absent as an actual spectral line; selecting the largest peak is not equivalent to estimating it. [P05, §§3.2–3.4, 4.3, 5; P04, §5.1]

**Author evidence.** Wu and Mendel normalize each channel's energy and extract harmonics 2–12. They model terrain-specific uncertainty with separate fuzzy rules rather than demonstrating removal of terrain information. Their test removes one run from each already represented terrain. [P13, chapter 3, §§2.2–2.3, 3, 5, Fig.4; PDF pp.63–65, 78]

**Author evidence.** Göksu reports complete classification of the selected four-car test samples using wavelet-packet norm entropy and an MLP. Training uses the first RPM increase/decrease episode and testing the second; both cover the same RPM range. More vehicles, heavy trucks and feature reduction are identified as future work. [P11, §§II, IV–V, Table 1]

**Review interpretation.** The stable candidates are conditional: harmonic-order patterns within suitable engine states, and learned distinctions across the sampled RPM sweeps. “Engine-speed independent” is an author claim supported here by repeated sweeps of the same cars, not by held-out RPM ranges, independent sessions, sensors or same-model exemplars. RPM, terrain and direction can alter the class-bearing structure itself, so they cannot simply be removed as irrelevant noise. [P11, §§II–V; P04, §5.3; P05, §5; P13, chapter 3, §2.3]

## Checks against the authoritative PDFs

- P05 Table 1 loses the `Lane 3` header in Markdown. PDF p.4 establishes four lanes, with concrete lane 0 and three sand lanes. Table 3's within-pass split was checked on PDF p.20.
- P04 §4.2 contains corrupted micro symbols in Markdown: PDF p.15 establishes a seismic background of **2 µm/s** and illustrative threshold of **4 µm/s**, not millimetres per second. PDF pp.5 and 20 were checked for propagation statements and Fig.10.
- P11 equations (7)–(9) lose absolute-value/square notation in Markdown. PDF p.4 establishes norm entropy as the sum of coefficient magnitudes raised to `p`, and log-energy entropy as the sum of logs of squared coefficients. No missing expression was reconstructed from the conversion.
- P13 chapter 3 feature-extraction notation and the training/testing pseudocode were checked on PDF pp.63–64 and 78. Table 6 was checked on PDF p.79. The pseudocode uses testing during parameter optimization; independent validation for this selection is unknown. This is our evaluation concern, not an author-acknowledged limitation.

## Handoff to Phase 2

**Review questions derived from Phase 1:** Does a representation preserve harmonic order, relative amplitude, broadband running-gear content, temporal evolution, or some mixture? Which physical transformation is it designed to tolerate? Was that tolerance tested on a held-out physical condition, or only with additional samples of known cars/runs? These questions follow directly from the differences between P05's within-pass split, P11's repeated RPM episodes, and P13's held-out runs on represented terrains. [P05, §4.4; P11, §II; P13, chapter 3, §5]

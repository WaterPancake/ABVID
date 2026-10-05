# ABVID and CAST: Essential Concepts

A practical glossary and study checklist for understanding the project, its experiments, and its limitations.

**Scope:** ABVID studies acoustic vehicle recognition and robustness. CAST fits an effective sound renderer to observations, then investigates distributions of those fitted controls. This document explains concepts; versioned experiment protocols remain authoritative for settings and acceptance criteria.

**How to use this:** for each concept, aim to explain it in your own words, identify where it appears in the project, and give an example of how misunderstanding it could produce a misleading result. Study sections 1–5 first; section 6 provides background on the historical multichannel work, and section 7 supports every experiment.

**Project references:** reports and protocols use the current repository layout. The original study guide and metric implementation use commit-pinned links to the preserved pre-restructure source.

## Start with these ten ideas

- [ ] A recording combines the source, propagation, environment, and sensor.
- [ ] Sampling rate and observation duration limit what can be measured.
- [ ] A spectrum, a spectrogram, and an amplitude envelope describe different properties.
- [ ] Windows from one recording are dependent observations.
- [ ] Model selection and final evaluation need separate data roles.
- [ ] Reconstruction, acoustic coverage, and classification transfer are separate claims.
- [ ] A good reconstruction does not uniquely identify its generating parameters.
- [ ] Matching individual feature distributions does not establish a matching joint distribution.
- [ ] Coverage, distribution distance, and classification accuracy measure different things.
- [ ] Reproducibility establishes what happened; independent evaluation establishes how far it generalizes.

## 1. Signals and acoustic observations

### Waveform, sample rate, and duration

A waveform is a sequence of amplitudes over time. The sample rate, fs, is the number of samples per second; a recording with N samples lasts N/fs seconds. Channels are synchronized signal streams, not extra independent recording sessions.

**Understand:** a two-second mono clip at 16 kHz contains 32,000 samples. Its duration and bandwidth constrain the patterns available to a model.

### Nyquist frequency, aliasing, and resampling

The Nyquist frequency is fs/2. Frequencies above it cannot be represented unambiguously and may appear as lower frequencies: aliasing. Downsampling requires appropriate low-pass filtering. Upsampling increases the sample count but cannot restore information already removed.

**CAST connection:** audio rendered at 8 kHz and resampled to 16 kHz retains the original approximately 4 kHz bandwidth; it does not acquire genuine 4–8 kHz content.

### Amplitude, power, RMS, DC, and decibels

Amplitude is the signal value. Mean squared amplitude is a power proxy; RMS is its square root. DC is the waveform's mean offset. Decibels express ratios logarithmically: use 10 log10 for power ratios and 20 log10 for amplitude ratios under equivalent conditions.

**Understand:** DC removal and unit-RMS normalization isolate aspects of waveform shape while discarding offset and global level. Normalized digital amplitude is not a calibrated sound-pressure level.

### Signal-to-noise ratio (SNR)

SNR compares a defined target signal's power with a defined noise signal's power:

`SNR_dB = 10 log10(P_signal / P_noise)`

At 0 dB the powers are equal; at −10 dB the noise power is ten times the signal power. Controlled mixing can set SNR when the components are available. Exact target SNR is usually not directly observable from an arbitrary field recording without additional assumptions.

### Fourier transform, spectrum, and phase

The Fourier transform describes frequency content; the FFT is an efficient algorithm for computing a discrete Fourier transform. Magnitude describes component strength, and phase describes their relative timing. A magnitude spectrum discards information that a full waveform retains.

**Understand:** two waveforms can have the same magnitude spectrum and different time-domain shapes. A good magnitude match does not imply sample-by-sample agreement.

### STFT, spectrogram, windowing, and spectral leakage

The short-time Fourier transform (STFT) computes spectra on successive windows. A spectrogram displays their magnitudes or powers over time. Window length trades temporal localization against frequency discrimination; the hop controls how often a window is evaluated. Window functions reduce leakage while changing spectral peak width.

**Understand:** FFT bin spacing is fs/N_fft, but zero-padding alone does not improve the ability to resolve two nearby tones. Longer observations and suitable windows matter.

### Power spectral density (PSD) and Welch's method

PSD describes the distribution of signal power across frequency. Welch's method averages spectra from windowed segments to stabilize the estimate. A long-term PSD emphasizes average spectral shape and can hide when particular events occurred.

**CAST connection:** PSD, STFT, and envelope diagnostics provide complementary views; one cannot substitute for all the others.

### Harmonics, harmonic spacing, and the missing fundamental

Harmonics occur at integer multiples of a fundamental frequency. The spacing between peaks can reveal periodic structure even when the lowest harmonic is weak or absent. Octave and subharmonic alternatives can make the underlying periodicity ambiguous.

**CAST connection:** fitted harmonic spacing is an effective acoustic control. Converting it to engine RPM requires independently justified knowledge of the relevant rotational or firing order.

### Amplitude envelope and modulation

An amplitude envelope describes how signal strength changes over time. Modulation describes variation in amplitude or another signal property; an envelope modulation spectrum measures the rates of amplitude fluctuation. Carrier frequency and modulation frequency are different quantities.

**Understand:** a 500 Hz tone pulsing three times per second has a 500 Hz carrier and 3 Hz amplitude modulation. Two seconds of audio provides limited evidence about much slower changes.

### Source, propagation, background, and sensor response

A useful approximation for a fixed acoustic path is:

`recorded_signal = source * impulse_response + background`

Here `*` means convolution. An impulse response describes how a linear, time-invariant path spreads and filters sound. Motion, changing geometry, clipping, and compression can require a more complicated model.

**Understand:** a spectral peak or envelope change may come from the source, path, background, or microphone. A roadside observation is not automatically a dry source that can be propagated again without duplicating effects. See the project's [physical signal model](reports/physical_signal_model.md).

**Study check:** explain why resampling 8 kHz audio to 16 kHz cannot recover missing high frequencies, and why RMS normalization can help shape comparison while removing potentially useful level information. For deeper DSP study, use [Julius Smith's Spectral Audio Signal Processing](https://dsprelated.com/freebooks/sasp/).

## 2. Features, representations, and recognition

### Acoustic descriptors, log-Mel features, and MFCCs

A descriptor summarizes some property of a signal, such as band energy or spectral centroid. A Mel filterbank pools spectral energy into perceptually motivated frequency bands; taking logarithms produces log-Mel features. MFCCs apply a cosine transform to log-Mel energies and commonly retain a subset of coefficients.

**Understand:** each representation emphasizes some distinctions and suppresses others. A smooth spectral summary may discard fine harmonic detail.

### Embeddings, pretrained encoders, and classifier heads

An encoder maps audio into a feature vector, or embedding. A classifier head maps that vector to class scores. A frozen encoder is unchanged during task training; fine-tuning updates some or all of its parameters. Pretraining supplies information learned outside the immediate experiment.

**ABVID connection:** a comparison using a pretrained encoder must report that training history. Read [BEATs](https://arxiv.org/abs/2212.09058) to understand the representation used by one of the project's baselines.

### Loss functions and decision rules

A training loss is the numerical objective optimized during fitting. Cross-entropy rewards assigning probability to the observed class. A decision rule converts model outputs into actions, such as a predicted class at a threshold. Neither the training loss nor the decision threshold is automatically an evaluation metric.

**Understand:** lower training loss may coexist with worse performance on a new session. Threshold selection belongs in development data.

### Confusion matrix, precision, recall, F1, and balanced accuracy

A confusion matrix counts predictions against true labels. For a designated positive class:

- Precision = TP / (TP + FP): how often a positive prediction is correct.
- Recall = TP / (TP + FN): how much of the positive class is recovered.
- F1 is the harmonic mean of precision and recall.
- Balanced accuracy is the unweighted mean of class recalls.

Define how zero denominators are handled. Also report class counts and per-session results. With 90% of examples in one class, always predicting that class gives 90% ordinary accuracy but 50% binary balanced accuracy.

### Calibration, aggregation, and detection

A calibrated probability agrees with empirical outcome frequencies on the population being evaluated. Temporal aggregation combines predictions from several windows; overlapping windows remain dependent. Vehicle-presence detection includes deciding whether any target is present, while classifying known vehicle clips answers a different question.

**ABVID connection:** class probabilities alone do not establish a vehicle detector. False-alarm evaluation requires vehicle-absent observations and a defined time/event denominator. An offline rule that uses future audio is not a causal streaming rule.

**Study check:** explain how an encoder can yield useful embeddings without calibrated class probabilities, and why good car/truck classification does not establish reliable vehicle-presence detection.

## 3. Experimental design and generalization

### Independent unit, recording session, and ancestry

The independent unit is the entity over which a generalization claim is made: perhaps a recording session, site, or physical vehicle. Several windows from one source share nuisance conditions and ancestry. Synthetic derivatives retain the ancestry of the recordings used to produce them.

**Understand:** 1,000 windows from one session do not provide 1,000 independent sessions. Site/date grouping is a useful proxy when stronger identities are unknown; it does not prove that physical vehicles are independent.

### Leakage and train-only preprocessing

Leakage occurs when information from an evaluation set influences training or selection in a way the evaluation is meant to exclude. It includes shared source recordings, fitted normalization, feature selection, sampler calibration, and repeated decisions made using test results.

**CAST connection:** class centers and descriptor scales must be computed from the permitted training subset within each fold. Holding out raw waveforms is insufficient if their descriptors influenced a choice.

### Training, validation, testing, and nested cross-validation

Training estimates model parameters. Validation chooses models or settings. A final test assesses the frozen procedure. Nested cross-validation uses inner folds for selection and outer folds for evaluation, with grouping respected at both levels.

Repeated adaptation to outer results turns those results into development evidence. See scikit-learn's [grouped cross-validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data) and [nested-validation example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

### Generalization axis and test domain

Generalization needs a named change: an unseen session, vehicle, state, microphone, site, or dataset. Holding out one does not automatically hold out the others. Always identify the training-to-evaluation domain:

| Domain | What the experiment asks |
| --- | --- |
| Real → real | Does learning from real recordings transfer to the specified held real groups? |
| Synthetic → synthetic | Does the method work within the specified simulation distributions? |
| Synthetic → real | Does learning from generated examples transfer to real recordings? |
| Synthetic + real → real | Does a declared mixture help on held real recordings? |

Real recordings used to calibrate a generator are real-data access, even if classifier training later uses only generated waveforms. Declare that access separately from encoder pretraining.

### Domain shift, confounding, and shortcut learning

Domain shift is a change in the distribution encountered by a model. A confound is a variable entangled with the label or experimental condition. Shortcut learning occurs when predictive cues that work in development fail under the intended deployment conditions.

**ABVID example:** if classes are mostly recorded using different microphones, microphone coloration can become a class cue. [Shortcut Learning in Deep Neural Networks](https://arxiv.org/abs/2004.07780) explains the broader problem.

### Augmentation, domain randomization, and adaptation

Augmentation transforms available examples. Domain randomization samples variation in simulated conditions. Domain adaptation uses information from a target domain. Each requires specifying which variables change, whether labels remain valid, and what target information is available.

**Understand:** adding noise does not create new vehicle identities. Randomizing an incorrect source model does not guarantee realistic diversity. Using unlabelled target audio still counts as target-domain access.

### Baselines, ablations, and matched controls

A baseline provides a reference. An ablation changes or removes a component to test its contribution. Matched controls keep relevant data, compute, preprocessing, random streams, and evaluation rules comparable.

**CAST connection:** a joint sampler, class prototype, and coordinate-resampling control ask different questions about retained variation and dependence. If both a candidate and its control change, report absolute scores as well as their relative difference.

### Uncertainty, group variation, and preregistration

Random-seed variation measures sensitivity to specified stochastic choices. Variation across independent groups addresses a different uncertainty. A group bootstrap resamples groups rather than pretending correlated windows are independent; very few groups still limit inference.

Preregistration records the question, candidates, selection rule, metrics, and acceptance criteria before the relevant evaluation. A mean can hide a failed group, so retain per-class, per-group, and worst-group results. A pass on reused development data does not become fresh confirmation.

**Study check:** explain why five synthesis seeds cannot replace five independent recording sessions, and why fitting normalization before splitting data can leak evaluation information.

## 4. CAST fitting and parameter interpretation

### Forward model and inverse problem

A forward model renders an observation from controls: `x_hat = renderer(theta, seed)`. The inverse problem estimates controls from an observation by comparing it with rendered candidates.

**CAST connection:** the renderer is intentionally restricted. Residual error can reflect model limitations as well as optimization failure. The [CAST plan](experiments/cast_coverage/PLAN.md) defines the observation-model interpretation.

### Effective parameters and identifiability

An effective parameter summarizes observable behavior without necessarily identifying a unique physical cause. Identifiability asks whether the observations and assumptions determine the parameter uniquely. Practical identifiability additionally concerns noise, limited duration, and finite precision.

**Understand:** different harmonic/noise mixtures can achieve similar spectral losses. Fitted spacing is not measured RPM, fitted noise is not separated tire noise, and an envelope is not measured throttle. An optimizer returning one answer does not resolve ambiguity.

### Analysis by synthesis and differentiable DSP

Analysis by synthesis estimates controls by repeatedly rendering and comparing. Differentiable DSP expresses signal-processing operations so gradients can guide those updates. Automatic differentiation computes derivatives; it does not establish that a model is physically correct or identifiable.

Read [DDSP](https://arxiv.org/abs/2001.04643) for differentiable harmonic and noise synthesis and spectral reconstruction objectives.

### Spectral reconstruction loss and multi-resolution analysis

Magnitude losses compare time-frequency magnitudes; log-magnitude losses increase the relative influence of weaker components. Multi-resolution objectives combine several STFT window sizes to assess different time/frequency scales. Floors and normalization affect the objective and must be fixed explicitly.

**Understand:** the loss defines what the fitter considers similar. Matching that loss does not establish perceptual realism, matching phase, correct source decomposition, or classifier usefulness. Inspect residuals and individual loss components.

### Optimization, initialization, and regularization

Gradients describe local sensitivity to parameter changes. Learning rate controls update size. A nonconvex objective can have several basins, making initialization and multiple starts consequential. Regularization adds preferences beyond reconstruction agreement; bounds restrict permissible controls.

**Understand:** better loss from more optimization is different from evidence of more plausible parameters. Keep the best valid state, record failures and boundary hits, and distinguish a poor renderer from a poor search. For deeper theory, read [Stuart's inverse-problems notes](https://warwick.ac.uk/fac/sci/maths/research/events/2010-2011/non_symp_wksp/pdes/stuartnotes.pdf).

### Parameter transforms, simplex constraints, and log ratios

Positive quantities can be represented in log coordinates; a fraction p in (0,1) can use `logit(p) = log(p/(1-p))`. Nonnegative weights summing to one lie on a simplex. Their components cannot vary freely while retaining that sum.

For a strictly positive composition w, centered log-ratio coordinates are `clr(w_i) = log(w_i/g(w))`, where g(w) is the geometric mean. CLR coordinates sum to zero, so their full covariance is singular. Zero handling must be declared. See the [CLR definition](https://scikit.bio/docs/dev/generated/skbio.stats.composition.clr.html).

**CAST connection:** harmonic/noise weight relationships matter. Transforming, perturbing, clipping, and renormalizing can each alter the implied distribution; they are modelling decisions rather than neutral formatting.

### Stochastic nuisance, common random numbers, and separate checks

Phase and noise realizations can change a rendered waveform while its effective controls stay fixed. Common random numbers give compared methods matching stochastic realizations, helping isolate the method change. Separate check realizations reveal whether fitting exploited particular noise draws.

**Understand:** a new synthesis seed creates a new realization, not new independent real ancestry. A fitted parameter bank is an empirical sampling resource, not automatically a Bayesian posterior over physical parameters.

### Reconstruction, coverage, and transfer

| Claim | Required kind of evidence |
| --- | --- |
| Reconstruction | Agreement with the observations used for fitting, under declared losses and diagnostics. |
| Acoustic coverage | Generated variation compared with specified real groups, using frozen sampling and metrics. |
| Classification transfer | A classifier trained under a declared data budget and evaluated on appropriate independent groups. |

Passing one row does not establish the next. These distinctions are part of the [CAST scientific contract](experiments/cast_coverage/PLAN.md).

**Study check:** give two reasons that several parameter vectors could reconstruct one recording equally well, then explain what additional evidence would be needed to call a control a physical measurement.

## 5. Distributions, sampling, and CAST's metrics

### Empirical distribution, CDF, quantiles, and support

An empirical distribution places probability mass on observed samples. A cumulative distribution function (CDF) gives `P(X <= x)`. A quantile identifies a value at a specified cumulative probability. Distribution support concerns where values can occur; an observed minimum–maximum range does not establish the full population support.

**Understand:** a 5th–95th percentile interval is a central interval of a distribution. It is not a confidence interval for an estimated mean.

### Marginal, joint, and conditional distributions

A marginal describes one variable. A joint describes variables together, including their dependence. A conditional describes a distribution given information such as class: `P(features | class)`.

**Example:** a population containing only (0,0) and (1,1), equally often, has the same one-coordinate marginals as one containing only (0,1) and (1,0). Their relationships are opposite. Every one-dimensional marginal W1 can therefore be zero while the joint distributions differ.

### Covariance, multimodality, and group effects

Covariance measures linear co-variation; zero covariance does not generally imply independence. A multimodal distribution contains several concentrations of probability that a single mean may conceal. Group effects are systematic differences associated with recording groups, alongside variation within each group.

**CAST connection:** independently resampling parameters can break combinations present in fitted examples. Equal-group weighting gives small and large groups equal influence; equal-clip weighting estimates a different population. Neither weighting creates additional groups.

### Prototype, joint sampler, and marginal control

A prototype uses a representative parameter vector. A joint sampler preserves complete-vector relationships under its declared construction. A marginal control resamples coordinates separately to test the value of those relationships.

**CAST caveat:** renormalizing resampled weight vectors reintroduces dependence. The control is not perfectly independent after all constraints. The original experiment describes these arms in its [sampling protocol](experiments/cast_coverage/protocols/generalization/PROTOCOL.md); later variants must be interpreted through their own frozen protocols.

### Wasserstein-1 distance (W1)

W1 measures the minimum average cost of moving probability mass between distributions, using distance along the value axis as the cost. In one dimension it equals the area between their CDFs. It can be computed directly from empirical samples without selecting histogram bins. Lower is better. See [SciPy's definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html).

**Example:** shifting every value by two units gives W1 = 2 before scaling. The distance initially has the units of the measured feature.

### CAST's scaled marginal W1

CAST computes one-dimensional W1 for each acoustic descriptor coordinate, divides by its training-real standard deviation with a declared floor, averages within each family, and gives four families equal weight:

1. Log spectral proportions.
2. Broad-band energy proportions.
3. Amplitude-envelope values.
4. Envelope-modulation magnitudes.

The score is dimensionless and lower is better; it is not a percentage or restricted to [0,1]. The report label **joint W1** identifies the sampler being scored. The metric itself remains an average of marginal comparisons. See the [metric implementation](https://github.com/WaterPancake/ABVID/blob/77daa1b339906d35b6fa0a59a166ecf2eac01174/CAST/generalization/src/cast_generalization/method.py) and [frozen definitions](experiments/cast_coverage/protocols/generalization/PROTOCOL.md).

### KL divergence and MMD

Kullback–Leibler divergence compares probability ratios: for discrete distributions, `KL(P || Q) = sum p_i log(p_i/q_i)`. It is asymmetric and can be infinite when Q assigns zero probability where P assigns positive probability. It is a divergence, not a symmetric distance metric. Applying it to continuous sample sets requires a suitable probability model or estimator. See [SciPy's relative-entropy definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html).

Maximum mean discrepancy (MMD) compares distributions using a kernel. With an appropriate characteristic kernel, its population value can distinguish different multivariate distributions. Finite-sample estimates depend on kernel choice, sample size, and sampling assumptions. [Gretton et al.](https://www.jmlr.org/papers/v13/gretton12a.html) develops MMD-based two-sample tests.

**CAST connection:** these are concepts to understand when comparing evaluation methods. The current metric above is W1; it does not become KL or a multivariate test by changing the sampler.

### Marginal coverage and spread

CAST coverage measures the fraction of held real values inside each generated coordinate's 5th–95th percentile interval, then averages within and across families. It does not count the fraction of complete clips covered jointly, and it is not classification accuracy.

Spread compares generated and held descriptor variability. CAST uses the ratio of their mean coordinate standard deviations after frozen training-based scaling, separately for each family. This differs from averaging the individual coordinate SD ratios.

**Understand:** widening a generator can cover more real values while distributing too much probability in the tails. Coverage can rise while W1 worsens. Both definitions are specified in the [coverage protocol](experiments/cast_coverage/protocols/generalization/PROTOCOL.md).

### Dispersion calibration and temperature

A width multiplier can expand transformed parameters around a center: `z_new = mu + T * (z - mu)`. Before constraints, T > 1 increases deviations. Bounds and nonlinear inverse transforms mean output audio variation need not increase proportionally.

**CAST example:** width-v15 applied this to noise, mixture, and envelope coordinates, preserving spacing and harmonic weights. Its width was selected on source folds. This use of temperature concerns parameter dispersion, not classifier probability calibration or physical temperature. See the [v15 protocol](experiments/cast_coverage/protocols/improvement/width_v15/PROTOCOL.md).

### Absolute change, relative improvement, and percentage points

For a lower-is-better distance, relative improvement against a control is `1 - D_candidate/D_control`. A fall from 0.80 to 0.72 is a 10% reduction. Coverage increasing from 78% to 82% is an increase of four percentage points, approximately 5.13% relative.

**Understand:** report the named baseline and absolute scores. A better relative margin can reflect a weaker control as well as a better candidate. Always preserve class and group detail behind the mean.

**Study check:** explain how coverage and W1 can move in opposite directions, and construct two populations with identical marginals but different joint relationships. For deeper distance theory, read [Computational Optimal Transport](https://arxiv.org/abs/1803.00567).

## 6. Background: multichannel acoustics

### Time difference of arrival (TDOA) and array geometry

Different microphones receive a sound at different times. Their relative delays, together with microphone positions and an assumed sound speed, constrain direction or position. Synchronization error, uncertain geometry, and reflections can distort those constraints.

### Direction of arrival, GCC-PHAT, and SRP-PHAT

Direction of arrival (DOA) describes where a sound appears to come from. GCC-PHAT estimates relative delay using a frequency-weighted cross-correlation. SRP-PHAT combines microphone-pair evidence while searching candidate directions or locations. Both depend on geometry and assumptions about propagation.

### Beamforming and simulated-array evidence

Beamforming combines microphone signals to emphasize a direction, often by compensating expected delays before summing. Its benefit depends on alignment, interference, reverberation, and the source geometry.

**Understand:** successful localization on simulated channels establishes performance under those simulated conditions. Physical-array performance requires measured multichannel validation. These concepts belong to the existing [ABVID study guide](https://github.com/WaterPancake/ABVID/blob/77daa1b339906d35b6fa0a59a166ecf2eac01174/docs/vehicle_acoustics_study_guide.md).

## 7. Reproducibility and scientific interpretation

### Provenance, manifests, hashes, and configuration

Provenance records where an artifact came from and how it was produced. A manifest connects files to labels, source identities, split roles, permissions, and processing. Hashes identify exact bytes; configuration, code state, dependency versions, and seeds describe the procedure.

**Understand:** a hash can establish that a file is unchanged. It cannot establish that its label is correct or its source is independent.

### Determinism, numerical validation, and reproducibility

Determinism means repeating a computation under specified conditions produces the same result. Numerical validation checks properties such as finite values, bounds, gradients, resampling, and known synthetic cases. Reproducibility also requires the inputs and procedure needed to repeat the experiment.

**CAST connection:** bit-exact replay under a pinned environment is a strong implementation check. Different hardware or library versions may require separately specified tolerances. Replay does not validate physical interpretation or generalization.

### Technical success and scientific success

Technical success means the implementation follows its specification and passes appropriate checks. Scientific success means the evidence supports the stated hypothesis under its declared protocol. A correct, reproducible experiment may yield a scientific failure.

**Understand:** retain unsuccessful fits, negative comparisons, runtime, and limitations. An acceptance threshold is an engineering decision unless separately validated as a scientific or perceptual standard. Consult [AGENTS.md](AGENTS.md) and each frozen protocol when interpreting project claims.

## Final understanding checklist

- [ ] Explain what information an STFT magnitude loss omits.
- [ ] Explain why an observed harmonic spacing is not automatically engine RPM.
- [ ] Distinguish per-window, per-session, and per-vehicle evidence.
- [ ] Describe a valid grouped, nested evaluation without fitting preprocessing on held data.
- [ ] Name the exact domain shift an experiment tests.
- [ ] Distinguish frozen representation learning from training a classifier head.
- [ ] Explain a plausible shortcut based on microphone or background.
- [ ] Distinguish an optimizer failure from model mismatch and parameter ambiguity.
- [ ] Explain why constrained parameter vectors require care when sampling.
- [ ] Explain empirical, marginal, joint, and class-conditional distributions.
- [ ] Define CAST's W1, coverage, and spread without calling them accuracy.
- [ ] Explain why widening a distribution may improve coverage while worsening W1.
- [ ] Distinguish stochastic variation across seeds from uncertainty across real groups.
- [ ] Explain why real-calibrated synthesis retains real-data access and ancestry.
- [ ] Describe what evidence would establish classification transfer beyond reconstruction.

## Project reading companions

- [ABVID study guide](https://github.com/WaterPancake/ABVID/blob/77daa1b339906d35b6fa0a59a166ecf2eac01174/docs/vehicle_acoustics_study_guide.md): a longer learning sequence and repository exercises.
- [Vehicle-acoustics literature review](reports/literature_review.md): paper-specific evidence and limitations.
- [CAST plan](experiments/cast_coverage/PLAN.md): scientific contract and initial renderer/fitting design.
- [CAST pilot report](reports/CAST_pilot.md): reconstruction results and ambiguity records.
- [CAST generalization report](reports/CAST_generalization.md): the initial held-group descriptor comparison.
- [CAST improvement report](reports/CAST_improvement.md): subsequent experiments, controls, successes, and failures.

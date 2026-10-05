# CAST — Characteristic Acoustic Signiture Transfer

**Planning handoff · 2026-10-04 · No CAST implementation or experiments performed by this document.**

CAST will fit a compact, interpretable sound model to the **observed acoustic characteristics** of vehicle recordings. It will first reconstruct observations, then estimate a distribution of fitted characteristics from which to synthesize additional examples. Its eventual purpose is to test whether these examples improve vehicle classification using frozen pretrained audio representations.

“Transfer” names the intended research outcome; it is not an established capability.

## 1. Scope and scientific contract

The first deliverable is an inspectable numerical fitter, not a neural audio generator. Given a short recording, it should return fitted parameters, a reconstruction, residual diagnostics, provenance, and explicit indications of ambiguity or failure.

```text
allowed real training recording
             |
      fixed preprocessing
             |
     observed acoustic descriptors
             |
   fit bounded renderer parameters <---- render and compare
             |
   parameter vector + reconstruction + diagnostics
             |
   later: joint distribution of training-only fitted vectors
             |
   sampled parameters + independent synthesis seeds
             |
      additional simulated observations
```

Three questions must be answered separately:

1. **Reconstruction:** Can a small model reproduce useful spectral and temporal characteristics of an observation?
2. **Coverage:** Does sampling its fitted parameter distribution reproduce variation beyond a few fitted exemplars?
3. **Transfer:** Does training on those examples improve classification of recordings from other groups or datasets?

Success at one does not establish success at the next.

An observation combines source, propagation, environment, and sensor effects. CAST v0 estimates an effective observation model. Its harmonic spacing is not measured RPM; its noise component is not isolated tire noise; its envelope is not measured throttle or distance. Unknown physical quantities remain `unknown`. Do not describe its output as a recovered dry engine source, inferred vehicle identity, or physically validated simulator.

The working hypothesis that source mismatch matters more than propagation mismatch remains untested. Reconstruction alone cannot resolve it. Nor can fitting or normalization manufacture independent recording sessions.

## 2. Read first and preserve

Read current [AGENTS.md](AGENTS.md), then:

| Reference | Purpose |
| --- | --- |
| [ABVID roadmap](ABVID_ROADMAP.md) | Current sequencing and unfinished research work |
| [Proposed experiments](reports/proposed_experiments.md) | H2/E4–E5 source/propagation comparison; H4/E8 calibration proposal |
| [Physical signal model](reports/physical_signal_model.md) | Source versus observation distinctions |
| [Representation taxonomy](reports/representation_taxonomy.md), [domain-shift taxonomy](reports/domain_shift_taxonomy.md) | What fitted variation might change in a representation |
| [Dataset protocol](experiments/DATASET_PROTOCOL.md), [admission findings](reports/dataset_admission_provenance.md) | Labels, linked groups, provenance limitations |
| [H1 results](reports/H1_baseline_results.md), [source diagnostics](reports/H1_source_diagnostics.md) | Established controls and limitations |
| [Existing source simulator](src/vehicle_audio/source_simulation.py), [profiles](configs/procedural_vehicles.yaml), [tests](tests/test_source_simulation.py) | Reusable synthesis concepts and compatibility boundary |

As inspected for this plan, the existing simulator uses hand-authored tracked/wheeled profiles, engine harmonics, mechanical noise, and class-specific mobility textures. It peak-normalizes the combined signal before distance attenuation. It has no real-recording parameter-fitting stage. Its Python dataclasses and scalar operations are not automatically an end-to-end trainable parameter interface merely because rendering uses PyTorch.

That simulator did **not** produce the supplied civilian procedural bank used in H1. Keep those two generators distinct. Do not reinterpret its tracked/wheeled branches as car/truck physics.

CAST is an additive research component. It does not advance the military milestone gates, replace H1, authorize another locked evaluation, or change existing experimental results. Preserve original simulator behavior and existing snapshots.

## 3. Data access and split boundaries

### Initial allowed data

Use the source-only [IDMT manifest](experiments/h1_diagnostics/frozen/source_diagnostics_20261003_v1/source_manifest.jsonl), verified against its [lock](experiments/h1_diagnostics/frozen/source_diagnostics_20261003_v1/lock.json). The current manifest contains 4,413 admitted IDMT sE8 CH34 excerpts: 3,902 car and 511 truck, grouped into six conservative site/date connected groups at three locations. These are grouping proxies, not proof of independent physical vehicle identities. See the source diagnostics and admission report above.

Use `car=0`, `truck=1`; continue excluding motorcycles. Labels condition a later parameter distribution, not different hand-coded renderer branches. Fit the same parameterization and bounds to both classes.

For the initial real-audio pilot:

1. Read H1's [frozen lock](experiments/h1/frozen/h1_common_budget_v1.3/lock.json) and select the existing **outer fold 0, selection seed 42, real training IDs**. Do not resample a replacement training set.
2. Exclude fold 0's held-out group, `connected_4001f06f57cfeee7`, before opening audio.
3. Within each remaining group and class, sort those selected IDs lexicographically and take up to five. This gives **at most 50 pilot recordings**, possibly fewer. Record exact IDs and shortages; do not fill shortages from outside the selection.
4. Use the pilot only for engineering development and in-sample reconstruction diagnostics. It has no validation or test score. Selecting a best latent fit to a clip is reconstruction, not classifier validation.

No new download is required for this bounded prototype. Unknown original sessions, vehicles, RPM, weather, or source assets do not block observation fitting; they limit physical interpretation and generalization claims.

### Excluded initially

Do not open MELAUDIS audio or features, mixed-domain H1 feature caches, supplied procedural/AudioLDM banks, or military protected/reserved recordings for CAST's initial work. In particular, do not use target means, equalization curves, embeddings, or performance to select a source-only fitter.

MELAUDIS has already served as exposed development data. It cannot later be presented as an untouched final test. Any future target-informed fitting must be declared adaptation and use a separate, group-disjoint adaptation subset.

Every reconstruction and generated derivative retains its original recording/group ancestry. A new noise seed or parameter perturbation does not create a new independent vehicle or session. No descendant may cross the split of its parent.

## 4. Proposed v0 observation renderer

Implement an additive, class-agnostic renderer in a new CAST namespace. Start with the established harmonic-plus-filtered-noise idea. DDSP provides a relevant precedent for differentiable signal components and spectral reconstruction objectives; its reported music results do not establish vehicle transfer. See [Engel et al., DDSP, ICLR 2020](https://arxiv.org/abs/2001.04643).

```text
effective harmonic-spacing trajectory -> additive harmonics --+
                                                              +-> amplitude envelope -> observation
seeded noise -> fixed bands with fitted relative energy -------+
```

Proposed engineering defaults below are starting choices, **not literature-established vehicle parameters**. Verify numerical behavior on synthetic fixtures and freeze their resolved values before real fitting.

| Control | Initial form | Interpretation and limitation |
| --- | --- | --- |
| Harmonic spacing | Three linearly interpolated values, bounded 10–400 Hz | Effective periodic structure; octave/subharmonic ambiguity is possible |
| Harmonic balance | Eight nonnegative weights, normalized to sum to one | Observed relative harmonic strengths; may include propagation/sensor coloration |
| Noise spectrum | Eight fixed frequency bands with normalized nonnegative energy weights | Observed broadband texture; source and background are not separated |
| Harmonic/noise mix | One bounded mixture coefficient | Effective decomposition, potentially non-unique |
| Slow amplitude envelope | Five positive interpolated knots, normalized to remove global scale | Observed temporal evolution, not an inferred pass-by geometry |
| Phase and noise realization | Recorded seeds; no free sample-by-sample waveform | Stochastic nuisance controls, not vehicle identity |

Render at 8 kHz internally for the first compatibility profile and use the fixed 8→16 kHz resampler for the final 32,000-sample waveform. Proposed noise-band edges are `[0, 125, 250, 500, 1000, 1500, 2000, 3000, 4000]` Hz; document the actual band-filter implementation. Suppress components at or above Nyquist and verify boundary behavior. Any later full-band renderer gets a separate version and matched controls.

Keep the first renderer deliberately small: no learned reverberation, source separation, explicit Doppler model, inferred cylinder count, vehicle-model presets, or unrestricted residual waveform. Those additions would introduce new identification problems before the basic fit has been assessed.

Retain all component signals for inspection. Normalize component energies explicitly so mixture weights are interpretable. Near-zero components must be handled without division instability. Fitting can legitimately choose a predominantly noise-based reconstruction; do not force every recording to have a credible fundamental frequency.

## 5. Preprocessing, descriptors, and objective

### Observation profile

Match H1's `core8_2s` input route: arithmetic mean of the two **stored** CH34 channels, native→8 kHz→16 kHz using `scipy.signal.resample_poly` with Kaiser beta 5, centered two seconds, no padding. CH3/CH4 are original channel identities, not indexes 3 and 4 into a two-channel array. Save the exact decoding/resampling/cropping specification and software versions.

For **fitting only**, retain the unnormalized observation and its RMS/DC metadata, then subtract DC and normalize to unit RMS for shape comparison. Apply the same shape normalization to the rendered observation. This deliberately removes global level from the initial fitting problem; it is not a claim that level is irrelevant to classification and does not modify H1 preprocessing. Flag silence/near-silence with a frozen threshold instead of amplifying it arbitrarily.

For listenable exports, use a documented common playback gain for an original/reconstruction pair. Do not independently peak-normalize examples and then compare their absolute amplitudes. Preserve float waveforms and gain metadata; do not silently clip.

### Fitting objective

Use a documented multi-resolution STFT magnitude objective inspired by the DDSP approach, not a new learned perceptual loss. Start with FFT sizes `[256, 512, 1024, 2048]`, Hann windows, hop `N/4`, `center=false`, and a fixed numerical floor `1e-5`. Average, with equal resolution weights:

- relative magnitude error: `mean(abs(S_real - S_render)) / (mean(S_real) + epsilon)`;
- log-magnitude error: `mean(abs(log(S_real + epsilon) - log(S_render + epsilon)))`.

These are proposed CAST loss definitions, not a claim to reproduce the original DDSP loss exactly. Specify FFT normalization, boundary handling, and any parameter regularization in the frozen config. Do not add BEATs embedding distance or target-domain statistics to v0's optimization objective.

Report additional diagnostics separately: long-term spectral envelope, harmonic salience where resolvable, band-energy proportions, amplitude-envelope error, and slow modulation differences. Two seconds provides limited evidence about slow dynamics; do not infer an entire operating cycle. Record per-resolution losses so an aggregate cannot conceal a missing frequency band.

## 6. Implementation stages and stopping points

### CAST-0 — Freeze access and interfaces

Create a CAST-only manifest and config from allowed IDs. Verify file hashes and group ancestry before audio access; reject unexpected datasets and paths. Snapshot git commit **and dirty state**, relevant source hashes, dependencies, seeds, dataset/parent-lock versions, and the selected IDs.

Define tensor-based renderer parameters, serialization, and the separation between physical metadata (`unknown` where unavailable) and effective fitted controls. Preserve immutable originals and all existing results.

**Deliverable:** validated input lock and a dry-run inventory. No audio generation or classifier work is required for this stage.

### CAST-1 — Verify fitting on known synthetic cases

Build fixtures from the new renderer: an unambiguous single tone, a harmonic mixture, filtered noise, a changing frequency/envelope, and mixtures. Include deliberately ambiguous cases: missing fundamental, alternative harmonic explanations, and gain/mixture tradeoffs.

Use bounded parameter transforms and numerical optimization. Starting proposal: Adam, learning rate `0.03`, 300 steps per start, four starts with seeds `[42, 123, 456, 789]`. Initialize spacing candidates using a fixed coarse harmonic-salience search; record the candidate-generation rule. Save the best objective state rather than blindly taking the final step. The step budget and initialization must be identical across classes.

Fix two noise realizations per fit and average their losses. Evaluate the fitted vector with two separately derived noise seeds to expose overfitting to a particular realization. Derive random streams from `(run seed, input ID, component, purpose)`; keep fitting, checking, and later sampling streams separate.

If gradient-based frequency fitting fails even on unambiguous fixtures, record that failure and repair initialization/numerics before real data. Do not respond by launching an architecture search. Freeze the tested settings after this stage; any later revision creates a new version.

**Acceptance:** deterministic reconstruction under a pinned environment; finite gradients/output; verified parameter bounds and no aliased harmonics; meaningful improvement over the best unoptimized start on predeclared nontrivial fixtures. The single-tone fixture must recover frequency within one bin of the longest STFT. Assess other parameter recovery only where identifiable, using fixture-specific tolerances declared before running. Ambiguous cases must expose alternatives or an ambiguity flag, not confident physical estimates.

### CAST-2 — Bounded real-observation pilot

Fit the at-most-50 IDMT recordings selected in Section 3. Compare every fitted reconstruction against its best unoptimized initialization using the same renderer, observation profile, and stochastic checking seeds. This baseline measures whether optimization helps; it is not evidence of transfer.

Save every result, including failed fits. Report losses and failure rates by class and source group, median and tail errors, runtime, parameter boundary hits, and the disagreement between equally good starts. Report spectral residuals and temporal residuals separately. Show preselected examples plus systematically selected best/median/worst cases; never publish only favorable examples.

A lower loss with implausible or inconsistent parameters is evidence that the observation fit is underdetermined. A persistent residual may indicate limited renderer capacity, optimizer failure, or unmodeled recording context. Distinguish these using the synthetic fixtures before adding components.

**First implementation handoff stops here:** produce `reports/CAST_pilot.md`, artifacts, and a concrete adequacy verdict. Do not automatically run classifier training, target evaluation, full-corpus fitting, or later CAST stages. A failed or inconclusive pilot is a valid result and should describe the smallest justified revision.

### CAST-3 — Estimate a training-only parameter distribution

Proceed later if the fitter has demonstrated useful, stable reconstruction. A table of per-clip fits is not yet a model that generates meaningful diversity.

Start with an empirical **joint** parameter bank: select a training class, select an allowed source group with equal probability, then select a fitted vector within that class/group. Resample complete vectors before trying independent parameter perturbations. This preserves observed correlations but remains a replay of fitted exemplars with stochastic variation; label it accordingly.

Compare this with a single class prototype and an independently resampled-marginals control. Introduce smooth perturbations or a low-dimensional distribution only after defining plausible ranges and testing held-group coverage. Do not condition on unavailable RPM, load, exact vehicle model, or environmental metadata.

Each sample records all parent IDs/groups, parameter-bank version, sampled vector, random streams, and output hash. Name its origin explicitly, e.g. `real_calibrated_observation_synthesis`; fitting on real audio is real-data access, even if classifier training later uses only generated waveforms.

### CAST-4 — Assess held-group acoustic coverage

Freeze the renderer and distribution rules before accessing the proposed held-out audio. Fold 0, excluded from the pilot, can provide the first CAST-held-group development check. It is already an exposed H1 development group, not a new final test.

Keep two evaluations distinct:

- **Conditional reconstruction:** fit latent parameters to a held-out recording to measure renderer capacity. This explicitly uses that recording at test time. Its fitted vector must never enter the training parameter bank.
- **Generative coverage:** sample only from the training bank, then compare generated descriptor distributions against held-out observations. No fitting, nearest-example selection, or parameter adjustment using the held-out audio is permitted for this score.

A later six-group evaluation must recreate all banks within each training fold. Pilot-driven design changes have already exposed several groups to development; report a six-group rerun as exploratory development, even if per-fold fitting is disjoint. Nested parameter selection controls fitting leakage but does not erase earlier human design exposure. Reserve genuinely new groups for future confirmation.

Report class/group/worst-group results, spread and tails, mode collapse, and retained sample counts. Six groups at three sites provide limited uncertainty estimates. When enough folds exist, use paired whole-group bootstrap intervals with a whole-site sensitivity analysis; do not bootstrap generated windows as independent recording sessions. A single held-out group supports no session-population confidence claim.

### CAST-5 — Test classification transfer under a separate frozen protocol

Only after the earlier evidence warrants it, define matched comparisons using frozen BEATs768 and the MFCC26 control with train-only scaling and the established logistic-regression head. Preserve class budgets, preprocessing, parent groups, real-data access, and seed selections across arms. Identify the test domain on every result.

Minimum useful controls are real-only training, conventional augmentation of the same real parents, an unfitted sampler using the same renderer, and CAST's fitted joint sampler. Distinguish generated-only training from real-plus-generated training. Count **all** real recordings used for calibration or validation when comparing data budgets.

Freeze selection using source-side groups; no tuning against MELAUDIS scores or target embedding distances. Any eventual MELAUDIS run is an exposed-development transfer evaluation. Use balanced accuracy, macro-F1, per-class recall, per-group and worst-group performance, and paired group-level uncertainty. Keep the five H1 selection seeds `[42, 123, 456, 789, 1024]` for a matched extension where feasible.

Predeclare a practically relevant gain and failure rule before classifier fitting. Acoustic matching improves without a transfer gain: the tested CAST calibration is insufficient for the classification objective. Conventional augmentation matches CAST: the added fitter has not demonstrated incremental value. A reproducible gain over both controls supports usefulness under that protocol, not recovery of clean-source physics or universal source-mismatch dominance.

**Propagation boundary:** CAST reconstructions already contain effective recording coloration. Passing them through another propagation/sensor chain can apply those effects twice. Treat such an extension as observation/recorded-asset augmentation with a no-additional-propagation control. It cannot silently substitute for the dry-source factor in H2/E4–E5. Testing source versus propagation causally needs separately controlled components, matched random streams and budgets, and their interaction—not reconstruction scores alone.

### CAST-6 — Optional learned parameter estimator

If the numerical fitter works but is slow, train a small estimator to predict its controls from a spectrogram, optionally refined by a few optimization steps. Compare accuracy, ambiguity, runtime, and memory with the original fitter on grouped data. Preserve the renderer and objective initially so the comparison isolates amortized fitting.

An estimator still needs an input recording. To generate without one, it also needs the parameter distribution from CAST-3 or another explicitly evaluated prior. Do not equate learning to reconstruct with learning independent vehicle identities. Larger neural generators or DDSP extensions need evidence of specific residual limitations and their own baselines.

## 7. Configuration, artifacts, and tests

Use one versioned CAST JSON configuration, with an immutable resolved config and lock for each run. Fit outputs and later sampled outputs must reference that same config hash. The following is a **schema sketch**, not an executable frozen experiment:

```json
{
  "schema_version": 1,
  "experiment_id": "cast_observation_pilot_v0",
  "phase": "CAST-2",
  "parent_protocol": "h1_common_budget_v1.3",
  "data": {
    "manifest": "experiments/h1_diagnostics/frozen/source_diagnostics_20261003_v1/source_manifest.jsonl",
    "manifest_sha256": null,
    "classes": {"car": 0, "truck": 1},
    "parent_fold": 0,
    "selection_seed": 42,
    "pilot_max_per_group_class": 5,
    "selected_ids_file": null,
    "target_access": "none"
  },
  "audio": {"profile": "core8_2s", "sample_rate_hz": 16000, "samples": 32000},
  "fit_normalization": "dc_removed_unit_rms_shape_only",
  "renderer": {"version": "cast_harmonic_noise_v0", "internal_sample_rate_hz": 8000},
  "loss": {"fft_sizes": [256, 512, 1024, 2048], "epsilon": 0.00001},
  "optimizer": {"name": "adam", "lr": 0.03, "steps": 300},
  "seeds": {"starts": [42, 123, 456, 789], "check_noise_stream": "disjoint"},
  "classifier": null,
  "distribution": null,
  "output_root": "experiments/cast/results/cast_observation_pilot_v0"
}
```

Before a real run, resolve every required placeholder and all omitted renderer bounds, filters, loss conventions, silence threshold, initialization, random-stream derivation, and numerical tolerances into the full config. Refuse unresolved configurations; never infer defaults from an existing result file. This CAST schema supplements existing experiment metadata requirements.

Planned additive paths, **not created by this plan**:

```text
src/vehicle_audio/cast/       renderer, descriptors, fitter, serialization, later sampler
experiments/cast/            runner, configs, immutable locks, results
tests/test_cast_*.py         meaningful numerical and provenance checks
reports/CAST_pilot.md        first implementation result and stopping decision
```

Per-run artifacts: resolved configuration, lock/hashes, environment and commit/dirty snapshot, selected manifest, timings, aggregated findings, and machine-readable verification. Per-recording artifacts: parameter JSON, identifiability/failure flags, optimizer history and alternative-start scores, component waveforms, reconstructed waveform, spectral/envelope comparison plots, original RMS/DC, and complete ancestry. Reproduction must not depend on notebook state.

Necessary tests cover deterministic rendering/replay, known synthetic fits and ambiguities, finite gradients, band limits and duration, channel/resampling behavior, silent/malformed input handling, serialization round trips, split/ancestry exclusion, and rejection of disallowed data. Preserve legacy simulator outputs; do not rewrite unrelated test suites.

## 8. Compute and implementation handoff

The initial fitter should run on CPU using NumPy/SciPy/PyTorch and existing audio I/O; a GPU, BEATs extraction, and TensorFlow/DDSP installation are unnecessary for the pilot. Runtime is **unknown until measured**. Time synthetic fitting and a single allowed real clip, then project the remaining pilot cost; report actual peak memory and seconds per fitted second. Do not promise a full-corpus schedule from classifier runtimes.

Inspect existing environments before choosing one. Do not upgrade or synchronize the historical reproduction environment to satisfy CAST dependencies. If isolation is needed, use a CAST-specific environment and lock its dependencies. The working tree contains substantial existing uncommitted/untracked work: do not reset, clean, stage, or overwrite it. A fresh git worktree alone may omit needed untracked research files; verify inputs instead of assuming it reproduces this checkout.

Suggested instruction for the next Codex instance:

> Read AGENTS.md and CAST_PLAN.md. Implement CAST-0 through CAST-2 only, using additive CAST paths and the source-only IDMT pilot selection specified here. First verify synthetic fitting, then run the bounded observation-reconstruction pilot. Preserve existing code behavior, locks, datasets, and baseline artifacts. Do not access target/protected audio, train classifiers, run transfer experiments, or advance the military milestones. Produce reproducible artifacts and reports/CAST_pilot.md, explain successful and failed fits, then stop at the pilot adequacy verdict. Treat later sections as a dependency-ordered research plan, not authorization to execute them all.

The immediate outcome should be an auditable answer to: **Can this small, interpretable renderer fit the acoustic characteristics we actually observe, with enough stability and fidelity to justify learning a distribution from its fits?**

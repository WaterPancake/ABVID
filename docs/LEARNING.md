# Learning ABVID: map, math and reading

Written 2026-10-31 from our learning conversation. The current project state
lives in [`reports/STATUS.md`](../reports/STATUS.md); this page is the background
for reading it.

Most of the repository's volume is process: freezes, hashes, seals and
verification records. The science fits on one page, below.

## 1. The project on one page

**The question:** if you train a car/truck classifier on simulated vehicle sound,
which parts of the simulator must be realistic for it to work on real recordings?

**The model behind everything:**

  x(t) = M[ H(s(t)) + n(t) ]

- **s:** the source (engine, tires).
- **H:** propagation (distance, ground reflection, Doppler).
- **n:** background noise.
- **M:** the microphone.

"The source matters most" is a hypothesis being tested, not an assumption.

**The fixed pipeline** used in almost every experiment: a 2 s clip goes into a
frozen pretrained encoder (BEATs → 768 numbers, or MFCC → 26 numbers), then a
logistic regression. Only the training data changes.

**The data:**
- **IDMT:** controlled German recordings, the main development set.
- **MELAUDIS:** harder Australian recordings, 3% trucks.

## 2. What was built and what it found

| Step | What we did | What it means |
|---|---|---|
| Admission / R0 | Audited 46,775 WAVs; replayed published checkpoints (17/18 matrices match) | Grouping clips by recording session matters; clips from one recording leak across a split |
| H1 | IDMT → MELAUDIS costs about 19 F1 points | Re-scored with AUROC, "synthetic beats real" shrank from +11.8 F1 to +2.6 AUROC [−8.5, 9.5]. With 3% trucks, F1 mostly measures threshold placement |
| H2 | Source vs. path 2×2; path appeared to win by about 35 points | An implementation artifact: filter latency (section 3). Without it, source vs. path can't be ranked |
| B1 | Stronger baseline recipe within IDMT | Full-band input plus strong regularization: AUROC 71.5 → 84.1. All synthetic work so far ran at the weaker 8 kHz setting |
| CAST-5A | Added CAST samples to real training data | No utility: +0.47 F1 [−0.97, 2.08]. Ordinary augmentation added nothing either |

**CAST in plain terms.** A small synthesizer turns 25 numbers into 2 s of sound:
- 3 pitch points over time (10–400 Hz);
- 8 harmonic loudnesses;
- 8 noise-band energies, with band edges at 0, 125, 250, 500, 1000, 1500, 2000,
  3000 and 4000 Hz;
- 1 harmonic-vs-noise mix;
- 5 loudness-envelope points.

It renders at 8 kHz and upsamples ×2.

For each real recording, gradient descent (Adam, 300 steps, 4 starts) finds the
25 numbers whose spectrogram best matches it. This is "analysis by synthesis",
the same idea as DDSP. The fitted vectors are pooled into a distribution, and new
vectors sampled from it are rendered as synthetic training data. Code:
`src/abvid/cast/renderer.py` and `src/abvid/cast/fit.py`.

**Where CAST fails.** The real-trained BEATs classifier's AUROC on 380 of its
own training clips:

| Stage | AUROC |
|---|---:|
| Originals (overfit head) | 100 |
| Fitted re-renders | 68 |
| After calibration | 77 |
| Population samples | 63 |

Most of the loss happens at reconstruction. Other measurements:
- 92% of fits had equally good starts that disagreed, so the 25 numbers are not
  identifiable physical quantities.
- In BEATs feature space, the car→truck direction of CAST data has cosine 0.29
  with the real one; ordinary augmentation scores 0.98.

**What we learned on Oct 31.** Through listening, per-clip classifier verdicts
and mel spectrograms:
- **Fitted re-renders often sound identical to the originals,** yet BEATs calls
  about half the re-rendered trucks "car" (91/190).
- **MFCC fails the other way:** cars become trucks (90/144).
- **Average spectra match closely.** What differs is change over time: for the
  trucks BEATs misses, the fit flattens the engine band's (100–400 Hz)
  frame-to-frame variation.
- **That matches a renderer limit:** harmonic and noise weights are fixed for the
  whole clip, so the spectral balance cannot evolve during a pass-by. A
  time-varying-weight renderer is being tested; results go in `STATUS.md`.
- **The extra content in the full-band version is mostly uniform hiss:** the
  recording's noise floor, not the vehicle.

## 3. Worked example: the H2 latency story

The "air absorption" filter was an 11-tap symmetric FIR filter applied causally.
A symmetric N-tap filter delays the signal by (N − 1)/2 samples: here 5 samples,
or 0.625 ms at 8 kHz. It was applied once on the direct path and twice on the
ground-reflected path, so it changed the relative delay τ between the two paths.

Two copies of a sound at relative delay τ interfere:

  H(f) = 1 + r·e^(−j2πfτ)

The response has notches at f = (2k + 1)/(2τ). If the relative delay were only the
extra 0.625 ms, the first notch would sit at 800 Hz. Moving the notches alone
reproduced the "propagation" gain: 35.43% F1 with latency only, against 3.68%
with attenuation only.

**The lesson:** declare and test every renderer component's gain, latency and
bandwidth. A numerical side effect can look like physics.

## 4. The math you need, and where to learn it

| Topic | Where it shows up | Learn it from |
|---|---|---|
| Sampling and Nyquist (8 kHz keeps nothing above 4 kHz) | B1, CAST, the hiss | [Think DSP](https://greenteapress.com/wp/think-dsp/) (Downey, Python, free); [Mathematics of the DFT](https://ccrma.stanford.edu/~jos/mdft/) (Smith) |
| Harmonics and Fourier series; engine firing frequency f0 = RPM × cylinders / 120 | CAST oscillator | [Mathematics of the DFT](https://ccrma.stanford.edu/~jos/mdft/) |
| DFT, windows, leakage | All spectrograms | [Harris 1978, windows](https://doi.org/10.1109/PROC.1978.10837); [Spectral Audio Signal Processing](https://ccrma.stanford.edu/~jos/sasp/) (Smith) |
| STFT and the time/frequency tradeoff; multi-resolution spectral loss | CAST fitting loss | [Spectral Audio Signal Processing](https://ccrma.stanford.edu/~jos/sasp/); [Welch 1967, averaged spectra](https://doi.org/10.1109/TAU.1967.1161901) |
| Filters, group delay, comb filtering | H2 latency | [Introduction to Digital Filters](https://ccrma.stanford.edu/~jos/filters/) (Smith); [Audio EQ Cookbook](https://www.w3.org/TR/audio-eq-cookbook/); [Laakso 1996, fractional delay](https://doi.org/10.1109/79.482137) |
| Mel filterbanks, MFCC, log-mel front ends | BEATs and MFCC inputs, the spectrogram page | [Davis & Mermelstein 1980, MFCC](https://doi.org/10.1109/TASSP.1980.1163420); [Wang 2017, PCEN](https://arxiv.org/abs/1607.05666) |
| Analysis by synthesis and inverse problems; non-identifiability | CAST fitting | [DDSP, Engel 2020](https://arxiv.org/abs/2001.04643); [Adam, Kingma & Ba](https://arxiv.org/abs/1412.6980) |
| Linear probes on frozen embeddings; logistic regression and L2 | Every experiment, the gap audit | [BEATs](https://arxiv.org/abs/2212.09058); [PANNs](https://arxiv.org/abs/1912.10211); [Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) (free) |
| Grouped evaluation, bootstrap over groups, AUROC vs. F1 at low prevalence | H1, target panel, all intervals | [scikit-learn grouped CV](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data); [Fawcett 2006, ROC](https://doi.org/10.1016/j.patrec.2005.10.010); [Efron & Tibshirani, bootstrap](https://dblp.dagstuhl.de/rec/books/sp/EfronT93.html); [EchoHawk](https://arxiv.org/abs/2606.29589) |
| Comparing distributions: per-feature W1 vs. joint MMD | CAST coverage, CAST-4B | [Gretton 2012, kernel two-sample test](https://www.jmlr.org/papers/v13/gretton12a.html); [Computational Optimal Transport](https://arxiv.org/abs/1803.00567); [scipy W1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html) |
| Shortcut learning and sim-to-real | The hiss hypothesis; "sounds identical but flips" | [Geirhos 2020, shortcut learning](https://arxiv.org/abs/2004.07780); [Tobin 2017, domain randomization](https://arxiv.org/abs/1703.06907); [Guo 2017, calibration](https://arxiv.org/abs/1706.04599) |
| Outdoor propagation: spreading, ground reflection, air absorption | H2 | [Lamancusa, outdoor propagation notes](https://www.angelofarina.it/Public/Acoustics-Course/Penn-State-Course/10_osp.pdf); [Attenborough 2014](https://link.springer.com/chapter/10.1007/978-1-4939-0755-7_4); [pyroadacoustics paper](https://www.dafx.de/paper-archive/2022/papers/DAFx20in22_paper_9.pdf) ([code](https://github.com/steDamiano/pyroadacoustics)) |
| How vehicles make sound: engine orders, tire/road noise | Physics prototype, signatures | [Tyre/Road Noise Reference Book](https://trid.trb.org/View/730140) (Sandberg & Ejsmont); [Physical Audio Signal Processing](https://ccrma.stanford.edu/~jos/pasp/) (Smith) |

## 5. Reading list

### Start with these five

1. **[DDSP](https://arxiv.org/abs/2001.04643)** (Engel et al. 2020), sections 1–3.
   The analysis-by-synthesis idea CAST is built on.
2. **[BEATs](https://arxiv.org/abs/2212.09058)** (Chen et al. 2022, P07). What the
   main encoder is and what it was trained to keep.
3. **[AI4TEN synthetic-to-real](https://doi.org/10.3390/app16147234)** (Khan,
   Ryzhikov & Kolehmainen 2026, P02). The synthetic-data baseline this project
   reproduced and questions.
4. **[EchoHawk](https://arxiv.org/abs/2606.29589)** (2026). How session leakage
   inflates results, and why groups, not clips, limit evidence.
5. **[A Kernel Two-Sample Test](https://www.jmlr.org/papers/v13/gretton12a.html)**
   (Gretton et al. 2012), section 2. Why matching each feature's range is not
   matching the distribution.

### Project literature

IDs match `literature/literature_matrix.csv` and the citations in reports.

**Physical signatures and military acoustics**
- [P05 Altmann, Linev & Weiß 2002](https://doi.org/10.1016/S0003-682X(02)00021-X): acoustic-seismic vehicle classification
- [P04 Altmann 2004](https://doi.org/10.1016/j.jsv.2003.05.002): signals of heavy military vehicles
- [P10 Wu & Mendel 2003](https://doi.org/10.1117/12.484909): binary ground-vehicle classifiers
- [P08 Wu & Mendel 2007](https://doi.org/10.1109/TFUZZ.2006.889760): acoustic features and fuzzy rules
- [P13 Wu & Mendel 2010](https://link.springer.com/book/10.1007/978-3-642-14084-6): battlefield vehicles from acoustic emissions (book)
- [P09 William & Hoffman 2011](https://digitalcommons.unl.edu/electricalengineeringfacpub/194/): time-domain harmonic amplitudes
- [P12 Guo, Nixon & Damarla 2011](https://eprints.soton.ac.uk/272713/): information fusion

**Representations and invariance**
- [P11 Göksu 2018](https://doi.org/10.1177/0020294018769080): engine-speed-independent signature
- [P15 Wieczorkowska et al. 2018](https://link.springer.com/article/10.1007/s10844-017-0459-2): spectral features for vehicle/engine classification
- [P03 Becker et al. 2020](https://rc.signalprocessingsociety.org/conferences/icassp-2020/spsicassp20vid0334): engine-noise feature extraction (PCEN, modulation)
- [AST, Gong et al. 2021](https://arxiv.org/abs/2104.01778): Audio Spectrogram Transformer
- [P07 BEATs, Chen et al. 2022](https://arxiv.org/abs/2212.09058)

**Domain shift and synthetic-to-real**
- [P16 Gharib et al. 2018](https://arxiv.org/abs/1808.05777): adversarial domain adaptation for scenes
- [P14 Ronchini et al. 2024](https://arxiv.org/abs/2403.17864): text-to-audio synthetic training sets
- [P06 AudioLDM, Liu et al. 2023](https://arxiv.org/abs/2301.12503): the generator behind the AudioLDM bank
- [P02 AI4TEN synthetic-to-real, 2026](https://doi.org/10.3390/app16147234)
- [P01 AI4TEN BEATs/AST fine-tuning, 2026](https://doi.org/10.3390/acoustics8030063)
- [EchoHawk 2026](https://arxiv.org/abs/2606.29589): session-level leakage

## 6. How to learn on this project

Learn by doing the real work. Each open item in `STATUS.md` teaches one piece:

| Task | What it teaches | Status |
|---|---|---|
| Listen to originals vs. CAST re-renders | What the renderer can represent; the 4 kHz ceiling | Done Oct 31 |
| Per-clip classifier verdicts; mel spectrograms by verdict | What BEATs keys on; perceptual vs. feature-space closeness | Done Oct 31 |
| Time-varying renderer weights | Renderer capacity vs. the class cue | Running |
| Review and commit the Oct 25 diagnostics | Linear probes, AUROC, class-direction geometry | Open |
| Review the CAST-4B MMD branch | Marginal vs. joint distributions | Open |
| Email the MVD authors (licence, sessions) | Why independent sessions limit every result | Open |
| Derive the comb-filter notches when H2 restarts | Group delay; implementation effects vs. physics | Later |

**Rules of thumb that keep coming back:**
- **The claim ladder:** reconstruction → coverage → transfer → independent
  confirmation. Evidence at one rung is not evidence at the next.
- **Groups, not clips:** uncertainty is limited by the number of independent
  recording sessions; more clips from the same groups barely help.
- **At low prevalence, prefer AUROC and balanced accuracy over F1:** F1 at a
  fixed threshold mostly measures where the threshold sits.
- **Implementation effects can look like physics:** check gains, delays and
  bandwidth before attributing a result to a mechanism.
- **Sounding the same is not looking the same to the classifier:** a re-render
  can be indistinguishable by ear and still sit far away in the encoder's
  feature space.
- **A fit to a roadside recording captures source, path, background and
  microphone together:** fitted parameters are effective, not physical.

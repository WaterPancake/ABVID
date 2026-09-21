# Passive Acoustic Perception for Autonomous Systems:
# Can Synthetic Audio Transfer to Real-World Vehicle Recognition?

> **DRAFT / TEMPLATE** — Milestone 5/6 results, ABVID project.
> Fill every `[PLACEHOLDER]`, prune the bracketed guidance, then this is ready for arXiv or a blog post.
> Status: draft v0.1. Author: `[PLACEHOLDER: your name / handle]`. Date: `[PLACEHOLDER]`.

---

## Abstract

> [Blog: make this a 3–4 sentence "TL;DR" pull quote. Paper: keep as a formal abstract.]

Passive acoustic sensing can detect and classify vehicles without radiating energy, making it a
natural sensing modality for autonomous systems that must operate covertly or under RF/optical
denial. A central open question is whether models trained on cheap, fully controlled synthetic
audio transfer to real-world recordings — the same sim-to-real problem that dominates robot
learning. We report a systematic, leakage-safe evaluation of tracked-vs-wheeled vehicle
classification across (i) a controlled synthetic augmentation pipeline, (ii) five training
objectives including a paired representation-invariance loss, and (iii) a small but carefully
curated corpus of native real recordings.

Our headline results are deliberately mixed and, we argue, more useful for that reason:

- **In-domain, synthetic → synthetic:** representation-invariance training improves balanced
  accuracy over clean-only supervised training (78.9% vs 77.7%) and substantially increases
  clean/corrupted embedding consistency (cosine similarity 0.845 → 0.916). The objective does
  what it asks.
- **Cross-domain, synthetic → real:** **no** training objective closes the gap. The invariance
  model is in fact the *worst* real-transfer model (12.5% balanced accuracy on held-out native
  real sessions). Augmentation-only training reaches 48.4% but by predicting "tracked" for
  almost everything.
- **Real → real (development only):** a leakage-safe, session-balanced probe over a frozen
  PANNs semantic representation reaches 72.9% mean balanced accuracy, and late fusion with
  classical MFCC/spectral features reaches **77.35%**, with tracked and wheeled recall both
  ~77%. But individual held-out sessions still fail (minimum recalls 22.2% / 29.4%), and this
  is a development result, not an independent test.

**Conclusion:** synthetic training alone is not yet a substitute for real recordings in this
task, and representation-invariance objectives that succeed in-domain can actively hurt
out-of-domain transfer. The bottleneck is source/session dependence and a real corpus too small
(4 tracked + 3 wheeled sessions) to support reliable category-level generalization. We document
the protocol, the failures, and the reproducibility controls so the next group can build on both
the method and the negative results.

---

## 1. Motivation

`[PLACEHOLDER: 1 paragraph. Write this last; it frames everything. Suggested angle below.]`

An autonomous system loitering in an area where traffic is expected needs to answer three
questions before acting: *is there a vehicle, is it tracked or wheeled, and where is it?*
Passive acoustic perception answers all three with a modality that emits nothing — it cannot be
jammed, detected, or visually occluded in the way RF and EO/IR can. But acoustic perception is
also the hardest modality to build training data for: real recordings of military vehicles are
scarce, hard to label, and impossible to collect at the scale modern classifiers want.

The natural answer is synthetic data: an augmentation engine that takes clean recordings and
applies controlled corruption (noise, filtering, clipping, compression, impulse responses,
microphone responses, resampling). This gives us arbitrarily many labeled samples at any desired
SNR. But synthetic data only helps if what the model learns *transfers*. This paper asks that
question directly and honestly, on a small but carefully controlled corpus.

`[Ethics / framing note for the blog version: this is dual-use sensing research, standard in the
open literature since the 1990s, and the same methods serve civilian traffic monitoring and
wildlife acoustics. The detection/classification/localization layer — what this paper covers —
is distinct from any autonomous-targeting application, and we do not address targeting here. See
the ethics note in Section 9.]`

---

## 2. Related Work

- **Unattended ground sensors (UGS).** Passive acoustic vehicle detection and tracked/wheeled
  classification is the canonical UGS task, studied since the 1990s (e.g. acoustic–seismic
  vehicle classification in the disarmament literature; DSIAC UGS surveys). Classic results cite
  tracked-vs-wheeled classification accuracy around 50–60% at ~500 m — evidence this is a hard,
  physically grounded problem, not an easy one.
- **Loitering munitions and acoustic fire control.** `[Cite AeroVironment Switchblade, IAI
  Harpy, Lancet; and the recent "acoustic detection → acoustic fire control" framing in
  counter-UAS literature. Note the transition of acoustic sensing onto UAV platforms, e.g.
  Weles WASP / Quantum Systems Vector AI.]`
- **Synthetic audio for acoustic classification.** `[Cite: AI4TEN (Khan et al., Appl. Sci.
  2026) — physics-based vs AI-generated synthetic training data for car/truck/motorcycle
  classification with cross-dataset evaluation; Synthio (ICLR 2025); text-to-audio augmentation
  (arXiv 2403.17864). Our Milestone 5 protocol resembles AI4TEN's cross-dataset, multi-seed
  design but focuses on the harder tracked/wheeled distinction and a stricter
  grouped-by-recording-session split.]`
- **Sim-to-real in robot learning.** The broader claim that synthetic pretraining reduces the
  need for real data has a strong track record in vision (domain randomization) but is much less
  established in audio. This paper is a negative-result contribution to that gap.

---

## 3. System Overview

### 3.1 The augmentation engine (Milestone 1)

Deterministic, seed-controlled pipeline transforming clean recordings into paired corrupted
observations:

```
clean target
     + background
     + propagation/environment effects
     + microphone effects
     = corrupted observation
```

Implemented effects: random gain; background-noise mixing with controlled SNR; low/high/band-pass
filtering; random EQ; resampling degradation; clipping; dynamic-range compression;
impulse-response convolution; temporal cropping; approximate microphone frequency response.
Every sample is reproducible from its metadata + seed, and every observation carries a paired
`clean_path` / `corrupted_path` and full metadata.

### 3.2 The synthetic corpus (Milestone 3)

- Controlled procedural corpus: `[PLACEHOLDER: exact size; README says a 2,688-observation
  factorial corpus of paired two-second observations was generated for the five-seed runs]`.
- Factorial design: fully crossed operating states and nuisances, `recording_session` as the
  group unit for all splits. `[PLACEHOLDER: confirm the exact vehicle/session inventory]`.

### 3.3 The real corpus (Milestones 5–6)

- 271 reviewed two-second windows from **4 tracked + 3 wheeled recording sessions**.
- Provenance preserved per window: session, source URL, license, source hashes, window
  boundaries, content-review status.
- The corpus audit fails closed unless ≥3 complete sessions per class with complete provenance.

### 3.4 The multichannel layer (Milestone 4, context only)

A four-microphone uniform linear array simulator (`[num_channels, num_samples]`), per-channel
delay/attenuation/impulse/frequency response, exact paired-SNR control, GCC-PHAT and SRP-PHAT
localization, and delay-and-sum beamforming baselines. `[PLACEHOLDER: include the multichannel
results here or keep them in a companion post; they matter for the "autonomous system" framing —
detection + bearing + class — but are secondary to the transfer question this paper answers.]`

---

## 4. Methods

### 4.1 Model family

- **Small CNN:** log-Mel spectrogram → CNN → 64-d pooled embedding → classifier. Same
  architecture, initialization, optimizer, split, and class weighting across all compared
  training objectives.
- **Frozen pretrained encoder:** PANNs Cnn14 (AudioSet). Two representations compared: the
  2,048-d embedding and the 527 AudioSet class outputs. Encoder frozen; only linear probes
  trained. `[Note: PANNs is pretrained on real AudioSet, so PANNs-based results are not a strict
  synthetic-only → real claim.]`
- **Classical branch:** 53-feature MFCC/spectral/statistical feature set + session-balanced
  logistic regression.

### 4.2 Training objectives compared

| Method | Training behavior |
|---|---|
| Standard supervised | Weighted tracked/wheeled cross-entropy on clean inputs |
| Augmentation only | Same cross-entropy on corrupted inputs |
| Paired supervised | Clean/corrupted classification, no explicit consistency term |
| Direct representation invariance | Classification + clean/corrupted cosine consistency (w=0.5), same-vehicle hard positives (w=0.1), metadata-matched hard-negative triplet (w=0.1, margin 0.2) |
| Projected representation invariance | Same invariance objective in a projection head |

### 4.3 Splitting: the leakage rule that everything else depends on

**Never randomly split windows from the same recording across train and test.** All windows from
one original recording belong to one `recording_session` and move together. Every result below
uses grouped, session-level splits. This rule is the difference between a real result and a
self-deception, and it is enforced by the tooling (aggregators reject mismatched commits,
datasets, splits, and configurations).

### 4.4 Evaluation domains

| Domain | Meaning |
|---|---|
| synthetic → synthetic | Train on augmented synthetic events, test on held-out synthetic sessions/corruptions |
| synthetic → real | Train on synthetic only, test on native real recordings (never seen in training) |
| real → real (development) | Fit on complete real sessions, evaluate on fully excluded real sessions (leakage-safe nested scheme) |

`[The AGENTS.md rule applies: never report performance without identifying the test domain.
Synthetic→synthetic and synthetic→real are different claims and must never be combined.]`

---

## 5. Results

### 5.1 In-domain: the invariance objective works on synthetic data

First controlled run (seed 42):

| Test domain | Standard | Augmentation only | Invariance |
|---|---:|---:|---:|
| Seen synthetic corruption | 70.5% | 66.7% | **72.7%** |
| Unseen synthetic noise | 40.0% | **50.0%** | 35.0% |
| Unseen synthetic mic | 70.3% | 65.4% | **76.9%** |
| Unseen synthetic env proxy | **91.3%** | 73.9% | 89.1% |
| All held-out synthetic corruptions | 77.7% | 73.4% | **78.9%** |
| **Native-real held-out sessions** | 29.0% | **48.4%** | 12.5% |

Embedding consistency (what the invariance loss explicitly optimizes):

| Method | Mean paired cosine | Mean normalized L2 |
|---|---:|---:|
| Standard supervised | 0.845 | 0.443 |
| Augmentation only | 0.890 | 0.356 |
| Invariance | **0.916** | **0.333** |

**Interpretation:** the objective achieves exactly what it is written to do on synthetic data —
and then transfers *worst* to real data. Optimizing in-domain consistency does not buy
out-of-domain robustness; in this experiment it appears to trade it away.

### 5.2 Five-seed confirmation (factorial corpus)

Seeds {7, 19, 42, 73, 101}, split seed fixed:

| Method | Synthetic all-corruption BA | Fixed native-real BA |
|---|---:|---:|
| Standard supervised | 75.14% ± 2.30% | 35.97% ± 9.43% |
| Augmentation only | 76.24% ± 0.76% | **38.79% ± 9.54%** |
| Paired supervised | **76.65% ± 1.39%** | 36.75% ± 5.96% |
| Direct invariance | 75.83% ± 1.72% | 35.66% ± 4.76% |
| Projected invariance | 76.60% ± 0.96% | 37.05% ± 6.84% |

No objective provides a repeatable synthetic→real improvement. All five methods average only
0.7–1.3% wheeled recall on the fixed real test — they all learn to predict "tracked".

### 5.3 Pretrained representation (frozen PANNs)

| Representation / probe | Synthetic BA | Fixed real BA | Tracked recall | Wheeled recall |
|---|---:|---:|---:|---:|
| 2,048-d embedding, corrupted | 79.91% ± 0.89% | 31.50% ± 6.18% | 40.00% | 23.00% |
| 2,048-d embedding, paired | 81.28% ± 2.22% | 36.96% ± 9.04% | 31.25% | 42.67% |
| 527 outputs, corrupted | **86.40% ± 1.24%** | 40.00% ± 8.17% | 25.00% | 55.00% |
| 527 outputs, paired | 83.57% ± 3.25% | **47.90% ± 8.76%** | 23.13% | 72.67% |

Adapting only the linear head with all 95 real development windows:

| Metric | Five-seed result |
|---|---:|
| Fixed real BA | **65.50% ± 13.85%** |
| Tracked / wheeled recall | 65.00% / 66.00% |

The data are the problem: 88 tracked windows came from 2 sessions; 7 wheeled windows came from 1
session. Adaptation fractions below 100% give a non-monotonic learning curve — `[this is the raw
material for a strong figure: learning curve vs % real data, showing it does NOT behave like
the classic "pretrain + fine-tune" curve]`.

### 5.4 Real → real, leakage-safe (development only)

Nested scheme: every outer fold excludes one complete tracked + one complete wheeled session
(all 4×3 = 12 pairs); regularization and threshold chosen only on inner held-out session pairs.

| Stage | Mean BA | Tracked recall | Wheeled recall | Notes |
|---|---:|---:|---:|---|
| Semantic probe (35 AudioSet outputs) | 72.92% | 89.81% | 56.03% | 8/12 pairs both sessions correct; Ford Model T near-zero |
| **Late fusion** (semantic + classical, 0.5/0.5) | **77.35%** | **77.42%** | **77.28%** | First leakage-safe run to pass both mean gates; min recalls 22.2%/29.4% |

**Key qualitative finding:** the two branches make complementary errors. The semantic PANN branch
recognizes Goodwood and Maserati but rejects the Ford Model T; the classical MFCC/spectral branch
recognizes Ford but fails Maserati. Fusing them raises wheeled recall by 21 points at the cost of
12 points of tracked recall.

### 5.5 Milestone 7 gates (why the average is not enough)

| Gate | Requirement | Best evidence | Status |
|---|---|---|---|
| ≥5 reviewed sessions/class | — | 4 tracked / 3 wheeled | Fail |
| Mean BA ≥ 75% (nested unseen session) | 77.35% | | **Pass** |
| Mean recall ≥ 70% both classes | 77.42/77.28 | | **Pass** |
| No session below 50% recall | 22.2/29.4% | | **Fail** |
| Confirm on newly admitted pair | none yet | | Fail |

Two average-performance gates pass; the individual-session gate fails catastrophically. A model
that can fail an entire unseen recording session is not a deployable classifier, whatever its
mean says.

---

## 6. Discussion: why transfer is hard here

1. **Source/session dependence dominates the mean.** Every method nearly fails Maserati while
   doing well on Ford and Goodwood. The classifier is memorizing recording conditions, not a
   robust tracked/wheeled invariant.
2. **In-domain invariance ≠ out-of-domain robustness.** The invariance objective moved
   clean/corrupted embeddings together but did not move synthetic/real together. Consistency in
   the training distribution can be satisfied trivially in ways that do not generalize.
3. **Data scarcity compounds it.** Seven real sessions is not enough to separate "vehicle class"
   from "recording session". 88 tracked windows from two sessions cannot support category-level
   claims.
4. **The augmentation engine models microphone and environmental effects, but not the full
   sim-to-real gap.** `[PLACEHOLDER: expand — what effects are missing? (e.g. real-world
   reverberant scenes, Doppler, sensor self-noise, propagation over terrain, source variability).
   This is the honest gap analysis that makes the negative result constructive.]`

---

## 7. Reproducibility

Every formal experiment records: git commit, configuration, random seed, dataset version, grouped
train/test split, model checkpoint, and metrics — in machine-readable form. Notable controls:

- leak-safe grouped splits enforced by tooling; aggregators reject mismatched commits/datasets/
  configurations;
- five-seed ablations with fixed split seed;
- a frozen confirmation protocol: one global model frozen (threshold 0.25) before admitting any
  new data, to be evaluated **exactly once** on a genuinely new provenance-complete tracked/
  wheeled pair;
- full suite passes: 70 tests at rundown time. `[PLACEHOLDER: re-run pytest and report current
  count]`.

---

## 8. Future work / what would change the conclusion

- **More independent real sessions** (target ≥5/class, ideally more); reserve a locked pair and
  run the frozen evaluator once. This is the single highest-value next experiment.
- **Domain randomization across the missing effects** (Section 6.4) rather than fixed
  corruption distributions.
- **Closer sim-to-real modeling** of the sources themselves (procedural engine models), and
  measuring whether that closes the gap that augmentation alone does not.
- **Milestone 8 integration:** stream-in → detect → localize (GCC-PHAT/SRP-PHAT) → classify →
  confidence aggregation, with latency/compute measurements — the step that turns this research
  into a system, and the natural subject of a follow-up post.

---

## 9. Ethics and dual-use note

`[Keep this short and matter-of-fact. Suggested content:]`

This work concerns passive acoustic **sensing, detection, and classification** — a modality and
task studied in the open literature for decades, with substantial civilian application (traffic
monitoring, noise analysis, wildlife acoustics). The methods here are dual-use, as is virtually
all sensing and signal-processing research. We do not address targeting, weapons employment, or
any use of this information to direct force; the "autonomous system" in the title refers to the
sensing/perception layer. We publish honest negative results specifically so that claims about
what acoustic vehicle classification can do are not oversold.

---

## 10. References

`[PLACEHOLDER: full citations. Seed list — verify and expand before publishing.]`

- Khan, Ryzhikov, Kolehmainen. *AI4TEN: Synthetic-to-Real Transfer for Acoustic Vehicle
  Classification Using Physics-Based and AI-Generated Training Data.* Appl. Sci. 2026, 16(14),
  7234. https://doi.org/10.3390/app16147234
- *Synthio: Augmenting small-scale audio classification with synthetic data.* ICLR 2025.
- *Synthetic training set generation using text-to-audio models.* arXiv:2403.17864.
- Acoustic–seismic detection and classification of military vehicles (disarmament literature,
  2000s). `[full cite]`
- DSIAC Unattended Ground Sensor Survey. `[full cite + the ~50–60% tracked/wheeled figure]`
- Kong et al. *PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern
  Recognition.* (AudioSet Cnn14.)
- `[UGS theses from whiterose.ac.uk and UNL digital commons, if cited in Section 2]`

---

## 11. Appendix: how this maps to the repo

| Paper section | Repo artifact |
|---|---|
| §3.1 augmentation engine | `src/vehicle_audio/augment.py`, `configs/*.yaml` |
| §3.2 synthetic corpus | `src/vehicle_audio/factorial_dataset.py` |
| §3.3 real corpus | `configs/real_corpus.yaml`, `data/real_eval_v2/` |
| §3.4 multichannel | `src/vehicle_audio/multichannel*.py` |
| §4.2 objectives | `src/vehicle_audio/invariance_*.py` |
| §5 results | `docs/milestone6_results_rundown.md`, `runs/m6_*/` |
| §7 reproducibility | `scripts/evaluate_*.py`, `scripts/aggregate_invariance.py` |

---

## Open TODOs before this is publishable

- [ ] `[PLACEHOLDER]` — author byline, affiliation, date.
- [ ] `[PLACEHOLDER]` — exact synthetic corpus size + session inventory (check `runs/` and manifests).
- [ ] `[PLACEHOLDER]` — fill Related Work citations (verify each, add DOIs).
- [ ] `[PLACEHOLDER]` — decide: include Milestone 4 multichannel results in this paper or a companion post.
- [ ] `[PLACEHOLDER]` — the learning-curve figure (BA vs % real data) and embedding-PCA figure.
- [ ] `[PLACEHOLDER]` — re-run test suite, record current count.
- [ ] `[PLACEHOLDER]` — blog version: cut to ~1,500 words, add a "why this matters" hook and plain-language figures.

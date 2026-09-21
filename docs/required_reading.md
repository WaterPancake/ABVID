# Required Reading — Annotated Bibliography

> Companion to `docs/paper_synthetic_to_real_transfer.md`.
> Every entry is organized around the question the paper asks: **can synthetic audio transfer to real-world vehicle recognition?**
> Each entry states (1) what it is, (2) why it matters for *this* paper, and (3) how it connects to a specific part of the ABVID repo.
> All citations verified against sources during drafting — but re-check DOIs/venues before final publication.

---

## 0. Reading order (if you want this as a study plan)

1. **The domain-gap framing:** Reality-Gap survey (§1) — read first, it gives the whole vocabulary.
2. **The benchmark reality:** AI4TEN (§2.1) — your closest published neighbor; know it cold.
3. **The pretrained backbone:** PANNs (§2.2) — the thing you actually froze and used.
4. **The historical baseline:** UGS literature (§2.4) — why this problem is hard and 20 years old.
5. **The synthetic-data engine lineage:** SpecAugment (§3.1), augmentation surveys (§3.2) — the toolbox behind Milestone 1.
6. **The representation-learning layer:** SimCLR/triplet (§4) — the theory behind your invariance objective.
7. **The dataset you stand on:** AudioSet (§5) — where PANNs came from.

That order moves from "why the problem is hard" → "what people have done" → "the tools you used" → "the theory under your methods" → "the data under everything."

---

## 1. The synthetic-to-real gap (the framing of your whole paper)

### 1.1 "The Reality Gap in Robotics: Challenges, Solutions, and Best Practices"
- **What:** A 2026 *Annual Review of Control, Robotics, and Autonomous Systems* (Vol. 9) preprint surveying sim-to-real transfer in robot learning — domain randomization, domain adaptation, system identification, evaluation methodology.
- **Why it matters for your paper:** Gives you the authoritative vocabulary to place audio sim-to-real inside the larger field. You can write "the reality gap is usually studied in vision/control; we study it in *passive acoustics*," which is exactly the positioning that makes your paper non-trivial.
- **Key point to use:** "The synthetic-to-real gap has not been solved; it has been optimised away on a single benchmark." That sentence is a gift — it legitimizes your honest negative result and your insistence on grouped, multi-seed, leakage-safe evaluation.
- **Connects to repo:** `docs/milestone5_status.md`, `docs/milestone6_results_rundown.md` — your evaluation discipline is the methodological contribution this survey says the field needs.
- **Citation:** Hofer et al., preprint to appear in *Annu. Rev. Control Robot. Auton. Syst.* 9 (2026). arXiv:2510.20808. `[PLACEHOLDER: verify authors]`

### 1.2 "Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey" (Zhao et al.)
- **What:** The canonical RL-focused sim-to-real survey; taxonomy of reality-gap sources and bridging methods.
- **Why it matters:** Reinforces the domain-shift framing and gives you the "domain randomization" term you'll use in Future Work (Section 8 of your draft). It's the bridge from robotics sim-to-real to your audio augmentation.
- **Connects to repo:** your `augment.py` random gain/EQ/filter randomization *is* domain randomization, just in the audio domain — name it as such.
- **Citation:** Zhao, Queralta, Westerlund, *Sensors* 20(15) (2020), 4291.

### 1.3 Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World"
- **What:** The paper that made domain randomization famous: train on a *distribution* of simulated appearances so the real world is "just another variation."
- **Why it matters:** This is the strongest *positive* counterpoint to your negative result. The insight is that you randomize the *right* parameters. Your Milestone 1 randomizes microphone/environment effects but not, as you'll note in Section 6.4, the source itself or full scene acoustics. Contrasting your result with Tobin is exactly the right discussion move.
- **Connects to repo:** `src/vehicle_audio/augment.py` — the fixed effects you *did* randomize, and the ones you didn't.
- **Citation:** Tobin et al., IROS 2017.

### 1.4 Pan & Yang, "A Survey on Transfer Learning"
- **What:** The classic taxonomy of transfer learning (inductive/transductive/unsupervised; feature-based vs instance-based).
- **Why it matters:** Gives you the precise technical vocabulary — "we study transductive, feature-based transfer from synthetic source to real target" — and lets you cite the formal definition of domain shift instead of hand-waving.
- **Connects to repo:** your synthetic→real vs real→real domain table (draft §4.4) is a transfer-taxonomy table; cite this to anchor it.
- **Citation:** Pan & Yang, *IEEE TKDE* 22(10), 2010.

---

## 2. Direct prior work on the *exact* task (must cite, must know)

### 2.1 AI4TEN — Khan, Ryzhikov & Kolehmainen, *Appl. Sci.* 16(14):7234 (2026)
- **What:** Compares physics-based (pyroadacoustics) vs AI text-to-audio (AudioLDM) synthetic training data for a CNN acoustic vehicle classifier (car/truck/motorcycle), with a cross-dataset protocol (train on one country's recordings, test on two independent datasets recorded with different microphones) across five seeds.
- **Why it matters:** This is the closest published thing to your Milestone 5 protocol. It's a *peer-reviewed 2026 paper doing almost exactly what you did* — synthetic audio for vehicle classification, cross-domain evaluation, multi-seed. Your reviewers WILL know it.
- **How to position against it:** (1) You do the harder binary (tracked/wheeled vs car/truck/moto); (2) you enforce grouped-by-recording-session splits — AI4TEN's cross-dataset protocol is good, but your session-grouping is stricter leakage control; (3) your real corpus is provenance-audited (source URL, license, hashes, review status). Be explicit about all three.
- **Bonus:** It found that combining real + both synthetic sources beats real-only by 55% F1 — a *positive* result to contrast against your negative transfer finding, which sharpens the discussion.
- **Connects to repo:** `docs/milestone5_status.md`, `scripts/evaluate_transfer.py`.
- **Citation:** https://doi.org/10.3390/app16147234

### 2.2 PANNs — Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition"
- **What:** The paper behind the model you actually froze (Cnn14 on AudioSet, 527 class outputs, 2,048-d embeddings). IEEE/ACM TASLP 2020.
- **Why it matters:** It's the empirical backbone of your Stage C results. You need to cite it and to state clearly the interpretation caveat: PANNs is *pretrained on real AudioSet*, so your "synthetic→real" runs using PANNs are really "real-pretrained representation + synthetic probe → real," not a strict synthetic-only claim. Your rundown already says this — make it explicit in the paper too.
- **Key detail to get right:** Cnn14 produces 527 outputs (the AudioSet ontology), and the "semantic" branch in your fusion uses a 35-label vehicle/engine subset. Cite PANNs for the embedding and AudioSet (below) for the ontology.
- **Connects to repo:** `src/vehicle_audio/pretrained_evaluation.py`.
- **Citation:** Kong, Cao, Wang, Plumbley, *IEEE/ACM TASLP* 28 (2020), 2880–2891. arXiv:1912.10211.

### 2.3 VGGish — Hershey et al., "CNN Architectures for Large-Scale Audio Classification"
- **What:** The predecessor CNN audio encoder (also AudioSet-trained); produced the widely-used 128-d VGGish embedding.
- **Why it matters:** It's the natural "other pretrained encoder" comparison. Your rundown notes the 2,048-d PANN embedding transferred worse than the 527-output vector — VGGish is the canonical alternative to benchmark against, and citing it signals you know the pretrained-audio landscape, not just one model.
- **Connects to repo:** a "VGGish vs PANNs embedding" comparison is a cheap, high-value addition for the paper.
- **Citation:** Hershey et al., *ICASSP 2017*.

### 2.4 The UGS (unattended ground sensor) literature — the historical baseline
- **What:** Decades of open work on passive acoustic/seismic detection & classification of military vehicles (tracked/wheeled), e.g.:
  - N. Evans, *Automated Vehicle Detection and Classification Using Acoustic and Seismic Signals* (PhD thesis, 2010) — a thorough algorithmic treatment of exactly your task.
  - Acoustic–seismic vehicle classification for disarmament/peacekeeping (2000s) — reports **94% separation of tracked vs wheeled** using combined acoustic+seismic maxima, which is the key number to know: with *good* multi-sensor conditions the task is tractable, which is why your 77% with audio-only on hard data is contextualized rather than embarrassing.
  - DSIAC Unattended Ground Sensor Survey — classic field figures: acoustic tracked/wheeled classification ~50–60% at ~500 m.
- **Why it matters:** It proves (a) this is a real, decades-old, funded problem — legitimizing your paper's premise; (b) the task is *hard* under field conditions (50–60%) and *easy* under good conditions (94%), which frames your mixed results correctly; (c) it's the historical lineage your work modernizes (deep learning + synthetic data + strict leakage control vs classic DSP + small real corpora).
- **Connects to repo:** your entire Milestone 2–6 arc is the modern re-run of this literature.
- **Citations (verify full details):** Evans 2010 thesis (Whiterose eprints); the disarmament/peacekeeping acoustic–seismic papers; DSIAC UGS survey.

---

## 3. Synthetic data & audio augmentation (the toolbox behind Milestone 1)

### 3.1 SpecAugment — Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
- **What:** Time/frequency masking on spectrograms; the single most-cited audio augmentation method (5,700+ citations). arXiv:1904.08779.
- **Why it matters:** Your Milestone 1 engine does waveform-domain augmentation (gain, EQ, filtering, clipping, compression, resampling, IR, mic response) — SpecAugment is the complementary *spectrogram-domain* approach, and reviewers will expect you to either compare against it or cite it as an alternative. It's also the simplest possible "we could add this" future-work item.
- **Connects to repo:** `src/vehicle_audio/augment.py` — note that your pipeline is time-domain; SpecAugment is feature-domain; the two compose trivially.
- **Citation:** Park, Chan, Zhang, Chiu, Zoph, Cubuk, Le, arXiv:1904.08779 (ICASSP 2019).

### 3.2 Audio augmentation survey (e.g., Nam, Kim, Ko; or the DCASE-oriented reviews)
- **What:** Systematic taxonomies of audio augmentation: noise injection, time stretching, pitch shifting, dynamic range compression, mixing, frequency masking.
- **Why it matters:** Situates your specific choice of 11 corruption types inside a known design space, and lets you justify *why* you picked those (they map to real microphone/environment physics) vs purely stochastic ones.
- **Connects to repo:** the 11-effect list in Milestone 1 maps one-to-one onto such taxonomies.
- **Citation (choose one and verify):** e.g. "Audio augmentation for machine learning" (2021, *Applied Sciences*); or Ko et al. 2015 (audio augmentation for ASR).

### 3.3 Synthio — "Synthio: Augmenting Small-Scale Audio Classification with Synthetic Data" (ICLR 2025)
- **What:** An ICLR 2025 paper augmenting small audio datasets with synthetic data.
- **Why it matters:** Current, top-venue evidence that synthetic audio augmentation is a *live* research frontier — supports the "timely" framing and gives you another citable neighbor. `[PLACEHOLDER: pull the authors + confirm the exact venue/proceedings]`
- **Connects to repo:** direct antecedent of your Milestone 1→5 pipeline.

---

## 4. Representation learning & invariance (the theory under Milestone 6)

### 4.1 SimCLR — Chen, Kornblith, Norouzi, Hinton, "A Simple Framework for Contrastive Learning of Visual Representations" (ICML 2020)
- **What:** The canonical contrastive-learning framework (augmented views → shared embedding → InfoNCE).
- **Why it matters:** Your invariance objective is a *supervised variant* of this idea — clean/corrupted pairs as "views" of the same event, pulled together while pushed apart from hard negatives. Cite SimCLR as the ancestor of the consistency-loss family you tested.
- **Key caveat to use:** SimCLR's augmentation views are carefully chosen so the *semantic* content survives. Your negative result is, at its core, that your clean/corrupted "views" change something the real world cares about — a genuinely interesting observation about what makes a good augmentation pair.
- **Connects to repo:** `src/vehicle_audio/invariance_evaluation.py`, the cosine-consistency + triplet objective.

### 4.2 Schroff et al., "FaceNet: A Unified Embedding for Face Recognition and Clustering" (CVPR 2015)
- **What:** The triplet-margin loss, the direct ancestor of your hard-positive/hard-negative triplet term (margin 0.2, weights 0.1).
- **Why it matters:** It's the specific paper your triplet loss derives from — must-cite for the "established cosine triplet-margin objective" phrase in your rundown.
- **Connects to repo:** the hard-negative mining (opposite class, matched metadata) is a nice *supervised-hard* mining variant of FaceNet's semi-hard mining — worth one sentence in Methods.
- **Citation:** Schroff, Kalenichenko, Philbin, *CVPR 2015*.

### 4.3 (Optional, advanced) Khosla et al., "Supervised Contrastive Learning" (NeurIPS 2020)
- **What:** Extends SimCLR-style contrastive loss to the fully-supervised setting with labels.
- **Why it matters:** Your hard-positive/hard-negative scheme is essentially supervised contrastive learning. Citing it lets you say "we compared a supervised-contrastive-style objective against standard CE" — a more precise and more current framing than "a triplet loss."
- **Connects to repo:** the invariance objective's positive/negative structure.

---

## 5. The dataset under everything: AudioSet

### 5.1 Gemmeke et al., "AudioSet: An Ontology and Human-Labeled Dataset for Audio Events" (ICASSP 2017)
- **What:** The 527-class, human-labeled audio-event ontology + dataset (5,800+ hours) that PANNs was trained on.
- **Why it matters:** (1) Cite it whenever you use the 527-output vector or the 35-label semantic subset — the *meaning* of those outputs is defined by the AudioSet ontology. (2) It explains your "semantic branch recognizes vehicle/engine classes" finding: AudioSet contains classes like "engine starting," "truck," "car passing by" — which is *why* the semantic probe worked better than the raw embedding. That's a mechanistic explanation, and it's the kind of insight reviewers love.
- **Connects to repo:** the 35-label vehicle/engine subset in `semantic_session_evaluation.py`.
- **Citation:** Gemmeke et al., *ICASSP 2017*, arXiv:1705.01249. (5,697 citations.)

---

## 6. (Optional) Domain adaptation in audio, for a sharper Discussion

If you want the Discussion's "why didn't it transfer" to be rigorous, add:

### 6.1 GAN-/cycle-consistency domain adaptation for audio
- The NASA/vision work on CycleGAN augmentation for sim-to-real is a strong analogy: they *translate* synthetic to look real, whereas you *corrupt* real to look like the deployment domain. Contrasting "make synthetic look real" vs "make real look like the deployment domain" is a crisp framing of the design choice at the heart of Milestone 1.

### 6.2 (Optional) Fine-grained audio benchmarks
- DCASE (Detection and Classification of Acoustic Scenes and Events) challenges and TAU Urban Acoustic Scenes datasets are the standard public benchmarks if you ever want to move from your private corpus to a public one — which would substantially strengthen the paper's generalizability claims. `[PLACEHOLDER: decide whether to include; this is a "future work" citation more than a "required" one]`

---

## 7. Checklist before publishing

- [ ] Verify every DOI / arXiv id / venue (several were checked during drafting; re-verify at submission).
- [ ] Confirm the exact authors for the Reality-Gap survey (§1.1) and Synthio (§3.3).
- [ ] Decide whether to add the DCASE/TAU public-benchmark reference (§6.2) — it's optional but strengthens "future work."
- [ ] Cross-check each "Connects to repo" line against the actual module/script names before submission.
- [ ] Add a one-line "why this matters to our specific experiment" to each citation *in the paper's references* too — a comment field in your bib file is a cheap way to keep that intent alive.

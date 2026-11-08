# ABVID status

Updated 2026-11-08. This page condenses the project's reports down to what is
still open or needs a decision. The original reports remain the detailed record
(settings, hashes, uncertainty). Most are in `archive/`; see the source map at
the end.

**The question:** which parts of an acoustic vehicle simulator must be realistic
for synthetic training data to help a car/truck classifier on real recordings?

**Where it stands:** the headline question (source vs. propagation) is still
unanswered. The H2 propagation effect was an implementation artifact, and CAST
adds no classification value in its current form. The frozen B1 reference
recipe is the first gain to hold outside IDMT: BEATs on MELAUDIS goes 61.6 → 73.3
AUROC. Every result so far is development evidence on already-exposed data.
Nothing has been confirmed on independent data.

## 1. Needs your action

These are owed reviews and decisions only you can make.

1. **Open a PR from `cast/diagnosis-20261102` to `main`.** The branch was
   pushed on 2026-11-08 (it starts from `3cc64ac`). It carries the
   Oct 25 – Nov 8 work:
   - `45d79c5`: report archive, link fixes, this page, `AGENTS.md`,
     `docs/LEARNING.md`
   - `f18a176`: CAST diagnosis code, frozen protocols, run scripts and the
     Oct 25 reports
   - `ec4ed5f`: the MVD download helper
   - `24f70ae`: the B1 reference recipe and the MELAUDIS check
   - `1153aea`: merges the CAST-4A serialization fix (`6a76b22`) and the
     CAST-4B MMD metric (`c60a682`). You refreshed the harness baseline
     in that commit: cast 723 → 748, baseline 756 → 781. These are the
     25 new tests in `tests/cast/test_cast4a_serialization_revision.py`
     and `tests/cast/test_joint.py`; name both in the PR description.
   - the commit after it: the two branch reports moved to `archive/`, their
     out-of-date delivery sections corrected, and this update

   Both review sets in `docs/CODE_REVIEW_PENDING.md` are now done.

   **Gaps:**
   - The new CAST modules (`stage_diagnostic`, `time_varying`,
     `utility_gap_audit`) have no tests.
   - `stage_diagnostic/report.py` still names the old `reports/` output paths.

   **Left untracked on purpose:** `.claude/`, `docs/.obsidian/`,
   `annotation.json`, `ABVID_CAST_core_concepts.md` and
   `docs/SIGNATURE_FOCUS_PLAN.md` (the last waits on the branch policy).
2. **Decide whether to rerun CAST-4A coverage.** The serialization fix is
   merged (`1153aea`). The original run failed only because libsndfile writes a
   timestamp into float WAV headers, which broke the byte-exact replay check;
   samples are now stored as `.npy`.
   - **Cost of the rerun:** all 1,864 fitted parents are reusable; only generation
     and scoring would be rerun.
   - **The question:** is it still worth running? It measures coverage, which
     CAST-5A already showed does not turn into classification value.
   - **CAST-4B MMD** (also merged) catches joint-distribution differences that
     the per-feature metric misses. It is verified on synthetic fixtures only,
     and using it on real data needs its own frozen experiment.
3. **Decide the branch policy.** The project memory says the military side project
   stays on `learned-audio-signature`, never merged wholesale. PRs #2 and #3
   merged that branch into `origin/main`. Pick one:
   - accept the merged state, or
   - restore the separation.

   `AGENTS.md` still describes the military program as retired. Separately, local
   `main` (`9faba71`) is behind `origin/main` (`3cc64ac`).
4. **Military P1: do the blind interval review.**
   - **What's waiting:** 54 recordings (2 h 26 min) across 7 caption-hidden packets.
   - **Why it matters:** until you mark the foreground/background intervals, the
     side project has zero admitted segments and zero verified independent
     recording groups, and nothing downstream can run.
   - **How to start:** the review server is documented only as
     `http://127.0.0.1:8766/`, with no launch command recorded.
5. **Housekeeping.**
   - **Stray file:** `annotation.json` at the repo root is a synthetic UI check
     from Oct 11.
   - **Untracked:** `docs/.obsidian/` and `.claude/` (ignore or commit).
   - **Stale branches:** `github-publication` and `list`.
   - **Study material:** `ABVID_CAST_core_concepts.md` is untracked, like the
     deleted course.

## 2. Scientific next steps

Ordered by how much each would change the picture. Each needs its own frozen
protocol before running.

1. **B1 recipe: the reference baseline (2026-11-03); positive on MELAUDIS for BEATs (2026-11-04).**
   - **The finding:** within IDMT, held-out-group AUROC rose from 71.5 to 84.1
     using full-band input (native → 16 kHz, not the 8 kHz route), all training
     data with balanced class weights, and strong L2 (C ≈ 0.001–0.01).
   - **Encoders:** MFCC26 generalized best; the BEATs layer mean was close.
   - **What's left:** this was recommended but never frozen as the reference.
     Every earlier synthetic comparison ran on the weaker 8 kHz route.
   - **Frozen and run (2026-11-03, `b1_reference_v1`): this is now the
     real-only reference.** All later "real-only reference" comparisons use it
     unchanged. The recipe code is in `src/abvid/learning/reference.py`.
     - **Recipe:** full-band input, all training data with balanced weights,
       MFCC26 and BEATs layer mean reported separately. C is chosen inside the
       training groups.
     - **IDMT held-group AUROC:**
       - **BEATs layer mean:** 84.9 [80.1, 89.4], or +13.4 [9.6, 18.1] over
         H1's reference. BA 77.1, AP 48.4, macro-F1 63.6; worst group 76.2
         (`b00d69dd`, 10 trucks). It chose C 0.001 in every fold.
       - **MFCC26:** 86.7 [82.4, 91.0], or +15.2 [8.2, 23.6]. BA 77.8,
         AP 48.1, macro-F1 60.1; worst group 79.1 (`e1daf436`). It chose C 1
         or 0.1.
       - Site-block intervals are in the run record.
     - **No more memorizing:**
       - **BEATs:** training AUROC is 90.9, against 100 under H1, and only
         6 points above its held score.
       - **MFCC:** training AUROC 86.0, held 86.7.
       - **Why it matters:** the confound behind the CAST stage diagnosis's
         100 → 68 should largely disappear under this recipe.
     - **Threshold still off:** macro-F1 at the argmax rule stays around
       60–64. In the two small-truck groups every truck is caught but car
       recall drops to 48–61%, a threshold effect at about 10% truck
       prevalence.
     - **Optimistic:** the route, budget and representations were picked after
       B1's outer results were seen. Only C is nested.
     - **Correction:** the frozen protocol calls the later cross-dataset check
       "the untouched test". That is wrong for MELAUDIS: H1 already scored
       there, so it is exposed development data. The only untouched target is
       sealed DCASE.
     - **All-IDMT heads** (trained on all six groups, scored on nothing):
       - BEATs: C 0.001, `heads/beats_avg.joblib`, SHA256 `ff49f793…`
       - MFCC: C 1, `heads/mfcc26.joblib`, SHA256 `0d770904…`
     - **Run record:** `artifact:baseline_strength/b1_reference_20261103_v1`
       (`results.json` `7f1ebc2c…`).
       - **Protocol:** `experiments/baseline_reference/PROTOCOL.md` (SHA256
         `1950f4e1…`).
       - **Code:** `run.py` `6b2b61f1…`, `reference.py` `02ba715d…`,
         `grouped_search.py` `7fea0de6…`; commit `daf72e9` plus a dirty tree.
       - **Inputs:** B1 features `aab4b3aa…` and results `46399d34…`.
       - **Statistics:** 10,000 fold and site-block draws, seed 20261103.
       - **Gates:** 288/288 fits reproduce B1 exactly; feature re-extraction
         difference 0.0.
     - **Gap:** `reference.py` has no tests. Adding them needs your
       human-only harness-baseline refresh.
   - **B1 on MELAUDIS (2026-11-04, `b1_melaudis_v1`): positive for BEATs.**
     This is the first gain that holds outside IDMT, on exposed development
     data.
     - **Primary:** IDMT-trained BEATs on MELAUDIS reaches AUROC 73.3
       [70.7, 82.9], against H1's 61.6. The gain is +11.7 [+6.6, +17.4]
       (paired whole-group bootstrap; no draws rejected).
       - **All four groups improve:** 84.4 vs 73.0, 78.5 vs 62.8, 71.1 vs
         59.5 (the 6,477-event group), and 93.2 vs 88.6 (the 1-truck group).
       - **Other BEATs measures, B1 vs H1:** BA 68.5 vs 57.4 (+11.1
         [6.7, 15.7]); AP 8.3 vs 5.7 (2.6× vs 1.8× prevalence); truck recall
         at 5% car false-positive rate 18.4 vs 11.9.
     - **MFCC is unresolved:** 68.9 vs 64.6, a gain of +4.3 [−8.6, +9.8].
       MFCC was the best encoder within IDMT (86.7) but loses 17.8 points on
       MELAUDIS, while BEATs loses 11.6. The pretrained representation
       carries over better; this is descriptive, from one target dataset.
     - **Still weak in absolute terms:** AP is only 2.6× prevalence. At the
       fixed rule, car recall is 59.7% at 3.2% trucks, so the threshold is
       still off.
     - **What it supports:** B1-BEATs is now a candidate for the one-time
       DCASE pass (item 6).
     - **Run record:** `artifact:baseline_strength/b1_melaudis_20261103_v1`
       (`results.json` `711d93e9…`).
       - **Protocol:** `experiments/b1_melaudis/PROTOCOL.md` (SHA256
         `ed8ca4f4…`, frozen 2026-11-04 on user approval).
       - **Code:** `run.py` `21aeda80…`, `reference.py` `02ba715d…`,
         `ranking.py` `eb726e1e…`.
       - **Features:** `c81af8a5…`.
       - **Heads:** `ff49f793…` (BEATs), `0d770904…` (MFCC).
       - **Statistics:** 10,000 draws, seed 20261104.
       - **Gates:**
         - all 8,066 members match both admission hashes, and the extracted
           copies were deleted;
         - the frozen heads, the target-panel scorer replay and the feature
           path are all exact.
2. **CAST: reconstruction and sampling both lose signal (revised 2026-11-01).**
   - **The stage diagnosis:** with the real-trained head scored on its own 380
     training clips (AUROC):

     | Stage | AUROC |
     |---|---:|
     | Original clips | 100 |
     | CAST reconstruction | 68 |
     | After calibration | 77 |
     | Population samples | 63 |

     The stage diagnosis read this as most class structure being lost in the
     fit. The Nov 1 held-out check (below) revises that: about 70% of the
     100 → 68 drop was memorized detail. On new recordings, re-rendering costs
     about 10 BEATs AUROC points, half the above-chance signal.
   - **Recommended change (my recommendation, not a frozen plan):** a full-band
     renderer. CAST synthesizes at 8 kHz, so it has no content above 4 kHz,
     and B1 shows that band matters.
   - **Caveat (measured on Nov 1):** the 100 comes from an overfit head. The
     held-out check puts the memorized share of the drop at about 70%.
   - **Listening check (2026-10-31, informal, one listener):**
     - smooth8 re-renders are often indistinguishable from the originals by ear,
       so the loss is not in the audible character.
     - The full-band versions add a uniform hiss on every clip, which looks like
       the recording noise floor. A full-band renderer would mostly be modeling
       that noise floor (environment/sensor), not the vehicle.
     - v11 with fresh noise sounds the same as v11.
     - **Untested hypothesis:** after unit-RMS normalization, the hiss level
       relative to the vehicle tracks vehicle loudness or site, which could be
       part of B1's full-band gain.
   - **Per-clip verdicts** (saved real-only heads on saved features; they
     reproduce all 44 published stage metrics exactly):
     - BEATs no longer recognizes about half the re-rendered trucks: 91/190 at
       smooth8 and 89/190 at v11 are called car. Calibration mainly repairs cars
       (57 → 34 flipped).
     - MFCC fails the other way: re-rendered cars become truck (90/144 at
       smooth8, 39/144 at v11), and trucks stay correct.
     - The CAST mismatch therefore has a direction, and that direction depends
       on the representation.
   - **Mel spectrograms** (BEATs's own filterbank; groups by BEATs verdict on
     smooth8: 62 very-wrong vs 69 still-right trucks; descriptive, not a test):
     - Average spectra match closely in both groups.
     - The clearest group difference is over time. For very-wrong trucks the
       fit keeps less of the original's frame-to-frame variation at 100–400 Hz
       (median ratio about 0.8–0.95 vs about 1.0).
     - All groups lose about 30% of their variation near 1–1.5 kHz.
     - Cars show no clear group difference.
     - This fits a structural limit in `src/abvid/cast/renderer.py`: harmonic
       and noise-band weights are fixed for the whole clip. Only pitch
       (3 points) and overall loudness (5 points) change over time, so the
       spectral balance cannot evolve during a pass-by.
   - **Time-varying weights tested (2026-10-31, `cast_tv_weights_v1`):**
     **small or ambiguous under the frozen rule.**
     - **Setup:** harmonic weights, noise-band weights and mix get 3 knots
       (59 controls instead of 25). Same fitter, starts, budget and noise
       streams as smooth8. Same 380 training parents and saved heads.
     - **Primary:** BEATs R-head AUROC 68.2 → 71.1 (+3.0). The frozen bar for
       "promising" was +5.
     - **Shift, not separation:** BEATs truck recall rose 53 → 70 while car
       recall fell 72 → 59, and BA rose only 62.8 → 64.5.
     - **Groups:** BEATs is mixed across the five groups (+4.0, −2.4, +3.1,
       −7.1, +3.3).
     - **MFCC improved more consistently:** AUROC 76.9 → 81.3, BA 59.7 → 70.0,
       and no group got worse.
     - **Mechanism:** the renderer did what it was designed to do. Every
       parent's check loss went down (median 1.374 → 1.321), and the engine-band
       variation kept rose from 0.91 to 0.97 in the very-wrong trucks. So fixed
       spectral weights explain only a small part of the lost BEATs structure.
     - **Run record:** `artifact:cast/diagnostics/cast_tv_weights_20261031_v1`.
       Protocol `experiments/cast_utility/time_varying/PROTOCOL.md` (SHA256
       `7333ad15…`); commit `3cc64ac` plus a dirty tree; parity gate max
       1.4e-6; training data only. No interval: one role, overfit heads.
   - **Held-out reconstruction check (2026-11-01, `cast_held_reconstruction_v1`):**
     **partial under the frozen rule.** Both explanations are real. Most of
     the 100 → 68 drop was memorized detail, but re-rendering also loses
     about half of the class signal that carries over to new recordings.
     - **Primary:** BEATs R-head AUROC on held groups fell from 71.3
       (originals) to 61.4 (re-renders). The drop is −9.9 points, 95% CI
       [−15.2, −3.5] (site-block [−15.6, −4.1]). "Keeps" needed a lower bound
       above −3, and "loses" needed an upper bound below −10.
     - **Memorization gap:** on the heads' own training clips the drop is
       −32.7 points, so about 23 of those points (≈70%) were memorized
       recording detail [17.1, 29.4]. The stage-diagnosis 68 overstated what
       reconstruction destroys.
     - **What generalizes is still cut:** held re-renders keep only 53% of
       BEATs's above-chance AUROC (21.3 → 11.4 points above 50). MFCC keeps
       70%: 76.6 → 68.5, drop −8.1 [−11.8, −5.8], with a small gap (−3.6).
     - **Other held metrics (BEATs R):** BA 65.2 → 56.1, AP 30.1 → 17.7,
       macro-F1 52.2 → 37.7.
     - **By fold:** BEATs drops 9–18 points in five folds and rises 3.3 in
       fold 4 (only 10 trucks). Fold 2 has only 14 trucks.
     - **Direction:**
       - **MFCC:** re-rendered cars get called truck (car recall 69.6 → 27.2,
         truck recall 66.7 → 87.7).
       - **BEATs:** the direction varies by role. Car recall fell 64.9 → 45.6
         with a CI touching zero, and truck recall stayed flat.
       - **Correction:** the stage diagnosis's "trucks become car" pattern
         was specific to role `utility_s42_0`. Across the 30 roles, BEATs's
         training-side truck recall ranges from 11% to 99%.
       - **Shared band limit:** originals and re-renders both go through the
         8 kHz route, so the band limit does not explain the shift.
     - **CAST-trained heads (descriptive):**
       - **BEATs:** the C head does no better on re-rendered held clips than
         on real ones (60.6 → 59.4) and is below the real head on those same
         re-renders (61.4). Its weakness is not just the render mismatch,
         which points at the population as well.
       - **MFCC:** the C head gains 53.0 → 66.7 on re-rendered held clips,
         so for MFCC much of C's failure is render mismatch.
     - **tv3 on fold 0 (one held group, descriptive):** it recovers part of
       the drop. BEATs went 78.6 → 65.1 with smooth8 and 68.7 with tv3; MFCC
       went 82.7 → 67.0 and 75.7. As before, the C heads break on tv3
       renders.
     - **What it supports (protocol: size both effects first):**
       - **Ceiling:** renderer work can win back at most about 10 BEATs
         points on held groups, capped by the real-head ceiling of 71 on this
         weak 8 kHz recipe. B1 reaches 84 with real data alone.
       - **Population:** separately, the BEATs C head result says sampling
         would still need work.
       - **My recommendation:** freeze B1 as the reference first (item 1).
         Then, if CAST continues, run a held-side tv3 re-test.
     - **Run record:** `artifact:cast/diagnostics/cast_held_reconstruction_20261101_v1`
       (1.2 GB).
       - **Protocol:** `experiments/cast_utility/held_reconstruction/PROTOCOL.md`
         (SHA256 `8d500238…`).
       - **Code:** `run.py` `cfcda2a1…`, `smooth_fit.py` `07d4d843…`,
         `time_varying.py` `6af1010a…`; commit `3cc64ac` plus a dirty tree.
       - **Data:** IDMT_V1 core8 2 s, 4,413 clips (511 trucks).
       - **Heads and statistics:** the 30 CAST-5A roles × saved R/C heads
         (manifest `9315e73b…`). Paired fold × selection bootstrap, 10,000
         draws, seed 20261101.
       - **Gates passed:**
         - all originals match their hashes;
         - all 120 held-original metrics reproduce CAST-5A exactly;
         - smooth8 replay max difference 0.0;
         - tv3 parity 1.2e-6.
       - **Limits:** six groups at three sites, already-exposed development
         data, and no new heads were fitted.
   - **tv3 on all held groups (2026-11-02, `cast_tv3_held_v1`):** **ambiguous
     under the frozen rule, so no renderer decision.** tv3 helps a little, but
     not enough to clear the bar, and it does not fix the main failure.
     - **Primary:** on held groups 1–5, BEATs R-head AUROC was 69.9 on
       originals, 60.6 with smooth8 and 62.8 with tv3. The gain is +2.2
       points, 95% CI [−2.9, +6.6]. "Helps" needed a lower bound above 0 and
       an estimate of at least +3. "Not enough" needed an upper bound below +3.
     - **Site-block interval:** [+0.4, +4.8]. The gain is probably real but
       small: tv3 wins back about a quarter of the lost points. Held re-renders
       keep 65% of BEATs's above-chance signal instead of 54%.
     - **By group:** gains of +4.8, +7.6, +0.7, −4.6 and +2.5 for groups 1–5.
       - The largest group (3, with 1,517 clips) gains only +0.7.
       - Group 4 (10 trucks) loses 4.6.
       - Group 0, already seen and excluded from the decision, was +3.6.
       - All six groups together: +2.4 [−1.9, +6.0].
     - **MFCC gains consistently:** +2.7 [+0.2, +4.7] on groups 1–5, +3.7 on
       all six. It keeps 85% of its above-chance signal instead of 74%.
     - **Main failure unfixed:** re-rendered cars still get called truck.
       - BEATs car recall: 63.8 originals, 45.8 smooth8, 44.6 tv3.
       - MFCC car recall: 70.7, 26.9, 34.5.
       - BEATs verdict changes per selection: tv3 repairs 74 trucks and
         breaks 35, but breaks 545 cars and repairs 463. As in the
         training-side study, it shifts the threshold more than it separates
         the classes.
     - **More fidelity, part of it memorizable:**
       - Every clip's fit and check loss went down (median check 1.383 →
         1.330).
       - The training-side gain (+6.6 [+4.3, +9.4]) is larger than the held
         gain, so some of tv3's extra detail is detail the overfit heads
         memorized.
       - About 7 BEATs points stay lost on held groups: −7.0 [−11.5, −2.6] vs.
         originals.
     - **C heads (not interpreted):** they were trained on smooth8 samples.
       BEATs C +3.1; MFCC C −14.4.
     - **What it supports:** the frozen rule gives no renderer decision.
       - **Time variation is not the main loss:** time-varying spectral weights
         are a modest, real improvement but do not explain most of the lost
         cue. The unexplained part is that re-rendered cars sound like trucks
         to both encoders.
       - **My recommendation:** B1 first, as before. If CAST continues, find
         out what the renderer removes from cars before adding more
         controls.
     - **Run record:** `artifact:cast/diagnostics/cast_tv3_held_20261102_v1`
       (992 MB).
       - **Protocol:** `experiments/cast_utility/tv3_held/PROTOCOL.md`
         (SHA256 `9f0e3065…`, frozen 2026-11-02 on user approval).
       - **Code:** `run.py` `1eeb6d0b…`, `time_varying.py` `6af1010a…`,
         `smooth_fit.py` `07d4d843…`; commit `3cc64ac` plus a dirty tree.
       - **Data:** IDMT_V1 core8 2 s, 4,413 clips; 3,843 new tv3 fits plus the
         570 fold-0 fits reused from `cast_held_reconstruction_v1`.
       - **Heads and statistics:** the 30 CAST-5A roles' saved heads. Paired
         fold × selection bootstrap, 10,000 draws, seed 20261102.
       - **Gates passed:**
         - all originals match their hashes;
         - all 500 reused metric rows reproduce exactly;
         - tv3 replay difference 0.0;
         - parity max 1.2e-6;
         - feature re-extraction difference 0.0.
       - **Limits:** five decision groups at three sites, already-exposed
         development data, the 8 kHz route, and a reconstruction test, not a
         utility test.
3. **Restart H2 (source vs. propagation) only on the corrected setup.**
   - **What happened:** the original "propagation wins by 35 F1 points" came from
     a 5-sample causal FIR latency (0.625 ms) shifting comb-filter notches.
   - **Current state:** with the latency removed, every BEATs cell sits at the
     always-truck floor, so source and path can't be ranked.
   - **Rerun needs:** the declared-latency renderer (`abvid.simulation.propagation_v2`),
     full-band synthesis and the B1 recipe.
4. **Admit MVD.**
   - **What's downloaded:** 4,229 files (1,005 car, 1,077 truck), 22.05 kHz,
     3-second clips; 53 byte-identical duplicate pairs, none crossing classes.
   - **Blockers:**
     - Audio licence: the repo's MIT licence covers code only.
     - Recording/session mapping: microphone tokens are not sessions.
     - An overlap audit against IDMT, MELAUDIS and the AI4TEN banks.
   - **First step:** ask the authors about the licence and sessions.
   - **Why it matters:** MVD is the only public truck-rich, near-balanced field
     dataset found.
5. **Plan the roadside collection.**
   - **Minimum spec:** 8 sessions, 4 sites, 2 dates, 300 trucks with at least
     20 per session.
   - **Capture setup:** camera labels, a reference microphone and background-only
     intervals.
   - **Why it matters:** it is the only target fully under project control, and
     site/session context is the best-evidenced weakness: a whole-site holdout
     cost BEATs 10 F1 points.
6. **Keep DCASE 2024 Task 10 sealed.**
   - **Unsealing needs:** a candidate with positive development evidence, a
     frozen protocol, an overlap audit and your written authorization.
   - **Candidate (2026-11-04):** the frozen all-IDMT B1-BEATs head. It has
     positive development evidence on MELAUDIS (+11.7 AUROC over H1).
     Remaining before a pass:
     - an overlap audit (DCASE vs. IDMT/MELAUDIS and BEATs pretraining data);
     - a frozen DCASE protocol;
     - your written authorization.
   - **Caveat:** DCASE's second class is "commercial vehicle", a proxy for truck.

Parked, with no action planned:
- **Physics engine prototype:** spectral loss 2.09 vs. CAST's 1.39. It needs real
  synchronized RPM and audio to validate.
- **P0 offline replay interface:** proposed, not started.
- **Unrun ladder steps:** E6–E9 (environment, sensor, real→sim→real, domain
  adaptation).

## 3. What is established

All of this is development evidence on exposed data.

| Finding | Number | Source |
|---|---|---|
| Real-to-real gap exists | IDMT→MELAUDIS −19 F1; whole-site holdout −10 F1 (BEATs) | H1, H1 diagnostics |
| More clips from the same groups barely help | 190→359/class: +0.8 F1 [−1.9, 3.1] | H1 diagnostics |
| On MELAUDIS (3% trucks), F1 at a fixed threshold mostly measures threshold placement | "procedural beats real" +11.8 F1 → +2.6 AUROC [−8.5, 9.5] | target panel |
| H2 path gain was filter latency, not physics | latency only 35.4% F1; attenuation only 3.7% | H2 latency control |
| Weak baseline was self-inflicted | AUROC 71.5 → 84.1 (+12.6 [8.3, 17.8]) | B1 |
| CAST adds no practical utility | RC − RA +0.47 F1 [−0.97, 2.08]; augmentation also adds ~0 | CAST-5A |
| CAST data carries some class signal, much less than real | CAST-only AUROC 60.6 vs. real 71.3 vs. unfitted 47.5 | CAST-5A |
| CAST class direction is misaligned with real | cosine 0.29 (ordinary augmentation 0.98) | gap audit |
| Loss happens mostly at reconstruction | 100 → 68 → 77 → 63 (above) | stage diagnosis |
| CAST fitted parameters are not identifiable | 92% of fits had disagreeing, equally good starts | CAST-4A |

## 4. Known gaps we live with

- **IDMT and MELAUDIS:** the original sessions are unknown.
  - IDMT has 6 conservative groups at 3 sites; two groups have only 10 and 14
    trucks.
  - MELAUDIS has 4 groups; the largest holds 6,477 of 8,066 events and one has
    a single truck.
  - All uncertainty intervals are conditional on these few groups.
- **R0 reproduction is partial.**
  - 17/18 released-checkpoint matrices match; ArcFace on IDMT CH34 is off by one
    truck.
  - No training run was reproduced.
  - P02's `soundclass_v1` module is missing from the release.
- **Provenance:** CAST-5A and B1 ran from a dirty working tree, and imported
  library modules weren't hashed at run time, so their exact identity is unknown.
  Hashing imported modules is a cheap fix for future runs.
- **BEATs pretraining:** its independence from AudioSet overlap is unknown for
  every dataset.
- **Archived reports:** three past verification records (physics prototype,
  military P0 follow-up and P1 handoff) name the old paths of four reports now
  in `archive/`. Those reports got link-only edits; their original bytes are in
  git history. No active check reads these records.

## Source map

The detailed record for each topic, newest first. All are historical except
where noted as current.

Most reports are in `archive/`. Twelve stay in this folder, byte-identical,
because frozen experiment configs or lock files pin their paths. Do not move or
edit them:
- `CAST_pilot`, `CAST_generalization`, `CAST_improvement`, `CAST_grouped_coverage_design`
- `H1_baseline_results`, `H2_source_path_results`, `H2_ground_air_collapse`, `H2_air_latency_control`
- `dataset_admission_integrity`, `dataset_admission_provenance`, `published_reproduction_status`
- `military_signature_P0`

`ARTIFACT_INDEX.md` and `refactor_symbol_map.json` also stay here.

Frozen protocol and run documents under `experiments/` still link to the old
report paths. They are hash-pinned, so those links stay stale on purpose.

| Topic | Reports |
|---|---|
| CAST diagnosis (current) | `CAST_stage_diagnosis_20261025`, `CAST_gap_and_data_decision_20261025` (+ JSON) |
| CAST utility | `CAST_source_classification` (5A), `CAST5A_precision_planning` |
| CAST coverage | `CAST4A_serialization_revision`, `CAST_joint_metric_fixtures` (4B MMD), `CAST4A_run_20261011`, `CAST_grouped_runner`, `CAST_fit_readiness`, `CAST_grouped_coverage_design`, `CAST_improvement` (v15), `CAST_generalization`, `CAST_pilot` |
| CAST physics | `CAST_physics_prototype`, `CAST_physics_data_audit` |
| Baselines | `B1_baseline_strength`, `target_panel`, `H1_regularization_transfer`, `H1_source_diagnostics`, `H1_baseline_results` |
| H2 | `H2_air_latency_control`, `H2_ground_air_collapse`, `H2_source_path_results` |
| Data | `MVD_download_20261025`, `dataset_scan_20261012`, `dataset_admission_dcase2024_t10`, `dataset_admission_provenance`, `dataset_admission_integrity`, `dataset_readiness_audit` |
| Reproduction | `published_reproduction_status` |
| Military side project | `military_signature_P1_admission`, `military_signature_P0_followup`, `military_signature_P0` |
| Planning and literature | `proposed_experiments`, `proposal_v0_1`, `proposal_v0_1_ste`, `research_gaps`, `domain_shift_taxonomy`, `representation_taxonomy`, `physical_signal_model`, `literature_review` |
| Infrastructure | `ABVID_prototype_v0_1`, `REFACTOR`, `ARTIFACT_INDEX` (artifact hashes; keep) |

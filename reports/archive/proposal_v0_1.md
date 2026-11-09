# ABVID: Measurement-First Evaluation of Sim-to-Real Transfer in Acoustic Vehicle Classification

**Research proposal — civilian v0.1**
Author: Daniel Merino · Date: 2026-10-04
Status: proposal; all cited results are frozen and verified (see §6).

---

## 1. Problem

Acoustic vehicle classification is routinely reported with high accuracy, yet deployed performance is dominated by a failure mode the standard evaluation design cannot see: recordings from a *new site or session* do not behave like held-out clips from a *seen site*. Window-level, random-split evaluation overstates generalization because windows from one recording share microphone, distance, road surface, compression, and background — the model partially learns the recording, not the vehicle.

Meanwhile, the obvious remedy — collect more data — is blocked by the scarcity of labeled vehicle audio, and the fashionable remedy — train on synthetic data rendered from simulators — is unproven: no existing work cleanly separates **source mismatch** (does my simulated vehicle sound like the real one?) from **propagation/environment mismatch** (does my simulated acoustic path match the deployment path?), and evidence is mounting that synthetic data's value is conditional on representation and operating point in ways the literature ignores.

This project builds the measurement science that both problems need: a leakage-safe evaluation protocol that (a) quantifies session/site-level transfer failure directly, (b) attributes synthetic-training value to identifiable factors rather than headline numbers, and (c) is honest about statistical power and exposure — development evaluation is never silently promoted to confirmation.

## 2. Research questions

- **Q1 (gap).** How large is the site/session transfer gap in vehicle-audio classification, measured with group-safe splits, and how does it compare to the within-site numbers typically reported?
- **Q2 (operating point).** How much of a reported "improvement" survives prevalence-robust metrics and fixed decision rules? Which claimed gains are majority-class tradeoffs?
- **Q3 (synthetic value).** Under what conditions does procedurally or generatively synthesized training data help real-domain classification, and is that value representation-conditional?
- **Q4 (attribution).** When synthetic training hurts, is the dominant cause source mismatch or propagation/environment mismatch? (Requires the paired physical factorial, later phase.)

## 3. Approach

Three principles govern every experiment:

1. **Group-safe identity.** All resampling and splitting is at the session/site level; provenance audits fail closed (unknown origin ⇒ not admitted to independent claims).
2. **Freeze before target access.** Every comparison's arms, budget, representations, decision rule, metrics, and margin are frozen (`PROTOCOL.md` + `lock.json`) before any target-domain scoring. No post-hoc winners, no combined "best" configurations.
3. **Co-primary metrics.** Macro-F1 *and* balanced accuracy, with class recall, ROC-AUC, average precision, worst-group, and ECE reported alongside. Threshold tuning is never used to claim discrimination gains.

### Phase A — completed: baseline and diagnostics (H1, D1–D5)

Car-vs-truck on admitted IDMT (4,413 excerpts, six date-groups, three sites) with frozen BEATs768 and MFCC26 front ends, 190 examples/class, five seeds, leave-one-group-out. Key frozen findings:

- **Site-holdout costs 10.2–11.5 F1 points** (both representations; site-level resampling intervals exclude zero) — the headline result.
- BEATs fits training perfectly (100% F1) yet holds out at 53% — recording-context overfitting, not fitting failure.
- More excerpts from the *same* sites do not help (190→359/class: ΔF1 intervals include zero); more *contexts* is the informative axis.
- Restoring native 16 kHz bandwidth helps modestly (+2.9 both representations).
- Threshold tuning buys +6.7–7.4 F1 while truck recall collapses (65.6→41.8%) — a majority-class tradeoff, not discrimination.
- On the exposed target (IDMT→MELAUDIS), procedurally-synthetic training beat matched real training by macro-F1 but the ordering is representation-conditional (reverses between BEATs and MFCC) — synthetic value is not monotone.

### Phase B — proposed now: frozen transfer sensitivity (T1)

One frozen run, IDMT→MELAUDIS, arms declared before target scoring:

- **A0** H1 exact replay (anchor); **A1** native-16 kHz observation; **A2** source-only nested-C regularization; **A3** A1+A2 interaction (declared now, in advance).
- Fixed argmax decisions; co-primary F1/BA; 3-point practical margin; group and site resampling; five seeds.
- MELAUDIS is declared **exposed development data**: T1 informs, it cannot confirm. The protocol states in writing which corpus could serve as untouched confirmation and under what admission standard.
- Includes an AudioSet-index overlap audit for the source video IDs (bounds pretraining contamination in either direction).

### Phase C — later, gated: paired physical factorial (E4/E5)

Cross two source alternatives × two propagation alternatives at matched bandwidth, budget, head, and decision rule, with explicit interaction estimates. Gated on T1: the factorial's reference arm and bandwidth matching are chosen using T1's outcome. Margins will be sized to the resolvable power of the target grouping, or the claims will be stated as descriptive.

## 4. Expected contributions

1. A reproducible, leakage-safe evaluation protocol for vehicle-audio classification (frozen selections, hashes, verification, per-model artifacts — all publicly retained).
2. The first controlled quantification of site-level transfer loss in this task family, with interval-level uncertainty, replicated across two independent corpora (civilian IDMT/MELAUDIS and a military-vehicle line under separate milestone gates).
3. Evidence that synthetic-training value is representation- and operating-point-conditional, with a falsifiable attribution design for source-vs-propagation mismatch.
4. Negative results published as results: budget scaling, threshold gains, and regularization each fail to fix the transfer gap, which redirects the field's attention from model capacity to context diversity.

## 5. Risks and limits

- **Statistical power.** Six groups at three sites give wide intervals; all claims are stated with resampling schemes and declared margins, and confirmation requires unexposed provenance-clean recordings not yet admitted.
- **Provenance incompleteness.** Original session independence and synthetic lineage are partially unknown; admission is qualified and independent-confirmation claims are withheld accordingly.
- **Pretraining contamination.** BEATs/PANNs were AudioSet-pretrained; overlap with target sources is auditable in part and will be reported either way.
- **Confounded location axis.** Site bundles fleet, road, weather, and acquisition context; the factorial (Phase C) is the only design that can attribute, and it is explicitly downstream.

## 6. Artifacts and reproducibility

Every phase ships: frozen protocol + selection lock, per-seed models and predictions, independent verification scripts, reproduction commands, and a report with paired interval tables. Computation is consumer-hardware (Apple M3 Pro, CPU-only; Phase A total under 7 minutes). All prior frozen artifacts are byte-protected across later runs.

## 7. Timeline

| Window | Deliverable |
|---|---|
| Week 1 | T1 protocol freeze + run + verification; AudioSet overlap audit |
| Weeks 2–3 | v0.1 writeup: site-gap headline, evidence chain (R0, admission, H1, D1–D5, T1); tagged release `v0.1-civilian` |
| Gated | E4/E5 factorial protocol (only after T1 interpretation); independent-corpus confirmation track |

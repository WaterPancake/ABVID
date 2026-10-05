# ABVID — Research Proposal in ASD-STE100 Simplified Technical English

**Rules applied:** approved verbs only (per ASD-STE100 Word Rules, with "measure", "compare", "show" as approved technical verbs), one topic per sentence, no adverbs of degree ("very", "significantly"), no "-ing" forms as verbs, no passive where an active alternative exists, no compound nouns longer than three words, numbers written as digits, no "only"/"alone" displaced modifiers, no idioms, no "we" (STE restricts personal pronouns in technical writing; procedures use imperative voice, descriptions use "the project"/"the system").

---

## 1. Purpose

This document gives the research plan for the ABVID project. ABVID does research on audio data that comes from vehicles. The project has two goals:

1. Measure how a classifier performs when the audio data comes from a new location.
2. Measure if training with synthetic audio data helps or harms the classifier.

## 2. Background

A classifier puts each audio excerpt into one class. The classes are "car" and "truck". Published studies report high scores. But the scores come from tests that use data from locations that the training data also used. The model learns properties of the recording, not properties of the vehicle. When the classifier receives data from a new location, the score becomes lower. The published scores do not show this failure.

Synthetic audio data is data that a computer program generates. Such data can increase the quantity of training data. But synthetic data does not always help. The project must find the conditions under which synthetic data helps.

## 3. Questions

Q1: How much does the score decrease when the test location is new?
Q2: Do the reported gains stay when the test uses metrics that do not change with class ratio? Or are the gains a change in the class that the model puts first?
Q3: Under which conditions does synthetic training data help the classifier on real data?
Q4: When synthetic training data harms the classifier, which part is the cause: the vehicle sound model or the sound path model?

## 4. Method

The project applies three rules to all experiments:

- **Rule 1: Group the data correctly.** Each recording session is one group. All statistics use the session as the unit. The audit rejects data with unknown origin.
- **Rule 2: Freeze the protocol before the test.** Each experiment defines its arms, data quantity, representations, decision rule, metrics, and margin before the project scores the target data. The system stores this definition with a hash. The project does not change the definition after it sees the results.
- **Rule 3: Use two primary metrics.** The project reports macro-F1 and balanced accuracy together. The project reports recall per class, ROC-AUC, average precision, worst-group score, and calibration error as secondary metrics. Threshold changes do not count as discrimination gains.

### 4.1 Phase A — Baseline and diagnostics (complete)

Phase A used 4,413 audio excerpts. The excerpts came from six recording dates at three locations. The front ends were BEATs768 and MFCC26. The training budget was 190 excerpts per class. The test procedure held out one group at a time. Five random seeds gave five training sets.

The results:

- **When the test holds out one location, the macro-F1 decreases by 10.2 to 11.5 points.** Both front ends show this decrease. This is the main result.
- BEATs768 reached 100 percent F1 on the training data but 53 percent on held-out groups. The model learned the recording context. The model did not fail to fit the training labels.
- More excerpts from the same locations did not help. The interval for the change in F1 contains zero. More locations are more informative than more excerpts.
- Audio at 16 kHz bandwidth gave 2.9 points more F1 than audio that passed through an 8 kHz step.
- Threshold tuning gave 6.7 to 7.4 points more F1, but truck recall decreased from 65.6 percent to 41.8 percent. This is a change in the class that the model puts first. This is not better discrimination.

### 4.2 Phase B — Transfer sensitivity (proposed)

Phase B runs one frozen experiment. The source data is IDMT. The target data is MELAUDIS. The project declares four arms before it scores the target:

- A0: replay of the Phase A procedure.
- A1: audio at native 16 kHz bandwidth.
- A2: regularization that the model selects on source data.
- A3: A1 and A2 together.

The decision rule stays fixed. The project does not tune thresholds. The margin is 3 points. MELAUDIS is development data. The project saw this data before. Thus Phase B informs the next steps. Phase B cannot confirm a final claim.

Phase B also audits the AudioSet index. The audit checks if the source videos of the target data occur in the data that trained the BEATs front end. The project reports the result in both cases.

### 4.3 Phase C — Physical factorial (conditional)

Phase C crosses two vehicle-sound models with two sound-path models. All cells use the same bandwidth, budget, head, and decision rule. The project estimates the effect of each factor and the interaction between the factors. Phase C starts only after Phase B is complete.

## 5. Expected contributions

1. An evaluation procedure for vehicle audio classification that prevents leakage. The procedure ships with frozen selections, hashes, verification scripts, and commands that reproduce each result.
2. The first measurement of the loss that a new location causes in this task. Two independent corpora show the same effect.
3. Evidence that the value of synthetic training data depends on the representation and the decision rule.
4. Negative results that the project publishes as results: more excerpts, threshold tuning, and regularization do not close the transfer gap. Context diversity is the open axis.

## 6. Risks and limits

- Six groups at three locations give wide intervals. The project states each claim with an interval and a margin. A confirmation claim requires data that the project did not see before. Such data does not exist now.
- The origin data for some recordings is not complete. The project does not claim independence where the audit cannot establish it.
- The BEATs front end trained on AudioSet. AudioSet can contain the same videos as the target data. The audit in Phase B gives a partial bound.
- The location holds more than one property: the fleet, the road, the weather, and the recording equipment change together. Phase C is the design that can attribute the loss to a factor.

## 7. Artifacts

Each phase stores: the frozen protocol, the selection lock with hashes, the models, the predictions, the verification scripts, and the commands that reproduce the run. A consumer laptop (Apple M3 Pro, CPU) runs the full Phase A in less than 7 minutes. Later runs do not change earlier artifacts.

## 8. Schedule

| Time | Deliverable |
|---|---|
| Week 1 | Freeze and run Phase B. Do the AudioSet audit. |
| Weeks 2–3 | Write the v0.1 report. Tag the release `v0.1-civilian`. |
| Conditional | Define the Phase C protocol. Start the confirmation track when new data passes the audit. |

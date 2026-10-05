"""Write the H1 report from saved, verified predictions. No fitting or tuning."""
import json
from pathlib import Path
import numpy as np
from freeze import ROOT,save,sha

RUN=ROOT/'experiments/h1/results/H1_20261002'
FREEZE=ROOT/'experiments/h1/frozen/h1_common_budget_v1.3'


def percent(x):return f'{100*x:.2f}'
def average(xs):return float(np.mean(list(xs)))
def interval(v):return f'{percent(v["estimate"])} [{percent(v["ci95"][0])}, {percent(v["ci95"][1])}]'


def main():
    r=json.loads((RUN/'metrics.json').read_text());meta=json.loads((RUN/'summary.json').read_text())
    verification=json.loads((RUN/'verification.json').read_text());assert verification['models_verified']==260
    lock=json.loads((FREEZE/'lock.json').read_text())
    rows={x['file_id']:x for x in map(json.loads,(FREEZE/'admitted.jsonl').open())}
    text='''# H1 baseline results

2026-10-02. The minimum frozen H1 run is complete: six IDMT held-out groups, five training selections, 190 cars + 190 trucks per single-origin fit, frozen BEATs and MFCC controls. All 260 scaler/heads passed independent artifact, metric and known-group-separation checks. MELAUDIS is exposed development evaluation, not untouched confirmation.

**BEATs loses 19.11 macro-F1 percentage points across datasets. A blanket synthetic-training deficit is not supported:** procedural-only training exceeds matched IDMT training on MELAUDIS, while AudioLDM-only falls below it. Absolute performance is weak and four uneven target groups limit uncertainty. H1 does not establish whether source or propagation mismatch dominates.

## Evidence and frozen scope

- [Executed protocol](../experiments/h1/frozen/h1_common_budget_v1.3/EXPERIMENTAL_PROTOCOL.md), [exact file selections and lock](../experiments/h1/frozen/h1_common_budget_v1.3/lock.json), [canonical configuration](../experiments/h1/frozen/h1_common_budget_v1.3/config.json).
- [Qualified admission and missing metadata](dataset_admission_provenance.md).
- [Complete metrics and per-fit/group records](../experiments/h1/results/H1_20261002/metrics.json), [run summary/hashes](../experiments/h1/results/H1_20261002/summary.json), [independent verification](../experiments/h1/results/H1_20261002/verification.json).
- [Commands](../experiments/h1/README.md), [pre-fit prose clarification](../experiments/h1/documentation_clarifications.md). Frozen document snapshots remain unchanged.

The same E0 fitted scaler/head is evaluated on its held-out IDMT group and on all admitted MELAUDIS excerpts. No validation, hyperparameter search, threshold tuning, target statistics, best-seed selection or score-based exclusions occur. Raw audio, R0 results and military evaluation sources are unchanged. No simulator interventions or encoder training were performed.

## Primary scores

Values are percentages. Brackets are 95% whole-group/selection percentile intervals, conditional on these fitted models. Source score averages six held-out-group F1s; target score averages those models on all 8,066 target excerpts. These are not ensembles. Synthetic rows average five actual fits, not thirty artificial repetitions. Balanced accuracy and recall are descriptive means.

| Representation / training | Evaluation | Macro-F1 [95% interval] | Balanced accuracy | Car recall | Truck recall |
|---|---|---:|---:|---:|---:|
'''
    arms={'real':'IDMT only','procedural':'procedural only','audioldm':'AudioLDM only','real_repeated':'IDMT repeated twice','real_plus_procedural':'IDMT + procedural','real_plus_audioldm':'IDMT + AudioLDM'}
    for rep in ['BEATs768','MFCC26']:
        for domain,arm,label in [('source','real','IDMT only')]+[('target',a,n) for a,n in arms.items()]:
            models=[m for m in r['models'] if m['representation']==rep and m['arm']==arm]
            ba=percent(average(m[domain]['balanced_accuracy'] for m in models))
            rec=[percent(average(m[domain]['per_class'][c]['recall'] for m in models)) for c in ['car','truck']]
            score=interval(r['inference'][rep]['estimates']['source' if domain=='source' else arm])
            text+=f'| {rep} / {label} | {"Held-out IDMT" if domain=="source" else "MELAUDIS"} | {score} | {ba} | {rec[0]} | {rec[1]} |\n'
    text+='''
MELAUDIS has **96.83% cars**. Always predicting car yields 49.19% macro-F1, 50% balanced accuracy, 100% car recall and 0% truck recall. Every learned arm scores below that constant predictor in target macro-F1, although several have balanced accuracy above 50%. Macro-F1 still depends on prevalence: the balanced-training fixed decision rule produces many false truck predictions. Procedural BEATs exceeds real-only by 11.82 F1 points but only 0.93 balanced-accuracy points. This is not evidence of generally reliable recognition. No threshold was adjusted after observing it.

Within-domain BEATs balanced accuracy is 65.87%, compared with 71.00% for MFCC. The weak within-domain baseline and constrained file budget limit mechanistic interpretation. Historical paper replay scores used different tasks, partitions and preprocessing and are not comparable to these binary results.

## Paired gap and mixing estimates

Values are macro-F1 percentage points. Positive domain/synthetic gap means the real-source reference exceeds its comparator. These conditional intervals have uncertain coverage with four uneven target groups; secondary comparisons are not multiplicity-adjusted confirmation claims.

| Contrast | BEATs estimate [95% interval] | MFCC estimate [95% interval] |
|---|---:|---:|
'''
    contrasts={'delta_domain':'Within IDMT minus cross-dataset','delta_sim2real_procedural':'Real minus procedural on target','delta_sim2real_audioldm':'Real minus AudioLDM on target','gain_real_plus_procedural':'Real + procedural minus real','gain_real_plus_audioldm':'Real + AudioLDM minus real','mixed_procedural_minus_repeated_real':'Real + procedural minus repeated real','mixed_audioldm_minus_repeated_real':'Real + AudioLDM minus repeated real'}
    for k,name in contrasts.items():
        text+=f'| {name} | {interval(r["inference"]["BEATs768"]["estimates"][k])} | {interval(r["inference"]["MFCC26"]["estimates"][k])} |\n'
    text+='''
The H1a domain-gap intervals exceed the prespecified 5-point practical margin for both representations. This establishes a descriptive loss under this protocol. Priors, fleet and recording context also change; acoustics are not separately identified.

For H1b, the BEATs procedural gap's upper bound is negative, contradicting an expected positive procedural-training deficit against this real baseline. The AudioLDM gap is positive, but its lower bound (3.85 points) is below the 5-point margin: a confidently ≥5-point deficit remains unresolved. MFCC reverses the AudioLDM ordering and also has a negative procedural point estimate. Do not select the representation that supports the working hypothesis.

BEATs procedural mixing gains 2.71 points, with a conditional interval above zero but crossing the broader proposal's 3-point intervention margin. AudioLDM mixing is inconclusive. MFCC procedural mixing hurts and AudioLDM mixing helps. Conventional real corruption augmentation was not run, so no advantage over ordinary augmentation is established. Repeating real features alone barely changes results. Mixed gains cannot be attributed to specific physics.

Paired BEATs-minus-MFCC real-only target F1 is 2.62 points, interval [−3.42, 14.21]. A general pretrained-representation advantage is not established. Every arm's representation contrast is in metrics.json.

## Source groups

Rows average five training selections. These are three locations across six recording dates, not six independent locations/known physical vehicles.

| Held-out IDMT site/date | Car/truck support | BEATs F1 | BEATs balanced accuracy | MFCC F1 |
|---|---:|---:|---:|---:|
'''
    for f in lock['folds']:
        row=rows[f['test_ids'][0]];name=f'{row["site_id"]} {row["calendar_date"]}'
        bm=[m for m in r['models'] if m['representation']=='BEATs768' and m['arm']=='real' and m['fold']==f['fold']]
        mm=[m for m in r['models'] if m['representation']=='MFCC26' and m['arm']=='real' and m['fold']==f['fold']]
        text+=f'| {name} | {f["test_counts"]["car"]}/{f["test_counts"]["truck"]} | {percent(average(m["source"]["macro_f1"] for m in bm))} | {percent(average(m["source"]["balanced_accuracy"] for m in bm))} | {percent(average(m["source"]["macro_f1"] for m in mm))} |\n'
    text+='''
## Target groups

Real-only BEATs results average the 30 fitted models. This does not create thirty independent recordings per group.

| MELAUDIS conservative group | Car/truck support | Macro-F1 | Car recall | Truck recall |
|---|---:|---:|---:|---:|
'''
    gmap={'connected_e345c615fb69bd2d':'Nine linked dates in 2023','connected_153e3dead5cb84d2':'2024-01-17','connected_8e14c1e672d4cb88':'2024-01-16','connected_e4bd1f6e5af7a1dc':'2024-02-09'}
    bm=[m for m in r['models'] if m['representation']=='BEATs768' and m['arm']=='real']
    for g,d in r['target_groups'].items():
        fs=percent(average(m['target_groups'][g]['macro_f1'] for m in bm))
        rec=[percent(average(m['target_groups'][g]['per_class'][c]['recall'] for m in bm)) for c in ['car','truck']]
        text+=f'| {gmap[g]} | {d["support"][0]}/{d["support"][1]} | {fs} | {rec[0]} | {rec[1]} |\n'
    text+='''
The February group has one truck: 100% average truck recall is one event repeatedly evaluated, not strong robustness evidence. The largest component supplies 6,477/8,066 events. Across individual real-only BEATs fits, minimum observed source class/group recall is 40%; minimum target class/group recall is 4.63%. MFCC minima are 30% and 0%. Means conceal severe failures.

## Verification and runtime

Eighteen admission regression tests, seven H1 tests and two existing BEATs adapter tests passed. The only shared adapter change removes an unnecessary benchmark/plotting import by using a local SHA256 helper; encoder loading/inference are unchanged. A deterministic artificial input gave bit-identical frozen encoder outputs.

The independent verifier checked all 260 model/prediction hashes, recomputed source/target confusion matrices and macro-F1/balanced accuracy using scikit-learn, checked known group/file/duplicate separation, verified every scaler against its exact training features, and replayed probabilities at three fixed target positions per model. These checks verify computation/known links, not unknown original-source independence. All prediction probabilities, training IDs, calibration metrics and per-group outcomes remain inspectable.

'''
    cache=json.loads((ROOT/'experiments/h1/cache/h1_common_budget_v1.3/summary.json').read_text())
    text+=f'Apple M3 Pro, 18 GiB RAM, CPU only: provenance audit 63.18 s; 14,630-excerpt BEATs/MFCC cache {cache["total_s"]:.2f} s (BEATs forwards {cache["timings"]["beats_forward_s"]:.2f} s); 260 fits, predictions and uncertainty {meta["total_s"]:.2f} s. No GPU/cloud use. NumPy 1.26.4, SciPy 1.17.1, scikit-learn 1.4.2, Torch/torchaudio 2.11.0, Librosa 0.11.0, SoundFile 0.13.1. Lock SHA256: `{meta["lock_sha256"]}`.\n'
    text+='''
The budget counts retained excerpts, not independent physical vehicles/templates. AudioLDM's 190 trucks recur in every selection; only 190-of-192 cars vary. Synthetic generation backend/source lineage and pretrained data overlap remain unknown. The bounded waveform search found no cross-corpus matches but cannot rule out arbitrary cropped/modified reuse. Four conservative target groups yield conditional, uncertain-coverage intervals. R0 exposure means all target results are development observations.

## Next decision

H1 provides an auditable gap measurement and a counterexample to assuming all synthetic training is worse. It does not establish source dominance, propagation accuracy, a performance ceiling, model-level recognition or field readiness. Address the weak baseline and prevalence/recording-context confounds using source-only diagnostics and a separately frozen sensitivity study before interpreting simulator-component changes. Do not optimize thresholds/simulator settings on these target results and call the same target untouched confirmation.

This minimum H1 completes E0, E1, released-bank E2 and the MFCC part of E3. Official AST, a newly trained CNN, conventional augmentation, full-real sensitivity and E4–E9 remain unrun. Later controlled interventions require backend-verified source/trajectory manifests and a paired factorial. The released banks alone cannot identify source versus propagation effects.
'''
    report=ROOT/'reports/H1_baseline_results.md';report.write_text(text)
    save(RUN/'report_provenance.json',dict(report_path=str(report.relative_to(ROOT)),report_sha256=sha(report),
         executed_report_script_sha256=sha(__file__),
         pre_fit_documentation_clarification_sha256=sha(ROOT/'experiments/h1/documentation_clarifications.md'),
         original_results_summary_sha256=sha(RUN/'summary.json'),no_refits=True,no_protocol_or_membership_changes=True))
    print(report)


if __name__=='__main__':main()

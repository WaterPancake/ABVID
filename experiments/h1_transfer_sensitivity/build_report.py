"""Render a short, evidence-linked report for the named completed transfer run."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'experiments/h1_transfer_sensitivity/results/regularization_transfer_20261004_v1'
FROZEN=ROOT/'experiments/h1_transfer_sensitivity/frozen/regularization_transfer_20261004_v1'
S=json.loads((RUN/'summary.json').read_text());V=json.loads((RUN/'verification.json').read_text());assert V['passed']
L=json.loads((FROZEN/'lock.json').read_text());C=json.loads((FROZEN/'config.json').read_text())
ROWS={r['file_id']:r for r in map(json.loads,(FROZEN/'manifest.jsonl').read_text().splitlines())}
P=[]
def add(s=''):P.append(s)
def pct(x):return f'{100*x:.2f}'
def est(rep,arm,domain,metric):return S['estimates'][rep][arm][domain][metric]['estimate']
def shown(d,signed=False):
    p=f'{100*d["estimate"]:+.2f}' if signed else pct(d['estimate'])
    return f'{p} [{100*d["ci95"][0]:.2f}, {100*d["ci95"][1]:.2f}]'
def table(headers,rows):
    add('| '+' | '.join(headers)+' |');add('| '+' | '.join('---' for _ in headers)+' |')
    for r in rows:add('| '+' | '.join(map(str,r))+' |')
    add()

add('# H1 regularization transfer sensitivity')
add('\n2026-10-04. Protocol `h1_regularization_transfer_v1`, run `regularization_transfer_20261004_v1`. **Complete: IDMT real → MELAUDIS real, car/truck, exposed development evaluation.**\n')
add('**BEATs regularization improves balanced accuracy and truck recall, but does not establish the primary macro-F1 improvement.** MFCC has no clear overall benefit. The source-domain gains therefore transfer only partly, and the domain gap remains. This is a classifier-control result; no simulator component was changed.\n')
add('## Frozen comparison and evidence\n')
add('- [Frozen protocol](../experiments/h1_transfer_sensitivity/frozen/regularization_transfer_20261004_v1/PROTOCOL.md), [configuration](../experiments/h1_transfer_sensitivity/frozen/regularization_transfer_20261004_v1/config.json), [lock and hashes](../experiments/h1_transfer_sensitivity/frozen/regularization_transfer_20261004_v1/lock.json).\n- [Exact saved models and training IDs](../experiments/h1_transfer_sensitivity/frozen/regularization_transfer_20261004_v1/model_references.json), [all per-model/group scores](../experiments/h1_transfer_sensitivity/results/regularization_transfer_20261004_v1/evaluations.json), [aggregates and paired intervals](../experiments/h1_transfer_sensitivity/results/regularization_transfer_20261004_v1/summary.json), [independent verification](../experiments/h1_transfer_sensitivity/results/regularization_transfer_20261004_v1/verification.json).\n- [Commands](../experiments/h1_transfer_sensitivity/README.md), [previous source-only diagnostics](H1_source_diagnostics.md), [unchanged H1](H1_baseline_results.md).\n')
add('Reuse the 190/class IDMT selections, six outer groups and five seeds, two-second mean-channel native→8 kHz→16 kHz observations, frozen BEATs768/MFCC26, and threshold .5. The reference uses C=1; the candidate reuses each saved outer head with C selected previously by five inner IDMT groups. BEATs selected .01 in 26/30 cases and .1 in four; MFCC selected .01/.1/1 in 3/14/13 cases. No new choices, fitting, extraction, augmentation, threshold adjustment or bandwidth changes occurred.\n')
add('All 8,066 admitted MELAUDIS excerpts are evaluated: 7,810 cars and 256 trucks. The target was already exposed in R0/H1; this freeze preceded new candidate inference, not all prior knowledge of the target. Candidate validation uses additional IDMT labels compared with the original untuned baseline. All target features enter only prediction/scoring, never scaler fitting or model selection.\n')
add('## Primary and class-level results\n')
add('Values are percentages. Source F1 averages six held-out group scores; target F1 is computed on the full target for each model and averaged over 30 source fold/seed combinations. These are averages of individual models, not ensembles. Brackets are conditional 95% paired-bootstrap intervals.\n')
table(['Representation / arm','Source F1','Target F1 [95% CI]','Target BA','Car recall','Truck recall','Truck precision'],
      [[rep+' / '+arm,pct(est(rep,arm,'source','macro_f1')),shown(S['estimates'][rep][arm]['target']['macro_f1'])]+[pct(est(rep,arm,'target',m)) for m in ['balanced_accuracy','car_recall','truck_recall','truck_precision']] for rep in C['representations'] for arm in C['arms']])
add('Paired candidate-minus-reference effects, in percentage points:\n')
table(['Representation','Macro-F1 gain','BA gain','Car-recall gain','Truck-recall gain'],
      [[rep]+[shown(S['paired_gains'][rep]['target'][m],True) for m in ['macro_f1','balanced_accuracy','car_recall','truck_recall']] for rep in C['representations']])
add('**BEATs:** +0.63 F1 points [−2.21,5.13] crosses both zero and the preregistered three-point practical reference. Neither a positive nor a practically important F1 benefit is established. BA improves +5.04 points [2.73,7.99] and truck recall +10.03 [4.14,14.80]; these secondary conditional intervals exclude zero. Mean car recall is almost unchanged, but its interval allows loss. The stricter claim “improves without sacrificing class recall” is not established.\n')
add('**MFCC:** F1 changes −0.98 points [−2.00,0.72]; this interval rules out the proposed ≥3-point practical F1 gain under this conditional estimator. BA is nearly unchanged. A +2.33-point truck-recall change accompanies −1.87 points in car recall; this is not a general improvement. These secondary comparisons are not multiplicity-adjusted confirmation.\n')
add('The always-car reference yields 49.19% F1, 50% BA and zero truck recall. Both learned candidates remain below it in F1 while detecting trucks; neither is a reliable classifier. BEATs truck precision is only 5.06%, reflecting many false truck predictions under the target’s 96.83% car prevalence.\n')

add('## Ranking, probability errors and residual domain gap\n')
table(['Representation / arm','ROC-AUC','Truck AP','Brier','Log loss','ECE10'],
      [[rep+' / '+arm]+[pct(S['descriptive'][rep][arm]['target'][m]) for m in ['roc_auc','average_precision']]+[f'{S["descriptive"][rep][arm]["target"][m]:.4f}' for m in ['brier_truck','log_loss','ece10_truck']] for rep in C['representations'] for arm in C['arms']])
add('These are descriptive means, without group-resampled intervals. BEATs ranks target classes better (.6156→.7003 AUC) and has lower Brier/log loss; however, ECE remains approximately .507. No probability calibrator was fitted. MFCC ranking changes little and AP slightly declines. The pooled truck prevalence reference for AP is 3.17%; higher AP does not by itself establish useful precision at the fixed decision rule.\n')
table(['Representation','Original source-minus-target F1 gap','Candidate gap'],
      [[rep,shown(S['source_minus_target_f1'][rep]['baseline']),shown(S['source_minus_target_f1'][rep]['nested_C'])] for rep in C['representations']])
add('The BEATs gap does not shrink: its point estimate is 19.11→20.30 points because source F1 improves more than target F1. The MFCC gap is 24.88→25.20. These gaps compare different class priors and aggregation schemes; they are descriptive domain differences, not isolated acoustic mechanisms.\n')

add('## Recording-group findings and worst cases\n')
labels={}
for g in L['target_groups']:
    rs=[ROWS[i] for i in L['target_ids'] if ROWS[i]['provenance_group_id']==g]
    dates=sorted({r['calendar_date'] for r in rs});labels[g]=dates[0] if len(dates)==1 else 'Nine linked 2023 dates'
for rep in C['representations']:
    add('### '+rep+'\n')
    rr=[]
    for g in L['target_groups']:
        rs=[ROWS[i] for i in L['target_ids'] if ROWS[i]['provenance_group_id']==g]
        a=S['descriptive'][rep]['baseline']['groups'][g];b=S['descriptive'][rep]['nested_C']['groups'][g]
        rr.append([labels[g],f'{sum(r["class_id"]==0 for r in rs)}/{sum(r["class_id"]==1 for r in rs)}']+
                  [pct(a[m]['mean'])+' → '+pct(b[m]['mean']) for m in ['macro_f1','car_recall','truck_recall']])
    table(['Target group','Car/truck','F1, original → candidate','Car recall','Truck recall'],rr)
add('The largest connected group contributes 6,477/8,066 excerpts. BEATs gains only +0.23 F1 points there and loses 0.63 points of car recall, although truck recall improves. The February group contains one truck: its repeated 100% BEATs recall is one event, not evidence of broad truck robustness.\n')
table(['Representation','Mean per-model worst group/class recall, original → candidate','Paired gain [95% CI]','Absolute minimum, original → candidate'],
      [[rep,pct(est(rep,'baseline','target','mean_model_worst_group_class_recall'))+' → '+pct(est(rep,'nested_C','target','mean_model_worst_group_class_recall')),
        shown(S['paired_gains'][rep]['target']['mean_model_worst_group_class_recall'],True),
        pct(S['descriptive'][rep]['baseline']['absolute_minimum_target_group_class_recall'])+' → '+pct(S['descriptive'][rep]['nested_C']['absolute_minimum_target_group_class_recall'])] for rep in C['representations']])
add('Worst-group improvement remains uncertain. The absolute worst cases persist: 4.63% car recall for BEATs and 0% truck recall for MFCC. The averages therefore do not justify automatic adoption or a robustness claim.\n')

add('## Verification and limits\n')
add(f'Seven pre-run tests passed. Independent verification reconstructed all 60 C choices from the 900 original IDMT inner-prediction records, checked all {V["unique_models_verified"]} distinct pipelines and {V["evaluations_verified"]} source/target evaluations, all 480 target-group score records, scaler training means/variances, saved probabilities, metrics, bootstrap samples and intervals. The 60 original H1 target probability arrays replay exactly (maximum error 0); all source predictions also replay exactly. **{V["protected_files_unchanged"]:,} protected H1/diagnostic artifacts remain byte-identical.**\n')
add(f'M3 Pro CPU, 18 GiB, one BLAS thread: inference/checks {S["prediction_and_checks_s"]:.2f} s; complete prediction/scoring/bootstrap {S["total_s"]:.2f} s; independent verification {V["verification_s"]:.2f} s. No new fits, feature extraction, GPU use or downloads. Peak RAM was not measured.\n')
add('Intervals use 10,000 paired draws (seed 314159) over four whole target groups, six source folds and five training selections; no draw lacked a class. They are conditional on fixed, overlapping source fits and a few uneven conservative groups, with uncertain coverage. Original recording/vehicle independence and pretrained-data overlap remain unknown. Checkpoint provenance retains the original mirror/official-checksum qualification. This is exposed development evaluation, not new independent confirmation.\n')

add('## Decision and preparation for the existing source × propagation design\n')
add('Keep H1 unchanged and retain source-selected regularization as a documented sensitivity, not a replacement baseline. There is evidence of a partial BEATs transfer benefit in balanced accuracy/recall, but no established primary F1 gain or resolution of worst-group failures. Close this bounded check; do not launch another C/threshold sweep on MELAUDIS.\n')
add('The next physical comparison remains the existing [H2 four-cell proposal](proposed_experiments.md#7-h2--compare-source-and-propagation-contributions-e4e5):\n')
table(['Cell','Source','Propagation'],[
    ['F00','S0: audited simple procedural dry source','P0: moving direct path, spreading/delay/Doppler'],
    ['F10','S1: paired source-envelope/fluctuation/component-balance diversity','Same P0'],
    ['F01','Same S0','P1: same path/trajectory plus ground reflection and atmospheric filtering'],
    ['F11','Same S1','Same P1']])
add('Before implementing that comparison, freeze the source/backend identity, numerical factor ranges and paired source/trajectory manifests, and a common head/decision rule. Keep bandwidth, class/event budget, environment and sensor treatment identical across cells; source/path effects and their interaction must all be reported. The current proposal’s RMS matching conditions out absolute attenuation/SNR effects and must remain explicit. The released synthetic banks have insufficient generation lineage to relabel them as these controlled cells. No unavailable anechoic corpus or BVP access is required for the proposed procedural comparison.\n')
add('If a later synthetic-training comparison uses IDMT-selected regularization, declare the additional real-label validation access and apply the same rule to every cell. Do not describe it as a zero-real-label experiment or select separate C values from target results. Source dominance remains a hypothesis; this study manipulated the classifier alone and supplies no causal source-versus-propagation ranking. No simulator was implemented or modified in this follow-up.\n')
(ROOT/'reports/H1_regularization_transfer.md').write_text('\n'.join(P))
print('Wrote reports/H1_regularization_transfer.md')

"""Render the completed diagnostic findings and two scientific plots from artifacts."""
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'experiments/h1_diagnostics/results/source_diagnostics_20261003_v1'
FROZEN = ROOT/'experiments/h1_diagnostics/frozen/source_diagnostics_20261003_v1'
S = json.loads((RUN/'summary.json').read_text())
V = json.loads((RUN/'verification.json').read_text()); assert V['passed']
L = json.loads((FROZEN/'lock.json').read_text())
CFG = json.loads((FROZEN/'config.json').read_text())
E = json.loads((RUN/'evaluations.json').read_text())
CHOICES = json.loads((RUN/'selections.json').read_text())
ROWS = [json.loads(s) for s in (FROZEN/'source_manifest.jsonl').read_text().splitlines()]
EXTRACT = json.loads((ROOT/'experiments/h1_diagnostics/cache/source_diagnostics_20261003_v1/summary.json').read_text())
P = []


def add(s=''): P.append(s)
def table(headers, rows):
    add('| ' + ' | '.join(headers) + ' |')
    add('| ' + ' | '.join('---' for _ in headers) + ' |')
    for row in rows: add('| ' + ' | '.join(str(v) for v in row) + ' |')
    add()
def held(rep, case, metric): return S['aggregate'][rep][case]['held_out'][metric]['mean']
def train(rep, case, metric): return S['aggregate'][rep][case]['training_at_threshold_0_5'][metric]
def pct(v): return f'{100*v:.2f}'
def delta(rep, case, metric='macro_f1', site=False):
    d = S['paired_contrasts'][rep][case][metric]
    lo, hi = d['site_seed_ci95' if site else 'group_seed_ci95']
    return f'{100*d["mean"]:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]'
def records(rep, case, fold=None):
    return [r for r in E if r['representation']==rep and r['case']==case and (fold is None or r['fold']==fold)]
def mean(rr, m): return float(np.mean([r['metrics'][m] for r in rr]))
def label(fold):
    rs = [r for r in ROWS if r['provenance_group_id']==fold['held_out_group']]
    return fold['site']+' '+rs[0]['calendar_date']


add('# H1 source-only diagnostic findings')
add('\n2026-10-03. Protocol `h1_source_diagnostics_v1`; run `source_diagnostics_20261003_v1`. **All five frozen checks are complete. The test domain throughout is IDMT real → held-out IDMT real, car versus truck.** No new MELAUDIS/synthetic evaluation was performed. This is a development diagnostic, not a transfer result.\n')
add('The main finding is a large loss when an entire location is unseen, alongside a substantial BEATs training/generalization gap. More excerpts from the existing groups barely improve the reference. Restoring bandwidth helps modestly; threshold tuning improves macro-F1 at the expense of truck recall. These results do not identify source-model mismatch or propagation-model mismatch as the dominant synthetic-to-real cause.\n')
add('## Evidence and interpretation rules\n')
add('- [Frozen protocol](../experiments/h1_diagnostics/frozen/source_diagnostics_20261003_v1/PROTOCOL.md), [exact selections and hashes](../experiments/h1_diagnostics/frozen/source_diagnostics_20261003_v1/lock.json), [config](../experiments/h1_diagnostics/frozen/source_diagnostics_20261003_v1/config.json).\n- [All aggregates and paired intervals](../experiments/h1_diagnostics/results/source_diagnostics_20261003_v1/summary.json), [every outer evaluation](../experiments/h1_diagnostics/results/source_diagnostics_20261003_v1/evaluations.json), [inner selections](../experiments/h1_diagnostics/results/source_diagnostics_20261003_v1/selections.json), [verification](../experiments/h1_diagnostics/results/source_diagnostics_20261003_v1/verification.json), [reproduction commands](../experiments/h1_diagnostics/README.md).\n- [Unchanged original H1](H1_baseline_results.md) and [qualified data admission](dataset_admission_provenance.md).\n')
add('Use all 4,413 admitted sE8 CH34 excerpts: 3,902 cars and 511 trucks, six conservative site/date groups at three locations. Five seeds vary training selection: 42, 123, 456, 789, 1024. Unless named otherwise, reuse H1’s exact 190/class selections, two-second mean-channel native→8 kHz→16 kHz waveform, frozen official BEATs768 or MFCC26, train-only StandardScaler and C=1 logistic regression. No augmentation, encoder training, relabelling or padding.\n')
add('Scores below are percentages, except Brier/log loss/ECE and explicitly stated raw units. Primary scores average each of six group metrics equally, then five seeds; they are not pooled clip scores or ensemble predictions. Paired differences are percentage points. Brackets are 95% conditional percentile intervals from 10,000 paired whole-group/seed resamples; a three-site block sensitivity carries both dates together. Six groups/three sites give uncertain coverage, and these diagnostic comparisons are not multiplicity-adjusted confirmation. The source groups were already evaluated in H1. Three F1 points is a practical reference, not a gate.\n')

add('## D1 — Can the head fit the training data, and does regularization help?\n')
table(['Representation', 'Train F1', 'Held-out F1', 'Train BA', 'Held-out BA', 'Train ROC-AUC', 'Held-out ROC-AUC'],
      [[r]+[pct(fn(r,'baseline',m)) for m,fn in [('macro_f1',train),('macro_f1',held),('balanced_accuracy',train),('balanced_accuracy',held),('roc_auc',train),('roc_auc',held)]] for r in CFG['representations']])
add('**Finding:** BEATs perfectly separates every balanced training selection but loses 34.13 balanced-accuracy points and 28.51 AUC points on held-out groups. Thus the linear head can fit these features; the poor held-out score is not simple failure to fit training labels. This is consistent with overfitting and recording-context sensitivity, without proving which cues were learned. MFCC has a smaller 8.32-point BA fit gap and stronger held-out discrimination. Training is balanced while testing is naturally imbalanced, so a train/test F1 difference alone would not establish overfitting.\n')
add('Nested regularization uses five inner held-out groups inside each outer fold and C∈{.01,.1,1}, with mean inner-group F1 selecting C before outer evaluation. Every final fit retains the original 190/class selection.\n')
table(['Representation', 'Selected C counts / 30', 'Nested F1', 'ΔF1 [group CI]', 'ΔF1 [site CI]', 'Nested BA', 'Nested AUC'],
      [[r, ', '.join(f'{c:g}: {n}' for c,n in sorted(Counter(x['chosen_C'] for x in CHOICES if x['representation']==r).items())), pct(held(r,'nested_C','macro_f1')),delta(r,'nested_C'),delta(r,'nested_C',site=True),pct(held(r,'nested_C','balanced_accuracy')),pct(held(r,'nested_C','roc_auc'))] for r in CFG['representations']])
add('BEATs benefits modestly from stronger regularization: +1.81 F1 points, with the conditional interval entirely below the three-point practical reference. Training BA falls to 90.07%, held-out BA rises to 68.39%, and AUC rises to 75.17%. MFCC does not show a corresponding F1 benefit. BEATs’ worst individual group/seed recall worsens from 40% to 30% despite the mean improvement. Inner tuning sees additional source validation labels; its label access differs from H1’s fixed untuned rule. This result does not justify silently replacing H1 or claiming transfer improvement.\n')

add('## D2 — Is the 190-per-class budget the main bottleneck?\n')
add('Training selections are classwise nested prefixes at N=25,50,100,190,300,359. The largest budget feasible in every outer fold is 359/class. The primary curve fixes C=1; the secondary C=190/N curve holds the installed logistic solver’s mean-loss L2 coefficient constant. Feature scales are still training-estimated. Both curves reuse the same six groups.\n')
table(['N / class', 'BEATs F1, C=1', 'BEATs F1, C=190/N', 'MFCC F1, C=1', 'MFCC F1, C=190/N'],
      [[n]+[pct(held(r,f'learning_{rule}_{n}','macro_f1')) for r in CFG['representations'] for rule in ['fixed','scaled']] for n in CFG['learning_curve_examples_per_class']])
table(['Representation', 'Contrast', 'ΔF1 [group CI]', 'ΔF1 [site CI]'],
      [[r,c.replace('_',' '),delta(r,c),delta(r,c,site=True)] for r in CFG['representations'] for c in ['fixed_359_minus_190','scaled_359_minus_190','fixed_359_minus_25','scaled_359_minus_25']])
add('**Finding:** nearly doubling 190→359 examples/class gives BEATs +0.76 F1 points (+1.16 under the penalty control), and MFCC −0.68 (−0.76 controlled); all intervals include zero. BEATs shows a gradual upward curve, but no demonstrated ≥3-point gain from relaxing the current budget. The controlled MFCC 25→359 contrast is positive, so this is not evidence that sample count never matters. It says that adding excerpts from these same groups is not an established remedy for the current baseline. Larger, more diverse collections remain untested. Do not replace the matched H1 budget or choose the highest-scoring N post hoc.\n')
add('![Learning curves](figures/h1_source_learning_curves.png)\n')

add('## D3 — Are bandwidth or context length hiding useful information?\n')
table(['Representation', 'Observation', 'F1', 'BA', 'ROC-AUC', 'ΔF1 [group CI]', 'ΔF1 [site CI]'],
      [[r,c,pct(held(r,c,'macro_f1')),pct(held(r,c,'balanced_accuracy')),pct(held(r,c,'roc_auc')),delta(r,c),delta(r,c,site=True)] for r in CFG['representations'] for c in ['native16_2s','core8_1s']])
add('**Bandwidth finding:** direct native→16 kHz gives +2.97 BEATs and +2.89 MFCC F1 points. The group and site intervals are above zero but cross the three-point practical reference. BA improves by 4.56 and 4.65 points, respectively. This supports sensitivity to the bandwidth/resampling route in IDMT. It does not isolate upper-frequency source information from road, propagation or sensor cues, because the resampling route changes too. The original 8 kHz intermediate made the procedural/real bandwidth comparable; a real-only full-band result cannot be substituted into that comparison.\n')
add('**Context finding:** one second versus two gives BEATs −1.01 and MFCC +0.66 F1 points, both intervals crossing zero. MFCC BA rises from 71.00% to 73.59%, but there is no consistent F1 advantage across representations. The one-second encoder sees true one-second input, not zero/repeat padding. Released excerpts are only about two seconds long, so this check cannot address longer pass-by context. No combination of regularization, bandwidth and threshold “winners” was run.\n')

add('## D4 — Is useful ranking hidden by the decision rule, and are probabilities reliable?\n')
table(['Representation / decision', 'F1', 'BA', 'Car precision', 'Car recall', 'Truck precision', 'Truck recall'],
      [[r+' / '+c]+[pct(held(r,c,m)) for m in ['macro_f1','balanced_accuracy','car_precision','car_recall','truck_precision','truck_recall']] for r in CFG['representations'] for c in ['baseline','nested_threshold']])
table(['Representation', 'ROC-AUC', 'Truck AP', 'Brier', 'Log loss', 'ECE10'],
      [[r,pct(held(r,'baseline','roc_auc')),pct(held(r,'baseline','average_precision'))]+[f'{held(r,"baseline",m):.4f}' for m in ['brier_truck','log_loss','ece10_truck']] for r in CFG['representations']])
add('The equal-group mean truck prevalence is 10.26% (the pooled dataset prevalence is 11.58%). Mean AP exceeds that prevalence reference for both representations, and every group’s mean AP exceeds its own prevalence. Thus there is useful within-group ranking. It is not strong enough to support reliable binary decisions: baseline truck precision is only 18.87% for BEATs and 21.34% for MFCC. The group-specific AP references appear below.\n')
add('The fixed C=1 inner predictions select thresholds on {.05,.10,…,.95}, without using the outer group. BEATs chooses .95 in 29/30 cases and .90 once; MFCC chooses .75 (4), .80 (8), .85 (16), .90 (2). The grid was not expanded when BEATs reached its upper boundary.\n')
table(['Representation', 'ΔF1 [group CI]', 'ΔF1 [site CI]', 'ΔBA [group CI]'],
      [[r,delta(r,'nested_threshold'),delta(r,'nested_threshold',site=True),delta(r,'nested_threshold','balanced_accuracy')] for r in CFG['representations']])
add('**Finding:** threshold tuning raises F1 by 6.73 BEATs / 7.40 MFCC points, but truck recall falls from 65.60→41.84% / 72.34→36.71%. BA decreases 2.54 / 6.27 points. Worst truck recall becomes 7.14% for BEATs and 0% for MFCC. This is a majority-class tradeoff, not improved acoustic discrimination. Ranking and all probability metrics remain exactly unchanged because only the decision threshold changed.\n')
add('Baseline probability errors are substantial: mean truck-probability ECE is .3041/.2918. For orientation, always predicting car has 47.27% mean group F1, 50% BA and zero truck recall; its Brier is .1026, lower than either learned baseline but uninformative for trucks. BEATs regularization improves Brier .2750→.2106 while ECE slightly worsens .3041→.3111; probability quality is not summarized by a single number. Balanced-training and test priors differ, and conditional distribution shift also remains possible. We did not isolate prior shift, fit a probability calibrator or estimate a target-domain threshold. Threshold selection is not calibration.\n')

add('## D5 — How much does performance depend on recording group and location?\n')
add('Each row below averages five selection seeds; brackets around F1 show the observed seed range, not a confidence interval. “Site-held F1” trains on the other two sites, evaluates this same date group, and uses the same N=190/class and C=1.\n')
for rep in CFG['representations']:
    add('### '+rep+'\n')
    rr = []
    for fold in L['folds']:
        rs = [r for r in ROWS if r['provenance_group_id']==fold['held_out_group']]
        br = records(rep,'baseline',fold['fold']); ls = records(rep,'held_site',fold['fold'])
        fs = [r['metrics']['macro_f1'] for r in br]
        rr.append([label(fold),f'{sum(r["class_id"]==0 for r in rs)}/{sum(r["class_id"]==1 for r in rs)}',
                   f'{pct(np.mean(fs))} [{pct(min(fs))}, {pct(max(fs))}]',pct(mean(br,'balanced_accuracy')),
                   pct(mean(br,'car_recall')),pct(mean(br,'truck_recall')),pct(mean(br,'roc_auc')),
                   pct(mean(br,'average_precision'))+' / '+pct(mean(br,'truck_prevalence')),pct(mean(ls,'macro_f1'))])
    table(['Held group','Car/truck','F1 [seed range]','BA','Car recall','Truck recall','AUC','AP / prevalence','Site-held F1'],rr)
add('### Entire-site holdout\n')
table(['Representation','Date-group F1','Site-held F1','ΔF1 [group CI]','ΔF1 [site CI]','Date-group BA','Site-held BA'],
      [[r,pct(held(r,'baseline','macro_f1')),pct(held(r,'held_site','macro_f1')),delta(r,'held_site'),delta(r,'held_site',site=True),pct(held(r,'baseline','balanced_accuracy')),pct(held(r,'held_site','balanced_accuracy'))] for r in CFG['representations']])
add('**Finding:** holding out a site costs 10.21 BEATs and 11.46 MFCC F1 points; both resampling schemes put these decreases below zero. The largest losses occur on Schleusinger-Allee, with additional losses on Fraunhofer. Langewiesener changes much less. Mean car recall collapses to 48.80% / 49.36%, while truck recall rises to 69.63% / 82.58%. BEATs AUC falls .7149→.6384; MFCC .8022→.7451. The MFCC BA-change interval includes zero, despite the clear F1 loss. The site-held result must therefore be read as more than a single aggregate number.\n')
add('Across individual group/seed fits, baseline minimum class recall is 40% BEATs / 30% MFCC. Site-held minima are 10% BEATs (truck) / 20.12% MFCC (car). Only 10 and 14 trucks occur in the two Fraunhofer groups, so recall is especially coarse and uncertain there. These are repeated predictions of the same retained events, not additional independent vehicles.\n')
table(['Group','Direction counts','Road token counts','Posted limit'],
      [[label(f), ', '.join(f'{k}:{v}' for k,v in sorted(Counter(r['direction'] for r in ROWS if r['provenance_group_id']==f['held_out_group']).items())),
        ', '.join(f'{k}:{v}' for k,v in sorted(Counter(r['road_condition_token'] for r in ROWS if r['provenance_group_id']==f['held_out_group']).items())),
        ', '.join(sorted({r['speed_limit_token'] for r in ROWS if r['provenance_group_id']==f['held_out_group']}))] for f in L['folds']])
add('Both directions occur in all groups. Posted limits are completely tied to location (30/50/70 km/h), and wet-road tokens occur only at Schleusinger. A posted limit is not measured vehicle speed; D/W is a road-condition annotation, not a weather measurement. Acquisition uses the same sE8 channel selection throughout, so this is not a held-out-sensor experiment. Actual vehicle identity, RPM, load, exact distance and original recording-session independence are not established.\n')
add('The location check also reduces training-group diversity from five groups to four. Location bundles fleet/source states, road/path/background and recording context. Consequently the loss demonstrates location-associated vulnerability under this split, but does not isolate propagation or any other physical factor. We did not train a location classifier or infer missing physical metadata.\n')
add('![Group versus site holdout](figures/h1_source_location_comparison.png)\n')

add('## Verification, runtime and scope\n')
add(f'Eight metadata/selection/threshold tests passed. Two artificial-waveform encoder checks established exact two-second agreement with the existing adapter and finite repeatable one-second output. Independent verification checked **{V["unique_fits_verified"]:,} training-only scalers/heads, {V["outer_predictions_verified"]:,} outer evaluations, {V["inner_predictions_verified"]:,} inner evaluations and all {V["nested_choices_verified"]} nested choices**, including saved probabilities, class/group separation, scalar metrics and all aggregate/paired intervals. All 60 H1 source prediction arrays reproduce exactly (maximum absolute error 0); 798 protected H1 files remain byte-identical. No target probability array was decoded.\n')
add(f'Apple M3 Pro CPU, 18 GiB, four Torch threads and one BLAS thread: extraction **{EXTRACT["total_s"]:.2f} s**, fitting/scoring/resampling **{S["total_s"]:.2f} s**, independent verification **{V["verification_s"]:.2f} s**. Three profiles × 4,413 files = 13,239 observations. No GPU, download or cloud service was needed. Each model, training list, score, feature hash, config, code snapshot and software version is retained. Peak RAM was not measured.\n')
add('These checks establish reproducible computation and separation of known provenance groups. They cannot certify unknown original-session/physical-vehicle independence, annotation correctness, pretrained-data non-overlap or performance outside these three sites. No source simulator, propagation model, background randomizer or sensor model was altered. H1 remains a historical fixed reference; military milestones and protected evaluation sources are unaffected.\n')

add('## Consequences for the next experimental phase\n')
add('1. **Keep H1, and add an explicitly frozen source-diagnostic successor before component attribution.** The real reference is limited by generalization as well as decision/prior effects. Preserve the fixed C=1 result; source-group-only regularization is a supported candidate control, not a demonstrated target improvement. MFCC remains necessary because it still outperforms BEATs within IDMT.\n2. **Do not spend the next phase merely increasing excerpt count or maximizing F1 through thresholds.** More independent contexts are more informative than more clips from these same groups. This is a research priority suggested by D2/D5, not a causal demonstration that collecting new sites will solve the gap. Keep class recall, BA, ROC-AUC, AP and worst-group outcomes alongside F1. Any future threshold rule needs a preregistered tradeoff and group-separated source validation.\n3. **Make bandwidth and operating context explicit in the source × propagation factorial.** Use matched bandwidth in every cell; carry a separate full-band sensitivity if all cells can supply it. Use the same head, threshold rule, event budget and physical-condition distributions across paired cells. Otherwise a roughly three-point bandwidth effect or a roughly ten-point site-associated loss could be mistaken for a simulator-component effect.\n4. **Do not infer that propagation dominates either.** Location is entangled with source fleet/state, road conditions, background and acquisition context. The source-dominance hypothesis remains untested. It still requires auditable source alternatives crossed with specified propagation alternatives, matched environment/sensor treatment, and an explicit interaction estimate.\n5. **Freeze any subsequent transfer comparison before target access.** No new MELAUDIS evaluation was run here. Existing MELAUDIS remains exposed development data; redesign cannot turn it into untouched confirmation. A later transfer sensitivity must declare its exposure, and genuinely unexposed provenance-clean recordings are needed for an independent confirmation claim. No combined “best” diagnostic configuration has been evaluated or promoted.\n')
add('The immediate change is therefore interpretive and procedural: the project now has a reproducible diagnosis of the weak source reference. It has not yet tested a physical source-versus-propagation intervention. Finish the next comparison’s freeze and provenance requirements before implementing that factorial; no new architecture is justified by this pass alone.\n')

# Plots show all predeclared budgets and sites, not a selected best result.
figdir = ROOT/'reports/figures'; figdir.mkdir(exist_ok=True)
fig, axes = plt.subplots(1,2,figsize=(10,4),sharey=True,layout='constrained')
ns = CFG['learning_curve_examples_per_class']
for ax, rep in zip(axes, CFG['representations']):
    for rule, lab, style in [('fixed','C = 1','o-'),('scaled','C = 190/N','s--')]:
        ax.plot(ns, [100*held(rep,f'learning_{rule}_{n}','macro_f1') for n in ns], style, label=lab)
    ax.axvline(190,color='gray',ls=':',lw=1); ax.set_title(rep); ax.set_xlabel('Training excerpts per class')
    ax.grid(alpha=.2); ax.legend(); ax.set_ylim(45,65)
axes[0].set_ylabel('Mean held-group macro-F1 (%)')
fig.suptitle('IDMT source-only learning curves: six groups, five selections')
fig.savefig(figdir/'h1_source_learning_curves.png',dpi=170); plt.close(fig)
fig, axes = plt.subplots(1,2,figsize=(11,5.4),sharex=True,sharey=True,layout='constrained')
for ax, rep in zip(axes, CFG['representations']):
    yy = np.arange(6)
    a = [100*mean(records(rep,'baseline',f['fold']),'macro_f1') for f in L['folds']]
    b = [100*mean(records(rep,'held_site',f['fold']),'macro_f1') for f in L['folds']]
    ax.hlines(yy,a,b,color='gray',alpha=.5); ax.scatter(a,yy,label='Held date group'); ax.scatter(b,yy,marker='s',label='Held entire site')
    ax.set_title(rep); ax.set_xlabel('Mean macro-F1 (%)'); ax.grid(axis='x',alpha=.2)
axes[0].set_yticks(np.arange(6),[label(f).replace('Langewiesener-Strasse','Langewiesener').replace('Schleusinger-Allee','Schleusinger') for f in L['folds']])
axes[0].invert_yaxis(); fig.suptitle('IDMT location sensitivity; same held-out events, N = 190/class')
fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=2)
fig.savefig(figdir/'h1_source_location_comparison.png',dpi=170); plt.close(fig)
(ROOT/'reports/H1_source_diagnostics.md').write_text('\n'.join(P))
print('Wrote reports/H1_source_diagnostics.md and two figures')

"""Summarize every declared diagnostic and make standalone scientific figures."""
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

FIGURES = ROOT/'reports/H2_ground_air_collapse_figures'
LABELS = ['Direct', 'Air only', 'Ground only', 'Ground + air']


def mean_fields(records, field='metrics'):
    keys = ['macro_f1', 'balanced_accuracy', 'car_recall', 'truck_recall',
            'predicted_truck_fraction', 'roc_auc', 'average_precision',
            'brier_truck', 'log_loss', 'ece10_truck']
    return {key: float(np.mean([r[field][key] for r in records])) for key in keys}


def describe(x):
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), median=float(np.median(x)),
                p05=float(np.quantile(x, .05)), p95=float(np.quantile(x, .95)),
                minimum=float(x.min()), maximum=float(x.max()))


def make_facts():
    audit = read(RESULTS/'verification.json')
    assert audit['passed'] and audit['result_summary_sha256'] == sha(RESULTS/'summary.json')
    evaluations = read(RESULTS/'evaluations.json')
    stats = read(RESULTS/'statistics.json')
    cross = read(RESULTS/'cross_path_validation.json')
    gain = read(RESULTS/'validation_gain.json')
    diagnostics = read(RESULTS/'collapse_diagnostics.json')
    fits = read(MODELS/'fits.json')
    facts = dict(target={}, cross_path={}, gain_stress={}, feature_shift={}, waveform={})
    for rep in REPS:
        facts['target'][rep] = {}
        facts['cross_path'][rep] = {}
        facts['gain_stress'][rep] = {}
        facts['feature_shift'][rep] = {}
        for source in [*SOURCES, 'mean_S']:
            facts['target'][rep][source] = {}
            for path in PATHS:
                facts['target'][rep][source][path] = {}
                for variant in ['native', 'matched']:
                    selected = [r for r in evaluations if r['representation'] == rep and r['path_level'] == path
                                and r['variant'] == variant and (source == 'mean_S' or r['source_level'] == source)]
                    facts['target'][rep][source][path][variant] = dict(
                        **mean_fields(selected, 'target'),
                        worst_group_class_recall=min(r['worst_group_class_recall'] for r in selected),
                        group_average_car_recall=float(np.mean([r['group_average_car_recall'] for r in selected])),
                        group_average_truck_recall=float(np.mean([r['group_average_truck_recall'] for r in selected])),
                        groups={group: {k:float(np.mean([r['groups'][group][k] for r in selected]))
                                        for k in ['macro_f1','balanced_accuracy','car_recall','truck_recall']}
                                for group in selected[0]['groups']})
        for path in PATHS:
            facts['cross_path'][rep][path] = {}
            facts['gain_stress'][rep][path] = {}
            facts['feature_shift'][rep][path] = {}
            for test_path in PATHS:
                selected = [r for r in cross if r['representation'] == rep and r['train_path'] == path and r['test_path'] == test_path]
                facts['cross_path'][rep][path][test_path] = dict(**mean_fields(selected),
                    worst_fit_class_recall=min(r['metrics'][k] for r in selected for k in ['car_recall', 'truck_recall']))
            for db in [-12., 0., 12.]:
                selected = [r for r in gain if r['representation'] == rep and r['path_level'] == path and r['gain_db'] == db]
                facts['gain_stress'][rep][path][str(int(db))] = dict(**mean_fields(selected),
                    mean_car_logit=float(np.mean([r['logit_by_class']['car']['mean'] for r in selected])),
                    mean_truck_logit=float(np.mean([r['logit_by_class']['truck']['mean'] for r in selected])))
            selected = [r for r in diagnostics if r['representation'] == rep and r['path_level'] == path]
            for domain in ['train', 'validation', 'native', 'matched']:
                facts['feature_shift'][rep][path][domain] = {}
                for name in ['car', 'truck']:
                    rr = [(r[domain] if domain in ['train', 'validation'] else r['targets'][domain])['classes'][name] for r in selected]
                    values = dict(
                        mean_logit=float(np.mean([r['logit']['mean'] for r in rr])),
                        mean_feature_norm=float(np.mean([r['standardized_feature_rms_norm']['mean'] for r in rr])),
                        above_train_p95=float(np.mean([r['above_training_norm_p95_fraction'] for r in rr])),
                        outside_train_coordinate_range=float(np.mean([r['mean_fraction_coordinates_outside_training_range'] for r in rr])),
                        mean_logit_shift=float(np.mean([r['mean_logit_shift_from_same_training_class'] for r in rr])))
                    if rep == 'MFCC26':
                        values['mean_C0_score_shift'] = float(np.mean([r['mean_logit_shift_contributions'][0] for r in rr]))
                    facts['feature_shift'][rep][path][domain][name] = values
    with np.load(CACHE/'features.npz', allow_pickle=False) as cache:
        ids = cache['job_ids'].tolist()
        obs = {r['job_id']: r for r in rows(FREEZE/'observations.jsonl')}
        names = cache['acoustic_names'].tolist()
        synthetic = cache['acoustic'].copy()
        for source in [*SOURCES, 'mean_S']:
            for path in PATHS:
                for label, name in enumerate(['car', 'truck']):
                    keep = [i for i,j in enumerate(ids) if obs[j]['role'] == 'train' and obs[j]['class_id'] == label
                            and obs[j]['path_level'] == path and (source == 'mean_S' or obs[j]['source_level'] == source)]
                    facts['waveform'][f'{source}.{path}.{name}'] = {key:describe(synthetic[keep, k]) for k,key in enumerate(names)}
    with np.load(CACHE/'target_gain/features.npz', allow_pickle=False) as cache:
        labels = np.array([r['class_id'] for r in rows(FREEZE/'target_manifest.jsonl')])
        for variant, key in [('native','original_acoustic'),('matched','normalized_acoustic')]:
            for label,name in enumerate(['car','truck']):
                facts['waveform'][f'{variant}.{name}'] = {n:describe(cache[key][labels == label,k]) for k,n in enumerate(names)}
        facts['target_gain_db'] = {name:describe(20*np.log10(cache['gain'][labels == k])) for k,name in enumerate(['car','truck'])}
    facts['validation'] = dict(heads=len(fits), minimum_macro_f1=min(r['validation']['macro_f1'] for r in fits),
                              maximum_macro_f1=max(r['validation']['macro_f1'] for r in fits))
    facts['statistics'] = stats
    facts['target_gain_metadata'] = read(CACHE/'target_gain/summary.json')
    facts['verification'] = audit
    facts['summary'] = read(RESULTS/'summary.json')
    facts['runtime_seconds'] = dict(generate=read(CORPUS/'summary.json')['elapsed_s'],
        complete_replay=read(CORPUS/'verification.json')['elapsed_s'],
        synthetic_features_and_gain=read(CACHE/'summary.json')['elapsed_s'],
        fit=read(MODELS/'lock.json')['elapsed_s'],target_gain=read(CACHE/'target_gain/summary.json')['elapsed_s'],
        evaluate=read(RESULTS/'summary.json')['elapsed_s'],audit=audit['elapsed_s'])
    save(RESULTS/'diagnostic_summaries.json', facts)
    return facts


def finish(fig, name):
    fig.savefig(FIGURES/(name+'.png'), dpi=170, bbox_inches='tight')
    fig.savefig(FIGURES/(name+'.svg'), bbox_inches='tight')
    plt.close(fig)


def figures(f):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    stats = f['statistics']['estimates']
    for offset, variant, color in [(-.17, 'native', '#2465a3'), (.17,'matched','#dd8530')]:
        values = [stats[variant]['BEATs768']['mean_S'][p]['macro_f1'] for p in PATHS]
        point = np.array([v['estimate'] for v in values])*100
        ci = np.array([v['ci95'] for v in values])*100
        axes[0].bar(np.arange(4)+offset, point, width=.32, color=color,
                    label='Native real level' if variant == 'native' else 'Fixed RMS matching')
        axes[0].vlines(np.arange(4)+offset, ci[:,0], ci[:,1], color='#252525', linewidth=1)
    axes[0].set(xticks=np.arange(4),xticklabels=LABELS,ylabel='Macro-F1 (%)',ylim=(0,100),title='BEATs: complete real target')
    axes[0].tick_params(axis='x',labelrotation=15)
    axes[0].axhline(f['summary']['constant_controls']['always_car']['macro_f1']*100, color='gray',ls=':',label='Always car')
    axes[0].legend(fontsize=8)
    keys = ['ground','air','interaction','ground_minus_air']
    for offset,variant,color in [(-.1,'native','#2465a3'),(.1,'matched','#dd8530')]:
        for i,key in enumerate(keys):
            item=stats[variant]['BEATs768']['mean_S'][key]['macro_f1']
            axes[1].plot(np.array(item['ci95'])*100,[i+offset]*2,color=color,lw=2)
            axes[1].scatter(item['estimate']*100,i+offset,color=color,s=25)
    axes[1].axvline(0,color='gray',ls=':')
    axes[1].set(yticks=np.arange(4),yticklabels=['Ground marginal','Air marginal','Interaction','Ground minus air'],
                xlabel='Macro-F1 change (percentage points)',title='Paired factor effects')
    axes[1].invert_yaxis()
    fig.suptitle('Mean of both fixed source levels and five banks; exposed MELAUDIS development\n95% paired four-group/five-bank intervals; no target tuning',fontsize=11)
    fig.tight_layout();finish(fig,'factor_effects')

    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    colors=['#333333','#b68417','#1a8b6b','#3d67a8']
    for ri,rep in enumerate(REPS):
        for path,label,color in zip(PATHS,LABELS,colors):
            vals=[f['gain_stress'][rep][path][str(db)] for db in [-12,0,12]]
            axes[ri,0].plot([-12,0,12],[v['car_recall']*100 for v in vals],'-o',label=label,color=color)
            axes[ri,1].plot([-12,0,12],[v['truck_recall']*100 for v in vals],'-o',label=label,color=color)
        for col,name in enumerate(['Car','Truck']):
            axes[ri,col].set(ylim=(-3,103),ylabel=f'{rep} {name.lower()} recall (%)',xticks=[-12,0,12])
            axes[ri,col].grid(alpha=.15)
    axes[0,0].legend(fontsize=8)
    for a in axes[-1]:a.set_xlabel('Fixed synthetic-validation gain change (dB)')
    fig.suptitle('Gain stress on disjoint synthetic validation templates\nSame waveforms, fixed heads, no refitting; mean of source levels and banks',fontsize=11)
    fig.tight_layout();finish(fig,'gain_stress')

    fig,axes=plt.subplots(1,2,figsize=(12,5.1),layout='constrained')
    for ax,rep in zip(axes,REPS):
        values=np.array([[f['cross_path'][rep][a][b]['macro_f1'] for b in PATHS] for a in PATHS])*100
        im=ax.imshow(values,vmin=0,vmax=100,cmap='viridis')
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{values[i,j]:.1f}',ha='center',va='center',color='white' if values[i,j]<55 else 'black')
        ax.set(xticks=range(4),xticklabels=LABELS,yticks=range(4),yticklabels=LABELS,title=rep,xlabel='Validation path',ylabel='Training path')
        ax.tick_params(axis='x',labelrotation=20)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Macro-F1 (%)',shrink=.7,pad=.025,fraction=.035)
    fig.suptitle('Unseen simulated path with source templates held out\nMean of both source levels and all five banks',fontsize=11)
    finish(fig,'cross_path_validation')

    fig,axes=plt.subplots(1,3,figsize=(14,4.7))
    categories=['Direct car','Direct truck','Real car','Real truck']
    keys=['mean_S.G0A0.car','mean_S.G0A0.truck','native.car','native.truck']
    for i,key in enumerate(keys):
        v=f['waveform'][key]['rms_dbfs']
        axes[0].plot([i,i],[v['p05'],v['p95']],color=colors[i],lw=3)
        axes[0].scatter(i,v['median'],color=colors[i],s=40)
    axes[0].axhline(-26,color='gray',ls=':')
    axes[0].set(xticks=range(4),xticklabels=categories,ylabel='RMS (dBFS)',title='Level: median and 5–95% range')
    axes[0].tick_params(axis='x',labelrotation=30)
    bands=['power_0_200','power_200_500','power_500_1000','power_1000_2000','power_2000_4000']
    for ax,cl in zip(axes[1:],['car','truck']):
        for source,label,color in [('mean_S.G0A0','Direct',colors[0]),('mean_S.G0A1','Air only',colors[1]),
                                   ('mean_S.G1A0','Ground only',colors[2]),('mean_S.G1A1','Ground + air',colors[3]),('native','Real','#b94470')]:
            ax.plot(range(5),[f['waveform'][source+'.'+cl][b]['mean']*100 for b in bands],'-o',label=label,color=color)
        ax.set(xticks=range(5),xticklabels=['0–200','200–500','500–1k','1k–2k','2k–4k'],
               title=f'{cl.title()}: mean normalized band power',ylabel='Share of 0–4 kHz power (%)',xlabel='Frequency (Hz)',ylim=(0,100))
        ax.tick_params(axis='x',labelrotation=30)
    axes[-1].legend(fontsize=8)
    fig.suptitle('Descriptive waveform differences; spectrum alone does not identify their physical origin',fontsize=11)
    fig.tight_layout();finish(fig,'waveform_differences')


def number(v):
    return f'{100*v:.2f}'


def interval(v):
    return f"{number(v['estimate'])} [{number(v['ci95'][0])}, {number(v['ci95'][1])}]"


def tables(f):
    lines=[]
    def add(*s):lines.extend(s)
    add('## Full-target cells', '', 'Percentages except Brier score. Point estimates equally average the two fixed source levels and five banks. Brackets are 95% paired whole-group/whole-bank intervals. Worst recall takes the minimum across sources, banks, groups and classes.', '',
        '| Representation | Path | Target level | Macro-F1 [95% CI] | Balanced accuracy | Car recall | Truck recall | Predicted truck | AUROC | Worst recall |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for rep in REPS:
        for path,label in zip(PATHS,LABELS):
            for variant in ['native','matched']:
                v=f['target'][rep]['mean_S'][path][variant]
                est=f['statistics']['estimates'][variant][rep]['mean_S'][path]['macro_f1']
                add(f"| {rep} | {label} | {variant} | {interval(est)} | "+' | '.join(number(v[k]) for k in ['balanced_accuracy','car_recall','truck_recall','predicted_truck_fraction','roc_auc','worst_group_class_recall'])+' |')
    add('', '![Paired factor results](H2_ground_air_collapse_figures/factor_effects.png)', '', '## Ground and air effects', '',
        'All values are macro-F1 percentage-point changes. The sole primary test is native BEATs, mean_S, ground minus air. Other intervals are descriptive.', '',
        '| Representation | Source | Target level | Ground marginal | Air marginal | Interaction | Ground minus air |',
        '|---|---|---|---:|---:|---:|---:|')
    for rep in REPS:
        for source in ['mean_S',*SOURCES]:
            for variant in ['native','matched']:
                v=f['statistics']['estimates'][variant][rep][source]
                add(f'| {rep} | {source} | {variant} | '+' | '.join(interval(v[k]['macro_f1']) for k in ['ground','air','interaction','ground_minus_air'])+' |')
    add('', '## Fixed target-level intervention', '',
        'Paired changes after setting each existing final 16 kHz crop to −26 dBFS RMS. No model changes, target-selected gain, clipping or threshold tuning.', '',
        '| Representation | Path | Macro-F1 change [95% CI] | Car recall change [95% CI] | Truck recall change [95% CI] |',
        '|---|---|---:|---:|---:|')
    for rep in REPS:
        for path,label in zip(PATHS,LABELS):
            v=f['statistics']['matched_minus_native'][rep]['mean_S'][path]
            add(f'| {rep} | {label} | '+' | '.join(interval(v[k]) for k in ['macro_f1','car_recall','truck_recall'])+' |')
    add('', '## Ranking and calibration', '',
        'AUROC/AP/Brier/ECE are averaged over individual heads; AP uses truck as positive and the target truck prevalence is 3.17%. Calibration numbers are descriptive on exposed development data.', '',
        '| Representation | Path | Level | AUROC % | Truck AP % | Brier | Log loss | ECE % |',
        '|---|---|---|---:|---:|---:|---:|---:|')
    for rep in REPS:
        for path,label in zip(PATHS,LABELS):
            for variant in ['native','matched']:
                v=f['target'][rep]['mean_S'][path][variant]
                add(f"| {rep} | {label} | {variant} | {number(v['roc_auc'])} | {number(v['average_precision'])} | {v['brier_truck']:.4f} | {v['log_loss']:.3f} | {number(v['ece10_truck'])} |")
    add('', '## Cross-path validation and synthetic gain stress', '',
        'All validation examples derive from source templates excluded from fitting. Rows in the figure are training paths; columns are validation paths. None of these synthetic results are real-domain validation.', '',
        '![Cross-path validation](H2_ground_air_collapse_figures/cross_path_validation.png)', '',
        '| Representation | Training path | Gain (dB) | Macro-F1 % | Car recall % | Truck recall % | Predicted truck % |',
        '|---|---|---:|---:|---:|---:|---:|')
    for rep in REPS:
        for path,label in zip(PATHS,LABELS):
            for db in ['-12','0','12']:
                v=f['gain_stress'][rep][path][db]
                add(f'| {rep} | {label} | {db} | '+' | '.join(number(v[k]) for k in ['macro_f1','car_recall','truck_recall','predicted_truck_fraction'])+' |')
    add('', '![Gain stress](H2_ground_air_collapse_figures/gain_stress.png)', '', '## Feature and waveform diagnostics', '',
        'Positive logistic scores predict truck. Norms use each head’s own training-only scaler. “Outside p95” is the fraction above that training feature-norm threshold, averaged across fixed source levels and banks. It is not a calibrated OOD detector.', '',
        '| Representation | Training path | Domain | Class | Mean score | Mean feature norm | Outside train norm p95 % | Coordinates outside train range % |',
        '|---|---|---|---|---:|---:|---:|---:|')
    for rep in REPS:
        for path,label in zip(PATHS,LABELS):
            for domain in ['train','validation','native','matched']:
                for name in ['car','truck']:
                    v=f['feature_shift'][rep][path][domain][name]
                    add(f"| {rep} | {label} | {domain} | {name} | {v['mean_logit']:.3f} | {v['mean_feature_norm']:.3f} | {number(v['above_train_p95'])} | {number(v['outside_train_coordinate_range'])} |")
    add('', 'MFCC26 comprises 13 coefficient means followed by 13 standard deviations; coordinate 0 is mean C0. The table below gives its contribution to the mean score shift from same-class synthetic training. Contributions sum with all other coordinates to the total; a large C0 term alone does not prove that level explains all errors. BEATs coordinates have no physical labels.', '',
        '| Path | Target level | Class | Total mean score shift | Mean C0 contribution |', '|---|---|---|---:|---:|')
    for path,label in zip(PATHS,LABELS):
        for variant in ['native','matched']:
            for name in ['car','truck']:
                v=f['feature_shift']['MFCC26'][path][variant][name]
                add(f"| {label} | {variant} | {name} | {v['mean_logit_shift']:.3f} | {v['mean_C0_score_shift']:.3f} |")
    add('', '![Waveform distributions](H2_ground_air_collapse_figures/waveform_differences.png)', '',
        'Exact per-source, per-bank and per-group metrics, feature-coordinate score contributions and waveform quantiles are retained in the linked machine-readable artifacts. Aggregated diagrams must not be interpreted as independent vehicle or recording-session evidence.', '')
    return '\n'.join(lines)


def main():
    f=make_facts();figures(f)
    detail=tables(f)
    (RESULTS/'report_tables.md').write_text(detail)
    prefix=(HERE/'REPORT_FINDINGS.md').read_text()
    # Narrative source lives beside the script; final report lives in reports/.
    prefix=prefix.replace('../../reports/','').replace('../../experiments/','../experiments/')
    rel='../experiments/h2_mechanisms/'
    artifact_folder=rel+'results/'+ID+'/'
    suffix='\n## Runtime and artifacts\n\n'
    suffix+='Measured on Apple M3 Pro, 18 GiB RAM: four render workers, four Torch CPU threads, one BLAS thread; no GPU.\n\n'
    suffix+='| Stage | Wall time (minutes) |\n|---|---:|\n'
    suffix+='\n'.join(f'| {k.replace("_"," ")} | {v/60:.2f} |' for k,v in f['runtime_seconds'].items())+'\n\n'
    suffix+='The audit preserved 22,316 parent H2 files and 8,257 historical files. Prediction replay error is zero; 504 primary/secondary F1, balanced-accuracy and truck-rate intervals were independently reconstructed from integer confusion counts.\n\n'
    for label,path in [
        ('Protocol',rel+'PROTOCOL.md'),('Configuration',rel+'config.json'),
        ('Design lock',rel+'frozen/'+ID+'/lock.json'),('Execution lock',rel+'execution/'+ID+'/lock.json'),
        ('Complete waveform replay',rel+'corpus/'+ID+'/verification.json'),('Model lock',rel+'models/'+ID+'/lock.json'),
        ('Independent result audit',artifact_folder+'verification.json'),
        ('Statistics and decisions',artifact_folder+'statistics.json'),
        ('Every target/class/group result',artifact_folder+'evaluations.json'),
        ('Every cross-path check',artifact_folder+'cross_path_validation.json'),
        ('Every synthetic gain check',artifact_folder+'validation_gain.json'),
        ('Per-head feature/score diagnostics',artifact_folder+'collapse_diagnostics.json'),
        ('Waveform and diagnostic summaries',artifact_folder+'diagnostic_summaries.json'),
        ('Post-result air-filter phase inspection',artifact_folder+'air_filter_phase_inspection.json'),
        ('Commands',rel+'README.md')]:
        suffix+=f'- [{label}]({path})\n'
    (ROOT/'reports/H2_ground_air_collapse.md').write_text(prefix+'\n'+detail+suffix)
    save(RESULTS/'report_generation.json',dict(report_sha256=sha(ROOT/'reports/H2_ground_air_collapse.md'),
        script_sha256=sha(Path(__file__)),narrative_sha256=sha(HERE/'REPORT_FINDINGS.md'),
        result_summary_sha256=sha(RESULTS/'summary.json'),verification_sha256=sha(RESULTS/'verification.json'),
        facts_sha256=sha(RESULTS/'diagnostic_summaries.json'),
        figures_sha256={p.name:sha(p) for p in FIGURES.iterdir() if p.is_file()}))
    print('Report facts, tables, and four PNG/SVG figures generated.',flush=True)


if __name__=='__main__':main()

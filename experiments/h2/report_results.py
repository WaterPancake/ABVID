"""Render locked H2 results and synthetic manipulation/physics figures; no fitting."""
import argparse
from collections import defaultdict
from pathlib import Path
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from metadata import HERE,ROOT,read,rows
from metrics import CELLS


def pct(value): return f'{100*value:.2f}'
def estimate(record): return f"{pct(record['estimate'])} [{pct(record['ci95'][0])}, {pct(record['ci95'][1])}]"
def link(label,path,parent): return f'[{label}]({os.path.relpath(path,parent)})'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--corpus',type=Path,required=True)
    ap.add_argument('--models',type=Path,required=True);ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists(): raise ValueError('Report exists; do not overwrite a published result')
    result=args.results;stats=read(result/'statistics.json');summary=read(result/'summary.json')
    audit=read(result/'verification.json');corpus=read(args.corpus/'summary.json');corpus_check=read(args.corpus/'verification.json')
    if not audit['passed'] or audit['full_waveform_replays']!=9600: raise ValueError('Final independent audit required')
    fits=read(args.models/'fits.json');evals=read(result/'evaluations.json')
    model_lock=read(args.models/'model_lock.json')
    features=read(Path(model_lock['features_path'])/'summary.json')
    figures=args.output.parent/(args.output.stem+'_figures');figures.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    # All cells and both representations; confidence intervals are paired group/bank bootstrap.
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,rep in zip(axes,['BEATs768','MFCC26']):
        values=[stats['estimates'][rep]['cells'][cell]['macro_f1'] for cell in CELLS]
        point=np.array([v['estimate'] for v in values]);bounds=np.array([v['ci95'] for v in values])
        ax.vlines(np.arange(4),bounds[:,0],bounds[:,1],color='#176b87')
        ax.scatter(np.arange(4),point,color='#176b87',zorder=3)
        ax.axhline(summary['constant_controls']['always_car']['macro_f1'],color='#888888',linestyle='--',label='Always car')
        ax.set_xticks(np.arange(4),['S0 P0','S0 P1','S1 P0','S1 P1']);ax.set_title(rep);ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Target macro-F1');axes[1].legend(loc='best')
    fig.suptitle('Synthetic → previously exposed MELAUDIS development data')
    fig.tight_layout();fig.savefig(figures/'target_cells.png');fig.savefig(figures/'target_cells.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for ri,rep in enumerate(['BEATs768','MFCC26']):
        keys=['delta_S','delta_P','interaction','D'];vals=[stats['estimates'][rep]['contrasts'][k]['macro_f1'] for k in keys]
        pt=np.array([v['estimate'] for v in vals]);bounds=np.array([v['ci95'] for v in vals])
        positions=np.arange(4)+(ri-.5)*.14;color=['#176b87','#c96a20'][ri]
        ax.vlines(positions,100*bounds[:,0],100*bounds[:,1],color=color)
        ax.scatter(positions,100*pt,label=rep,color=color,zorder=3)
    ax.axhline(0,color='gray',linewidth=1);ax.set_xticks(np.arange(4),['Source marginal','Path marginal','Interaction','Source minus path'])
    ax.set_ylabel('Macro-F1 change (percentage points)');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(figures/'factor_contrasts.png');fig.savefig(figures/'factor_contrasts.svg');plt.close(fig)
    # Preselected first seed/template/geometry, independent of target performance.
    timing=HERE/'results/paired_timing_20261004_v3'
    fig,axes=plt.subplots(2,2,figsize=(11,6))
    for ri,label in enumerate(['car','truck']):
        for level,color in [('S0','#176b87'),('S1','#c96a20')]:
            x=np.load(timing/'sources'/f'h2.r42.validation.t000.{label}.{level}.npy')
            f,p=welch(x[32000:48000],fs=8000,nperseg=4096)
            axes[ri,0].plot(f,10*np.log10(np.maximum(p,1e-20)),label=level,color=color,alpha=.85)
        for cell,color in [('F00','#176b87'),('F01','#c96a20')]:
            x=np.load(timing/'observations'/f'h2.r42.validation.t000.g00.{label}.{cell}.npy')
            width=320;envelope=np.sqrt(np.mean(x.reshape(-1,width).astype(float)**2,axis=1))
            axes[ri,1].plot((np.arange(len(envelope))+.5)*.02,envelope,label='P0' if cell=='F00' else 'P1',color=color)
        axes[ri,0].set_xlim(0,4000);axes[ri,0].set_ylabel(label+' PSD (dB/Hz)');axes[ri,0].legend()
        axes[ri,1].set_ylabel(label+' receiver RMS');axes[ri,1].legend()
    axes[0,0].set_title('Dry source changes; common RMS');axes[0,1].set_title('Paired S0 path changes; common crop RMS')
    axes[1,0].set_xlabel('Frequency (Hz)');axes[1,1].set_xlabel('Time within observation (s)')
    fig.tight_layout();fig.savefig(figures/'source_path_manipulations.png');fig.savefig(figures/'source_path_manipulations.svg');plt.close(fig)
    physical=read(HERE/'results/renderer_validation_20261004_v2/report.json')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for component,color in [('direct_filtered','#176b87'),('reflected','#c96a20'),('P1','#599b70')]:
        rr=[r for r in physical['records'] if r['check']=='static_actual_transfer' and r['component']==component]
        axes[0].scatter([r['frequency_hz'] for r in rr],[r['magnitude_error_db'] for r in rr],s=18,alpha=.6,label=component,color=color)
    axes[0].axhline(.5,color='gray',linestyle='--');axes[0].axhline(-.5,color='gray',linestyle='--');axes[0].set_ylabel('Static magnitude error (dB)');axes[0].set_xlabel('Frequency (Hz)');axes[0].legend(fontsize=8)
    for component,color in [('P0','#176b87'),('reflected','#c96a20')]:
        rr=[r for r in physical['records'] if r['check']=='moving_retarded_frequency' and r['component']==component]
        axes[1].scatter([r['frequency_hz'] for r in rr],[100*r['max_relative_error'] for r in rr],s=18,alpha=.6,label=component,color=color)
    axes[1].axhline(1,color='gray',linestyle='--');axes[1].set_ylabel('Maximum frequency error (%)');axes[1].set_xlabel('Frequency (Hz)');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(figures/'physical_checks.png');fig.savefig(figures/'physical_checks.svg');plt.close(fig)
    primary=stats['estimates']['BEATs768'];decision=stats['decision']
    if decision['practical_source_dominance_supported']:
        verdict='The prespecified practical source-dominance criterion is met for these interventions on this exposed development population.'
    elif decision['directional_source_dominance_contradicted']:
        verdict='The conditional interval contradicts directional source dominance for these interventions: the source-minus-path upper bound is nonpositive.'
    elif decision['practical_three_point_margin_falsified']:
        verdict='The prespecified three-percentage-point practical source-dominance margin is falsified for these interventions under the conditional interval.'
    else:
        verdict='The primary source-dominance comparison is inconclusive under the prespecified conditional uncertainty rule.'
    lines=['# H2: paired source × propagation results','',
        '**Test domain: controlled synthetic → previously exposed MELAUDIS development data.**','',verdict,'',
        f"Primary BEATs source-minus-path contrast D: **{estimate(primary['contrasts']['D']['macro_f1'])} percentage points**.",
        f"Source marginal: {estimate(primary['contrasts']['delta_S']['macro_f1'])}; path marginal: {estimate(primary['contrasts']['delta_P']['macro_f1'])}; interaction: {estimate(primary['contrasts']['interaction']['macro_f1'])} percentage points.",'',
        'This result tests the specified source and path bundles. It does not identify an intrinsic share of the real domain gap, establish that S1 is empirically more realistic, or rank environmental and sensor modeling, which were fixed. It is not independent confirmation.','',
        'The relative propagation gain does not establish a reliable classifier: all cells retain a zero-recall seed/group/class case. The BEATs direct-only cells nearly always predict truck; MFCC nearly always predicts truck at every factor setting. The BEATs path intervention restores substantial car recall, while reducing truck recall. All four mean BEATs macro-F1 scores remain below the always-car reference, despite the path-enriched cells exceeding its balanced accuracy.','',
        '## What was held fixed','',
        'Four paired cells; five frozen banks (42, 123, 456, 789, 1024); 190 car and 190 truck training observations per head; 50/class synthetic validation observations from disjoint source templates. The complete corpus contains 240 base templates, 480 dry source waveforms and 9,600 observations. These are generated sources and events, not independent real vehicles. Motorcycles are excluded.','',
        'S1 changes engine harmonic rolloff, 0–2% firing-frequency modulation and component balance. P1 adds ground reflection and atmospheric absorption to the common moving direct path. Both factors are bundles. Source RMS, receiver-crop RMS, geometry, ideal mono microphone, atmosphere, 8 kHz synthesis / 16 kHz model input, two-second observation length, fixed BEATs768/MFCC26 and C=1 logistic heads remain common. Encoders are frozen; each scaler/head sees only its own 380 training rows. No target tuning, adaptation or model selection was performed.','',
        '| Cell | Source | Propagation |','|---|---|---|',
        '| F00 | S0: baseline procedural | P0: moving direct path |',
        '| F01 | S0: baseline procedural | P1: direct + ground + air |',
        '| F10 | S1: declared source changes | P0: moving direct path |',
        '| F11 | S1: declared source changes | P1: direct + ground + air |','',
        '## Full-target performance','',
        'All numbers below are percentages. Brackets are 95% paired group/bank bootstrap intervals; point estimates average five complete-target seed scores. Worst recall is the minimum over every seed, target group and class.','',
        '| Representation | Cell | Macro-F1 [95% CI] | Balanced accuracy | Car recall | Truck recall | Worst recall |',
        '|---|---|---:|---:|---:|---:|---:|']
    for rep in ['BEATs768','MFCC26']:
        for cell in CELLS:
            v=stats['estimates'][rep]['cells'][cell]
            lines.append(f"| {rep} | {cell} | {estimate(v['macro_f1'])} | {pct(v['balanced_accuracy']['estimate'])} | {pct(v['car_recall']['estimate'])} | {pct(v['truck_recall']['estimate'])} | {pct(v['absolute_worst_bank_group_class_recall'])} |")
    lines+=['',f"Always-car macro-F1 is {pct(summary['constant_controls']['always_car']['macro_f1'])}%; always-truck is {pct(summary['constant_controls']['always_truck']['macro_f1'])}%. Both have 50% balanced accuracy on this two-class population.",'',
        '![Full target performance]('+os.path.relpath(figures/'target_cells.png',args.output.parent)+')','',
        '## Factor effects and falsification','',
        'Positive effects mean higher macro-F1. With Qij denoting the five-bank mean for source i and path j: source marginal = ((Q10−Q00)+(Q11−Q01))/2; path marginal = ((Q01−Q00)+(Q11−Q10))/2; interaction = Q11−Q10−Q01+Q00. The primary contrast D is source minus path, equivalently Q10−Q01. Conditional effects show whether a change behaves differently at the other factor level.','',
        '| Representation | Contrast | Macro-F1 change [95% CI], percentage points |','|---|---|---:|']
    for rep in ['BEATs768','MFCC26']:
        for key in ['source_at_P0','source_at_P1','path_at_S0','path_at_S1','delta_S','delta_P','interaction','D']:
            lines.append(f"| {rep} | {key} | {estimate(stats['estimates'][rep]['contrasts'][key]['macro_f1'])} |")
    lines+=['',f"Decision flags: practical dominance supported = `{decision['practical_source_dominance_supported']}`; ≥3-point margin falsified = `{decision['practical_three_point_margin_falsified']}`; directional dominance contradicted = `{decision['directional_source_dominance_contradicted']}`; useful source improvement contradicted = `{decision['useful_source_improvement_contradicted']}`.",'',
        'Support requires lower CI(D) ≥3 points and lower CI(source marginal)>0. If both changes hurt, a less harmful source change is not useful source enrichment. Conditional source/path effects and interaction are reported even when they disagree with a simple ranking.','',
        '![Factor effects]('+os.path.relpath(figures/'factor_contrasts.png',args.output.parent)+')','',
        '## Target groups and class failure cases','',
        '| Group | Cars | Trucks |','|---|---:|---:|']
    groups=list(summary['target_groups'])
    for group,counts in summary['target_groups'].items():lines.append(f'| {group} | {counts[0]} | {counts[1]} |')
    lines+=['','| Representation | Cell | Group | Mean car recall | Mean truck recall | Worst seed/class recall |','|---|---|---|---:|---:|---:|']
    for rep in ['BEATs768','MFCC26']:
        for cell in CELLS:
            rr=[r for r in evals if r['representation']==rep and r['cell']==cell]
            for g in groups:
                car=np.mean([r['groups'][g]['car_recall'] for r in rr]);truck=np.mean([r['groups'][g]['truck_recall'] for r in rr])
                worst=min(r['groups'][g][k] for r in rr for k in ['car_recall','truck_recall'])
                lines.append(f'| {rep} | {cell} | {g} | {pct(car)} | {pct(truck)} | {pct(worst)} |')
    lines+=['','The four groups are conservative provenance components; original recording sessions are unknown. One group contains 6,290 cars and 187 trucks, and another has only one truck. Repeated events are not independent recording sessions. Bootstrap uncertainty is conditional and weak with this support. Group tables expose failures that a mean F1 can hide.','',
        '## Synthetic validation and per-bank results','',
        'Validation is generated synthetic → disjoint synthetic templates. It was not used for fitting, stopping or parameter selection. All 40 heads achieve 100% validation macro-F1. This establishes separation under these generated class priors, while the target results show that this validation success is insufficient evidence of real transfer.','',
        '| Representation | Cell | Seed | Synthetic validation F1 | Real development F1 | Real balanced accuracy |','|---|---|---:|---:|---:|---:|']
    fit_index={r['fit_id']:r for r in fits}
    for r in evals:
        lines.append(f"| {r['representation']} | {r['cell']} | {r['seed']} | {pct(fit_index[r['fit_id']]['validation']['macro_f1'])} | {pct(r['target']['macro_f1'])} | {pct(r['target']['balanced_accuracy'])} |")
    lines+=['','## Physical and source checks','',
        'The indexing, total image-path spreading, second-leg initialization, common speed of sound, Sinc weighting and air-frequency corrections are recorded in the execution patch. The initial 11-tap amplitude failure and last-bit replay failures remain available; they were resolved before bulk generation, with no target scores used. The common Sinc length is 31 taps.','',
        '- All 480 source cases pass harmonic-ratio, waveform-derived firing-frequency, RMS, reference-source and deterministic replay checks. Maximum released-reference waveform difference is about 2.73e−12; this numerical agreement does not establish empirical realism.',
        '- All 151 renderer checks pass. Maximum moving-frequency error is 0.539% (limit 1%); maximum static transfer magnitude error is 0.0143 dB (limit 0.5 dB). Scalar and batched outputs agree within 3e−17.',
        '- All 480 dry sources and 9,600 complete renders were independently regenerated exactly. Every saved file, class/cell/geometry pairing, source-parent role, crop, dtype, resampling operation and peak/RMS check passed.',
        '- The full ground-field coefficient is not capped to a plane-wave energy ratio. Its gains and the passivity clarification remain disclosed; raw ground units and real-road calibration are unknown.','',
        '![Source and path manipulations]('+os.path.relpath(figures/'source_path_manipulations.png',args.output.parent)+')','',
        'The example is the prespecified seed 42, validation template 0, geometry 0 for each class; it was not chosen by target performance.','',
        '![Numerical physical checks]('+os.path.relpath(figures/'physical_checks.png',args.output.parent)+')','',
        '## Reproducibility and limits','',
        'All forty heads were frozen before target evaluation. The independent final audit verifies training-only scaler moments, source-parent separation, exact fit populations, fixed hyperparameters, saved validation/target predictions, all paired bootstrap samples and the protected historical files. A separate integer-confusion calculation reconstructs every macro-F1 draw, contrast and interval without the evaluation score/bootstrap helpers. Prediction replay allows 1e−10 numerical error; waveform replay is exact.','',
        f"Generation: {corpus['elapsed_s']/60:.2f} min; complete corpus verification/replay: {corpus_check['elapsed_s']/60:.2f} min; feature extraction: {features['elapsed_s']/60:.2f} min; fitting: {model_lock['elapsed_s']:.2f} s; target scoring/statistics: {summary['elapsed_s']:.2f} s; final audit: {audit['elapsed_s']:.2f} s. Rendering used {corpus['workers']} CPU workers on the existing Apple M3 Pro; the encoder used four CPU threads, batch 8 and no GPU.",'',
        f"The generated corpus occupies {corpus['output_bytes']/1e9:.2f} GB before features and results. Generation recorded peak parent RSS {corpus['peak_parent_rss_bytes']/2**20:.1f} MiB and peak child RSS {corpus['peak_child_rss_bytes']/2**20:.1f} MiB. These are separate process maxima, not simultaneous total memory or encoder memory measurements.",'',
        'No untouched confirmation population was available. H2 retains H1 target preprocessing and IDs, but its deliberately normalized new factorial corpus differs from the released H1 procedural bank. F00 is not a replay of that earlier result. Scalar normalization removes absolute attenuation/SNR as tested benefits; environment, sensor effects, real engine identity, load, road material calibration and recorded-source alternatives remain untreated. The current-position delay and finite FIR/angle discretization remain approximate.','',
        'Analyst interpretation: the bounded source intervention has not earned priority over propagation in this setting. Separating ground reflection from atmospheric filtering, and understanding the direct-only classifier collapse, are justified follow-up questions. The present run cannot attribute the path gain to either mechanism separately. Its source intervention retains the same procedural architecture and is not a test of a learned or recorded source model. These follow-ups require a new declared protocol; none was run here.','',
        'The next experimental decision should use the conditional gains, class failures and uncertainty above. It should not promote a component as universally dominant or change C, thresholds, priors or sample counts from these target scores without a new declared development experiment. The military milestone gates remain separate and unchanged.','',
        '## Artifacts','']
    for label,path in [('Protocol',HERE/'PROTOCOL.md'),('Implementation amendments',HERE/'IMPLEMENTATION_AMENDMENTS.md'),
        ('Execution lock',HERE/'execution/source_path_20261004_r1/lock.json'),('Corpus manifest',args.corpus/'observation_manifest.jsonl'),
        ('Corpus verification',args.corpus/'verification.json'),('Model lock',args.models/'model_lock.json'),
        ('Feature lock and encoder provenance',Path(model_lock['features_path'])/'summary.json'),
        ('All per-head/group metrics',result/'evaluations.json'),('All scalar metrics and confidence intervals',result/'statistics.json'),
        ('Raw predictions',result/'target_predictions.npz'),('Paired bootstrap indices',result/'bootstrap_indices.npz'),
        ('Final independent audit',result/'verification.json'),('Execution commands',HERE/'RUN_H2.md')]:
        lines.append('- '+link(label,path,args.output.parent))
    args.output.write_text('\n'.join(lines)+'\n');print(args.output,flush=True)


if __name__=='__main__':main()

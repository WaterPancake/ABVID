"""Read-only reporting of the audited attenuation/latency control.

This post-freeze program summarizes saved predictions and metrics. It never
fits, selects, normalizes target audio, or changes the frozen execution files.
"""
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from phase_common import *

FIGURES = ROOT / 'reports/H2_air_latency_control_figures'
LABELS = ['Neither', 'Attenuation only', 'Latency only', 'Both']
EFFECTS = ['attenuation', 'latency', 'interaction', 'latency_minus_attenuation']
CONDITIONAL = ['attenuation_at_L0', 'attenuation_at_L1', 'latency_at_A0', 'latency_at_A1']


def mean_metrics(records, field):
    return {key: np.asarray([r[field][key] for r in records], dtype=float).mean(axis=0).tolist()
            for key in records[0][field]}


def facts():
    audit = read(RESULTS / 'verification.json')
    summary = read(RESULTS / 'summary.json')
    assert audit['passed'] and audit['result_summary_sha256'] == sha(RESULTS / 'summary.json')
    evaluations = read(RESULTS / 'evaluations.json')
    cross = read(RESULTS / 'cross_cell_validation.json')
    fits = read(MODELS / 'fits.json')
    statistics = read(RESULTS / 'statistics.json')
    out = dict(target={}, cross_cell={}, statistics=statistics, summary=summary, audit=audit)
    for rep in REPS:
        out['target'][rep] = {}
        out['cross_cell'][rep] = {}
        for source in ['mean_S', *SOURCES]:
            out['target'][rep][source] = {}
            out['cross_cell'][rep][source] = {}
            for cell in CELLS:
                out['target'][rep][source][cell] = {}
                out['cross_cell'][rep][source][cell] = {}
                for variant in ['native', 'matched']:
                    selected = [r for r in evaluations if r['representation'] == rep
                                and r['cell'] == cell and r['variant'] == variant
                                and (source == 'mean_S' or r['source_level'] == source)]
                    assert len(selected) == (10 if source == 'mean_S' else 5)
                    v = mean_metrics(selected, 'target')
                    for metric, est in statistics['estimates'][variant][rep][source][cell].items():
                        np.testing.assert_allclose(v[metric], est['estimate'], rtol=0, atol=1e-12)
                    v['worst_group_class_recall'] = min(r['worst_group_class_recall'] for r in selected)
                    v['groups'] = {}
                    for group in selected[0]['groups']:
                        group_rows = [dict(metrics=r['groups'][group]) for r in selected]
                        gv = mean_metrics(group_rows, 'metrics')
                        gv['worst_head_class_recall'] = min(r['metrics'][key] for r in group_rows
                                                          for key in ['car_recall', 'truck_recall'])
                        v['groups'][group] = gv
                    out['target'][rep][source][cell][variant] = v
                for test_cell in CELLS:
                    selected = [r for r in cross if r['representation'] == rep
                                and r['train_cell'] == cell and r['test_cell'] == test_cell
                                and (source == 'mean_S' or r['source_level'] == source)]
                    assert len(selected) == (10 if source == 'mean_S' else 5)
                    out['cross_cell'][rep][source][cell][test_cell] = dict(
                        **mean_metrics(selected, 'metrics'),
                        worst_head_class_recall=min(r['metrics'][key] for r in selected
                                                   for key in ['car_recall', 'truck_recall']))
    out['validation'] = dict(heads=len(fits),
        minimum_macro_f1=min(r['validation']['macro_f1'] for r in fits),
        maximum_macro_f1=max(r['validation']['macro_f1'] for r in fits))
    out['runtime_seconds'] = dict(
        generation=read(CORPUS / 'summary.json')['elapsed_s'],
        complete_replay=read(CORPUS / 'verification.json')['elapsed_s'],
        synthetic_features=read(CACHE / 'summary.json')['elapsed_s'],
        fit=read(MODELS / 'lock.json')['elapsed_s'],
        evaluate=summary['elapsed_s'], audit=audit['elapsed_s'])
    save(RESULTS / 'report_facts.json', out)
    return out


def finish(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(FIGURES / (name + '.' + ext), dpi=175, bbox_inches='tight')
    plt.close(fig)


def figures(f):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'native': '#2465a3', 'matched': '#dd8530'}
    stats = f['statistics']['estimates']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for ax, rep in zip(axes, REPS):
        for offset, variant in [(-.18, 'native'), (.18, 'matched')]:
            vals = [stats[variant][rep]['mean_S'][c]['macro_f1'] for c in CELLS]
            pos = np.arange(4) + offset
            ax.bar(pos, [v['estimate'] * 100 for v in vals], width=.33, color=colors[variant],
                   label='Native target' if variant == 'native' else 'Fixed RMS target')
            ax.vlines(pos, [v['ci95'][0] * 100 for v in vals], [v['ci95'][1] * 100 for v in vals], color='#222', lw=1)
        ax.axhline(100 * f['summary']['constant_controls']['always_car']['macro_f1'],
                   color='#777', ls=':', label='Always car')
        ax.set(xticks=range(4), xticklabels=['Neither\nA0L0', 'Attenuation\nA1L0', 'Latency\nA0L1', 'Both\nA1L1'],
               ylabel='Macro-F1 (%)', ylim=(0, 100), title=rep)
    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle('Ground reflection on in every cell; mean of fixed S0/S1 and five banks\n'
                 'Synthetic → exposed MELAUDIS development; 95% paired group/bank intervals', fontsize=11)
    finish(fig, 'cell_performance')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for offset, variant in [(-.10, 'native'), (.10, 'matched')]:
        for ax, keys in zip(axes, [EFFECTS, CONDITIONAL]):
            for i, key in enumerate(keys):
                v = stats[variant]['BEATs768']['mean_S'][key]['macro_f1']
                ax.plot(np.array(v['ci95']) * 100, [i + offset] * 2, color=colors[variant], lw=2)
                ax.scatter(v['estimate'] * 100, i + offset, color=colors[variant], s=30,
                           label=('Native target' if variant == 'native' else 'Fixed RMS target') if i == 0 else None)
    for ax, labels in zip(axes, [
            ['Attenuation marginal', 'Latency marginal', 'Interaction', 'Latency minus attenuation'],
            ['Attenuation, latency off', 'Attenuation, latency on', 'Latency, attenuation off', 'Latency, attenuation on']]):
        ax.axvline(0, color='#888', ls=':')
        ax.set(yticks=range(4), yticklabels=labels, xlabel='Macro-F1 change (percentage points)')
        ax.invert_yaxis()
    axes[0].set_title('Marginal effects and primary contrast')
    axes[1].set_title('Conditional effects')
    axes[0].legend(fontsize=8)
    fig.suptitle('BEATs: attenuation and numerical latency are separate interventions\n'
                 'Mean of both fixed source levels; conditional 95% intervals', fontsize=11)
    finish(fig, 'factor_effects')

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), layout='constrained')
    for ax, rep in zip(axes, REPS):
        vals = np.array([[f['cross_cell'][rep]['mean_S'][a][b]['macro_f1'] for b in CELLS] for a in CELLS]) * 100
        im = ax.imshow(vals, vmin=0, vmax=100, cmap='viridis')
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{vals[i,j]:.1f}', ha='center', va='center',
                        color='white' if vals[i,j] < 55 else 'black')
        ax.set(xticks=range(4), xticklabels=LABELS, yticks=range(4), yticklabels=LABELS,
               xlabel='Validation cell', ylabel='Training cell', title=rep)
        ax.tick_params(axis='x', labelrotation=20)
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Macro-F1 (%)', shrink=.7, pad=.025, fraction=.035)
    fig.suptitle('Synthetic validation: disjoint source templates, all cell pairs\n'
                 'Same source level and bank; no refitting; mean of S0/S1 and five banks', fontsize=11)
    finish(fig, 'cross_cell_validation')


def pct(v):
    return f'{100 * v:.2f}'


def ci(v):
    return f"{pct(v['estimate'])} [{pct(v['ci95'][0])}, {pct(v['ci95'][1])}]"


def tables(f):
    lines = []
    def add(*text): lines.extend(text)
    add('## Complete-target results', '',
        'All four cells include ground reflection. A means air-amplitude shaping and L means its finite-FIR implementation latency. '
        'Percentages are means over both fixed source levels and five banks, not pooled predictions. Brackets show paired 95% intervals. '
        'Worst recall is the minimum across individual heads, groups and classes.', '',
        '| Representation | Cell | Target level | Macro-F1 [95% CI] | Balanced accuracy | Car recall | Truck recall | Predicted truck | Worst recall |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|')
    for rep in REPS:
        for cell, label in zip(CELLS, LABELS):
            for variant in ['native', 'matched']:
                v = f['target'][rep]['mean_S'][cell][variant]
                est = f['statistics']['estimates'][variant][rep]['mean_S'][cell]['macro_f1']
                add(f'| {rep} | {cell}: {label} | {variant} | {ci(est)} | ' + ' | '.join(pct(v[k]) for k in
                    ['balanced_accuracy', 'car_recall', 'truck_recall', 'predicted_truck_fraction', 'worst_group_class_recall']) + ' |')
    add('', '![All four cells](H2_air_latency_control_figures/cell_performance.png)', '',
        '## Individual source levels', '',
        'Native target is primary. The full equivalent tables for the matched target and all metrics are in the linked `report_facts.json` and `statistics.json`.', '',
        '| Representation | Source | Cell | Macro-F1 [95% CI] | Car recall | Truck recall | Worst recall |',
        '|---|---|---|---:|---:|---:|---:|')
    for rep in REPS:
        for source in SOURCES:
            for cell in CELLS:
                v = f['target'][rep][source][cell]['native']
                est = f['statistics']['estimates']['native'][rep][source][cell]['macro_f1']
                add(f'| {rep} | {source} | {cell} | {ci(est)} | ' +
                    ' | '.join(pct(v[k]) for k in ['car_recall', 'truck_recall', 'worst_group_class_recall']) + ' |')
    add('', '## Factor effects and source interactions', '',
        'Changes are macro-F1 percentage points. The sole primary contrast is native BEATs, mean_S, latency minus attenuation. '
        'All remaining intervals are descriptive. Marginal effects average the two settings of the other factor; conditional effects keep it fixed.', '',
        '| Representation | Source | Target | Attenuation marginal | Latency marginal | Interaction | Latency minus attenuation |',
        '|---|---|---|---:|---:|---:|---:|')
    for rep in REPS:
        for source in ['mean_S', *SOURCES]:
            for variant in ['native', 'matched']:
                v = f['statistics']['estimates'][variant][rep][source]
                add(f'| {rep} | {source} | {variant} | ' + ' | '.join(ci(v[k]['macro_f1']) for k in EFFECTS) + ' |')
    add('', '| Representation | Source | Target | Attenuation, L off | Attenuation, L on | Latency, A off | Latency, A on |',
        '|---|---|---|---:|---:|---:|---:|')
    for rep in REPS:
        for source in ['mean_S', *SOURCES]:
            for variant in ['native', 'matched']:
                v = f['statistics']['estimates'][variant][rep][source]
                add(f'| {rep} | {source} | {variant} | ' + ' | '.join(ci(v[k]['macro_f1']) for k in CONDITIONAL) + ' |')
    add('', 'Source interactions below are S1 effect minus S0 effect, using the same paired draws. Source levels are fixed interventions and are not resampled.', '',
        '| Representation | Target | Change in attenuation effect | Change in latency effect | Change in interaction | Change in primary contrast |',
        '|---|---|---:|---:|---:|---:|')
    for rep in REPS:
        for variant in ['native', 'matched']:
            v = f['statistics']['source_interactions'][variant][rep]
            add(f'| {rep} | {variant} | ' + ' | '.join(ci(v[k]['macro_f1']) for k in EFFECTS) + ' |')
    add('', '![Factor and conditional effects](H2_air_latency_control_figures/factor_effects.png)', '',
        '## Fixed target RMS check', '',
        'Changes after reusing the already frozen −26 dBFS target cache; no new target preprocessing, target selection, threshold fitting or gain sweep.', '',
        '| Representation | Cell | Macro-F1 change [95% CI] | Car recall change | Truck recall change |',
        '|---|---|---:|---:|---:|')
    for rep in REPS:
        for cell in CELLS:
            v = f['statistics']['matched_minus_native'][rep]['mean_S'][cell]
            add(f'| {rep} | {cell} | ' + ' | '.join(ci(v[k]) for k in ['macro_f1', 'car_recall', 'truck_recall']) + ' |')
    add('', '## Ranking, calibration and confusion counts', '',
        'Truck is the positive class, with 3.17% target prevalence. Metrics average individual heads. '
        'Confusion entries are **mean counts per head**, ordered [true car→car, car→truck; truck→car, truck→truck]; they are not ensemble predictions. '
        'Every integer per-head matrix is retained in `evaluations.json`. Brier and log loss are unscaled; other columns are percentages.', '',
        '| Representation | Cell | Target | Accuracy | AUROC | Truck AP | Brier | Log loss | ECE | Mean confusion counts |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---|')
    for rep in REPS:
        for cell in CELLS:
            for variant in ['native', 'matched']:
                v = f['target'][rep]['mean_S'][cell][variant]
                cm = v['confusion_matrix']
                ctext = f'[{cm[0][0]:.1f}, {cm[0][1]:.1f}; {cm[1][0]:.1f}, {cm[1][1]:.1f}]'
                add(f"| {rep} | {cell} | {variant} | {pct(v['accuracy'])} | {pct(v['roc_auc'])} | {pct(v['average_precision'])} | "
                    f"{v['brier_truck']:.4f} | {v['log_loss']:.3f} | {pct(v['ece10_truck'])} | {ctext} |")
    add('', '## Group-specific results', '',
        'Groups are conservative connected components, not verified original recording sessions. G1–G4 abbreviate the IDs below. '
        'Car/truck recall averages the ten heads; worst recall is the lowest class recall among those heads for that group. '
        'The one-truck group cannot support a precise truck-recall estimate.', '',
        '| Group | Original identifier | Cars | Trucks |', '|---|---|---:|---:|')
    groups = list(f['summary']['target_groups'])
    for i, group in enumerate(groups):
        car, truck = f['summary']['target_groups'][group]
        add(f'| G{i+1} | {group} | {car} | {truck} |')
    for variant in ['native', 'matched']:
        add('', f'**{variant.capitalize()} target:** recall percentages.', '',
            '| Representation | Cell | Group | Car recall | Truck recall | Worst head/class recall |',
            '|---|---|---|---:|---:|---:|')
        for rep in REPS:
            for cell in CELLS:
                for i, group in enumerate(groups):
                    v = f['target'][rep]['mean_S'][cell][variant]['groups'][group]
                    add(f'| {rep} | {cell} | G{i+1} | ' +
                        ' | '.join(pct(v[k]) for k in ['car_recall', 'truck_recall', 'worst_head_class_recall']) + ' |')
    add('', '## Synthetic validation across every cell pair', '',
        'These checks use source templates excluded from fitting, with the same source level and bank. '
        'No head is refitted on the alternate validation cell. Full class metrics, source-specific means and worst-head recall are in `report_facts.json`; '
        'all 320 individual evaluations are retained in `cross_cell_validation.json`.', '',
        '![Cross-cell validation](H2_air_latency_control_figures/cross_cell_validation.png)', '',
        '## Measured runtime', '',
        'Apple M3 Pro, 18 GiB RAM; four render workers, four Torch CPU threads, one BLAS thread, encoder batch eight. '
        'These stage times exclude the previously completed study and admission fixtures.', '',
        '| Stage | Seconds | Minutes |', '|---|---:|---:|')
    for key, value in f['runtime_seconds'].items():
        add(f'| {key.replace("_", " ")} | {value:.2f} | {value/60:.2f} |')
    value = sum(f['runtime_seconds'].values())
    add(f'| Total recorded stages | {value:.2f} | {value/60:.2f} |', '')
    return '\n'.join(lines)


def main():
    f = facts()
    figures(f)
    (RESULTS / 'report_tables.md').write_text(tables(f))
    paths = [RESULTS / 'report_facts.json', RESULTS / 'report_tables.md', *sorted(FIGURES.glob('*'))]
    save(RESULTS / 'report_generation.json', dict(
        completed_utc=datetime.now(timezone.utc).isoformat(),
        generator_sha256=sha(Path(__file__)), verification_sha256=sha(RESULTS / 'verification.json'),
        artifacts_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
        new_fits=0, target_tuning=False))
    print(f['statistics']['decision'])
    print('Runtime seconds:', f['runtime_seconds'])


if __name__ == '__main__':
    main()

"""Audit saved inference artifacts and archived metrics; no training or inference."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments/reproduction'

def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())

def f1(cm):
    cm = np.asarray(cm, dtype=float)
    denom = cm.sum(0) + cm.sum(1)
    return float(np.divide(2 * np.diag(cm), denom, out=np.zeros(len(cm)), where=denom != 0).mean())

def prediction_check(path, n_classes, expected, expected_hash, indices=None):
    assert digest(path) == expected_hash
    with np.load(path, allow_pickle=False) as a:
        y, pred, probs = a['y_true'], a['y_pred'], a['probabilities']
        assert np.isfinite(probs).all()
        assert np.array_equal(probs.argmax(1), pred)
        np.testing.assert_allclose(probs.sum(1), 1, atol=1e-5)
        if indices is not None:
            take = a[indices]
            y, pred = y[take], pred[take]
        cm = np.bincount(y * n_classes + pred, minlength=n_classes ** 2).reshape(n_classes, n_classes)
        assert np.array_equal(cm, expected)
    return {'path': str(path.relative_to(ROOT)), 'sha256': expected_hash,
            'n_evaluated': int(cm.sum()), 'macro_f1': f1(cm), 'prediction_artifact_verified': True}

def main():
    audit = {'scope': 'release integrity, archived arithmetic, and saved inference artifacts; no retraining',
        'date': '2026-09-28', 'platform': platform.platform(),
        'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'working_tree': 'dirty; replay scripts and inputs separately hashed',
        'script_sha256': digest(__file__), 'release_downloads': [], 'prediction_checks': [], 'arithmetic_checks': []}
    for paper, names in [('P01',['code_and_results.zip','models.zip','features.zip','data.zip','README.md','requirements_colab.txt']),('P02',['code_and_results.zip'])]:
        metadata = read(BASE/'releases'/f'{paper}_zenodo_record.json')
        entries = {v['key']: v for v in metadata['files']}
        for name in names:
            path = BASE/'releases'/f'{paper}_{name}'
            actual = digest(path,'md5')
            assert f'md5:{actual}' == entries[name]['checksum']
            audit['release_downloads'].append({'paper':paper,'file':name,'bytes':path.stat().st_size,'md5':actual,'matches_provider':True})
    p02 = read(BASE/'results/P02/replay.json')
    for relative, expected in p02['inputs'].items(): assert digest(ROOT/relative) == expected
    for entry in p02['snapshots']:
        release = BASE/'releases/P02/code_and_results'
        assert digest(release/'models'/entry['checkpoint']) == entry['checkpoint_sha256']
        config = entry['config']
        archived_path = release/'results'/f'{config}_experiment_results.json'
        assert digest(archived_path) == entry['archived_results_sha256']
        archived = read(archived_path)
        audit['prediction_checks'].append(prediction_check(BASE/'results/P02'/f'{config}_predictions.npz',3,entry['full_set_confusion_matrix'],entry['predictions_sha256']))
        balanced, errors = [], []
        for run in archived['all_runs']:
            errors += [abs(f1(run['confusion_matrix'])-run['test_f1_macro']),abs(f1(run['balanced_confusion_matrix'])-run['balanced_test_f1_macro'])]
            balanced.append(f1(run['balanced_confusion_matrix']))
        assert max(errors) < 1e-12
        table = entry['archived_table4_recalculation']
        audit['arithmetic_checks'].append({'paper':'P02','reference':'Table 4, PDF p.8','config':config,'runs':len(balanced),'max_metric_error':max(errors),'recomputed_mean':float(np.mean(balanced)),'recomputed_std_ddof0':float(np.std(balanced)),'rounds_to_paper':round(float(np.mean(balanced)),2)==table['paper_mean'] and round(float(np.std(balanced)),2)==table['paper_std']})
    cnn = read(BASE/'results/P01_CNN/replay.json')
    release = BASE/'releases/P01'
    assert digest(release/'models/baseline_cnn_best.pt') == cnn['checkpoint_sha256']
    archived_cnn = read(release/'code_and_results/results/baseline_results.json')
    for domain, entry in cnn['domains'].items():
        for name, expected in entry['input_sha256'].items(): assert digest(release/'features'/name) == expected
        audit['prediction_checks'].append(prediction_check(BASE/'results/P01_CNN'/f'{domain}_predictions.npz',5,entry['balanced_confusion_matrix'],entry['predictions_sha256'],'balanced_indices'))
        scores = [f1(r[domain]['confusion_matrix']) for r in archived_cnn['runs']]
        errors = [abs(score-r[domain]['balanced_f1']) for score,r in zip(scores,archived_cnn['runs'])]
        assert max(errors) <= 0.00005
        paper_mean,paper_std = (0.26,0.03) if domain=='melaudis' else (0.52,0.04)
        audit['arithmetic_checks'].append({'paper':'P01','reference':'Table 3, PDF p.10','model':'CNN','domain':domain,'runs':len(scores),'max_rounding_error':max(errors),'recomputed_mean':float(np.mean(scores)),'recomputed_std_ddof0':float(np.std(scores)),'rounds_to_paper':round(float(np.mean(scores)),2)==paper_mean and round(float(np.std(scores)),2)==paper_std})
    for directory in ['P01_transformers','P01_AST_compatibility']:
        result_path = BASE/'results'/directory/'replay.json'
        result = read(result_path)
        for model in result['models']:
            for domain, entry in model['domains'].items():
                audit['prediction_checks'].append(prediction_check(result_path.parent/f"{model['model']}_{domain}_predictions.npz",5,entry['confusion_matrix'],entry['predictions_sha256']))
    controlled = read(BASE/'releases/P02/code_and_results/results/revision_controlled/controlled_experiment_results.json')
    expected_rows = {'AudioLDM only (200/class)':(0.20,0.04),'Pyroad only (200/class)':(0.18,0.02),'Real + AudioLDM (200/class)':(0.24,0.09),'Real + Pyroad (200/class)':(0.29,0.04)}
    for name,entry in controlled['results'].items():
        scores = [f1(r['confusion_matrix']) for r in entry['runs']]
        # Keep the saved aggregate comparison independent of checkpoint replay.
        audit['arithmetic_checks'].append({'paper':'P02','reference':'Table 5, PDF p.11','config':name,'runs':len(scores),'inference_reproduced':False,'recomputed_mean':float(np.mean(scores)),'recomputed_std_ddof0':float(np.std(scores)),'max_run_rounding_error':max(abs(s-r['balanced_f1']) for s,r in zip(scores,entry['runs'])),'rounds_to_paper':(round(float(np.mean(scores)),2),round(float(np.std(scores)),2))==expected_rows[name]})
    audit['published_training_runs_reproduced'] = 0
    audit['custom_training_or_simulator_ablations_run'] = False
    audit['script_hashes'] = {str(p.relative_to(ROOT)):digest(p) for p in sorted(BASE.glob('*.py'))}
    (BASE/'results/reproduction_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('Verified downloads:',len(audit['release_downloads']))
    print('Verified prediction artifacts:',len(audit['prediction_checks']))
    print('Archived arithmetic rows:',len(audit['arithmetic_checks']))
    print('Rows differing from printed paper:',[r for r in audit['arithmetic_checks'] if not r['rounds_to_paper']])

if __name__=='__main__': main()

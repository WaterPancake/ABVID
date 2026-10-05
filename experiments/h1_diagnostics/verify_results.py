"""Independently check saved IDMT fits, predictions, choices, metrics and roles."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from threadpoolctl import threadpool_limits

from common import ROOT, sha, save, read_lock, assert_separation
from run_checks import load_features


def independent_metrics(y, p, threshold):
    pred = p > threshold
    bins = np.minimum(np.floor(p*10).astype(int), 9)
    ece = sum(abs(float(p[bins == b].sum()-y[bins == b].sum())) / len(y)
              for b in range(10) if np.any(bins == b))
    result = dict(macro_f1=f1_score(y, pred, average='macro', zero_division=0),
        balanced_accuracy=balanced_accuracy_score(y, pred), accuracy=accuracy_score(y, pred),
        roc_auc=roc_auc_score(y, p), average_precision=average_precision_score(y, p),
        truck_prevalence=float(y.mean()), brier_truck=brier_score_loss(y, p),
        log_loss=float(-np.log(np.clip(np.where(y == 1, p, 1-p), 1e-15, 1)).mean()), ece10_truck=ece,
        confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())
    for c, name in enumerate(['car', 'truck']):
        result[f'{name}_recall'] = recall_score(y, pred, labels=[0, 1], average=None, zero_division=0)[c]
        result[f'{name}_precision'] = precision_score(y, pred, labels=[0, 1], average=None, zero_division=0)[c]
    return result


def check_metrics(stored, y, p, threshold):
    expected = independent_metrics(y, p, threshold)
    for key, value in expected.items():
        np.testing.assert_allclose(stored[key], value, rtol=0, atol=1e-10, err_msg=key)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--lock', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True); ap.add_argument('--results', type=Path, required=True)
    args = ap.parse_args(); started = time.perf_counter()
    lock, cfg, rows = read_lock(args.lock); out = args.results
    summary = json.loads((out/'summary.json').read_text())
    assert summary['lock_sha256'] == sha(args.lock) and summary['target_paths_accessed'] == []
    for name, h in summary['artifacts_sha256'].items(): assert sha(out/name) == h, name
    x = load_features(args.cache, args.lock, rows, cfg)
    rowmap = {r['file_id']: r for r in rows}; pos = {r['file_id']: i for i, r in enumerate(rows)}
    yy = np.array([r['class_id'] for r in rows]); positions = lambda ids: np.array([pos[i] for i in ids])
    fits = json.loads((out/'fit_index.json').read_text()); fi = {r['fit_id']: r for r in fits}
    evals = json.loads((out/'evaluations.json').read_text())
    inners = json.loads((out/'inner_evaluations.json').read_text())
    choices = json.loads((out/'selections.json').read_text())
    ci = {(r['representation'], r['fold'], r['seed']): r for r in choices}
    models = {}; h1_probability_error = 0.; h1_replayed = 0
    with threadpool_limits(limits=1):
        for r in fits:
            assert sha(ROOT/r['model_path']) == r['model_sha256']
            model = joblib.load(ROOT/r['model_path']); models[r['fit_id']] = model
            ip = positions(r['train_ids']); xx = x[r['profile'], r['representation']][ip]
            np.testing.assert_allclose(model[0].mean_, xx.mean(axis=0), rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(model[0].var_, xx.var(axis=0), rtol=1e-10, atol=1e-10)
            assert model[1].C == r['C'] and model[1].n_iter_.max() < cfg['classifier']['max_iter']
            assert np.bincount(yy[ip]).tolist() == r['train_class_counts']
            with np.load(out/'models'/(r['fit_id']+'_training.npz')) as z:
                assert z['ids'].tolist() == r['train_ids']; np.testing.assert_array_equal(z['y'], yy[ip])
                np.testing.assert_allclose(z['p'], model.predict_proba(xx)[:, 1], rtol=0, atol=1e-12)
                check_metrics(r['training_metrics'], z['y'], z['p'], .5)
        print(f'Verified {len(fits)} training-only scalers, models and training scores', flush=True)
        ep = {}; inner_scores = defaultdict(dict)
        for r in evals + inners:
            assert sha(ROOT/r['predictions_path']) == r['predictions_sha256']
            fit = fi[r['fit_id']]; profile = fit['profile']; rep = r['representation']
            with np.load(ROOT/r['predictions_path']) as z:
                ids = z['ids'].tolist(); y = z['y']; p = z['p']
            assert_separation(fit['train_ids'], ids, rowmap)
            ip = positions(ids); np.testing.assert_array_equal(y, yy[ip])
            np.testing.assert_allclose(p, models[r['fit_id']].predict_proba(x[profile, rep][ip])[:, 1], atol=1e-12, rtol=0)
            check_metrics(r['metrics'], y, p, r.get('threshold', .5))
            if 'case' in r:
                fold = lock['folds'][r['fold']]; seed = str(r['seed'])
                assert ids == fold['test_ids']
                case = r['case']
                if case == 'held_site':
                    sl = next(s for s in lock['locations'] if s['site'] == fold['site'])
                    expected = sl['train_ids'][seed]
                else:
                    n = int(case.split('_')[-1]) if case.startswith('learning_') else 190
                    expected = fold['train_ids'][str(n)][seed]
                assert fit['train_ids'] == expected
                if case == 'nested_C': assert r['C'] == ci[rep, r['fold'], r['seed']]['chosen_C']
                if case == 'nested_threshold': assert r['threshold'] == ci[rep, r['fold'], r['seed']]['chosen_threshold']
                ep[rep, case, r['seed'], r['fold']] = (y, p)
                if case == 'baseline':
                    # Decode source-only members. Target arrays remain unopened.
                    oldpath = ROOT/'experiments/h1/results/H1_20261002/predictions'/f'{rep}__real__seed{r["seed"]}__fold{r["fold"]}.npz'
                    with np.load(oldpath) as old:
                        assert ids == old['source_ids'].tolist()
                        np.testing.assert_array_equal(y, old['source_true'])
                        oldp = old['source_probabilities'][:, 1]
                    error = float(np.max(np.abs(p-oldp))); h1_probability_error = max(h1_probability_error, error)
                    np.testing.assert_allclose(p, oldp, rtol=0, atol=1e-5)
                    np.testing.assert_array_equal(p > .5, oldp > .5)
                    h1_replayed += 1
            else:
                fold = lock['folds'][r['outer_fold']]
                inner = next(i for i in fold['inner'] if i['validation_group'] == r['validation_group'])
                assert ids == inner['validation_ids'] and fit['train_ids'] == inner['train_ids'][str(r['seed'])]
                assert_separation(ids, fold['test_ids'], rowmap)
                assert_separation(fit['train_ids'], fold['test_ids'], rowmap)
                inner_scores[rep, r['outer_fold'], r['seed']].setdefault(str(r['C']), []).append((y, p))
        print(f'Verified {len(evals)} outer and {len(inners)} inner prediction/role/metric records', flush=True)
        for choice in choices:
            rep, fold, seed = choice['representation'], choice['fold'], choice['seed']
            byc = inner_scores[rep, fold, seed]
            cs = {c: np.mean([f1_score(y, p > .5, average='macro', zero_division=0) for y, p in pairs]) for c, pairs in byc.items()}
            best_c = min(float(c) for c, v in cs.items() if abs(v-max(cs.values())) <= 1e-12)
            assert choice['chosen_C'] == best_c
            for c, v in cs.items(): np.testing.assert_allclose(v, choice['C_inner_scores'][c], atol=1e-12, rtol=0)
            ts = {t: np.mean([f1_score(y, p > t, average='macro', zero_division=0) for y, p in byc['1.0']]) for t in cfg['threshold_grid']}
            best_t = min((t for t, v in ts.items() if abs(v-max(ts.values())) <= 1e-12), key=lambda t: (round(abs(t-.5), 12), t))
            assert choice['chosen_threshold'] == best_t
            for t, v in ts.items(): np.testing.assert_allclose(v, choice['threshold_inner_scores'][str(t)], atol=1e-12, rtol=0)
            np.testing.assert_array_equal(ep[rep, 'baseline', seed, fold][1], ep[rep, 'nested_threshold', seed, fold][1])
    # Independently recalculate every reported aggregate and paired interval from records.
    with np.load(out/'bootstrap_indices.npz') as b:
        si, gi, li = b['seed'], b['group'], b['site']
    metric_arrays = {}
    def verify_estimate(values, estimate):
        np.testing.assert_allclose(values.mean(), estimate['mean'], atol=1e-12, rtol=0)
        for indices, name in [(gi, 'group_seed_ci95'), (li, 'site_seed_ci95')]:
            samples = values[si[:, :, None], indices[:, None, :]].mean(axis=(1, 2))
            np.testing.assert_allclose(np.percentile(samples, [2.5, 97.5]), estimate[name], atol=1e-12, rtol=0)
    for rep, cases in summary['aggregate'].items():
        for case, result in cases.items():
            rr = [r for r in evals if r['representation'] == rep and r['case'] == case]
            for metric, estimate in result['held_out'].items():
                a = np.zeros((5, 6))
                for r in rr: a[cfg['seeds'].index(r['seed']), r['fold']] = r['metrics'][metric]
                metric_arrays[rep, case, metric] = a; verify_estimate(a, estimate)
    for rep, contrasts in summary['paired_contrasts'].items():
        for name, metrics in contrasts.items():
            if '_minus_' in name:
                a, b = name.split('_minus_'); rule, n = a.split('_'); a = f'learning_{rule}_{n}'
                b = 'baseline' if b == '190' else f'learning_{rule}_{b}'
            else: a, b = name, 'baseline'
            for m, estimate in metrics.items(): verify_estimate(metric_arrays[rep, a, m]-metric_arrays[rep, b, m], estimate)
    protected = json.loads((args.lock.parent/'protected_h1_hashes.json').read_text())
    assert all(sha(ROOT/r['path']) == r['sha256'] for r in protected)
    save(out/'verification.json', dict(passed=True, source_only=True, unique_fits_verified=len(fits),
        outer_predictions_verified=len(evals), inner_predictions_verified=len(inners), nested_choices_verified=len(choices),
        h1_source_predictions_replayed=h1_replayed, h1_source_max_absolute_probability_error=h1_probability_error,
        h1_source_decisions_identical=True, all_training_scalers_verified=True, all_aggregate_intervals_verified=True,
        protected_h1_artifacts_unchanged=len(protected), no_target_arrays_decoded=True,
        verification_s=time.perf_counter()-started, verifier_sha256=sha(__file__)))
    print(json.dumps(json.loads((out/'verification.json').read_text()), indent=2))


if __name__ == '__main__': main()

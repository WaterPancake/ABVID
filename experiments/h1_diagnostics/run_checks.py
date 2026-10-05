"""Frozen IDMT-only diagnostic grid; validation choices never see an outer group."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from common import ROOT, HERE, sha, save, read_lock, assert_separation
from evaluate import fixed_fit, cm_for, cm_metrics


SCALARS = ['macro_f1', 'balanced_accuracy', 'accuracy', 'car_recall', 'truck_recall',
           'car_precision', 'truck_precision', 'roc_auc', 'average_precision',
           'truck_prevalence', 'brier_truck', 'log_loss', 'ece10_truck']


def score(y, p, threshold=0.5):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    assert y.shape == p.shape and set(y) == {0, 1}
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    result = cm_metrics(cm_for(y, p > threshold))
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(np.mean(bins == b) * abs(p[bins == b].mean() - y[bins == b].mean())
              for b in range(10) if np.any(bins == b))
    result.update(roc_auc=float(roc_auc_score(y, p)),
                  average_precision=float(average_precision_score(y, p)),
                  truck_prevalence=float(y.mean()), brier_truck=float(np.mean((p-y)**2)),
                  log_loss=float(-np.mean(y*np.log(np.clip(p, 1e-15, 1)) +
                                         (1-y)*np.log(np.clip(1-p, 1e-15, 1)))),
                  ece10_truck=float(ece))
    for name in ['car', 'truck']:
        for metric in ['precision', 'recall']:
            result[f'{name}_{metric}'] = result['per_class'][name][metric]
    return result


def macro(y, p, threshold):
    return cm_metrics(cm_for(y, np.asarray(p) > threshold))['macro_f1']


def choose_C(inner, grid):
    """Only inner records are accepted; no outer prediction is an input."""
    scores = {str(c): float(np.mean([macro(r['y'], r['p'], .5) for r in inner[str(c)]]))
              for c in grid}
    best = max(scores.values())
    chosen = min(c for c in grid if np.isclose(scores[str(c)], best, rtol=0, atol=1e-12))
    return chosen, scores


def choose_threshold(inner, grid):
    scores = {str(t): float(np.mean([macro(r['y'], r['p'], t) for r in inner])) for t in grid}
    best = max(scores.values())
    chosen = min((t for t in grid if np.isclose(scores[str(t)], best, rtol=0, atol=1e-12)),
                 key=lambda t: (round(abs(t-.5), 12), t))
    return chosen, scores


def load_features(cache, lock_path, rows, config):
    summary = json.loads((cache/'summary.json').read_text())
    assert summary['lock_sha256'] == sha(lock_path) and summary['source_only']
    assert summary['target_paths_accessed'] == []
    for name, h in summary['artifacts_sha256'].items():
        assert sha(cache/name) == h, name
    ids = [r['file_id'] for r in rows]
    features = {}
    for profile in config['feature_profiles']:
        with np.load(cache/(profile+'.npz'), allow_pickle=False) as data:
            assert data['file_ids'].tolist() == ids
            for rep in config['representations']:
                x = np.asarray(data[rep], dtype=np.float64)
                assert len(x) == len(ids) and np.isfinite(x).all()
                features[profile, rep] = x
    return features


class Runs:
    def __init__(self, out, rows, features, config):
        self.out, self.features, self.config = out, features, config
        self.rows = {r['file_id']: r for r in rows}
        self.pos = {r['file_id']: i for i, r in enumerate(rows)}
        self.y = np.array([r['class_id'] for r in rows])
        self.models, self.fit_records, self.evaluations, self.inner_records = {}, [], [], []
        for sub in ['models', 'predictions', 'inner_predictions']:
            (out/sub).mkdir()

    def positions(self, ids):
        return np.array([self.pos[i] for i in ids])

    def fit(self, rep, profile, C, ids):
        key = dict(representation=rep, profile=profile, C=float(C), train_ids=ids)
        fid = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:20]
        if fid in self.models:
            return fid
        x = self.features[profile, rep]; ip = self.positions(ids); y = self.y[ip]
        assert np.bincount(y).tolist()[0] == np.bincount(y).tolist()[1]
        begin = time.perf_counter()
        model = fixed_fit(x[ip], y, dict(self.config['classifier'], C=C))
        mp = self.out/'models'/(fid+'.joblib'); joblib.dump(model, mp, compress=3)
        p = model.predict_proba(x[ip])[:, 1]
        np.savez_compressed(self.out/'models'/(fid+'_training.npz'), ids=np.array(ids), y=y, p=p)
        record = dict(fit_id=fid, **key, train_class_counts=np.bincount(y, minlength=2).tolist(),
                      train_groups=sorted({self.rows[i]['provenance_group_id'] for i in ids}),
                      fit_s=time.perf_counter()-begin, model_sha256=sha(mp),
                      model_path=str(mp.relative_to(ROOT)), n_iter=model[1].n_iter_.tolist(),
                      training_metrics=score(y, p), fitted_only_on_training=True)
        save(self.out/'models'/(fid+'.json'), record)
        self.models[fid] = model
        self.fit_records.append(record)
        return fid

    def predict(self, fid, rep, profile, ids, train_ids):
        assert_separation(train_ids, ids, self.rows)
        ip = self.positions(ids)
        return self.y[ip], self.models[fid].predict_proba(self.features[profile, rep][ip])[:, 1]

    def outer(self, case, rep, profile, C, train, fold, seed, threshold=.5, fit_id=None):
        fid = fit_id or self.fit(rep, profile, C, train)
        ids = fold['test_ids']; y, p = self.predict(fid, rep, profile, ids, train)
        name = f'{rep}__{case}__seed{seed}__fold{fold["fold"]}'
        path = self.out/'predictions'/(name+'.npz')
        np.savez_compressed(path, ids=np.array(ids), y=y, p=p)
        record = dict(case=case, representation=rep, profile=profile, C=C, seed=seed,
                      fold=fold['fold'], group=fold['held_out_group'], site=fold['site'],
                      fit_id=fid, threshold=threshold, n_per_class=len(train)//2,
                      metrics=score(y, p, threshold), predictions_path=str(path.relative_to(ROOT)),
                      predictions_sha256=sha(path))
        self.evaluations.append(record)
        return record


def bootstrap_indices(lock, cfg):
    rng = np.random.default_rng(cfg['bootstrap']['seed'])
    draws = cfg['bootstrap']['draws']; ns = len(cfg['seeds']); nf = len(lock['folds'])
    seed = rng.integers(ns, size=(draws, ns))
    group = rng.integers(nf, size=(draws, nf))
    sites = sorted({f['site'] for f in lock['folds']})
    site_folds = np.array([[f['fold'] for f in lock['folds'] if f['site'] == s] for s in sites])
    site = site_folds[rng.integers(len(sites), size=(draws, len(sites)))].reshape(draws, nf)
    return seed, group, site


def estimates(a, indices):
    seed, group, site = indices
    draws = a[seed[:, :, None], group[:, None, :]].mean(axis=(1, 2))
    site_draws = a[seed[:, :, None], site[:, None, :]].mean(axis=(1, 2))
    return dict(mean=float(a.mean()), group_seed_ci95=np.quantile(draws, [.025, .975]).tolist(),
                site_seed_ci95=np.quantile(site_draws, [.025, .975]).tolist()), draws, site_draws


def summarize(runs, lock, cfg):
    idx = bootstrap_indices(lock, cfg)
    aggregate, arrays, raw = {}, {}, {}
    fits = {f['fit_id']: f for f in runs.fit_records}
    for rep in cfg['representations']:
        aggregate[rep] = {}
        cases = sorted({r['case'] for r in runs.evaluations if r['representation'] == rep})
        for case in cases:
            recs = [r for r in runs.evaluations if r['representation'] == rep and r['case'] == case]
            assert len(recs) == len(cfg['seeds'])*len(lock['folds'])
            aggregate[rep][case] = dict(held_out={}, training_at_threshold_0_5={}, pooled_per_seed=[])
            for metric in SCALARS:
                a = np.zeros((len(cfg['seeds']), len(lock['folds'])))
                train = np.zeros_like(a)
                for r in recs:
                    ix = cfg['seeds'].index(r['seed']), r['fold']
                    a[ix] = r['metrics'][metric]
                    train[ix] = fits[r['fit_id']]['training_metrics'][metric]
                e, b, sb = estimates(a, idx)
                aggregate[rep][case]['held_out'][metric] = e
                aggregate[rep][case]['training_at_threshold_0_5'][metric] = float(train.mean())
                arrays[rep, case, metric] = a
                if metric in ['macro_f1', 'balanced_accuracy', 'car_recall', 'truck_recall']:
                    raw[f'{rep}__{case}__{metric}__group'] = b
                    raw[f'{rep}__{case}__{metric}__site'] = sb
            for seed in cfg['seeds']:
                rr = sorted((r for r in recs if r['seed'] == seed), key=lambda r: r['fold'])
                yy, pp, pred = [], [], []
                for r in rr:
                    with np.load(ROOT/r['predictions_path']) as z:
                        yy.extend(z['y']); pp.extend(z['p']); pred.extend(z['p'] > r['threshold'])
                pooled = score(yy, pp)
                # Thresholds may differ across outer folds; overwrite decision metrics.
                dm = cm_metrics(cm_for(yy, pred)); pooled.update(dm)
                for name in ['car', 'truck']:
                    for m in ['precision', 'recall']: pooled[f'{name}_{m}'] = dm['per_class'][name][m]
                aggregate[rep][case]['pooled_per_seed'].append(dict(seed=seed, metrics=pooled))
    contrasts = {}
    contrast_cases = [(c, c, 'baseline') for c in ['nested_C', 'nested_threshold', 'native16_2s', 'core8_1s', 'held_site']]
    for rule in ['fixed', 'scaled']:
        contrast_cases += [(f'{rule}_359_minus_190', f'learning_{rule}_359', 'baseline'),
                           (f'{rule}_359_minus_25', f'learning_{rule}_359', f'learning_{rule}_25')]
    for rep in cfg['representations']:
        contrasts[rep] = {}
        for name, a, b in contrast_cases:
            contrasts[rep][name] = {m: estimates(arrays[rep, a, m]-arrays[rep, b, m], idx)[0]
                                     for m in SCALARS}
    np.savez_compressed(runs.out/'bootstrap_samples.npz', **raw)
    np.savez_compressed(runs.out/'bootstrap_indices.npz', seed=idx[0], group=idx[1], site=idx[2])
    controls = []
    for fold in lock['folds']:
        y = runs.y[runs.positions(fold['test_ids'])]
        controls.append(dict(group=fold['held_out_group'], site=fold['site'],
                             always_car=score(y, np.zeros(len(y))), always_truck=score(y, np.ones(len(y)))))
    return dict(aggregate=aggregate, paired_contrasts=contrasts, constant_controls=controls)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--lock', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); begin = time.perf_counter()
    lock, cfg, rows = read_lock(args.lock)
    features = load_features(args.cache, args.lock, rows, cfg)
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    runs = Runs(out, rows, features, cfg); selections = []
    for path in [Path(__file__), HERE/'common.py', ROOT/'experiments/h1/evaluate.py']:
        shutil.copy2(path, out/('executed_'+path.name))
    with threadpool_limits(limits=cfg['hardware']['blas_threads']):
        for rep in cfg['representations']:
            for fold in lock['folds']:
                for seed in cfg['seeds']:
                    train = fold['train_ids']['190'][str(seed)]
                    base = runs.outer('baseline', rep, 'core8_2s', 1., train, fold, seed)
                    for n in cfg['learning_curve_examples_per_class']:
                        ids = fold['train_ids'][str(n)][str(seed)]
                        for rule, C in [('fixed', 1.), ('scaled', 190/n)]:
                            runs.outer(f'learning_{rule}_{n}', rep, 'core8_2s', C, ids, fold, seed)
                    for profile in ['native16_2s', 'core8_1s']:
                        runs.outer(profile, rep, profile, 1., train, fold, seed)
                    inner = {str(c): [] for c in cfg['nested_C_grid']}
                    for infold in fold['inner']:
                        itrain = infold['train_ids'][str(seed)]; val = infold['validation_ids']
                        assert_separation(itrain, fold['test_ids'], runs.rows)
                        assert_separation(val, fold['test_ids'], runs.rows)
                        for C in cfg['nested_C_grid']:
                            fid = runs.fit(rep, 'core8_2s', C, itrain)
                            y, p = runs.predict(fid, rep, 'core8_2s', val, itrain)
                            path = out/'inner_predictions'/f'{fid}__{infold["validation_group"]}.npz'
                            np.savez_compressed(path, ids=np.array(val), y=y, p=p)
                            record = dict(representation=rep, outer_fold=fold['fold'], seed=seed, C=C,
                                          fit_id=fid, validation_group=infold['validation_group'],
                                          validation_ids=val, predictions_path=str(path.relative_to(ROOT)),
                                          predictions_sha256=sha(path), metrics=score(y, p))
                            runs.inner_records.append(record)
                            inner[str(C)].append(dict(y=y, p=p))
                    C, cscore = choose_C(inner, cfg['nested_C_grid'])
                    threshold, tscore = choose_threshold(inner['1.0'], cfg['threshold_grid'])
                    selections.append(dict(representation=rep, fold=fold['fold'], seed=seed,
                                           chosen_C=C, C_inner_scores=cscore, chosen_threshold=threshold,
                                           threshold_inner_scores=tscore, selection_uses_outer=False))
                    runs.outer('nested_C', rep, 'core8_2s', C, train, fold, seed)
                    runs.outer('nested_threshold', rep, 'core8_2s', 1., train, fold, seed,
                               threshold=threshold, fit_id=base['fit_id'])
                print(f'{rep}: group {fold["fold"]+1}/6 complete; {len(runs.fit_records)} fits; {time.perf_counter()-begin:.1f}s', flush=True)
            for site in lock['locations']:
                for seed in cfg['seeds']:
                    train = site['train_ids'][str(seed)]
                    for fold in lock['folds']:
                        if fold['site'] == site['site']:
                            runs.outer('held_site', rep, 'core8_2s', 1., train, fold, seed)
    save(out/'fit_index.json', runs.fit_records)
    save(out/'evaluations.json', runs.evaluations)
    save(out/'inner_evaluations.json', runs.inner_records)
    save(out/'selections.json', selections)
    summaries = summarize(runs, lock, cfg)
    protected = json.loads((args.lock.parent/'protected_h1_hashes.json').read_text())
    assert all(sha(ROOT/r['path']) == r['sha256'] for r in protected), 'Previous H1 artifact changed'
    save(out/'summary.json', dict(protocol_id=cfg['protocol_id'], source_only=True,
        domain='IDMT_real_to_IDMT_real', lock_path=str(args.lock), lock_sha256=sha(args.lock),
        cache_summary_sha256=sha(args.cache/'summary.json'), total_s=time.perf_counter()-begin,
        completed_utc=datetime.now(timezone.utc).isoformat(), unique_fits=len(runs.fit_records),
        outer_evaluations=len(runs.evaluations), inner_evaluations=len(runs.inner_records),
        source_files=len(rows), protected_h1_artifacts_unchanged=len(protected), target_paths_accessed=[],
        **summaries, artifacts_sha256={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print(f'Complete: {len(runs.fit_records)} fits in {time.perf_counter()-begin:.1f}s', flush=True)


if __name__ == '__main__':
    main()

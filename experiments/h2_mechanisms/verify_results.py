"""Read-only verification added after the experimental executable was frozen.

No rendering, fitting, transformation selection or target threshold selection.
Primary intervals also use an independent integer-confusion-count calculation.
"""
import hashlib
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import joblib
from scipy.special import expit
from threadpoolctl import threadpool_limits

from common import *
from effect_statistics import calculate, full_score


def equal(actual, expected, name='root'):
    if isinstance(actual, dict):
        assert set(actual) == set(expected), name
        for key in actual:
            equal(actual[key], expected[key], name+'.'+key)
    elif isinstance(actual, list):
        assert len(actual) == len(expected), name
        for i, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, name+f'[{i}]')
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12, err_msg=name)
    else:
        assert actual == expected, name


def independent_intervals(y, groups, probabilities, stats, raw, indices):
    """No calls to the production scoring, contrast or bootstrap functions."""
    rng = np.random.Generator(np.random.PCG64(314159))
    gd = np.empty((10000, 4), np.int64)
    bd = np.empty((10000, 5), np.int64)
    for i in range(10000):
        gd[i] = rng.integers(4, size=4)
        bd[i] = rng.integers(5, size=5)
    np.testing.assert_array_equal(gd, indices['target_groups'])
    np.testing.assert_array_equal(bd, indices['banks'])

    def measures(cm):
        tp = np.diagonal(cm, axis1=-2, axis2=-1)
        true = cm.sum(axis=-1)
        predicted = cm.sum(axis=-2)
        f1 = np.divide(2*tp, true+predicted, out=np.zeros_like(tp, dtype=float),
                       where=true+predicted > 0).mean(axis=-1)
        return dict(macro_f1=f1, balanced_accuracy=(tp/true).mean(axis=-1),
                    predicted_truck_fraction=predicted[..., 1]/true.sum(axis=-1))

    def contrasts(q):
        a, b, c, d = q
        return dict(ground_at_A0=c-a, ground_at_A1=d-b, air_at_G0=b-a,
                    air_at_G1=d-c, ground=(c-a+d-b)/2, air=(b-a+d-c)/2,
                    interaction=d-c-b+a, ground_minus_air=c-b)

    saved = {}
    count = 0
    for variant, reps in probabilities.items():
        for rep, pr in reps.items():
            counts = np.zeros((2, 4, 5, 4, 2, 2), dtype=np.int64)
            for s in range(2):
                for p in range(4):
                    for b in range(5):
                        for g in range(4):
                            keep = groups == g
                            counts[s, p, b, g] = np.bincount(
                                2*y[keep]+(pr[s, p, b, keep] > .5), minlength=4).reshape(2, 2)
            points = {m: v.mean(axis=2) for m, v in measures(counts.sum(axis=3)).items()}
            # Select literal repeated groups for every repeated bank, then score each bank.
            samples_by_source = []
            for s in range(2):
                repeated = counts[s][:, bd[:, :, None], gd[:, None, :]].sum(axis=3)
                samples_by_source.append({m: v.mean(axis=2) for m, v in measures(repeated).items()})
            samples = {m: np.stack([v[m] for v in samples_by_source]) for m in points}
            saved[variant, rep] = (points, samples)
            for si, source in enumerate([*SOURCES, 'mean_S']):
                for metric in points:
                    point = points[metric][si] if si < 2 else points[metric].mean(axis=0)
                    draw = samples[metric][si] if si < 2 else samples[metric].mean(axis=0)
                    references = {p: (point[j], draw[j]) for j, p in enumerate(PATHS)}
                    cp, cd = contrasts(point), contrasts(draw)
                    references.update({k: (cp[k], cd[k]) for k in cp})
                    for name, (p, d) in references.items():
                        key = f'{variant}__{rep}__{source}__{name}__{metric}'
                        np.testing.assert_allclose(raw[key], d, rtol=0, atol=1e-12)
                        equal(stats['estimates'][variant][rep][source][name][metric],
                              dict(estimate=float(p), ci95=np.quantile(d, [.025, .975], method='linear').tolist()))
                        count += 1
    for rep in REPS:
        for metric in ['macro_f1', 'balanced_accuracy', 'predicted_truck_fraction']:
            p = saved['matched', rep][0][metric]-saved['native', rep][0][metric]
            d = saved['matched', rep][1][metric]-saved['native', rep][1][metric]
            for si, source in enumerate([*SOURCES, 'mean_S']):
                pp = p[si] if si < 2 else p.mean(axis=0)
                dd = d[si] if si < 2 else d.mean(axis=0)
                for pi, path in enumerate(PATHS):
                    key = f'matched_minus_native__{rep}__{source}__{path}__{metric}'
                    np.testing.assert_allclose(raw[key], dd[pi], rtol=0, atol=1e-12)
                    equal(stats['matched_minus_native'][rep][source][path][metric],
                          dict(estimate=float(pp[pi]), ci95=np.quantile(dd[pi], [.025, .975], method='linear').tolist()))
                    count += 1
    return count


def check_diagnostic(model, x, y, train, train_y, report):
    z = (x-model[0].mean_)/model[0].scale_
    zt = (train-model[0].mean_)/model[0].scale_
    coefficient = model[1].coef_[0]
    logits = z@coefficient+model[1].intercept_[0]
    np.testing.assert_allclose(expit(logits), model.predict_proba(x)[:, 1], rtol=0, atol=1e-12)
    norms = np.linalg.norm(z, axis=1)/np.sqrt(z.shape[1])
    tn = np.linalg.norm(zt, axis=1)/np.sqrt(zt.shape[1])
    threshold = np.quantile(tn, .95)
    equal(report['training_norm_p95'], float(threshold))
    for label, name in enumerate(['car', 'truck']):
        mask = y == label
        r = report['classes'][name]
        contributions = (z[mask].mean(axis=0)-zt[train_y == label].mean(axis=0))*coefficient
        np.testing.assert_allclose(r['mean_logit_shift_contributions'], contributions, rtol=1e-10, atol=1e-10)
        equal(r['count'], int(mask.sum()))
        equal(r['logit'], percentile_summary(logits[mask]))
        equal(r['standardized_feature_rms_norm'], percentile_summary(norms[mask]))
        equal(r['above_training_norm_p95_fraction'], float(np.mean(norms[mask] > threshold)))
        equal(r['mean_fraction_coordinates_outside_training_range'],
              float(np.mean((z[mask] < zt.min(axis=0)) | (z[mask] > zt.max(axis=0)))))
        equal(r['predicted_truck_fraction'], float(np.mean(logits[mask] > 0)))
        equal(r['mean_logit_shift_from_same_training_class'], float(contributions.sum()))


def main():
    out = RESULTS/'verification.json'
    if out.exists():
        raise ValueError('Audit already exists; preserve it')
    begin = time.perf_counter()
    check_execution()
    corpus = read(CORPUS/'summary.json')
    replay = read(CORPUS/'verification.json')
    assert corpus['passed'] and corpus['stage'] == 'full'
    assert corpus['observations'] == corpus['endpoints_identical'] == 9600
    assert corpus['failed_geometries'] == corpus['redraws'] == 0
    assert replay['passed'] and replay['complete_waveform_replays'] == 9600
    assert replay['summary_sha256'] == sha(CORPUS/'summary.json')
    assert corpus['manifest_sha256'] == sha(CORPUS/'observation_manifest.jsonl')
    assert corpus['execution_lock_sha256'] == sha(EXECUTION/'lock.json')
    feature_summary = read(CACHE/'summary.json')
    model_lock = read(MODELS/'lock.json')
    result_summary = read(RESULTS/'summary.json')
    normalized_summary = read(CACHE/'target_gain/summary.json')
    for folder, lock in [(CACHE, feature_summary), (MODELS, model_lock), (RESULTS, result_summary)]:
        for name, digest in lock['artifacts_sha256'].items():
            assert sha(folder/name) == digest, str(folder/name)
    assert feature_summary['corpus_verification_sha256'] == sha(CORPUS/'verification.json')
    assert model_lock['features_summary_sha256'] == sha(CACHE/'summary.json')
    assert result_summary['model_lock_sha256'] == normalized_summary['model_lock_sha256'] == sha(MODELS/'lock.json')
    assert normalized_summary['features_sha256'] == sha(CACHE/'target_gain/features.npz')
    assert not model_lock['target_access'] and not result_summary['selection_or_tuning']

    plans = {r['job_id']: r for r in rows(FREEZE/'observations.jsonl')}
    observations = {r['job_id']: r for r in joined_observations()}
    assert len(observations) == len(plans) == 19200 and set(observations) == set(plans)
    assert Counter(r['role'] for r in observations.values()) == {'train': 15200, 'validation': 4000}
    assert sum(r['reused'] for r in observations.values()) == 9600
    roles = defaultdict(set)
    for jid, row in observations.items():
        for key, value in plans[jid].items():
            assert row[key] == value
        roles[row['source_parent_group']].add(row['role'])
        for key in ['observation', 'render']:
            assert sha(Path(row['resolved_'+key])) == row[key+'_file_sha256'], jid
    assert all(len(v) == 1 for v in roles.values())
    print('Audit: all 19,200 observations and source-parent roles verified', flush=True)

    f = np.load(CACHE/'features.npz', allow_pickle=False)
    fids = f['job_ids'].tolist()
    pos = {j: i for i, j in enumerate(fids)}
    assert set(pos) == set(plans) and len(pos) == 19200
    matrices = {rep: np.asarray(f[rep], float) for rep in REPS}
    with np.load(ROOT/CONFIG['parent_features']/'features.npz', allow_pickle=False) as parent:
        pp = {j: i for i, j in enumerate(parent['job_ids'].tolist())}
        reused = [j for j in fids if observations[j]['reused']]
        for rep in REPS:
            np.testing.assert_array_equal(matrices[rep][[pos[j] for j in reused]],
                                          parent[rep][[pp[observations[j]['parent_job_id']] for j in reused]])
    for j, digest in zip(fids, f['waveform_sha256']):
        assert digest == observations[j]['observation_sha256']

    target = rows(FREEZE/'target_manifest.jsonl')
    tids = [r['file_id'] for r in target]
    y = np.array([r['class_id'] for r in target])
    group_names = sorted({r['provenance_group_id'] for r in target})
    g = np.array([group_names.index(r['provenance_group_id']) for r in target])
    assert np.bincount(y).tolist() == [7810, 256] and len(group_names) == 4
    from learning import h1_cache_compatibility
    compatibility = h1_cache_compatibility()
    with np.load(compatibility['feature_cache'], allow_pickle=False) as old:
        old_pos = {j: i for i, j in enumerate(old['file_ids'].tolist())}
        target_x = {'native': {rep: np.asarray(old[rep][[old_pos[j] for j in tids]], float) for rep in REPS}}
    normalized = np.load(CACHE/'target_gain/features.npz', allow_pickle=False)
    assert normalized['file_ids'].tolist() == tids
    target_x['matched'] = {rep: np.asarray(normalized[rep], float) for rep in REPS}
    predictions = np.load(RESULTS/'target_predictions.npz', allow_pickle=False)
    assert predictions['target_ids'].tolist() == tids
    np.testing.assert_array_equal(predictions['y'], y)
    np.testing.assert_array_equal(predictions['group_index'], g)
    probabilities = {v: {rep: predictions[v+'__'+rep].copy() for rep in REPS} for v in target_x}
    parent_predictions = np.load(ROOT/CONFIG['parent_results']/'target_predictions.npz', allow_pickle=False)
    for rep in REPS:
        for si in range(2):
            for pi, old_pi in [(0, 2*si), (3, 2*si+1)]:
                np.testing.assert_allclose(probabilities['native'][rep][si, pi], parent_predictions[rep][old_pi], rtol=0, atol=1e-12)

    refs = read(MODELS/'fits.json')
    fit_plans = {r['fit_id']: r for r in rows(FREEZE/'fits.jsonl')}
    evaluations = {(r['fit_id'], r['variant']): r for r in read(RESULTS/'evaluations.json')}
    diags = {r['fit_id']: r for r in read(RESULTS/'collapse_diagnostics.json')}
    cross_metrics = {(r['fit_id'], r['test_path']): r for r in read(RESULTS/'cross_path_validation.json')}
    gain_metrics = {(r['fit_id'], r['gain_db']): r for r in read(RESULTS/'validation_gain.json')}
    sp = np.load(RESULTS/'synthetic_diagnostic_predictions.npz', allow_pickle=False)
    stress = np.load(CACHE/'validation_gain.npz', allow_pickle=False)
    stress_pos = {j: i for i, j in enumerate(stress['job_ids'].tolist())}
    assert len(refs) == 80 and sum(r['reused'] for r in refs) == 40
    assert len(evaluations) == 160 and len(cross_metrics) == 320 and len(gain_metrics) == 240
    assert sp['fit_ids'].tolist() == [r['fit_id'] for r in refs]
    assert sp['paths'].tolist() == PATHS and sp['gains_db'].tolist() == [-12., 0., 12.]
    assert stress['gains_db'].tolist() == [-12., 12.]
    max_error = 0.
    with threadpool_limits(limits=1):
        for fi, ref in enumerate(refs):
            plan = fit_plans[ref['fit_id']]
            for key, value in plan.items():
                equal(ref[key], value, key)
            rep = ref['representation']
            train, val = ref['train_job_ids'], ref['validation_job_ids']
            assert len(train) == 380 and len(val) == 100 and not set(train) & set(val)
            assert not {observations[j]['source_parent_group'] for j in train} & {observations[j]['source_parent_group'] for j in val}
            assert all(observations[j]['role'] == 'train' for j in train)
            assert all(observations[j]['role'] == 'validation' for j in val)
            ty = np.array([observations[j]['class_id'] for j in train])
            vy = np.array([observations[j]['class_id'] for j in val])
            assert np.bincount(ty).tolist() == [190, 190] and np.bincount(vy).tolist() == [50, 50]
            tx = matrices[rep][[pos[j] for j in train]]
            vx = matrices[rep][[pos[j] for j in val]]
            assert sha(ROOT/ref['model_path']) == ref['model_sha256']
            model = joblib.load(ROOT/ref['model_path'])
            params = model[1].get_params()
            for key, value in BASE['classifier'].items():
                if key != 'type':
                    assert params[key] == value
            assert params['random_state'] == ref['replicate_seed'] and model.classes_.tolist() == [0, 1]
            assert np.all(model[1].n_iter_ < params['max_iter'])
            np.testing.assert_allclose(model[0].mean_, tx.mean(axis=0), rtol=0, atol=1e-12)
            np.testing.assert_allclose(model[0].var_, tx.var(axis=0), rtol=1e-12, atol=1e-12)
            check_diagnostic(model, tx, ty, tx, ty, diags[ref['fit_id']]['train'])
            check_diagnostic(model, vx, vy, tx, ty, diags[ref['fit_id']]['validation'])
            equal(ref['validation'], full_score(vy, model.predict_proba(vx)[:, 1]))
            with np.load(ROOT/ref['validation_path'], allow_pickle=False) as pv:
                assert pv['job_ids'].tolist() == val
                np.testing.assert_array_equal(pv['y'], vy)
                np.testing.assert_allclose(pv['p'], model.predict_proba(vx)[:, 1], rtol=0, atol=1e-12)
            si, pi, bi = SOURCES.index(ref['source_level']), PATHS.index(ref['path_level']), SEEDS.index(ref['replicate_seed'])
            for variant, matrix in target_x.items():
                pr = model.predict_proba(matrix[rep])[:, 1]
                stored = probabilities[variant][rep][si, pi, bi]
                max_error = max(max_error, float(abs(pr-stored).max()))
                np.testing.assert_allclose(pr, stored, rtol=0, atol=1e-12)
                ev = evaluations[ref['fit_id'], variant]
                equal(ev['target'], full_score(y, pr))
                for gi, group in enumerate(group_names):
                    equal(ev['groups'][group], full_score(y[g == gi], pr[g == gi]))
                equal(ev['worst_group_class_recall'], min(v[c+'_recall'] for v in ev['groups'].values() for c in ['car', 'truck']))
                check_diagnostic(model, matrix[rep], y, tx, ty, diags[ref['fit_id']]['targets'][variant])
            np.testing.assert_array_equal(sp['y'][fi], vy)
            for pj, path in enumerate(PATHS):
                jj = [j.rsplit('.', 1)[0]+'.'+path for j in val]
                np.testing.assert_array_equal(vy, [observations[j]['class_id'] for j in jj])
                assert all(observations[j]['role'] == 'validation' for j in jj)
                pr = model.predict_proba(matrices[rep][[pos[j] for j in jj]])[:, 1]
                np.testing.assert_allclose(sp['cross_path'][fi, pj], pr, rtol=0, atol=1e-12)
                equal(cross_metrics[ref['fit_id'], path]['metrics'], full_score(vy, pr))
            for gj, gain in enumerate([-12., 0., 12.]):
                xx = vx if gain == 0 else np.asarray(stress[rep][0 if gain < 0 else 1, [stress_pos[j] for j in val]], float)
                pr = model.predict_proba(xx)[:, 1]
                np.testing.assert_allclose(sp['gain'][fi, gj], pr, rtol=0, atol=1e-12)
                equal(gain_metrics[ref['fit_id'], gain]['metrics'], full_score(vy, pr))
                for label, name in enumerate(['car', 'truck']):
                    equal(gain_metrics[ref['fit_id'], gain]['logit_by_class'][name], percentile_summary(model.decision_function(xx[vy == label])))
    print('Audit: all 80 heads, 160 target evaluations, 320 cross-path and 240 gain checks verified', flush=True)

    stats, raw, indices = calculate(y, g, probabilities)
    equal(stats, read(RESULTS/'statistics.json'))
    with np.load(RESULTS/'bootstrap_samples.npz', allow_pickle=False) as saved:
        assert set(saved.files) == set(raw)
        for key, value in raw.items():
            np.testing.assert_allclose(saved[key], value, rtol=0, atol=1e-12)
    with np.load(RESULTS/'bootstrap_indices.npz', allow_pickle=False) as saved:
        for key, value in indices.items():
            np.testing.assert_array_equal(saved[key], value)
    reconstructed = independent_intervals(y, g, probabilities, stats, raw, indices)
    print('Audit: paired uncertainty independently reconstructed from integer confusion counts', flush=True)

    # Rebuild every original crop; use a direct scalar formula rather than normalize_target().
    sys.path.insert(0, str(ROOT/'experiments/h1'))
    from extract import load
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (wave, mfcc, digest) in enumerate(pool.map(lambda r: load(r, BASE), target)):
            assert digest == normalized['original_waveform_sha256'][i] == compatibility['target_observation_sha256'][i]
            np.testing.assert_array_equal(mfcc, target_x['native']['MFCC26'][i])
            gain = 10**(-26/20)/np.sqrt(np.mean(wave.astype(float)**2))
            rebuilt = (wave.astype(float)*gain).astype(np.float32)
            assert hashlib.sha256(rebuilt.tobytes()).hexdigest() == normalized['normalized_waveform_sha256'][i]
            equal(float(gain), float(normalized['gain'][i]))
            np.testing.assert_allclose(20*np.log10(np.sqrt(np.mean(rebuilt.astype(float)**2))), -26., rtol=0, atol=1e-5)
    np.testing.assert_allclose(normalized['original_acoustic'][:, 2:], normalized['normalized_acoustic'][:, 2:], rtol=2e-6, atol=1e-4)
    np.testing.assert_allclose(normalized['normalized_acoustic'][:, 0], -26, rtol=0, atol=1e-5)
    assert normalized_summary['clipped_events'] == 0
    assert normalized_summary['peak_gt_one_events'] == int((normalized['normalized_acoustic'][:, 1] > 1).sum())
    preserved = verify_parent_files()
    report = dict(passed=True, observed_population=19200, new_full_waveform_replays=9600,
        endpoints_identical=9600, heads=80, new_heads=40, reused_heads=40,
        source_parent_roles_and_train_only_scalers_verified=True, fixed_parameters_and_convergence_verified=True,
        target_evaluations_replayed=160, cross_path_evaluations_replayed=320, gain_evaluations_replayed=240,
        native_endpoint_predictions_preserved=40, max_prediction_replay_error=max_error,
        target_crop_and_gain_replays=8066, manual_logit_and_contribution_reconstruction_verified=True,
        bootstrap_draws=10000, independently_reconstructed_intervals=reconstructed,
        **preserved, target_role='previously_exposed_development', target_tuning=0, new_fits=0,
        execution_lock_sha256=sha(EXECUTION/'lock.json'), result_summary_sha256=sha(RESULTS/'summary.json'),
        verifier_sha256=sha(Path(__file__)), elapsed_s=time.perf_counter()-begin)
    save(out, report)
    print(report, flush=True)


if __name__ == '__main__':
    main()

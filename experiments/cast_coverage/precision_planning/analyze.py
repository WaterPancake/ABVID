"""Read saved H1 scalar scores/paired draws; never train, infer or open audio."""
import hashlib
import json
import argparse
from pathlib import Path
import platform
import subprocess

import numpy as np

from abvid.evaluation.baselines import f1
from abvid.evaluation.precision import paired_summary, crossed_components, normal_scenarios
from abvid.paths import resolve, roots
from abvid.protocol import load


OUT = 'artifact:cast/precision/cast5a_h1_precision_20261011_v1'
DIAG = 'archive:experiments/h1_diagnostics/results/source_diagnostics_20261003_v1/'
H1 = 'archive:experiments/h1/results/H1_20261002/'
FROZEN = 'archive:experiments/h1_diagnostics/frozen/source_diagnostics_20261003_v1/'
REPS = ('BEATs768', 'MFCC26')
CONTRASTS = [(name, name, 'baseline') for name in
             ('nested_C','native16_2s','core8_1s','nested_threshold','held_site')]
CONTRASTS += [(f'{rule}_359_minus_{n}', f'learning_{rule}_359',
               'baseline' if n == 190 else f'learning_{rule}_25')
              for rule in ('fixed','scaled') for n in (190,25)]
TARGET = [('gain_real_plus_procedural','real_plus_procedural','real'),
          ('mixed_procedural_minus_repeated_real','real_plus_procedural','real_repeated'),
          ('gain_real_plus_audioldm','real_plus_audioldm','real'),
          ('mixed_audioldm_minus_repeated_real','real_plus_audioldm','real_repeated')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=OUT, help='new artifact: directory; existing outputs are refused')
    args = parser.parse_args()
    if not args.output.startswith('artifact:'):
        raise ValueError('Planning output must use artifact storage')
    # Canonical H1 wrapper is validated, not executed. The saved diagnostics
    # retain their own original config, split lock and result-file hashes.
    load('configs/experiments/h1_baselines.json', check_files=True)
    observed_hashes = {}
    def read(ref, expected=None):
        data = resolve(ref).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError('Changed saved input: '+ref)
        observed_hashes[ref] = digest
        return data
    def obj(ref, expected=None):
        return json.loads(read(ref, expected))
    def arrays(ref, expected):
        from io import BytesIO
        return np.load(BytesIO(read(ref, expected)), allow_pickle=False)
    diag = obj(DIAG+'summary.json')
    h1 = obj(H1+'summary.json')
    dv, hv = obj(DIAG+'verification.json'), obj(H1+'verification.json')
    assert dv['passed'] and dv['all_aggregate_intervals_verified']
    assert hv['hashes_verified'] and observed_hashes[H1+'summary.json'] == hv['results_summary_sha256']
    lock = obj(FROZEN+'lock.json',diag['lock_sha256'])
    config = obj(FROZEN+'config.json',lock['artifacts_sha256']['config.json'])
    assert config['bootstrap']['draws'] == 10000 and config['bootstrap']['seed'] == 314159
    assert len(lock['folds']) == 6 and len(config['seeds']) == 5
    assert lock['h1_lock_sha256'] == h1['lock_sha256']
    rows = obj(DIAG+'evaluations.json',diag['artifacts_sha256']['evaluations.json'])
    estimates = obj(H1+'metrics.json',h1['artifacts_sha256']['metrics.json'])
    for row in rows:
        assert abs(float(f1(row['metrics']['confusion_matrix']))-row['metrics']['macro_f1']) < 1e-14
    sites = sorted({f['site'] for f in lock['folds']})
    site_folds = np.array([[f['fold'] for f in lock['folds'] if f['site']==s] for s in sites])
    assert site_folds.shape == (3,2)
    with arrays(DIAG+'bootstrap_indices.npz',diag['artifacts_sha256']['bootstrap_indices.npz']) as archive:
        ix = {k:archive[k].copy() for k in archive.files}
    rng = np.random.default_rng(314159)
    assert np.array_equal(ix['seed'],rng.integers(5,size=(10000,5)))
    assert np.array_equal(ix['group'],rng.integers(6,size=(10000,6)))
    assert np.array_equal(ix['site'],site_folds[rng.integers(3,size=(10000,3))].reshape(10000,6))
    matrices = {}
    cases = sorted({x for _,a,b in CONTRASTS for x in (a,b)})
    for rep in REPS:
        for case in cases:
            selected = [r for r in rows if r['representation']==rep and r['case']==case]
            assert len(selected)==30 and len({(r['seed'],r['fold']) for r in selected})==30
            matrix = np.empty((5,6))
            for row in selected:
                assert row['group'] == lock['folds'][row['fold']]['held_out_group']
                assert row['site'] == lock['folds'][row['fold']]['site']
                matrix[config['seeds'].index(row['seed']),row['fold']]=row['metrics']['macro_f1']
            matrices[rep,case] = matrix
    results = []
    with arrays(DIAG+'bootstrap_samples.npz',diag['artifacts_sha256']['bootstrap_samples.npz']) as saved:
        for rep in REPS:
            for name, a, b in CONTRASTS:
                d = matrices[rep,a]-matrices[rep,b]
                point = float(d.mean()*100)
                for scheme in ('group','site'):
                    generated = [matrices[rep,c][ix['seed'][:,:,None],ix[scheme][:,None,:]].mean(axis=(1,2)) for c in (a,b)]
                    original = [saved[f'{rep}__{c}__macro_f1__{scheme}'] for c in (a,b)]
                    assert all(np.array_equal(x,y) for x,y in zip(generated,original))
                    result = paired_summary(original[0]*100,original[1]*100,point)
                    stated = diag['paired_contrasts'][rep][name]['macro_f1']
                    assert np.allclose(result['ci95_pp'],np.array(stated[scheme+'_seed_ci95'])*100,rtol=0,atol=1e-11)
                    assert abs(point-stated['mean']*100)<1e-11
                    block_matrix = d*100 if scheme=='group' else (d[:,site_folds].mean(axis=2)*100)
                    components = crossed_components(block_matrix)
                    # Monte Carlo SD should approximate the exact empirical-crossed SD;
                    # 10k draws are not required to be bitwise equal to that expectation.
                    assert abs(components['exact_bootstrap_sd_pp']/result['paired_estimator_sd_pp']-1)<.04
                    results.append({'domain':'IDMT_source','representation':rep,'contrast':name,
                                    'scheme':scheme,**result,'variance_components':components,
                                    'paired_differences_selection_by_group_pp':(d*100).tolist(),
                                    'normal_location_scale_scenarios':normal_scenarios(result['paired_estimator_sd_pp'])})
    for rep in REPS:
        with arrays(H1+rep+'_bootstrap.npz',h1['artifacts_sha256'][rep+'_bootstrap.npz']) as saved:
            for name,a,b in TARGET:
                point=estimates['inference'][rep]['estimates'][name]['estimate']*100
                result=paired_summary(saved[a]*100,saved[b]*100,point)
                stated=estimates['inference'][rep]['estimates'][name]['ci95']
                assert np.allclose(result['ci95_pp'],np.array(stated)*100,rtol=0,atol=1e-11)
                results.append({'domain':'MELAUDIS_target_context_only','representation':rep,
                                'contrast':name,'scheme':'target_groups_source_folds_selections',**result,
                                'normal_location_scale_scenarios':normal_scenarios(result['paired_estimator_sd_pp'])})
    # Closed-form fixtures: group-only and seed-only variation have the known
    # variance; paired identical arms have exactly zero contrast uncertainty.
    assert crossed_components(np.tile(np.arange(6),(5,1)))['exact_bootstrap_variance_pp2'] == np.var(np.arange(6))/6
    assert crossed_components(np.tile(np.arange(5)[:,None],(1,6)))['exact_bootstrap_variance_pp2'] == np.var(np.arange(5))/5
    assert paired_summary(np.arange(10),np.arange(10),0)['paired_estimator_variance_pp2']==0
    # Hash the exact analysis and inherited resampling implementation too.
    for ref in ('repo:src/abvid/evaluation/precision.py','repo:experiments/cast_coverage/precision_planning/analyze.py',
                'repo:src/abvid/evaluation/baselines.py','repo:CAST_ROADMAP.md',
                DIAG+'executed_run_checks.py','repo:configs/experiments/h1_baselines.json'):
        read(ref,diag['artifacts_sha256'].get('executed_run_checks.py') if ref==DIAG+'executed_run_checks.py' else None)
    for ref,digest in observed_hashes.items():
        assert hashlib.sha256(resolve(ref).read_bytes()).hexdigest()==digest
    output = resolve(args.output,must_exist=False)
    assert output.is_relative_to(roots()['artifact']) and not output.is_relative_to(roots()['archive'])
    output.mkdir(parents=True,exist_ok=False)
    summary={'analysis_id':'cast5a_h1_precision_planning_v1','scope':'existing scalar scores and paired bootstrap draws only',
             'hypothesis_margin_pp':3,'prospective_RC_minus_RA_variance':'unknown',
             'new_fits':0,'new_inference':0,'audio_or_feature_arrays_accessed':False,
             'bootstrap_replay':'18 source contrasts x 2 schemes and 8 target contrasts reproduced',
             'rows':results,'seeds':config['seeds'],'folds':[{'fold':f['fold'],'group':f['held_out_group'],'site':f['site']} for f in lock['folds']],
             'normal_scenarios_status':'conditional planning approximation; neither validated power nor a CAST effect estimate',
             'environment':{'python':platform.python_version(),'numpy':np.__version__},
             'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             'git_dirty_status':subprocess.check_output(['git','status','--porcelain'],text=True),
             'input_sha256':observed_hashes}
    payload=(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    (output/'precision.json').write_bytes(payload)
    for ref in ('repo:src/abvid/evaluation/precision.py','repo:experiments/cast_coverage/precision_planning/analyze.py'):
        (output/Path(ref).name).write_bytes(resolve(ref).read_bytes())
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())}
    (output/'lock.json').write_text(json.dumps({'payload_sha256':hashes,'input_sha256':observed_hashes},indent=2,sort_keys=True)+'\n')
    print(json.dumps({'output':str(output),'contrast_scheme_rows':len(results),'all_checks_passed':True},indent=2))
    for row in results:
        if row['representation']=='BEATs768':
            print(row['domain'],row['contrast'],row['scheme'],'effect/CI/SD/width/threshold/MDE80:',
                  round(row['estimate_pp'],3),[round(x,3) for x in row['ci95_pp']],
                  *[round(row[k],3) for k in ('paired_estimator_sd_pp','width_pp','observed_point_to_clear_margin_pp','normal_true_effect_for_80pct_margin_support_pp')])


if __name__=='__main__':
    main()

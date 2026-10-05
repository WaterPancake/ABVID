"""Read-only post-freeze audit of the complete attenuation/latency comparison."""
import importlib.util
import time
from collections import Counter,defaultdict
import hashlib

import joblib
from scipy.special import expit
from threadpoolctl import threadpool_limits

from phase_common import *
from phase_statistics import calculate,full_score,NAMES
from phase_learning import target_data
from metrics import interval

spec=importlib.util.spec_from_file_location('ground_air_integer_audit',PARENT_HERE/'verify_results.py')
prior_audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior_audit)
equal=prior_audit.equal


def independent_intervals(y,g,p,stats,raw,indices):
    """Relabel numerical axes for the independent integer-count reconstruction.

The first numerical factor is latency; the second is amplitude. The independent
reconstruction computes scores from raw predictions and literal repeated groups,
not from the production score/contrast/bootstrap implementation.
"""
    inverse={v:k for k,v in NAMES.items()}
    def translate(value):
        if isinstance(value,dict):return {inverse.get(k,k):translate(v) for k,v in value.items()}
        return value
    old_stats={k:translate(stats[k]) for k in ['estimates','matched_minus_native']}
    old_raw={'__'.join(inverse.get(k,k) for k in key.split('__')):value
             for key,value in raw.items() if not key.startswith('S1_minus_S0')}
    count=prior_audit.independent_intervals(y,g,p,old_stats,old_raw,indices)
    extra=0
    for variant in p:
        for rep in REPS:
            for effect,metrics in stats['source_interactions'][variant][rep].items():
                for metric,result in metrics.items():
                    a=stats['estimates'][variant][rep]['S0'][effect][metric]['estimate']
                    b=stats['estimates'][variant][rep]['S1'][effect][metric]['estimate']
                    draws=raw[f'{variant}__{rep}__S1__{effect}__{metric}']-raw[f'{variant}__{rep}__S0__{effect}__{metric}']
                    equal(result,interval(b-a,draws))
                    np.testing.assert_array_equal(raw[f'S1_minus_S0__{variant}__{rep}__{effect}__{metric}'],draws)
                    extra+=1
    primary=stats['estimates']['native']['BEATs768']['mean_S']
    d=primary['latency_minus_attenuation']['macro_f1']['ci95']
    latency=primary['latency']['macro_f1']['ci95'];amplitude=primary['attenuation']['macro_f1']['ci95']
    e=primary['attenuation_at_L1']['macro_f1'];margin=CONFIG['practical_margin']
    expected=dict(primary=CONFIG['primary'],latency_larger=d[0]>0,attenuation_larger=d[1]<0,
        practical_latency_dominance=d[0]>=margin and latency[0]>0,
        practical_attenuation_dominance=d[1]<=-margin and amplitude[0]>0,
        latency_only_equivalent_to_joint_within_three_points=e['ci95'][0]>=-margin and e['ci95'][1]<=margin,
        equivalence_diagnostic=e,independent_confirmation=False)
    equal(stats['decision'],expected)
    return count,extra


def main():
    out=RESULTS/'verification.json'
    if out.exists():raise ValueError('Preserve existing audit')
    begin=time.perf_counter();check_execution()
    corpus=read(CORPUS/'summary.json');replay=read(CORPUS/'verification.json')
    feature_meta=read(CACHE/'summary.json');model_lock=read(MODELS/'lock.json')
    summary=read(RESULTS/'summary.json')
    assert corpus['passed'] and corpus['stage']=='full'
    assert corpus['observations']==corpus['endpoints_identical']==9600
    assert corpus['failed_geometries']==corpus['redraws']==0
    assert replay['passed'] and replay['complete_waveform_replays']==replay['exact_observation_replays']==9600
    assert replay['summary_sha256']==sha(CORPUS/'summary.json')
    assert corpus['manifest_sha256']==sha(CORPUS/'observations.jsonl')
    assert corpus['execution_lock_sha256']==feature_meta['execution_lock_sha256']==sha(EXECUTION/'lock.json')
    assert feature_meta['corpus_verification_sha256']==sha(CORPUS/'verification.json')
    assert feature_meta['features_sha256']==sha(CACHE/'features.npz')
    assert model_lock['features_summary_sha256']==sha(CACHE/'summary.json')
    assert summary['model_lock_sha256']==sha(MODELS/'lock.json')
    assert summary['parent_target_gain_summary_sha256']==sha(parent.CACHE/'target_gain/summary.json')
    assert not feature_meta['target_access'] and not model_lock['target_access'] and not summary['selection_or_tuning']
    assert not corpus['complete_new_render_arrays_retained'] and corpus['complete_new_render_hashes_retained']
    for folder,lock in [(MODELS,model_lock),(RESULTS,summary)]:
        for name,digest in lock['artifacts_sha256'].items():
            assert sha(folder/name)==digest,str(folder/name)

    plans={r['job_id']:r for r in rows(FREEZE/'observations.jsonl')}
    obs={r['job_id']:r for r in joined_observations()}
    original={r['job_id']:r for r in parent.joined_observations()}
    assert len(plans)==len(obs)==19200 and set(plans)==set(obs)
    assert Counter(r['role'] for r in obs.values())=={'train':15200,'validation':4000}
    assert sum(r['reused'] for r in obs.values())==9600
    assert len(list((CORPUS/'observations').glob('*.npy')))==9600
    roles=defaultdict(set);geometry=defaultdict(set)
    source_fields=['event_id','geometry_id','template_id','dry_source_id','source_parent_group',
                   'source_level','role','class_id','replicate_seed']
    for jid,row in obs.items():
        for key,value in plans[jid].items():equal(row[key],value)
        assert row['ground'] and row['amplitude']==(row['cell'][1]=='1') and row['latency']==(row['cell'][3]=='1')
        roles[row['source_parent_group']].add(row['role'])
        geometry[row['geometry_id']].add((row['source_level'],row['cell'],row['class_id']))
        path=Path(row['resolved_observation'])
        assert sha(path)==row['observation_file_sha256']
        wave=np.load(path,allow_pickle=False)
        assert wave.shape==(32000,) and wave.dtype==np.float32 and np.isfinite(wave).all()
        assert hashlib.sha256(wave.tobytes()).hexdigest()==row['observation_sha256']
        assert abs(wave).max()<=.98
        if row['reused']:
            old=original[row['parent_job_id']]
            for key in source_fields:equal(row[key],old[key],jid+'.'+key)
            assert old['path_level']==CONFIG['parent_cell_mapping'][row['cell']]
            assert row['rendered_waveform_sha256']==old['rendered_waveform_sha256']
        else:
            assert row['full_render_samples']==80000 and row['full_render_dtype']=='float64'
            assert not row['full_render_array_retained']
    assert len(roles)==240 and all(len(v)==1 for v in roles.values())
    assert len(geometry)==1200 and all(len(v)==16 for v in geometry.values())
    print('Audit: full populations, source roles and all observation hashes verified',flush=True)

    with np.load(CACHE/'features.npz',allow_pickle=False) as file:
        fids=file['job_ids'].tolist();pos={j:i for i,j in enumerate(fids)}
        data={rep:np.asarray(file[rep],float) for rep in REPS}
        fhashes=file['waveform_sha256'].tolist()
    assert len(pos)==19200 and set(pos)==set(obs)
    for j,digest in zip(fids,fhashes):assert digest==obs[j]['observation_sha256']
    with np.load(parent.CACHE/'features.npz',allow_pickle=False) as old:
        pp={j:i for i,j in enumerate(old['job_ids'].tolist())}
        reused=[j for j in fids if obs[j]['reused']]
        for rep in REPS:
            assert np.isfinite(data[rep]).all()
            np.testing.assert_array_equal(data[rep][[pos[j] for j in reused]],old[rep][[pp[obs[j]['parent_job_id']] for j in reused]])
    equal(feature_meta['encoder_provenance'],read(parent.CACHE/'summary.json')['encoder_provenance'])

    target=rows(FREEZE/'target_manifest.jsonl')
    tids=[r['file_id'] for r in target];y=np.array([r['class_id'] for r in target])
    groups=sorted({r['provenance_group_id'] for r in target})
    g=np.array([groups.index(r['provenance_group_id']) for r in target])
    assert len(tids)==8066 and np.bincount(y).tolist()==[7810,256] and len(groups)==4
    ti,ty,tg,tgi,targets=target_data()
    assert ti==tids and tg==groups;np.testing.assert_array_equal(ty,y);np.testing.assert_array_equal(tgi,g)
    with np.load(RESULTS/'target_predictions.npz',allow_pickle=False) as file:
        assert file['target_ids'].tolist()==tids
        np.testing.assert_array_equal(file['y'],y);np.testing.assert_array_equal(file['group_index'],g)
        probabilities={v:{rep:file[v+'__'+rep].copy() for rep in REPS} for v in targets}
    with np.load(parent.RESULTS/'target_predictions.npz',allow_pickle=False) as file:
        for variant in targets:
            for rep in REPS:
                for ci,pj in [(0,2),(3,3)]:
                    np.testing.assert_allclose(probabilities[variant][rep][:,ci],file[variant+'__'+rep][:,pj],rtol=0,atol=1e-12)
    fits=read(MODELS/'fits.json');fit_plans={r['fit_id']:r for r in rows(FREEZE/'fits.jsonl')}
    old_fits={r['fit_id']:r for r in read(parent.MODELS/'fits.json')}
    ev={(r['fit_id'],r['variant']):r for r in read(RESULTS/'evaluations.json')}
    cross={(r['fit_id'],r['test_cell']):r for r in read(RESULTS/'cross_cell_validation.json')}
    cp=np.load(RESULTS/'cross_cell_predictions.npz',allow_pickle=False)
    assert len(fits)==80 and len({r['fit_id'] for r in fits})==80 and set(fit_plans)=={r['fit_id'] for r in fits}
    assert sum(r['reused'] for r in fits)==40 and len(ev)==160 and len(cross)==320
    assert cp['fit_ids'].tolist()==[r['fit_id'] for r in fits] and cp['cells'].tolist()==CELLS
    error=0.
    with threadpool_limits(limits=1):
        for fi,ref in enumerate(fits):
            plan=fit_plans[ref['fit_id']]
            for key,value in plan.items():equal(ref[key],value)
            train=ref['train_job_ids'];val=ref['validation_job_ids'];rep=ref['representation']
            assert len(train)==380 and len(val)==100 and not set(train)&set(val)
            assert not {obs[j]['source_parent_group'] for j in train}&{obs[j]['source_parent_group'] for j in val}
            assert all(obs[j]['role']=='train' for j in train) and all(obs[j]['role']=='validation' for j in val)
            yy=np.array([obs[j]['class_id'] for j in train]);vy=np.array([obs[j]['class_id'] for j in val])
            assert np.bincount(yy).tolist()==[190,190] and np.bincount(vy).tolist()==[50,50]
            xx=data[rep][[pos[j] for j in train]];vx=data[rep][[pos[j] for j in val]]
            assert sha(ROOT/ref['model_path'])==ref['model_sha256']
            if ref['reused']:
                old=old_fits[ref['parent_fit_id']]
                assert ref['model_path']==old['model_path'] and ref['model_sha256']==old['model_sha256']
            model=joblib.load(ROOT/ref['model_path']);params=model[1].get_params()
            for key,value in BASE['classifier'].items():
                if key!='type':assert params[key]==value
            assert params['random_state']==ref['replicate_seed'] and model.classes_.tolist()==[0,1]
            assert np.all(model[1].n_iter_<params['max_iter'])
            np.testing.assert_allclose(model[0].mean_,xx.mean(axis=0),rtol=0,atol=1e-12)
            np.testing.assert_allclose(model[0].var_,xx.var(axis=0),rtol=1e-12,atol=1e-12)
            assert sha(ROOT/ref['validation_path'])==ref['validation_sha256']
            with np.load(ROOT/ref['validation_path'],allow_pickle=False) as file:
                assert file['job_ids'].tolist()==val;np.testing.assert_array_equal(file['y'],vy)
                pv=model.predict_proba(vx)[:,1]
                np.testing.assert_allclose(file['p'],pv,rtol=0,atol=1e-12)
                equal(ref['validation'],full_score(vy,pv))
            si,ci,bi=SOURCES.index(ref['source_level']),CELLS.index(ref['cell']),SEEDS.index(ref['replicate_seed'])
            for variant,matrix in targets.items():
                x=matrix[rep]
                z=(x-model[0].mean_)/model[0].scale_
                manual=expit(z@model[1].coef_[0]+model[1].intercept_[0])
                p=model.predict_proba(x)[:,1];saved=probabilities[variant][rep][si,ci,bi]
                np.testing.assert_allclose(manual,p,rtol=0,atol=1e-12)
                np.testing.assert_allclose(saved,p,rtol=0,atol=1e-12)
                error=max(error,float(abs(p-saved).max()))
                record=ev[ref['fit_id'],variant]
                equal(record['target'],full_score(y,p))
                for gi,group in enumerate(groups):equal(record['groups'][group],full_score(y[g==gi],p[g==gi]))
                equal(record['worst_group_class_recall'],min(v[k+'_recall'] for v in record['groups'].values() for k in ['car','truck']))
            np.testing.assert_array_equal(cp['y'][fi],vy)
            for cj,cell in enumerate(CELLS):
                ids=[j.rsplit('.',1)[0]+'.'+cell for j in val]
                assert all(obs[j]['role']=='validation' and obs[j]['source_level']==ref['source_level'] for j in ids)
                np.testing.assert_array_equal(vy,[obs[j]['class_id'] for j in ids])
                p=model.predict_proba(data[rep][[pos[j] for j in ids]])[:,1]
                np.testing.assert_allclose(cp['p'][fi,cj],p,rtol=0,atol=1e-12)
                equal(cross[ref['fit_id'],cell]['metrics'],full_score(vy,p))
    print('Audit: all 80 heads, 160 target evaluations and 320 cross-cell checks reproduced',flush=True)

    stats,raw,indices=calculate(y,g,probabilities)
    equal(stats,read(RESULTS/'statistics.json'))
    with np.load(RESULTS/'bootstrap_samples.npz',allow_pickle=False) as old:
        assert set(old.files)==set(raw)
        for key,value in raw.items():np.testing.assert_allclose(old[key],value,rtol=0,atol=1e-12)
    with np.load(RESULTS/'bootstrap_indices.npz',allow_pickle=False) as old:
        for key,value in indices.items():np.testing.assert_array_equal(old[key],value)
    count,extra=independent_intervals(y,g,probabilities,stats,raw,indices)
    for name,p in [('always_car',0.),('always_truck',1.)]:equal(summary['constant_controls'][name],full_score(y,np.full(len(y),p)))
    print('Audit: integer-count intervals, source interactions and declared decisions verified',flush=True)
    protected=check_preserved()
    result=dict(passed=True,observations=19200,new_complete_waveform_replays=9600,new_exact_observation_replays=9600,
        endpoints_identical=9600,heads=80,new_heads=40,reused_heads=40,
        train_only_scaler_roles_fixed_parameters_and_convergence_verified=True,
        target_evaluations_replayed=160,cross_cell_evaluations_replayed=320,
        original_target_vectors_preserved=80,max_prediction_replay_error=error,
        manual_linear_score_probabilities_verified=True,paired_bootstrap_draws=10000,
        independently_reconstructed_integer_count_intervals=count,
        additional_source_interaction_intervals_verified=extra,declared_decision_rules_verified=True,
        **protected,new_fits=0,new_target_tuning=0,target_role='previously_exposed_development',
        execution_lock_sha256=sha(EXECUTION/'lock.json'),result_summary_sha256=sha(RESULTS/'summary.json'),
        verifier_sha256=sha(Path(__file__)),integer_verifier_sha256=sha(PARENT_HERE/'verify_results.py'),
        elapsed_s=time.perf_counter()-begin)
    save(out,result);print(result,flush=True)


if __name__=='__main__':main()

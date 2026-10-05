"""Read-only post-run H2 audit, independent of the execution entry points."""
import argparse
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from metadata import HERE,ROOT,read,rows,save,sha
from generate_corpus import checked_execution
from metrics import score,bootstrap,CELLS


def independent_f1_reconstruction(y, group_index, probabilities, stats, samples, indices):
    """Reconstruct the primary estimand directly from integer confusion counts.

    This does not call score(), contrasts() or bootstrap(). It checks the actual
    saved population, all paired draws and intervals against a separate path.
    """
    rng=np.random.Generator(np.random.PCG64(314159))
    group_draws=np.empty((10000,4),dtype=np.int64)
    bank_draws=np.empty((10000,5),dtype=np.int64)
    for draw in range(10000):
        group_draws[draw]=rng.integers(0,4,size=4)
        bank_draws[draw]=rng.integers(0,5,size=5)
    np.testing.assert_array_equal(group_draws,indices['target_groups'])
    np.testing.assert_array_equal(bank_draws,indices['banks'])

    def f1_from_counts(cm):
        numerator=2*np.diagonal(cm,axis1=-2,axis2=-1)
        denominator=cm.sum(axis=-1)+cm.sum(axis=-2)
        return np.divide(numerator,denominator,out=np.zeros_like(numerator,dtype=float),
            where=denominator>0).mean(axis=-1)

    for rep,p in probabilities.items():
        counts=np.empty((4,5,4,2,2),dtype=np.int64)
        for ci in range(4):
            for bank in range(5):
                for group in range(4):
                    selected=group_index==group
                    counts[ci,bank,group]=np.bincount(
                        2*y[selected]+(p[ci,bank,selected]>.5),minlength=4).reshape(2,2)
        point=f1_from_counts(counts.sum(axis=2)).mean(axis=1)
        repeated=counts[:,bank_draws[:,:,None],group_draws[:,None,:]].sum(axis=3)
        draws=f1_from_counts(repeated).mean(axis=2)
        q00,q01,q10,q11=draws
        p00,p01,p10,p11=point
        references={
            'source_at_P0':(p10-p00,q10-q00),
            'source_at_P1':(p11-p01,q11-q01),
            'path_at_S0':(p01-p00,q01-q00),
            'path_at_S1':(p11-p10,q11-q10),
            'delta_S':((p10-p00+p11-p01)/2,(q10-q00+q11-q01)/2),
            'delta_P':((p01-p00+p11-p10)/2,(q01-q00+q11-q10)/2),
            'interaction':(p11-p10-p01+p00,q11-q10-q01+q00),
            'D':(p10-p01,q10-q01)}
        for ci,cell in enumerate(CELLS):
            references[cell]=(point[ci],draws[ci])
        for name,(estimate,values) in references.items():
            np.testing.assert_allclose(samples[f'{rep}__{name}__macro_f1'],values,rtol=0,atol=1e-12)
            kind='cells' if name in CELLS else 'contrasts'
            compare_nested(stats['estimates'][rep][kind][name]['macro_f1'],dict(
                estimate=float(estimate),ci95=np.quantile(values,[.025,.975],method='linear').tolist()))


def compare_nested(actual,expected,path='root'):
    if isinstance(actual,dict):
        if set(actual)!=set(expected): raise ValueError('Keys differ at '+path)
        for k in actual: compare_nested(actual[k],expected[k],path+'.'+k)
    elif isinstance(actual,list):
        if len(actual)!=len(expected): raise ValueError('Lengths differ at '+path)
        for i,(a,b) in enumerate(zip(actual,expected)): compare_nested(a,b,path+f'[{i}]')
    elif isinstance(actual,(int,float)) and not isinstance(actual,bool):
        if not np.isclose(actual,expected,rtol=1e-10,atol=1e-12): raise ValueError('Values differ at '+path)
    elif actual!=expected: raise ValueError('Value differs at '+path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execution-lock',type=Path,required=True)
    ap.add_argument('--corpus',type=Path,required=True);ap.add_argument('--features',type=Path,required=True)
    ap.add_argument('--models',type=Path,required=True);ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists(): raise ValueError('Audit destination exists')
    begin=time.perf_counter();execution=checked_execution(args.execution_lock)
    corpus=read(args.corpus/'summary.json');check=read(args.corpus/'verification.json')
    if not check['passed'] or check['observations_checked']!=9600 or check['complete_waveform_replays']!=9600 or check['source_replays']!=480:
        raise ValueError('Complete waveform replay has not passed')
    if corpus['execution_lock_sha256']!=sha(args.execution_lock) or check['corpus_summary_sha256']!=sha(args.corpus/'summary.json'):
        raise ValueError('Corpus execution/replay mismatch')
    feature_meta=read(args.features/'summary.json');model_lock=read(args.models/'model_lock.json');summary=read(args.results/'summary.json')
    if feature_meta['features_sha256']!=sha(args.features/'features.npz'): raise ValueError('Feature file changed')
    if feature_meta['corpus_summary_sha256']!=sha(args.corpus/'summary.json') or feature_meta['corpus_verification_sha256']!=sha(args.corpus/'verification.json'):
        raise ValueError('Feature source changed')
    if model_lock['features_summary_sha256']!=sha(args.features/'summary.json') or summary['model_lock_sha256']!=sha(args.models/'model_lock.json'):
        raise ValueError('Model/feature lock mismatch')
    for folder,lock in [(args.models,model_lock),(args.results,summary)]:
        for name,digest in lock['artifacts_sha256'].items():
            if sha(folder/name)!=digest: raise ValueError('Changed output '+str(folder/name))
    cfg=read(HERE/'frozen/source_path_20261004_v1/config.json')
    plans={p['fit_id']:p for p in rows(HERE/'frozen/source_path_20261004_v1/fit_plan.jsonl')}
    observations={r['job_id']:r for r in rows(args.corpus/'observation_manifest.jsonl')}
    fits=read(args.models/'fits.json');evaluations=read(args.results/'evaluations.json')
    if len(fits)!=40 or {r['fit_id'] for r in fits}!=set(plans) or {r['fit_id'] for r in evaluations}!=set(plans):
        raise ValueError('Missing/duplicate factorial fits or evaluations')
    target=rows(HERE/'frozen/source_path_20261004_v1/target_manifest.jsonl')
    tids=[r['file_id'] for r in target];y=np.array([r['class_id'] for r in target]);groups=sorted({r['provenance_group_id'] for r in target})
    group_index=np.array([groups.index(r['provenance_group_id']) for r in target])
    target_cache=read(args.features/'target_cache_compatibility.json')
    if sha(Path(target_cache['feature_cache']))!=target_cache['feature_cache_sha256']: raise ValueError('Target cache changed')
    with np.load(target_cache['feature_cache'],allow_pickle=False) as cache:
        pos={fid:i for i,fid in enumerate(cache['file_ids'].tolist())}
        target_features={rep:np.asarray(cache[rep][[pos[i] for i in tids]],dtype=np.float64) for rep in cfg['representations']}
    with np.load(args.features/'features.npz',allow_pickle=False) as feature_file:
        ids=feature_file['job_ids'].tolist();position={jid:i for i,jid in enumerate(ids)}
        features={rep:np.asarray(feature_file[rep],dtype=np.float64) for rep in cfg['representations']}
    with np.load(args.results/'target_predictions.npz',allow_pickle=False) as predictions:
        if predictions['target_ids'].tolist()!=tids: raise ValueError('Target ID order differs')
        np.testing.assert_array_equal(y,predictions['y']);np.testing.assert_array_equal(group_index,predictions['group_index'])
        probabilities={rep:predictions[rep].copy() for rep in cfg['representations']}
    ev={r['fit_id']:r for r in evaluations};max_error=0.;audited=[]
    with threadpool_limits(limits=1):
        for ref in fits:
            plan=plans[ref['fit_id']]; rep=ref['representation'];train=ref['train_job_ids'];validation=ref['validation_job_ids']
            if train!=plan['train_job_ids'] or validation!=plan['validation_job_ids']: raise ValueError('Fit population changed')
            if {observations[i]['source_parent_group'] for i in train}&{observations[i]['source_parent_group'] for i in validation}:
                raise ValueError('Source-parent leakage')
            if any(observations[i]['role']!='train' for i in train) or any(observations[i]['role']!='validation' for i in validation):
                raise ValueError('Fit role leakage')
            model=joblib.load(args.models/ref['model_path']);parameters=model[1].get_params()
            for key,value in cfg['classifier'].items():
                if key!='type' and parameters[key]!=value: raise ValueError('Classifier parameter changed')
            if parameters['random_state']!=plan['replicate_seed'] or model.classes_.tolist()!=[0,1]: raise ValueError('Seed/class mapping changed')
            x=features[rep][[position[i] for i in train]]
            np.testing.assert_allclose(model[0].mean_,x.mean(axis=0),rtol=0,atol=1e-12)
            np.testing.assert_allclose(model[0].var_,x.var(axis=0),rtol=1e-12,atol=1e-12)
            if np.any(model[1].n_iter_>=cfg['classifier']['max_iter']): raise ValueError('Nonconverged model')
            with np.load(args.models/ref['validation_predictions'],allow_pickle=False) as val:
                if val['job_ids'].tolist()!=validation: raise ValueError('Validation IDs changed')
                vy=np.array([observations[i]['class_id'] for i in validation]);np.testing.assert_array_equal(vy,val['y'])
                vp=model.predict_proba(features[rep][[position[i] for i in validation]])[:,1]
                np.testing.assert_allclose(vp,val['p'],rtol=0,atol=1e-10);compare_nested(score(vy,vp),ref['validation'])
            ci=CELLS.index(ref['cell']);si=cfg['seeds'].index(ref['replicate_seed'])
            replay=model.predict_proba(target_features[rep])[:,1]
            saved=probabilities[rep][ci,si];max_error=max(max_error,float(abs(replay-saved).max()))
            np.testing.assert_allclose(replay,saved,rtol=0,atol=1e-10)
            compare_nested(score(y,saved),ev[ref['fit_id']]['target'])
            for gi,g in enumerate(groups):
                mask=group_index==gi;compare_nested(score(y[mask],saved[mask]),ev[ref['fit_id']]['groups'][g])
            audited.append(ref['fit_id'])
    stats,samples,indices=bootstrap(y,group_index,probabilities)
    compare_nested(stats,read(args.results/'statistics.json'))
    with np.load(args.results/'bootstrap_samples.npz',allow_pickle=False) as old:
        if set(old.files)!=set(samples): raise ValueError('Bootstrap sample keys changed')
        for key,value in samples.items(): np.testing.assert_allclose(value,old[key],rtol=0,atol=1e-12)
    with np.load(args.results/'bootstrap_indices.npz',allow_pickle=False) as old:
        for key,value in indices.items(): np.testing.assert_array_equal(value,old[key])
    independent_f1_reconstruction(y,group_index,probabilities,stats,samples,indices)
    protected=read(HERE/'frozen/source_path_20261004_v1/protected_artifacts.json')
    for record in protected:
        if sha(ROOT/record['path'])!=record['sha256']: raise ValueError('Protected artifact changed')
    report=dict(passed=True,model_count=len(audited),source_replays=480,full_waveform_replays=9600,
        train_only_scaler_and_roles_verified=True,all_fixed_hyperparameters_verified=True,
        raw_prediction_and_metric_replay_verified=True,max_prediction_replay_error=max_error,
        paired_bootstrap_draws_verified=10000,protected_files_unchanged=len(protected),
        independent_integer_confusion_f1_and_intervals_verified=True,
        new_fits=0,new_target_tuning=0,target_role='exposed_development',
        execution_lock_sha256=sha(args.execution_lock),result_summary_sha256=sha(args.results/'summary.json'),
        verifier_sha256=sha(Path(__file__)),elapsed_s=time.perf_counter()-begin)
    save(args.output,report);print(report,flush=True)


if __name__=='__main__':main()

"""Fixed synthetic fits followed by the complete native/matched comparison."""
import argparse
from datetime import datetime,timezone
import time
import warnings
import joblib
from scipy.special import expit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from phase_common import *
from phase_statistics import full_score,calculate


def features():
    summary=read(CACHE/'summary.json')
    if summary['features_sha256']!=sha(CACHE/'features.npz') or summary['corpus_verification_sha256']!=sha(CORPUS/'verification.json'):
        raise ValueError('Feature provenance changed')
    with np.load(CACHE/'features.npz',allow_pickle=False) as file:
        ids=file['job_ids'].tolist();matrix={rep:np.asarray(file[rep],float) for rep in REPS}
    if len(ids)!=len(set(ids)) or len(ids)!=19200:raise ValueError('Feature population')
    return matrix,{j:i for i,j in enumerate(ids)}


def fit():
    check_execution();data,pos=features();begin=time.perf_counter()
    MODELS.mkdir(parents=True,exist_ok=False);(MODELS/'models').mkdir();(MODELS/'validation').mkdir()
    obs={r['job_id']:r for r in rows(FREEZE/'observations.jsonl')}
    old={r['fit_id']:r for r in read(parent.MODELS/'fits.json')};records=[]
    with threadpool_limits(limits=1):
        for plan in rows(FREEZE/'fits.jsonl'):
            train=plan['train_job_ids'];val=plan['validation_job_ids'];rep=plan['representation']
            if len(train)!=380 or len(val)!=100 or set(train)&set(val):raise ValueError('Budget/ID overlap')
            if {obs[j]['source_parent_group'] for j in train}&{obs[j]['source_parent_group'] for j in val}:raise ValueError('Source leakage')
            if any(obs[j]['role']!='train' for j in train) or any(obs[j]['role']!='validation' for j in val):raise ValueError('Role leakage')
            x=data[rep][[pos[j] for j in train]];vx=data[rep][[pos[j] for j in val]]
            y=np.array([obs[j]['class_id'] for j in train]);vy=np.array([obs[j]['class_id'] for j in val])
            if np.bincount(y).tolist()!=[190,190] or np.bincount(vy).tolist()!=[50,50]:raise ValueError('Class balance')
            if plan['reused']:
                ref=old[plan['parent_fit_id']];path=ROOT/ref['model_path']
                if sha(path)!=ref['model_sha256']:raise ValueError('Parent model changed')
                model=joblib.load(path)
            else:
                params={k:v for k,v in BASE['classifier'].items() if k!='type'}
                model=make_pipeline(StandardScaler(),LogisticRegression(**params,random_state=plan['replicate_seed']))
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning);model.fit(x,y)
                path=MODELS/'models'/(plan['fit_id']+'.joblib');joblib.dump(model,path,compress=3)
            np.testing.assert_allclose(model[0].mean_,x.mean(axis=0),rtol=0,atol=1e-12)
            np.testing.assert_allclose(model[0].var_,x.var(axis=0),rtol=1e-12,atol=1e-12)
            if model.classes_.tolist()!=[0,1] or np.any(model[1].n_iter_>=5000):raise ValueError('Classes/convergence')
            p=model.predict_proba(vx)[:,1]
            if plan['reused']:
                with np.load(ROOT/ref['validation_path'],allow_pickle=False) as previous:
                    np.testing.assert_array_equal(previous['y'],vy);np.testing.assert_allclose(previous['p'],p,rtol=0,atol=1e-12)
            vpath=MODELS/'validation'/(plan['fit_id']+'.npz');np.savez_compressed(vpath,job_ids=np.array(val),y=vy,p=p)
            records.append(dict(plan,model_path=str(path.relative_to(ROOT)),model_sha256=sha(path),
                n_iter=model[1].n_iter_.tolist(),validation=full_score(vy,p),
                validation_path=str(vpath.relative_to(ROOT)),validation_sha256=sha(vpath)))
            print('fixed attenuation/latency head',len(records),'/80',flush=True)
    save(MODELS/'fits.json',records)
    save(MODELS/'lock.json',dict(models=80,new_models=40,reused_models=40,target_access=False,
        completed_utc=datetime.now(timezone.utc).isoformat(),execution_lock_sha256=sha(EXECUTION/'lock.json'),
        features_summary_sha256=sha(CACHE/'summary.json'),elapsed_s=time.perf_counter()-begin,
        artifacts_sha256={str(p.relative_to(MODELS)):sha(p) for p in MODELS.rglob('*') if p.is_file()}))


def checked_models():
    lock=read(MODELS/'lock.json')
    if lock['models']!=80 or lock['features_summary_sha256']!=sha(CACHE/'summary.json'):raise ValueError('Model lock')
    for name,digest in lock['artifacts_sha256'].items():
        if sha(MODELS/name)!=digest:raise ValueError('Model artifacts changed')
    refs=read(MODELS/'fits.json')
    for row in refs:
        if sha(ROOT/row['model_path'])!=row['model_sha256']:raise ValueError('Model changed')
    return refs


def target_data():
    # Read only after the caller has checked the complete new model lock.
    import fit_evaluate as previous
    result=previous.target_data()
    if result[0]!=[r['file_id'] for r in rows(FREEZE/'target_manifest.jsonl')]:raise ValueError('Target population changed')
    return result


def evaluate():
    check_execution();refs=checked_models();data,pos=features()
    tids,y,groups,g,targets=target_data()
    RESULTS.mkdir(parents=True,exist_ok=False);begin=time.perf_counter()
    obs={r['job_id']:r for r in rows(FREEZE/'observations.jsonl')}
    probabilities={v:{rep:np.empty((2,4,5,len(y))) for rep in REPS} for v in targets}
    records=[];cross=[];cross_p=np.empty((80,4,100));val_y=np.empty((80,100),np.int64)
    with np.load(parent.RESULTS/'target_predictions.npz',allow_pickle=False) as old:
        previous={v:{rep:old[v+'__'+rep].copy() for rep in REPS} for v in targets}
    with threadpool_limits(limits=1):
        for fi,ref in enumerate(refs):
            model=joblib.load(ROOT/ref['model_path']);rep=ref['representation']
            si,ci,bi=SOURCES.index(ref['source_level']),CELLS.index(ref['cell']),SEEDS.index(ref['replicate_seed'])
            val=ref['validation_job_ids'];vy=np.array([obs[j]['class_id'] for j in val]);val_y[fi]=vy
            for variant,matrix in targets.items():
                x=matrix[rep];p=model.predict_proba(x)[:,1]
                np.testing.assert_allclose(p,expit(model[0].transform(x)@model[1].coef_[0]+model[1].intercept_[0]),rtol=0,atol=1e-12)
                probabilities[variant][rep][si,ci,bi]=p
                scores={group:full_score(y[g==i],p[g==i]) for i,group in enumerate(groups)}
                records.append(dict(fit_id=ref['fit_id'],representation=rep,source_level=ref['source_level'],
                    cell=ref['cell'],seed=ref['replicate_seed'],variant=variant,target=full_score(y,p),groups=scores,
                    worst_group_class_recall=min(r[k+'_recall'] for r in scores.values() for k in ['car','truck'])))
                if ref['reused']:
                    pj=parent.PATHS.index(CONFIG['parent_cell_mapping'][ref['cell']])
                    np.testing.assert_allclose(p,previous[variant][rep][si,pj,bi],rtol=0,atol=1e-12)
            for cj,cell in enumerate(CELLS):
                jobs=[j.rsplit('.',1)[0]+'.'+cell for j in val]
                if any(obs[j]['role']!='validation' or obs[j]['source_level']!=ref['source_level'] for j in jobs):raise ValueError('Cross-cell role')
                np.testing.assert_array_equal(vy,[obs[j]['class_id'] for j in jobs])
                p=model.predict_proba(data[rep][[pos[j] for j in jobs]])[:,1];cross_p[fi,cj]=p
                cross.append(dict(fit_id=ref['fit_id'],representation=rep,source_level=ref['source_level'],seed=ref['replicate_seed'],
                    train_cell=ref['cell'],test_cell=cell,metrics=full_score(vy,p)))
            print('evaluated attenuation/latency head',fi+1,'/80',flush=True)
    np.savez_compressed(RESULTS/'target_predictions.npz',target_ids=np.array(tids),y=y,group_index=g,
        **{v+'__'+rep:p for v,reps in probabilities.items() for rep,p in reps.items()})
    np.savez_compressed(RESULTS/'cross_cell_predictions.npz',fit_ids=np.array([r['fit_id'] for r in refs]),cells=np.array(CELLS),y=val_y,p=cross_p)
    save(RESULTS/'evaluations.json',records);save(RESULTS/'cross_cell_validation.json',cross)
    stats,samples,indices=calculate(y,g,probabilities)
    save(RESULTS/'statistics.json',stats);np.savez_compressed(RESULTS/'bootstrap_samples.npz',**samples)
    np.savez_compressed(RESULTS/'bootstrap_indices.npz',**indices)
    save(RESULTS/'summary.json',dict(passed=True,protocol_id=ID,models=80,new_models=40,target_events=len(y),
        reused_target_vectors_preserved=80,model_lock_sha256=sha(MODELS/'lock.json'),
        parent_target_gain_summary_sha256=sha(parent.CACHE/'target_gain/summary.json'),
        target_groups={group:np.bincount(y[g==i],minlength=2).tolist() for i,group in enumerate(groups)},
        target_role='previously_exposed_development',selection_or_tuning=False,elapsed_s=time.perf_counter()-begin,
        completed_utc=datetime.now(timezone.utc).isoformat(),
        constant_controls={name:full_score(y,np.full(len(y),p)) for name,p in [('always_car',0.),('always_truck',1.)]},
        artifacts_sha256={p.name:sha(p) for p in RESULTS.iterdir() if p.is_file()}))
    print(stats['decision'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['fit','evaluate']);a=ap.parse_args()
    {'fit':fit,'evaluate':evaluate}[a.stage]()

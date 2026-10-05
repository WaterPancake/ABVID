"""Fit only new synthetic heads, then evaluate the complete declared comparison."""
import argparse
from datetime import datetime,timezone
import time
import warnings
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from scipy.special import expit
from threadpoolctl import threadpool_limits
from common import *
from effect_statistics import full_score,calculate


def feature_data():
    summary=read(CACHE/'summary.json')
    for name,digest in summary['artifacts_sha256'].items():
        if sha(CACHE/name)!=digest:raise ValueError('Synthetic feature artifact changed')
    data=np.load(CACHE/'features.npz',allow_pickle=False)
    ids=data['job_ids'].tolist()
    if len(ids)!=19200 or len(set(ids))!=19200:raise ValueError('Feature population')
    return data,{j:i for i,j in enumerate(ids)}


def fit():
    check_execution();data,pos=feature_data();begin=time.perf_counter()
    MODELS.mkdir(parents=True,exist_ok=False);(MODELS/'models').mkdir();(MODELS/'validation').mkdir()
    obs={r['job_id']:r for r in rows(FREEZE/'observations.jsonl')}
    old={r['fit_id']:r for r in read(ROOT/CONFIG['parent_models']/'fits.json')}
    records=[]
    with threadpool_limits(limits=1):
        for plan in rows(FREEZE/'fits.jsonl'):
            ids=plan['train_job_ids'];vids=plan['validation_job_ids'];rep=plan['representation']
            if len(ids)!=380 or len(vids)!=100 or set(ids)&set(vids):raise ValueError('Fit budget/role mismatch')
            if {obs[i]['source_parent_group'] for i in ids}&{obs[i]['source_parent_group'] for i in vids}:raise ValueError('Source leakage')
            x=np.asarray(data[rep][[pos[i] for i in ids]],dtype=np.float64)
            y=np.array([obs[i]['class_id'] for i in ids]);vy=np.array([obs[i]['class_id'] for i in vids])
            if np.bincount(y).tolist()!=[190,190] or np.bincount(vy).tolist()!=[50,50]:raise ValueError('Class mapping/balance')
            if plan['reused']:
                reference=old[plan['parent_fit_id']];path=ROOT/CONFIG['parent_models']/reference['model_path']
                if sha(path)!=reference['model_sha256']:raise ValueError('Parent model changed')
                model=joblib.load(path)
            else:
                params={k:v for k,v in BASE['classifier'].items() if k!='type'}
                model=make_pipeline(StandardScaler(),LogisticRegression(**params,random_state=plan['replicate_seed']))
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning);model.fit(x,y)
                path=MODELS/'models'/(plan['fit_id']+'.joblib');joblib.dump(model,path,compress=3)
            np.testing.assert_allclose(model[0].mean_,x.mean(axis=0),rtol=0,atol=1e-12)
            np.testing.assert_allclose(model[0].var_,x.var(axis=0),rtol=1e-12,atol=1e-12)
            if model.classes_.tolist()!=[0,1] or np.any(model[1].n_iter_>=5000):raise ValueError('Model classes/convergence')
            vx=np.asarray(data[rep][[pos[i] for i in vids]],float);p=model.predict_proba(vx)[:,1]
            if plan['reused']:
                with np.load(ROOT/CONFIG['parent_models']/reference['validation_predictions'],allow_pickle=False) as ref:
                    np.testing.assert_array_equal(vy,ref['y']);np.testing.assert_allclose(p,ref['p'],rtol=0,atol=1e-12)
            vpath=MODELS/'validation'/(plan['fit_id']+'.npz');np.savez_compressed(vpath,job_ids=np.array(vids),y=vy,p=p)
            records.append(dict(plan,model_path=str(path.relative_to(ROOT)),model_sha256=sha(path),
                n_iter=model[1].n_iter_.tolist(),validation=full_score(vy,p),
                validation_path=str(vpath.relative_to(ROOT)),validation_sha256=sha(vpath)))
            print('fixed mechanism head',len(records),'/80',flush=True)
    save(MODELS/'fits.json',records)
    save(MODELS/'lock.json',dict(models=80,new_models=40,reused_models=40,
        completed_utc=datetime.now(timezone.utc).isoformat(),target_access=False,
        execution_lock_sha256=sha(EXECUTION/'lock.json'),features_summary_sha256=sha(CACHE/'summary.json'),
        fits_sha256=sha(MODELS/'fits.json'),elapsed_s=time.perf_counter()-begin,
        artifacts_sha256={str(p.relative_to(MODELS)):sha(p) for p in MODELS.rglob('*') if p.is_file()}))


def checked_models():
    lock=read(MODELS/'lock.json')
    if lock['models']!=80 or lock['features_summary_sha256']!=sha(CACHE/'summary.json'):raise ValueError('Model lock invalid')
    for name,digest in lock['artifacts_sha256'].items():
        if sha(MODELS/name)!=digest:raise ValueError('Model artifact changed')
    records=read(MODELS/'fits.json')
    for record in records:
        if sha(ROOT/record['model_path'])!=record['model_sha256']:raise ValueError('Referenced model changed')
    return records


def target_data():
    from learning import h1_cache_compatibility
    compatibility=h1_cache_compatibility();target=rows(FREEZE/'target_manifest.jsonl')
    ids=[r['file_id'] for r in target];y=np.array([r['class_id'] for r in target]);groups=sorted({r['provenance_group_id'] for r in target})
    g=np.array([groups.index(r['provenance_group_id']) for r in target])
    matched_summary=read(CACHE/'target_gain/summary.json')
    if matched_summary['model_lock_sha256']!=sha(MODELS/'lock.json') or matched_summary['features_sha256']!=sha(CACHE/'target_gain/features.npz'):
        raise ValueError('Target diagnostic lock changed')
    with np.load(compatibility['feature_cache'],allow_pickle=False) as old:
        pos={fid:i for i,fid in enumerate(old['file_ids'].tolist())}
        native={rep:np.asarray(old[rep][[pos[i] for i in ids]],float) for rep in REPS}
    with np.load(CACHE/'target_gain/features.npz',allow_pickle=False) as matched:
        if matched['file_ids'].tolist()!=ids:raise ValueError('Target population mismatch')
        normalized={rep:np.asarray(matched[rep],float) for rep in REPS}
    return ids,y,groups,g,dict(native=native,matched=normalized)


def domain_diagnostic(model,x,y,train_x,train_y):
    z=model[0].transform(x);zt=model[0].transform(train_x)
    coefficient=model[1].coef_[0];intercept=float(model[1].intercept_[0])
    logits=model.decision_function(x);reconstruct=z@coefficient+intercept
    np.testing.assert_allclose(logits,reconstruct,rtol=0,atol=1e-10)
    np.testing.assert_allclose(expit(logits),model.predict_proba(x)[:,1],rtol=0,atol=1e-12)
    norm=np.sqrt(np.mean(z*z,axis=1));train_norm=np.sqrt(np.mean(zt*zt,axis=1))
    limit=float(np.quantile(train_norm,.95));lo=zt.min(axis=0);hi=zt.max(axis=0)
    report=dict(intercept=intercept,training_norm_p95=limit,classes={})
    for label,name in [(0,'car'),(1,'truck')]:
        mask=y==label;reference=zt[train_y==label]
        shift=(z[mask].mean(axis=0)-reference.mean(axis=0))*coefficient
        expected=float(logits[mask].mean()-model.decision_function(train_x[train_y==label]).mean())
        if abs(float(shift.sum())-expected)>1e-9:raise ValueError('Score-shift decomposition failed')
        report['classes'][name]=dict(count=int(mask.sum()),logit=percentile_summary(logits[mask]),
            standardized_feature_rms_norm=percentile_summary(norm[mask]),
            above_training_norm_p95_fraction=float(np.mean(norm[mask]>limit)),
            mean_fraction_coordinates_outside_training_range=float(np.mean((z[mask]<lo)|(z[mask]>hi))),
            predicted_truck_fraction=float(np.mean(logits[mask]>0)),
            mean_logit_shift_from_same_training_class=expected,
            mean_logit_shift_contributions=shift.tolist(),
            top_positive_coordinates=np.argsort(shift)[-10:][::-1].tolist(),
            top_negative_coordinates=np.argsort(shift)[:10].tolist())
    return report


def evaluate():
    check_execution();refs=checked_models();data,pos=feature_data();ids,y,groups,g,targets=target_data()
    RESULTS.mkdir(parents=True,exist_ok=False);begin=time.perf_counter()
    obs={r['job_id']:r for r in rows(FREEZE/'observations.jsonl')}
    probabilities={variant:{rep:np.empty((2,4,5,len(y))) for rep in REPS} for variant in targets}
    records=[];diagnostics=[];cross=[];gain_records=[]
    cross_p=np.empty((80,4,100));gain_p=np.empty((80,3,100));validation_y=np.empty((80,100),np.int64)
    with np.load(CACHE/'validation_gain.npz',allow_pickle=False) as gain:
        gain_ids={j:i for i,j in enumerate(gain['job_ids'].tolist())};stress={rep:gain[rep].copy() for rep in REPS}
    with np.load(ROOT/CONFIG['parent_results']/'target_predictions.npz',allow_pickle=False) as old:
        parent_p={rep:old[rep].copy() for rep in REPS}
    with threadpool_limits(limits=1):
        for fi,ref in enumerate(refs):
            model=joblib.load(ROOT/ref['model_path']);rep=ref['representation']
            si=SOURCES.index(ref['source_level']);pi=PATHS.index(ref['path_level']);bi=SEEDS.index(ref['replicate_seed'])
            train_ids=ref['train_job_ids'];vids=ref['validation_job_ids']
            tx=np.asarray(data[rep][[pos[i] for i in train_ids]],float);ty=np.array([obs[i]['class_id'] for i in train_ids])
            vx=np.asarray(data[rep][[pos[i] for i in vids]],float);vy=np.array([obs[i]['class_id'] for i in vids])
            validation_y[fi]=vy
            diag=dict(fit_id=ref['fit_id'],representation=rep,source_level=ref['source_level'],path_level=ref['path_level'],
                seed=ref['replicate_seed'],train=domain_diagnostic(model,tx,ty,tx,ty),
                validation=domain_diagnostic(model,vx,vy,tx,ty),targets={})
            for variant,matrix in targets.items():
                p=model.predict_proba(matrix[rep])[:,1];probabilities[variant][rep][si,pi,bi]=p
                scores={group:full_score(y[g==j],p[g==j]) for j,group in enumerate(groups)}
                records.append(dict(fit_id=ref['fit_id'],representation=rep,source_level=ref['source_level'],
                    path_level=ref['path_level'],seed=ref['replicate_seed'],variant=variant,target=full_score(y,p),groups=scores,
                    worst_group_class_recall=min(r[name+'_recall'] for r in scores.values() for name in ['car','truck']),
                    group_average_car_recall=float(np.mean([r['car_recall'] for r in scores.values()])),
                    group_average_truck_recall=float(np.mean([r['truck_recall'] for r in scores.values()]))))
                diag['targets'][variant]=domain_diagnostic(model,matrix[rep],y,tx,ty)
                if ref['reused'] and variant=='native':
                    old_cell=['F00','F01','F10','F11'].index(ref['parent_fit_id'].split('.')[-2])
                    np.testing.assert_allclose(p,parent_p[rep][old_cell,bi],rtol=0,atol=1e-12)
            diagnostics.append(diag)
            for pj,path in enumerate(PATHS):
                cross_ids=[i.rsplit('.',1)[0]+'.'+path for i in vids]
                if any(obs[i]['role']!='validation' or obs[i]['source_level']!=ref['source_level'] for i in cross_ids):raise ValueError('Cross-path validation role changed')
                p=model.predict_proba(np.asarray(data[rep][[pos[i] for i in cross_ids]],float))[:,1]
                cross_p[fi,pj]=p
                cross.append(dict(fit_id=ref['fit_id'],representation=rep,source_level=ref['source_level'],
                    train_path=ref['path_level'],test_path=path,seed=ref['replicate_seed'],metrics=full_score(vy,p)))
            gain_x=[np.asarray(stress[rep][0,[gain_ids[j] for j in vids]],float),vx,
                np.asarray(stress[rep][1,[gain_ids[j] for j in vids]],float)]
            for gj,(db,x) in enumerate(zip([-12.,0.,12.],gain_x)):
                p=model.predict_proba(x)[:,1];gain_p[fi,gj]=p
                gain_records.append(dict(fit_id=ref['fit_id'],representation=rep,source_level=ref['source_level'],
                    path_level=ref['path_level'],seed=ref['replicate_seed'],gain_db=db,metrics=full_score(vy,p),
                    logit_by_class={name:percentile_summary(model.decision_function(x[vy==k])) for k,name in enumerate(['car','truck'])}))
            print('evaluated fixed mechanism head',fi+1,'/80',flush=True)
    packed={variant+'__'+rep:p for variant,reps in probabilities.items() for rep,p in reps.items()}
    np.savez_compressed(RESULTS/'target_predictions.npz',target_ids=np.array(ids),y=y,group_index=g,**packed)
    np.savez_compressed(RESULTS/'synthetic_diagnostic_predictions.npz',fit_ids=np.array([r['fit_id'] for r in refs]),
        paths=np.array(PATHS),gains_db=np.array([-12.,0.,12.]),y=validation_y,cross_path=cross_p,gain=gain_p)
    save(RESULTS/'evaluations.json',records);save(RESULTS/'collapse_diagnostics.json',diagnostics)
    save(RESULTS/'cross_path_validation.json',cross);save(RESULTS/'validation_gain.json',gain_records)
    stats,samples,indices=calculate(y,g,probabilities)
    save(RESULTS/'statistics.json',stats)
    np.savez_compressed(RESULTS/'bootstrap_samples.npz',**samples);np.savez_compressed(RESULTS/'bootstrap_indices.npz',**indices)
    save(RESULTS/'summary.json',dict(passed=True,protocol_id=ID,models=80,new_models=40,target_events=len(ids),
        endpoint_predictions_reproduced=40,target_groups={group:np.bincount(y[g==j],minlength=2).tolist() for j,group in enumerate(groups)},
        model_lock_sha256=sha(MODELS/'lock.json'),target_gain_feature_summary_sha256=sha(CACHE/'target_gain/summary.json'),
        target_role='previously_exposed_development',selection_or_tuning=False,
        constant_controls={name:full_score(y,np.full(len(y),p)) for name,p in [('always_car',0.),('always_truck',1.)]},
        elapsed_s=time.perf_counter()-begin,completed_utc=datetime.now(timezone.utc).isoformat(),
        artifacts_sha256={p.name:sha(p) for p in RESULTS.iterdir() if p.is_file()}))
    print(stats['decision'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['fit','evaluate']);a=ap.parse_args()
    {'fit':fit,'evaluate':evaluate}[a.stage]()

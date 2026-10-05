"""Predict with frozen source-only models; no fit/selection API is used here."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from common import ROOT,HERE,sha,save,read_json,load_lock,score,cm_scores,protected_check,assert_separation


def feature_view(lock,cfg,index):
    path=ROOT/cfg['feature_cache']/'features.npz'
    assert sha(path)==lock['feature_cache_sha256']
    with np.load(path,allow_pickle=False) as data:
        ids=data['file_ids'].tolist(); assert ids==lock['feature_ids']
        features={r:np.asarray(data[r],dtype=np.float64) for r in cfg['representations']}
    for x in features.values(): assert len(x)==len(ids) and np.isfinite(x).all()
    return features,{i:p for p,i in enumerate(ids)}


def interval(point,samples):
    return dict(estimate=float(point),ci95=np.quantile(samples,[.025,.975]).tolist())


def mean_worst(cm):
    scores=cm_scores(cm)
    return np.minimum(scores['car_recall'],scores['truck_recall']).min(axis=-1).mean()


def bootstrap(source,target,cfg):
    """source[rep,arm]: [seed,fold,2,2]; target: [seed,fold,group,2,2]."""
    reps=cfg['representations']; arms=cfg['arms']; ns,nf,ng=target[reps[0],arms[0]].shape[:3]
    rng=np.random.default_rng(cfg['uncertainty']['seed']); count=cfg['uncertainty']['draws']
    indices={k:np.empty((count,n),dtype=np.int64) for k,n in [('selection',ns),('fold',nf),('target_group',ng)]}
    keys=list(cm_scores(source[reps[0],arms[0]]))
    raw={};points={}
    for rep in reps:
        for arm in arms:
            for domain in ['source','target']:
                cm=source[rep,arm] if domain=='source' else target[rep,arm].sum(axis=2)
                for k,v in cm_scores(cm).items():
                    name=f'{rep}__{arm}__{domain}__{k}'; raw[name]=np.empty(count);points[name]=float(v.mean())
            name=f'{rep}__{arm}__target__mean_model_worst_group_class_recall'
            raw[name]=np.empty(count);points[name]=float(mean_worst(target[rep,arm]))
    rejected=0
    for b in range(count):
        rs=rng.integers(ns,size=ns);rf=rng.integers(nf,size=nf)
        while True:
            rg=rng.integers(ng,size=ng)
            if (target[reps[0],arms[0]][0,0,rg].sum(axis=(0,2))>0).all():break
            rejected+=1
        indices['selection'][b]=rs;indices['fold'][b]=rf;indices['target_group'][b]=rg
        for rep in reps:
            for arm in arms:
                sc=source[rep,arm][rs[:,None],rf[None,:]]
                tc=target[rep,arm][rs[:,None],rf[None,:]][:,:,rg]
                for domain,cm in [('source',sc),('target',tc.sum(axis=2))]:
                    for k,v in cm_scores(cm).items():raw[f'{rep}__{arm}__{domain}__{k}'][b]=float(v.mean())
                raw[f'{rep}__{arm}__target__mean_model_worst_group_class_recall'][b]=mean_worst(tc)
    summaries={};contrasts={};gaps={}
    for rep in reps:
        summaries[rep]={}; contrasts[rep]={};gaps[rep]={}
        for arm in arms:
            summaries[rep][arm]={}
            for domain in ['source','target']:
                metrics=keys+(['mean_model_worst_group_class_recall'] if domain=='target' else [])
                summaries[rep][arm][domain]={k:interval(points[f'{rep}__{arm}__{domain}__{k}'],raw[f'{rep}__{arm}__{domain}__{k}']) for k in metrics}
            sn=f'{rep}__{arm}__source__macro_f1';tn=f'{rep}__{arm}__target__macro_f1'
            gaps[rep][arm]=interval(points[sn]-points[tn],raw[sn]-raw[tn])
        for domain in ['source','target']:
            metrics=keys+(['mean_model_worst_group_class_recall'] if domain=='target' else [])
            contrasts[rep][domain]={}
            for k in metrics:
                a=f'{rep}__nested_C__{domain}__{k}';b=f'{rep}__baseline__{domain}__{k}'
                contrasts[rep][domain][k]=interval(points[a]-points[b],raw[a]-raw[b])
    return dict(estimates=summaries,paired_gains=contrasts,source_minus_target_f1=gaps,class_missing_draws_rejected=rejected),raw,indices


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lock',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    begin=time.perf_counter();lock,cfg,index=load_lock(args.lock)
    refs=read_json(args.lock.parent/'model_references.json')
    features,pos=feature_view(lock,cfg,index)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'predictions').mkdir()
    for p in [Path(__file__),HERE/'common.py']:shutil.copy2(p,out/('executed_'+p.name))
    tids=lock['target_ids'];tp=np.array([pos[i] for i in tids]);ty=np.array([index[i]['class_id'] for i in tids])
    groups=lock['target_groups']; masks={g:np.array([index[i]['provenance_group_id']==g for i in tids]) for g in groups}
    source={(r,a):np.zeros((5,6,2,2),dtype=np.int64) for r in cfg['representations'] for a in cfg['arms']}
    target={(r,a):np.zeros((5,6,len(groups),2,2),dtype=np.int64) for r in cfg['representations'] for a in cfg['arms']}
    records=[];source_error=0.;h1_error=0.
    with threadpool_limits(limits=1):
        for ref in refs:
            rep=ref['representation'];arm=ref['arm'];seed=ref['seed'];fold=ref['fold'];si=cfg['seeds'].index(seed)
            assert sha(ROOT/ref['model_path'])==ref['model_sha256']; model=joblib.load(ROOT/ref['model_path'])
            train=ref['train_ids'];source_ids=ref['source_test_ids']
            assert_separation(train,tids,index);assert_separation(train,source_ids,index)
            ip=np.array([pos[i] for i in train]);trainx=features[rep][ip]
            np.testing.assert_allclose(model[0].mean_,trainx.mean(axis=0),rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(model[0].var_,trainx.var(axis=0),rtol=1e-10,atol=1e-10)
            assert model[1].C==ref['C'] and np.array_equal(model.classes_,[0,1])
            sp=np.array([pos[i] for i in source_ids]);sy=np.array([index[i]['class_id'] for i in source_ids])
            srcp=model.predict_proba(features[rep][sp])[:,1]
            # Source replay is checked before this head's new target prediction.
            with np.load(ROOT/ref['source_predictions_path']) as z:
                assert z['ids'].tolist()==source_ids;np.testing.assert_array_equal(z['y'],sy)
                err=float(np.max(np.abs(srcp-z['p'])));source_error=max(source_error,err)
                np.testing.assert_allclose(srcp,z['p'],rtol=0,atol=1e-10)
            pred=model.predict_proba(features[rep][tp])[:,1]
            if arm=='baseline':
                old=ROOT/cfg['h1_results']/'predictions'/f'{rep}__real__seed{seed}__fold{fold}.npz'
                with np.load(old) as z:
                    assert z['target_ids'].tolist()==tids;np.testing.assert_array_equal(z['target_true'],ty)
                    oldp=z['target_probabilities'][:,1]
                err=float(np.max(np.abs(pred-oldp)));h1_error=max(h1_error,err)
                np.testing.assert_allclose(pred,oldp,rtol=0,atol=1e-10)
            name=f'{rep}__{arm}__seed{seed}__fold{fold}';path=out/'predictions'/(name+'.npz')
            np.savez_compressed(path,source_ids=np.array(source_ids),source_true=sy,source_p=srcp,
                                target_ids=np.array(tids),target_true=ty,target_p=pred)
            sr=score(sy,srcp);tr=score(ty,pred);gr={g:score(ty[masks[g]],pred[masks[g]]) for g in groups}
            source[rep,arm][si,fold]=sr['confusion_matrix']
            for gi,g in enumerate(groups): target[rep,arm][si,fold,gi]=gr[g]['confusion_matrix']
            records.append(dict(**ref,source=sr,target=tr,target_groups=gr,threshold=.5,
                worst_target_group_class_recall=min(gr[g][c+'_recall'] for g in groups for c in ['car','truck']),
                predictions_path=str(path.relative_to(ROOT)),predictions_sha256=sha(path)))
    prediction_s=time.perf_counter()-begin;save(out/'evaluations.json',records)
    print(f'{len(records)} evaluations complete; baseline target replay max error {h1_error:g}; {prediction_s:.2f}s',flush=True)
    stats,raw,indices=bootstrap(source,target,cfg)
    np.savez_compressed(out/'confusions.npz',**{f'{r}__{a}__source':v for (r,a),v in source.items()},**{f'{r}__{a}__target':v for (r,a),v in target.items()})
    np.savez_compressed(out/'bootstrap_samples.npz',**raw);np.savez_compressed(out/'bootstrap_indices.npz',**indices)
    descriptive={}
    for rep in cfg['representations']:
        descriptive[rep]={}
        for arm in cfg['arms']:
            rr=[r for r in records if r['representation']==rep and r['arm']==arm]
            desc={d:{m:float(np.mean([r[d][m] for r in rr])) for m in ['roc_auc','average_precision','brier_truck','log_loss','ece10_truck','truck_prevalence']} for d in ['source','target']}
            desc['absolute_minimum_target_group_class_recall']=min(r['worst_target_group_class_recall'] for r in rr)
            desc['groups']={g:{m:dict(mean=float(np.mean([r['target_groups'][g][m] for r in rr])),
                                            range=[float(min(r['target_groups'][g][m] for r in rr)),float(max(r['target_groups'][g][m] for r in rr))])
                                  for m in list(cm_scores(target[rep,arm]))+['roc_auc','average_precision','brier_truck','log_loss','ece10_truck']}
                           for g in groups}
            descriptive[rep][arm]=desc
    protected=protected_check(args.lock.parent/'protected_artifacts.json')
    save(out/'summary.json',dict(protocol_id=cfg['protocol_id'],domain=cfg['domain'],target_role=cfg['target_role'],
        completed_utc=datetime.now(timezone.utc).isoformat(),lock_sha256=sha(args.lock),unique_models=107,evaluation_records=120,
        new_fits=0,new_encoder_extractions=0,target_files=len(tids),target_class_counts=np.bincount(ty).tolist(),
        source_replay_max_probability_error=source_error,h1_target_replay_max_probability_error=h1_error,
        protected_files_unchanged=protected,prediction_and_checks_s=prediction_s,total_s=time.perf_counter()-begin,
        descriptive=descriptive,constant_controls={name:score(ty,np.full(len(ty),p)) for name,p in [('always_car',0.),('always_truck',1.)]},
        **stats,artifacts_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print(f'Complete in {time.perf_counter()-begin:.2f}s',flush=True)


if __name__=='__main__':main()

"""Fixed H1 heads and paired, conditional group-resampling inference.

Target features are read only for prediction after each training-only fit.
No hyperparameter search or best-run selection is implemented.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from freeze import ROOT, sha, save, assert_separation


def cm_for(y, pred):
    return np.bincount(np.asarray(y,dtype=int)*2+np.asarray(pred,dtype=int),minlength=4).reshape(2,2)


def f1(cm):
    cm=np.asarray(cm,dtype=float)
    tp=np.diagonal(cm,axis1=-2,axis2=-1)
    denominator=cm.sum(axis=-1)+cm.sum(axis=-2)
    return np.divide(2*tp,denominator,out=np.zeros_like(tp),where=denominator!=0).mean(axis=-1)


def cm_metrics(cm):
    cm=np.asarray(cm,dtype=float)
    support=cm.sum(axis=1); predicted=cm.sum(axis=0); tp=cm.diagonal()
    recall=np.divide(tp,support,out=np.zeros(2),where=support!=0)
    precision=np.divide(tp,predicted,out=np.zeros(2),where=predicted!=0)
    fs=np.divide(2*tp,support+predicted,out=np.zeros(2),where=(support+predicted)!=0)
    return dict(macro_f1=float(f1(cm)),accuracy=float(tp.sum()/cm.sum()),
                balanced_accuracy=float(recall.mean()) if (support>0).all() else None,
                confusion_matrix=cm.astype(int).tolist(),
                per_class={name:dict(precision=float(precision[c]),recall=float(recall[c]) if support[c]>0 else None,
                                     f1=float(fs[c]),support=int(support[c])) for c,name in enumerate(['car','truck'])})


def metrics(y,p):
    result=cm_metrics(cm_for(y,p.argmax(axis=1)))
    result.update(brier_truck=float(np.mean((p[:,1]-y)**2)),
                  log_loss=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1)).mean()))
    return result


def fixed_fit(features, labels, config):
    if sorted(np.unique(labels).tolist()) != [0,1]: raise ValueError('Both training classes required')
    clf=LogisticRegression(C=config['C'],solver=config['solver'],tol=config['tol'],max_iter=config['max_iter'],
                           fit_intercept=config['fit_intercept'],class_weight=config['class_weight'],penalty='l2')
    model=make_pipeline(StandardScaler(),clf)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='scipy.optimize: The .*',category=DeprecationWarning)
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(features,labels)
    assert np.array_equal(model.classes_,[0,1])
    np.testing.assert_allclose(model[0].mean_,features.mean(axis=0),rtol=1e-6,atol=1e-7)
    return model


def bootstrap(source,targets,config):
    """Arrays: source [seed,fold,2,2], targets[arm] [seed,fold,group,2,2]."""
    seeds,folds=source.shape[:2]
    groups=targets['real'].shape[2]
    rng=np.random.default_rng(config['seed'])
    source_point=float(f1(source).mean())
    points={'source':source_point,**{arm:float(f1(cms.sum(axis=2)).mean()) for arm,cms in targets.items()}}
    names=list(points)
    samples={k:np.empty(config['draws']) for k in names}
    rejected=0
    for b in range(config['draws']):
        rs=rng.integers(seeds,size=seeds); rf=rng.integers(folds,size=folds)
        while True:
            rg=rng.integers(groups,size=groups)
            support=targets['real'][0,0,rg].sum(axis=(0,2))
            if (support>0).all(): break
            rejected+=1
        samples['source'][b]=float(f1(source[rs[:,None],rf[None,:]]).mean())
        for arm,cms in targets.items():
            if cms.shape[1]==1:
                # Synthetic fits are replicated only across the five selections.
                sampled=cms[rs,0][:,rg].sum(axis=1)
            else:
                sampled=cms[rs[:,None],rf[None,:]][:,:,rg].sum(axis=2)
            samples[arm][b]=float(f1(sampled).mean())
    contrasts={'delta_domain':('source','real'),
               'delta_sim2real_procedural':('real','procedural'),
               'delta_sim2real_audioldm':('real','audioldm'),
               'gain_real_plus_procedural':('real_plus_procedural','real'),
               'gain_real_plus_audioldm':('real_plus_audioldm','real'),
               'mixed_procedural_minus_repeated_real':('real_plus_procedural','real_repeated'),
               'mixed_audioldm_minus_repeated_real':('real_plus_audioldm','real_repeated')}
    estimates={k:dict(estimate=points[k],ci95=np.quantile(samples[k],[.025,.975]).tolist()) for k in names}
    for key,(a,b) in contrasts.items():
        estimates[key]=dict(estimate=points[a]-points[b],ci95=np.quantile(samples[a]-samples[b],[.025,.975]).tolist())
    return estimates,samples,rejected


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--lock',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    started=time.perf_counter(); folder=args.lock.parent
    lock=json.loads(args.lock.read_text())
    for name,h in lock['artifacts_sha256'].items():
        if sha(folder/name)!=h: raise ValueError(f'Frozen input changed: {name}')
    config=json.loads((folder/'config.json').read_text())
    cache_meta=json.loads((args.cache/'summary.json').read_text())
    assert cache_meta['lock_sha256']==sha(args.lock)
    for name,h in cache_meta['artifacts_sha256'].items(): assert sha(args.cache/name)==h
    data=np.load(args.cache/'features.npz',allow_pickle=False)
    feature_ids=data['file_ids'].tolist();assert feature_ids==lock['feature_ids']
    position={fid:i for i,fid in enumerate(feature_ids)}
    rows={r['file_id']:r for r in map(json.loads,(folder/'admitted.jsonl').open())}
    yall=np.array([rows[i]['class_id'] for i in feature_ids])
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'models').mkdir();(out/'predictions').mkdir()
    for path in [Path(__file__),Path(__file__).with_name('freeze.py')]: shutil.copy2(path,out/('executed_'+path.name))
    target_ids=lock['target_ids']; tpos=np.array([position[i] for i in target_ids]);ty=yall[tpos]
    target_groups=sorted(set(rows[i]['provenance_group_id'] for i in target_ids))
    target_masks={g:np.array([rows[i]['provenance_group_id']==g for i in target_ids]) for g in target_groups}
    reps=config['representations'];seeds=config['seeds'];nf=len(lock['folds'])
    all_models=[];inference={};raw_bootstrap={};timings={}
    with threadpool_limits(limits=config['hardware']['blas_threads']):
        for rep in reps:
            begin=time.perf_counter()
            x=np.asarray(data[rep],dtype=np.float64)
            if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Invalid features')
            source_cm=np.zeros((len(seeds),nf,2,2),dtype=int)
            target_cm={arm:np.zeros((len(seeds),1 if arm in ['procedural','audioldm'] else nf,len(target_groups),2,2),dtype=int) for arm in config['arms']}

            def run_fit(arm,seed,fold,train_ids,test_ids=None):
                assert all(rows[i]['admitted_for_training'] for i in train_ids)
                assert_separation(train_ids,target_ids,rows)
                if test_ids is not None:assert_separation(train_ids,test_ids,rows)
                train_pos=np.array([position[i] for i in train_ids]);y=yall[train_pos]
                n=config['budget']['examples_per_class']*(2 if arm.startswith('real_') else 1)
                assert np.array_equal(np.bincount(y,minlength=2),[n,n])
                name=f'{rep}__{arm}__seed{seed}__fold{fold if fold is not None else "bank"}'
                before=time.perf_counter();model=fixed_fit(x[train_pos],y,config['classifier'])
                fitting=time.perf_counter()-before
                model_path=out/'models'/f'{name}.joblib';joblib.dump(model,model_path,compress=3)
                # Only prediction sees held-out features. Scaler was fitted above.
                p=model.predict_proba(x[tpos]);pred=p.argmax(axis=1)
                record=dict(model_id=name,representation=rep,arm=arm,seed=seed,fold=fold,
                            train_ids=train_ids,train_class_counts=np.bincount(y,minlength=2).tolist(),
                            train_unique_files=len(set(train_ids)),validation=None,target_role='exposed_development',
                            fitted_only_on_training=True,model_sha256=sha(model_path),
                            model_path=str(model_path.relative_to(ROOT)),fit_s=fitting,
                            n_iter=model[1].n_iter_.tolist(),target=metrics(ty,p),target_groups={})
                si=seeds.index(seed);fi=0 if fold is None else fold
                for gi,g in enumerate(target_groups):
                    mask=target_masks[g];cm=cm_for(ty[mask],pred[mask]);target_cm[arm][si,fi,gi]=cm
                    record['target_groups'][g]=metrics(ty[mask],p[mask])
                payload=dict(target_ids=np.array(target_ids),target_true=ty,target_probabilities=p)
                if test_ids is not None:
                    ip=np.array([position[i] for i in test_ids]);sp=model.predict_proba(x[ip]);sy=yall[ip]
                    source_cm[si,fold]=cm_for(sy,sp.argmax(axis=1));record['source']=metrics(sy,sp)
                    record['source_group']=lock['folds'][fold]['held_out_group']
                    payload.update(source_ids=np.array(test_ids),source_true=sy,source_probabilities=sp)
                pred_path=out/'predictions'/f'{name}.npz';np.savez_compressed(pred_path,**payload)
                record['predictions_path']=str(pred_path.relative_to(ROOT));record['predictions_sha256']=sha(pred_path)
                record['total_fit_and_evaluation_s']=time.perf_counter()-before
                save(out/'models'/f'{name}.json',record)
                all_models.append(record)

            # E0 then E1 from the same model; finish real baselines before E2.
            for fold in lock['folds']:
                for seed in seeds:run_fit('real',seed,fold['fold'],fold['train_ids'][str(seed)],fold['test_ids'])
                print(f'{rep} E0/E1 fold {fold["fold"]+1}/{nf}',flush=True)
            print(f'{rep} source F1 {f1(source_cm).mean():.6f}; target F1 {f1(target_cm["real"].sum(axis=2)).mean():.6f}',flush=True)
            for arm in ['procedural','audioldm']:
                for seed in seeds:run_fit(arm,seed,None,lock['synthetic_train_ids'][arm][str(seed)])
                print(f'{rep} E2 {arm} complete',flush=True)
            for arm in ['real_repeated','real_plus_procedural','real_plus_audioldm']:
                for fold in lock['folds']:
                    for seed in seeds:
                        real=fold['train_ids'][str(seed)]
                        extra=real if arm=='real_repeated' else lock['synthetic_train_ids'][arm.removeprefix('real_plus_')][str(seed)]
                        run_fit(arm,seed,fold['fold'],real+extra)
                print(f'{rep} E2 {arm} complete',flush=True)
            np.savez_compressed(out/f'{rep}_group_confusions.npz',source=source_cm,**target_cm)
            estimates,samples,rejected=bootstrap(source_cm,target_cm,config['uncertainty'])
            inference[rep]=dict(estimates=estimates,rejected_missing_class_draws=rejected,
                                source_oof_pooled_by_seed=[cm_metrics(cm.sum(axis=0)) for cm in source_cm])
            raw_bootstrap[rep]=samples
            np.savez_compressed(out/f'{rep}_bootstrap.npz',**samples)
            timings[rep]=time.perf_counter()-begin
            print(rep,json.dumps(estimates),flush=True)
    comparisons={}
    for arm in ['source']+config['arms']:
        a=raw_bootstrap['BEATs768'][arm]-raw_bootstrap['MFCC26'][arm]
        comparisons[arm]=dict(estimate=inference['BEATs768']['estimates'][arm]['estimate']-inference['MFCC26']['estimates'][arm]['estimate'],
                              ci95=np.quantile(a,[.025,.975]).tolist())
    controls={}
    for label,name in enumerate(['always_car','always_truck']):
        p=np.zeros((len(ty),2));p[:,label]=1;controls[name]=metrics(ty,p)
    save(out/'metrics.json',dict(protocol_id=config['protocol_id'],inference=inference,
         beats_minus_mfcc=comparisons,target_constant_controls=controls,
         target_groups={g:dict(support=np.bincount(ty[mask],minlength=2).tolist()) for g,mask in target_masks.items()},
         models=all_models))
    assert len(all_models)==260
    save(out/'summary.json',dict(protocol_id=config['protocol_id'],completed_utc=datetime.now(timezone.utc).isoformat(),
         lock_sha256=sha(args.lock),feature_cache_sha256=sha(args.cache/'features.npz'),
         frozen_config_sha256=sha(folder/'config.json'),models=len(all_models),target_used_for_fitting=False,
         target_role='exposed_development',validation=None,simulation_components_tested=False,
         hardware=platform.platform(),representation_runtime_s=timings,total_s=time.perf_counter()-started,
         git_commit=lock['git_commit'],versions=lock['versions'],
         artifacts_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(f'Complete {len(all_models)} models in {time.perf_counter()-started:.1f}s',flush=True)


if __name__=='__main__':main()

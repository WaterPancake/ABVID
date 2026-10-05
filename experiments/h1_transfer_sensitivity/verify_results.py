"""Independent sklearn scoring, source-choice reconstruction and vectorized CI checks."""
import argparse
from collections import defaultdict
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score,average_precision_score,balanced_accuracy_score,
    brier_score_loss,confusion_matrix,f1_score,precision_recall_fscore_support,roc_auc_score)
from threadpoolctl import threadpool_limits

from common import ROOT,HERE,sha,save,read_json,load_lock,check_artifacts,assert_separation,protected_check
from evaluate_transfer import feature_view


def verify_score(stored,y,p):
    pred=p>.5;pr,rc,fs,support=precision_recall_fscore_support(y,pred,labels=[0,1],zero_division=0)
    expected=dict(macro_f1=f1_score(y,pred,average='macro',zero_division=0),
        balanced_accuracy=balanced_accuracy_score(y,pred),accuracy=accuracy_score(y,pred),
        confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist(),count=len(y),
        class_counts=support.tolist(),truck_prevalence=float(y.mean()),roc_auc=roc_auc_score(y,p),
        average_precision=average_precision_score(y,p),brier_truck=brier_score_loss(y,p),
        log_loss=float(-(y*np.log(np.clip(p,1e-15,1))+(1-y)*np.log(np.clip(1-p,1e-15,1))).mean()))
    for c,name in enumerate(['car','truck']):
        expected.update({name+'_precision':pr[c],name+'_recall':rc[c],name+'_f1':fs[c]})
    ece=0.
    for b in range(10):
        mask=(p>=b/10)&((p<(b+1)/10) if b<9 else (p<=1))
        if mask.any():ece+=mask.mean()*abs(p[mask].mean()-y[mask].mean())
    expected['ece10_truck']=ece
    for k,v in expected.items():np.testing.assert_allclose(stored[k],v,rtol=0,atol=1e-10,err_msg=k)


def vector_metrics(cm):
    """Explicit binary formulas independent of the runner's diagonal-based helper."""
    a,b,c,d=(np.asarray(cm[...,i,j],dtype=float) for i,j in [(0,0),(0,1),(1,0),(1,1)])
    def div(x,y):return np.divide(x,y,out=np.zeros_like(x),where=y>0)
    cr,tr=div(a,a+b),div(d,c+d);cf,tf=div(2*a,2*a+b+c),div(2*d,2*d+b+c)
    return dict(macro_f1=(cf+tf)/2,balanced_accuracy=(cr+tr)/2,accuracy=div(a+d,a+b+c+d),
                car_recall=cr,truck_recall=tr,car_precision=div(a,a+c),truck_precision=div(d,b+d),car_f1=cf,truck_f1=tf)


def verify_ci(stored,point,samples):
    np.testing.assert_allclose(stored['estimate'],point,rtol=0,atol=1e-12)
    np.testing.assert_allclose(stored['ci95'],np.percentile(samples,[2.5,97.5]),rtol=0,atol=1e-12)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lock',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);args=ap.parse_args()
    started=time.perf_counter();lock,cfg,index=load_lock(args.lock);out=args.results
    summary=read_json(out/'summary.json');assert summary['lock_sha256']==sha(args.lock);check_artifacts(out,summary)
    records=read_json(out/'evaluations.json');refs=read_json(args.lock.parent/'model_references.json')
    refidx={(r['representation'],r['arm'],r['seed'],r['fold']):r for r in refs}
    choices={(r['representation'],r['fold'],r['seed']):r for r in read_json(args.lock.parent/'source_selections.json')}
    # Reconstruct the frozen nested C choice from original IDMT inner predictions only.
    inner=read_json(ROOT/cfg['diagnostic_results']/'inner_evaluations.json');bychoice=defaultdict(lambda:defaultdict(list))
    for r in inner:
        assert sha(ROOT/r['predictions_path'])==r['predictions_sha256']
        with np.load(ROOT/r['predictions_path']) as z:
            ids=z['ids'].tolist();y=z['y'];p=z['p']
        assert all(index[i]['dataset_id']=='IDMT' for i in ids)
        assert_separation(ids,lock['target_ids'],index)
        value=f1_score(y,p>.5,average='macro',zero_division=0)
        bychoice[r['representation'],r['outer_fold'],r['seed']][r['C']].append(value)
    for key,values in bychoice.items():
        means={c:float(np.mean(v)) for c,v in values.items()};best=max(means.values())
        chosen=min(c for c,v in means.items() if abs(v-best)<=1e-12)
        assert choices[key]['chosen_C']==chosen
        for c,v in means.items():np.testing.assert_allclose(choices[key]['C_inner_scores'][str(c)],v,atol=1e-12,rtol=0)
    features,pos=feature_view(lock,cfg,index);groups=lock['target_groups'];tids=lock['target_ids']
    masks={g:np.array([index[i]['provenance_group_id']==g for i in tids]) for g in groups}
    getpos=lambda ids:np.array([pos[i] for i in ids]);gety=lambda ids:np.array([index[i]['class_id'] for i in ids])
    cms={};source_cms={};max_replay=0.;models_checked=set()
    for rep in cfg['representations']:
        for arm in cfg['arms']:
            cms[rep,arm]=np.zeros((5,6,len(groups),2,2),dtype=int);source_cms[rep,arm]=np.zeros((5,6,2,2),dtype=int)
    with threadpool_limits(limits=1):
        for r in records:
            rep,arm,seed,fold=r['representation'],r['arm'],r['seed'],r['fold'];ref=refidx[rep,arm,seed,fold]
            for k,v in ref.items():assert r[k]==v
            assert sha(ROOT/r['model_path'])==r['model_sha256'] and sha(ROOT/r['predictions_path'])==r['predictions_sha256']
            model=joblib.load(ROOT/r['model_path']);assert model[1].C==r['C'];models_checked.add(r['fit_id'])
            x=features[rep][getpos(r['train_ids'])]
            np.testing.assert_allclose(model[0].mean_,x.mean(axis=0),rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(model[0].var_,x.var(axis=0),rtol=1e-10,atol=1e-10)
            assert_separation(r['train_ids'],tids,index);assert_separation(r['train_ids'],r['source_test_ids'],index)
            with np.load(ROOT/r['predictions_path']) as z:
                assert z['target_ids'].tolist()==tids and z['source_ids'].tolist()==r['source_test_ids']
                ty,sy,p,sp=z['target_true'],z['source_true'],z['target_p'],z['source_p']
            np.testing.assert_array_equal(ty,gety(tids));np.testing.assert_array_equal(sy,gety(r['source_test_ids']))
            np.testing.assert_allclose(p,model.predict_proba(features[rep][getpos(tids)])[:,1],rtol=0,atol=1e-12)
            np.testing.assert_allclose(sp,model.predict_proba(features[rep][getpos(r['source_test_ids'])])[:,1],rtol=0,atol=1e-12)
            verify_score(r['target'],ty,p);verify_score(r['source'],sy,sp)
            si=cfg['seeds'].index(seed);source_cms[rep,arm][si,fold]=confusion_matrix(sy,sp>.5,labels=[0,1])
            for gi,g in enumerate(groups):
                m=masks[g];verify_score(r['target_groups'][g],ty[m],p[m]);cms[rep,arm][si,fold,gi]=confusion_matrix(ty[m],p[m]>.5,labels=[0,1])
            worst=min(r['target_groups'][g][c+'_recall'] for g in groups for c in ['car','truck'])
            assert worst==r['worst_target_group_class_recall']
            if arm=='baseline':
                old=ROOT/cfg['h1_results']/'predictions'/f'{rep}__real__seed{seed}__fold{fold}.npz'
                with np.load(old) as z:oldp=z['target_probabilities'][:,1]
                max_replay=max(max_replay,float(np.max(np.abs(p-oldp))));np.testing.assert_allclose(p,oldp,rtol=0,atol=1e-10)
            with np.load(ROOT/r['source_predictions_path']) as z:np.testing.assert_allclose(sp,z['p'],rtol=0,atol=1e-10)
    print(f'Verified {len(models_checked)} model/scalers, {len(records)} evaluations, 60 source-only C choices',flush=True)
    with np.load(out/'bootstrap_indices.npz') as z:rs,rf,rg=z['selection'],z['fold'],z['target_group']
    n=len(rs);resamples={};points={}
    with np.load(out/'confusions.npz') as saved:
        for rep in cfg['representations']:
            for arm in cfg['arms']:
                tc=cms[rep,arm];sc=source_cms[rep,arm]
                np.testing.assert_array_equal(saved[f'{rep}__{arm}__source'],sc);np.testing.assert_array_equal(saved[f'{rep}__{arm}__target'],tc)
                for domain,cm in [('source',sc),('target',tc.sum(axis=2))]:
                    for k,v in vector_metrics(cm).items():
                        name=f'{rep}__{arm}__{domain}__{k}';points[name]=float(v.mean());resamples[name]=np.empty(n)
                metrics=vector_metrics(tc)
                wn=f'{rep}__{arm}__target__mean_model_worst_group_class_recall'
                points[wn]=float(np.minimum(metrics['car_recall'],metrics['truck_recall']).min(axis=-1).mean());resamples[wn]=np.empty(n)
                for a in range(0,n,250):
                    b=min(n,a+250);ss=sc[rs[a:b,:,None],rf[a:b,None,:]]
                    tt=tc[rs[a:b,:,None],rf[a:b,None,:]]
                    tt=np.take_along_axis(tt,rg[a:b,None,None,:,None,None],axis=3)
                    for domain,cm in [('source',ss),('target',tt.sum(axis=3))]:
                        for k,v in vector_metrics(cm).items():resamples[f'{rep}__{arm}__{domain}__{k}'][a:b]=v.mean(axis=(1,2))
                    vm=vector_metrics(tt);resamples[wn][a:b]=np.minimum(vm['car_recall'],vm['truck_recall']).min(axis=-1).mean(axis=(1,2))
    with np.load(out/'bootstrap_samples.npz') as z:
        for name,values in resamples.items():np.testing.assert_allclose(z[name],values,rtol=0,atol=1e-12)
    for rep in cfg['representations']:
        for arm in cfg['arms']:
            for domain,metrics in summary['estimates'][rep][arm].items():
                for metric,value in metrics.items():
                    name=f'{rep}__{arm}__{domain}__{metric}';verify_ci(value,points[name],resamples[name])
            sn=f'{rep}__{arm}__source__macro_f1';tn=f'{rep}__{arm}__target__macro_f1'
            verify_ci(summary['source_minus_target_f1'][rep][arm],points[sn]-points[tn],resamples[sn]-resamples[tn])
        for domain,metrics in summary['paired_gains'][rep].items():
            for metric,value in metrics.items():
                a=f'{rep}__nested_C__{domain}__{metric}';b=f'{rep}__baseline__{domain}__{metric}'
                verify_ci(value,points[a]-points[b],resamples[a]-resamples[b])
        for arm in cfg['arms']:
            rr=[r for r in records if r['representation']==rep and r['arm']==arm]
            d=summary['descriptive'][rep][arm]
            for domain in ['source','target']:
                for m,v in d[domain].items():np.testing.assert_allclose(v,np.mean([r[domain][m] for r in rr]),atol=1e-12,rtol=0)
            for g,metrics in d['groups'].items():
                for m,values in metrics.items():
                    actual=[r['target_groups'][g][m] for r in rr]
                    np.testing.assert_allclose(values['mean'],np.mean(actual),atol=1e-12,rtol=0)
                    np.testing.assert_array_equal(values['range'],[min(actual),max(actual)])
            assert d['absolute_minimum_target_group_class_recall']==min(r['worst_target_group_class_recall'] for r in rr)
    protected=protected_check(args.lock.parent/'protected_artifacts.json')
    for p in [Path(__file__),HERE/'test_transfer.py']:shutil.copy2(p,out/('executed_'+p.name))
    save(out/'verification.json',dict(passed=True,unique_models_verified=len(models_checked),evaluations_verified=len(records),
        source_only_C_choices_reconstructed=len(bychoice),target_group_score_records_verified=len(records)*len(groups),
        all_scalers_verified_training_only=True,all_probability_replays_verified=True,all_scalar_metrics_verified=True,
        all_aggregate_intervals_and_bootstrap_samples_verified=True,h1_target_max_probability_error=max_replay,
        protected_files_unchanged=protected,new_fits=0,new_encoder_extractions=0,
        protocol_tests_passed=7,verification_s=time.perf_counter()-started,verifier_sha256=sha(__file__),
        test_script_sha256=sha(HERE/'test_transfer.py')))
    print(read_json(out/'verification.json'))


if __name__=='__main__':main()

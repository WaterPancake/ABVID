"""Prespecified paired H2 scores and whole-group/whole-bank bootstrap."""
from itertools import product
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

CELLS=['F00','F01','F10','F11']
SCALARS=['macro_f1','balanced_accuracy','accuracy','car_precision','truck_precision',
    'car_recall','truck_recall','car_f1','truck_f1','roc_auc','average_precision',
    'brier_truck','log_loss','ece10_truck']


def score(y,p,weights=None):
    y=np.asarray(y,dtype=np.int64); p=np.asarray(p,dtype=np.float64)
    if y.shape!=p.shape or not set(np.unique(y))<={0,1} or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid binary predictions')
    w=np.ones(len(y)) if weights is None else np.asarray(weights,dtype=float)
    if w.shape!=y.shape or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0: raise ValueError('Invalid weights')
    cm=np.bincount(y*2+(p>.5),weights=w,minlength=4).reshape(2,2)
    true=cm.sum(axis=1); predicted=cm.sum(axis=0);tp=cm.diagonal()
    recall=np.divide(tp,true,out=np.zeros(2),where=true>0)
    precision=np.divide(tp,predicted,out=np.zeros(2),where=predicted>0)
    f1=np.divide(2*tp,true+predicted,out=np.zeros(2),where=(true+predicted)>0)
    bins=np.minimum((p*10).astype(int),9)
    result=dict(macro_f1=float(f1.mean()),balanced_accuracy=float(recall.mean()) if (true>0).all() else None,
        accuracy=float(tp.sum()/cm.sum()),confusion_matrix=cm.tolist(),class_counts=true.tolist(),count=float(w.sum()),
        brier_truck=float(np.sum(w*(p-y)**2)/w.sum()),
        log_loss=float(-np.sum(w*np.log(np.clip(np.where(y==1,p,1-p),1e-15,1)))/w.sum()),
        ece10_truck=float(sum(abs(np.sum(w[bins==b]*(p[bins==b]-y[bins==b]))) for b in range(10))/w.sum()),
        roc_auc=float(roc_auc_score(y,p,sample_weight=w)) if (true>0).all() else None,
        average_precision=float(average_precision_score(y,p,sample_weight=w)) if (true>0).all() else None)
    for i,name in enumerate(['car','truck']):
        result[name+'_recall']=float(recall[i]) if true[i]>0 else None
        result[name+'_precision']=float(precision[i]);result[name+'_f1']=float(f1[i])
    return result


def contrasts(q):
    """q[cell,...], fixed order F00,F01,F10,F11; no pooled-seed probabilities."""
    q00,q01,q10,q11=q
    source=((q10-q00)+(q11-q01))/2
    path=((q01-q00)+(q11-q10))/2
    return dict(source_at_P0=q10-q00,source_at_P1=q11-q01,
        path_at_S0=q01-q00,path_at_S1=q11-q10,delta_S=source,delta_P=path,
        interaction=q11-q10-q01+q00,D=q10-q01)


def interval(point,samples):
    return dict(estimate=float(point),ci95=np.quantile(samples,[.025,.975],method='linear').tolist())


def bootstrap(y,group_index,probabilities,draws=10000,seed=314159):
    """probabilities[rep] is [cell,bank,event]; group weights retain all events.

    There are only 35 distinct multiplicity vectors when drawing four groups four
    times. Cache their exact weighted scores, including ranking/calibration, then
    apply the identical sampled groups and banks to every representation/cell.
    """
    group_index=np.asarray(group_index); groups=int(group_index.max())+1
    banks=next(iter(probabilities.values())).shape[1]
    rng=np.random.Generator(np.random.PCG64(seed))
    indices_g=np.empty((draws,groups),dtype=np.int64); indices_b=np.empty((draws,banks),dtype=np.int64)
    for i in range(draws):
        indices_g[i]=rng.integers(groups,size=groups);indices_b[i]=rng.integers(banks,size=banks)
    multiplicities=[v for v in product(range(groups+1),repeat=groups) if sum(v)==groups]
    combination={v:i for i,v in enumerate(multiplicities)}
    draw_combo=np.array([combination[tuple(np.bincount(g,minlength=groups))] for g in indices_g])
    original_combo=combination[tuple([1]*groups)]
    estimates={};all_raw={};per_rep={}
    for rep,pr in probabilities.items():
        if pr.shape!=(4,banks,len(y)): raise ValueError('Incomplete factorial predictions')
        values=np.empty((4,banks,len(multiplicities),len(SCALARS)))
        for cell in range(4):
            for bank in range(banks):
                for gi,multiplicity in enumerate(multiplicities):
                    scored=score(y,pr[cell,bank],np.asarray(multiplicity)[group_index])
                    values[cell,bank,gi]=[scored[k] for k in SCALARS]
        point=values[:,:,original_combo,:].mean(axis=1)
        samples=np.stack([v[indices_b,draw_combo[:,None]].mean(axis=1) for v in values])
        per_rep[rep]=dict(points=point,samples=samples)
        estimates[rep]=dict(cells={},contrasts={})
        for ci,cell in enumerate(CELLS):
            estimates[rep]['cells'][cell]={k:interval(point[ci,mi],samples[ci,:,mi]) for mi,k in enumerate(SCALARS)}
            for mi,k in enumerate(SCALARS): all_raw[f'{rep}__{cell}__{k}']=samples[ci,:,mi]
            recall=np.empty((banks,groups,2))
            for bank in range(banks):
                for gi in range(groups):
                    mask=group_index==gi;scored=score(y[mask],pr[ci,bank,mask])
                    recall[bank,gi]=[scored['car_recall'],scored['truck_recall']]
            selected=recall[indices_b[:,:,None],indices_g[:,None,:]]
            worst=selected.min(axis=(2,3)).mean(axis=1)
            average=selected.mean(axis=(1,2))
            extra={
                'mean_bank_worst_group_class_recall':(recall.min(axis=(1,2)).mean(),worst),
                'group_average_car_recall':(recall[:,:,0].mean(),average[:,0]),
                'group_average_truck_recall':(recall[:,:,1].mean(),average[:,1])}
            for key,(estimate,draw_values) in extra.items():
                estimates[rep]['cells'][cell][key]=interval(estimate,draw_values)
                all_raw[f'{rep}__{cell}__{key}']=draw_values
            estimates[rep]['cells'][cell]['absolute_worst_bank_group_class_recall']=float(recall.min())
        pc=contrasts(point); sc=contrasts(samples)
        for name in pc:
            estimates[rep]['contrasts'][name]={k:interval(pc[name][mi],sc[name][:,mi]) for mi,k in enumerate(SCALARS)}
            for mi,k in enumerate(SCALARS): all_raw[f'{rep}__{name}__{k}']=sc[name][:,mi]
    d=estimates['BEATs768']['contrasts']['D']['macro_f1']
    source=estimates['BEATs768']['contrasts']['delta_S']['macro_f1']
    decision=dict(practical_source_dominance_supported=d['ci95'][0]>=.03 and source['ci95'][0]>0,
        practical_three_point_margin_falsified=d['ci95'][1]<.03,
        directional_source_dominance_contradicted=d['ci95'][1]<=0,
        useful_source_improvement_contradicted=source['ci95'][1]<=0,
        primary='BEATs768 macro_f1 D',independent_confirmation=False)
    return dict(estimates=estimates,decision=decision),all_raw,dict(target_groups=indices_g,banks=indices_b)

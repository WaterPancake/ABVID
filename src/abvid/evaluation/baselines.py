"""Shared numeric functions from experiments/h1/evaluate.py."""
import numpy as np

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

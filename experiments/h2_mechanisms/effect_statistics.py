"""Paired whole-group/whole-bank uncertainty for independent ground and air factors."""
from itertools import product
from common import *
from metrics import score,SCALARS,interval

METRICS=SCALARS+['predicted_truck_fraction']


def full_score(y,p,weights=None):
    result=score(y,p,weights);cm=np.asarray(result['confusion_matrix'])
    result['predicted_truck_fraction']=float(cm[:,1].sum()/cm.sum())
    return result


def effects(q):
    q00,q01,q10,q11=q
    return dict(ground_at_A0=q10-q00,ground_at_A1=q11-q01,
        air_at_G0=q01-q00,air_at_G1=q11-q10,
        ground=((q10-q00)+(q11-q01))/2,air=((q01-q00)+(q11-q10))/2,
        interaction=q11-q10-q01+q00,ground_minus_air=q10-q01)


def calculate(y,group_index,probabilities):
    """probabilities[variant][representation] = [source,path,bank,event]."""
    rng=np.random.Generator(np.random.PCG64(314159));draws=10000
    gi=np.empty((draws,4),np.int64);bi=np.empty((draws,5),np.int64)
    for d in range(draws):gi[d]=rng.integers(4,size=4);bi[d]=rng.integers(5,size=5)
    multiplicities=[t for t in product(range(5),repeat=4) if sum(t)==4]
    lookup={v:i for i,v in enumerate(multiplicities)}
    combo=np.array([lookup[tuple(np.bincount(v,minlength=4))] for v in gi]);original=lookup[(1,1,1,1)]
    estimates={};raw={};point_cache={};sample_cache={}
    for variant,reps in probabilities.items():
        estimates[variant]={}
        for rep,p in reps.items():
            if p.shape!=(2,4,5,len(y)):raise ValueError('Incomplete factorial probabilities')
            values=np.empty((2,4,5,35,len(METRICS)))
            for si in range(2):
                for pi in range(4):
                    for bank in range(5):
                        for mi,m in enumerate(multiplicities):
                            s=full_score(y,p[si,pi,bank],np.asarray(m)[group_index])
                            values[si,pi,bank,mi]=[s[k] for k in METRICS]
            point=values[:,:,:,original,:].mean(axis=2)
            sampled=np.array([[values[si,pi][bi,combo[:,None]].mean(axis=1) for pi in range(4)] for si in range(2)])
            point_cache[variant,rep]=point;sample_cache[variant,rep]=sampled
            estimates[variant][rep]={}
            for si,source in enumerate([*SOURCES,'mean_S']):
                pp=point[si] if si<2 else point.mean(axis=0)
                ss=sampled[si] if si<2 else sampled.mean(axis=0)
                series={path:(pp[i],ss[i]) for i,path in enumerate(PATHS)}
                ep,es=effects(pp),effects(ss)
                series.update({key:(ep[key],es[key]) for key in ep})
                estimates[variant][rep][source]={}
                for key,(pt,draw) in series.items():
                    estimates[variant][rep][source][key]={metric:interval(pt[m],draw[:,m]) for m,metric in enumerate(METRICS)}
                    for m,metric in enumerate(METRICS):raw[f'{variant}__{rep}__{source}__{key}__{metric}']=draw[:,m]
    paired={}
    for rep in REPS:
        paired[rep]={}
        pp=point_cache['matched',rep]-point_cache['native',rep]
        ss=sample_cache['matched',rep]-sample_cache['native',rep]
        for si,source in enumerate([*SOURCES,'mean_S']):
            pt=pp[si] if si<2 else pp.mean(axis=0);dr=ss[si] if si<2 else ss.mean(axis=0)
            paired[rep][source]={}
            for pi,path in enumerate(PATHS):
                paired[rep][source][path]={metric:interval(pt[pi,m],dr[pi,:,m]) for m,metric in enumerate(METRICS)}
                for m,metric in enumerate(METRICS):raw[f'matched_minus_native__{rep}__{source}__{path}__{metric}']=dr[pi,:,m]
    native=estimates['native']['BEATs768']['mean_S'];d=native['ground_minus_air']['macro_f1']['ci95']
    ground=native['ground']['macro_f1']['ci95'];air=native['air']['macro_f1']['ci95']
    decision=dict(primary='native BEATs mean_S ground_minus_air macro_f1',
        ground_larger=d[0]>0,air_larger=d[1]<0,
        practical_ground_dominance=d[0]>=.03 and ground[0]>0,
        practical_air_dominance=d[1]<=-.03 and air[0]>0,
        independent_confirmation=False)
    return dict(estimates=estimates,matched_minus_native=paired,decision=decision),raw,dict(target_groups=gi,banks=bi)

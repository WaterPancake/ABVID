"""Separated extract/fit/evaluate stages for the frozen H2 design.

All forty heads must be locked before evaluate can load target features. The
extract and fit stages have no target-score selection path.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
from pathlib import Path
import shutil
import sys
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from metadata import HERE,ROOT,read,rows,save,sha
from generate_corpus import checked_execution
from metrics import score,bootstrap,CELLS

FREEZE=HERE/'frozen/source_path_20261004_v1'


def h1_cache_compatibility():
    """Read hash/ID metadata only; do not access embeddings or their statistics."""
    folder=ROOT/'experiments/h1/frozen/h1_common_budget_v1.3'
    cache=ROOT/'experiments/h1/cache/h1_common_budget_v1.3'
    cfg=read(FREEZE/'config.json'); h1=read(folder/'config.json'); lock=read(folder/'lock.json'); summary=read(cache/'summary.json')
    for key in ['preprocessing','mfcc','encoder']:
        if cfg[key]!=h1[key]: raise ValueError('H1 representation/preprocessing mismatch: '+key)
    if summary['lock_sha256']!=sha(folder/'lock.json'): raise ValueError('H1 cache lock mismatch')
    if sha(cache/'features.npz')!=summary['artifacts_sha256']['features.npz']: raise ValueError('H1 cache changed')
    target=rows(FREEZE/'target_manifest.jsonl'); tids=[r['file_id'] for r in target]
    if tids!=lock['target_ids']: raise ValueError('H1 target order changed')
    originals={r['file_id']:r for r in rows(folder/'admitted.jsonl')}
    for row in target:
        for key in ['source_path','source_file_sha256','class_id','provenance_group_id']:
            if row[key]!=originals[row['file_id']][key]: raise ValueError('Target provenance changed')
    with np.load(cache/'features.npz',allow_pickle=False) as data:
        ids=data['file_ids'].tolist(); hashes=data['waveform_sha256'].tolist()
    if ids!=lock['feature_ids'] or len(hashes)!=len(ids): raise ValueError('H1 ID/waveform metadata mismatch')
    pos={fid:i for i,fid in enumerate(ids)}
    return dict(passed=True,feature_cache=str(cache/'features.npz'),feature_cache_sha256=sha(cache/'features.npz'),
        target_ids=tids,target_observation_sha256=[hashes[pos[fid]] for fid in tids],
        target_manifest_sha256=sha(FREEZE/'target_manifest.jsonl'),feature_moments_or_predictions_computed=False)


def validate_features(path):
    summary=read(path/'summary.json')
    if sha(path/'features.npz')!=summary['features_sha256']: raise ValueError('Feature cache changed')
    data=np.load(path/'features.npz',allow_pickle=False)
    ids=data['job_ids'].tolist()
    if ids!=summary['job_ids'] or len(ids)!=9600: raise ValueError('Feature population mismatch')
    return summary,data,{jid:i for i,jid in enumerate(ids)}


def extract(args,cfg):
    import torch
    sys.path.insert(0,str(ROOT/'experiments/h1'))
    from extract import mfcc_features
    sys.path.insert(0,str(ROOT/'src'))
    from vehicle_audio.beats_adapter import load_encoder,extract_embeddings
    corpus=args.corpus; summary=read(corpus/'summary.json'); check=read(args.corpus_verification)
    if summary['stage']!='full' or not check['passed'] or check['observations_checked']!=9600 or check['source_replays']!=480:
        raise ValueError('Complete verified corpus required')
    if check['corpus_summary_sha256']!=sha(corpus/'summary.json'): raise ValueError('Corpus verification mismatch')
    manifest=rows(corpus/'observation_manifest.jsonl')
    if sha(corpus/'observation_manifest.jsonl')!=summary['observation_manifest_sha256']: raise ValueError('Corpus changed')
    index={r['job_id']:r for r in manifest}; ids=sorted(index)
    out=args.output; out.mkdir(parents=True,exist_ok=False);begin=time.perf_counter()
    save(out/'target_cache_compatibility.json',h1_cache_compatibility())
    torch.set_num_threads(4);torch.set_num_interop_threads(1);np.random.seed(42);torch.manual_seed(42)
    encoder,provenance=load_encoder(cfg['encoder'])
    beats=np.empty((9600,768),dtype=np.float32);mfcc=np.empty((9600,26),dtype=np.float32);wave_hashes=[]
    for lo in range(0,len(ids),8):
        waves=[]
        for offset,jid in enumerate(ids[lo:lo+8]):
            row=index[jid];path=corpus/row['observation_path']
            if sha(path)!=row['observation_file_sha256']: raise ValueError('Observation file changed')
            wave=np.load(path,allow_pickle=False)
            digest=hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest()
            if digest!=row['observation_sha256']: raise ValueError('Observation changed')
            waves.append(wave);wave_hashes.append(digest);mfcc[lo+offset]=mfcc_features(wave,cfg['mfcc'])
        batch=torch.from_numpy(np.stack(waves));encoded=extract_embeddings(encoder,batch).numpy()
        if lo==0:
            np.testing.assert_array_equal(encoded,extract_embeddings(encoder,batch).numpy())
        beats[lo:lo+len(waves)]=encoded
        if lo%800==0: print(f'H2 features {lo+len(waves)}/9600; {time.perf_counter()-begin:.1f}s',flush=True)
    if not np.isfinite(beats).all() or not np.isfinite(mfcc).all(): raise ValueError('Nonfinite embeddings')
    np.savez_compressed(out/'features.npz',job_ids=np.array(ids),waveform_sha256=np.array(wave_hashes),BEATs768=beats,MFCC26=mfcc)
    save(out/'summary.json',dict(passed=True,job_ids=ids,features_sha256=sha(out/'features.npz'),
        corpus_summary_sha256=sha(corpus/'summary.json'),corpus_verification_sha256=sha(args.corpus_verification),
        execution_lock_sha256=sha(args.execution_lock),target_cache_compatibility_sha256=sha(out/'target_cache_compatibility.json'),
        encoder_provenance=provenance,encoder_frozen=True,target_feature_extraction=False,
        target_statistics_computed=False,elapsed_s=time.perf_counter()-begin))


def fit(args,cfg):
    meta,data,position=validate_features(args.features)
    if meta['execution_lock_sha256']!=sha(args.execution_lock): raise ValueError('Feature execution lock mismatch')
    plans=rows(FREEZE/'fit_plan.jsonl');render_index={r['job_id']:r for r in rows(FREEZE/'render_plan.jsonl')}
    out=args.output;out.mkdir(parents=True,exist_ok=False);(out/'models').mkdir();(out/'validation').mkdir()
    records=[];begin=time.perf_counter()
    with threadpool_limits(limits=1):
        for plan in plans:
            ids=plan['train_job_ids'];vids=plan['validation_job_ids'];rep=plan['representation']
            if len(ids)!=380 or len(vids)!=100 or set(ids)&set(vids): raise ValueError('Incorrect fit roles/budget')
            if any(render_index[i]['role']!='train' for i in ids) or any(render_index[i]['role']!='validation' for i in vids): raise ValueError('Role leakage')
            x=np.asarray(data[rep][[position[i] for i in ids]],dtype=np.float64)
            y=np.array([render_index[i]['class_id'] for i in ids]); vy=np.array([render_index[i]['class_id'] for i in vids])
            if np.bincount(y,minlength=2).tolist()!=[190,190]: raise ValueError('Training balance')
            parameters={k:v for k,v in cfg['classifier'].items() if k!='type'}
            model=make_pipeline(StandardScaler(),LogisticRegression(**parameters,random_state=plan['replicate_seed']))
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning);model.fit(x,y)
            np.testing.assert_allclose(model[0].mean_,x.mean(axis=0),rtol=0,atol=1e-12)
            np.testing.assert_allclose(model[0].var_,x.var(axis=0),rtol=1e-12,atol=1e-12)
            if model.classes_.tolist()!=[0,1]: raise ValueError('Class mapping changed')
            path=out/'models'/(plan['fit_id']+'.joblib');joblib.dump(model,path,compress=3)
            p=model.predict_proba(np.asarray(data[rep][[position[i] for i in vids]],dtype=np.float64))[:,1]
            predpath=out/'validation'/(plan['fit_id']+'.npz');np.savez_compressed(predpath,job_ids=np.array(vids),y=vy,p=p)
            record=dict(plan,model_path=str(path.relative_to(out)),model_sha256=sha(path),status='fitted',
                checkpoint_sha256=sha(path),n_iter=model[1].n_iter_.tolist(),training_only_scaler_verified=True,
                validation=score(vy,p),validation_predictions=str(predpath.relative_to(out)),validation_sha256=sha(predpath))
            records.append(record)
            print('H2 fixed head',len(records),'/40',plan['fit_id'],flush=True)
    if len(records)!=40: raise ValueError('Incomplete factorial fits')
    save(out/'fits.json',records)
    save(out/'model_lock.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),models=40,
        execution_lock_sha256=sha(args.execution_lock),features_summary_sha256=sha(args.features/'summary.json'),
        features_path=str(args.features.resolve()),fits_sha256=sha(out/'fits.json'),threshold=.5,C=1,
        target_scoring_started=False,target_used_for_fitting=False,elapsed_s=time.perf_counter()-begin,
        artifacts_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))


def evaluate(args,cfg):
    lock=read(args.models/'model_lock.json')
    if lock['models']!=40 or lock['execution_lock_sha256']!=sha(args.execution_lock): raise ValueError('Model freeze mismatch')
    for name,digest in lock['artifacts_sha256'].items():
        if sha(args.models/name)!=digest: raise ValueError('Frozen model artifact changed')
    compatibility=h1_cache_compatibility();feature_summary=read(Path(lock['features_path'])/'summary.json')
    if sha(Path(lock['features_path'])/'summary.json')!=lock['features_summary_sha256']: raise ValueError('Feature lock changed')
    if read(Path(lock['features_path'])/'target_cache_compatibility.json')!=compatibility: raise ValueError('Target cache compatibility changed')
    target=rows(FREEZE/'target_manifest.jsonl');tids=[r['file_id'] for r in target]
    y=np.array([r['class_id'] for r in target]);groups=sorted({r['provenance_group_id'] for r in target})
    gindex=np.array([groups.index(r['provenance_group_id']) for r in target]); masks={g:gindex==i for i,g in enumerate(groups)}
    with np.load(compatibility['feature_cache'],allow_pickle=False) as cache:
        ids=cache['file_ids'].tolist();pos={fid:i for i,fid in enumerate(ids)}
        target_features={rep:np.asarray(cache[rep][[pos[fid] for fid in tids]],dtype=np.float64) for rep in cfg['representations']}
    out=args.output;out.mkdir(parents=True,exist_ok=False);begin=time.perf_counter();records=[]
    probabilities={rep:np.empty((4,5,len(y)),dtype=np.float64) for rep in cfg['representations']}
    with threadpool_limits(limits=1):
        for ref in read(args.models/'fits.json'):
            model=joblib.load(args.models/ref['model_path']);rep=ref['representation']
            p=model.predict_proba(target_features[rep])[:,1]
            ci=CELLS.index(ref['cell']);si=cfg['seeds'].index(ref['replicate_seed']);probabilities[rep][ci,si]=p
            scores={g:score(y[m],p[m]) for g,m in masks.items()}
            records.append(dict(fit_id=ref['fit_id'],representation=rep,cell=ref['cell'],seed=ref['replicate_seed'],
                target=score(y,p),groups=scores,
                worst_group_class_recall=min(v[name+'_recall'] for v in scores.values() for name in ['car','truck']),
                group_average_car_recall=float(np.mean([v['car_recall'] for v in scores.values()])),
                group_average_truck_recall=float(np.mean([v['truck_recall'] for v in scores.values()]))))
    np.savez_compressed(out/'target_predictions.npz',target_ids=np.array(tids),y=y,group_index=gindex,**probabilities)
    save(out/'evaluations.json',records)
    stats,samples,indices=bootstrap(y,gindex,probabilities)
    np.savez_compressed(out/'bootstrap_samples.npz',**samples);np.savez_compressed(out/'bootstrap_indices.npz',**indices)
    save(out/'statistics.json',stats)
    save(out/'summary.json',dict(passed=True,domain=cfg['domain'],target_role='previously_exposed_development',
        models=40,target_events=len(y),target_groups={g:np.bincount(y[m],minlength=2).tolist() for g,m in masks.items()},
        execution_lock_sha256=sha(args.execution_lock),model_lock_sha256=sha(args.models/'model_lock.json'),
        feature_cache_sha256=compatibility['feature_cache_sha256'],target_used_for_selection=False,
        constant_controls={name:score(y,np.full(len(y),p)) for name,p in [('always_car',0.),('always_truck',1.)]},
        elapsed_s=time.perf_counter()-begin,completed_utc=datetime.now(timezone.utc).isoformat(),
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(stats['decision'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['extract','fit','evaluate'])
    ap.add_argument('--execution-lock',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--corpus',type=Path);ap.add_argument('--corpus-verification',type=Path)
    ap.add_argument('--features',type=Path);ap.add_argument('--models',type=Path);args=ap.parse_args()
    checked_execution(args.execution_lock);cfg=read(FREEZE/'config.json')
    {'extract':extract,'fit':fit,'evaluate':evaluate}[args.stage](args,cfg)


if __name__=='__main__':main()

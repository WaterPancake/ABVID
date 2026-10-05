"""Hash-checked fixed H1 observations, BEATs768 and MFCC26. No fitting."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from math import gcd
from pathlib import Path
import json
import shutil
import sys
import time

import librosa
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch

from freeze import ROOT, sha, save
sys.path.insert(0,str(ROOT/'src'))
from vehicle_audio.beats_adapter import load_encoder, extract_embeddings


def observation(x,sr,config):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim==2: x=x.mean(axis=1)
    if x.ndim!=1 or not np.isfinite(x).all(): raise ValueError('Invalid waveform')
    for rate in [config['intermediate_rate_hz'],config['sample_rate_hz']]:
        if sr!=rate:
            g=gcd(int(sr),int(rate))
            x=resample_poly(x,rate//g,sr//g,window=('kaiser',config['window_beta']),padtype='constant')
        sr=rate
    n=config['samples']
    if len(x)<n: raise ValueError('Short waveform; padding is forbidden')
    start=(len(x)-n)//2
    x=np.asarray(x[start:start+n],dtype=np.float32)
    if not np.isfinite(x).all() or not np.any(x): raise ValueError('Invalid processed waveform')
    return x


def mfcc_features(x,config):
    mel=librosa.feature.melspectrogram(y=x,sr=16000,n_fft=config['n_fft'],hop_length=config['hop_length'],
        win_length=config['n_fft'],window=config['window'],center=config['center'],pad_mode=config['pad_mode'],
        power=config['power'],n_mels=config['n_mels'],fmin=config['fmin'],fmax=config['fmax'],
        norm=config['mel_norm'],htk=config['htk'])
    db=librosa.power_to_db(mel,ref=1.0,top_db=config['top_db'])
    mf=librosa.feature.mfcc(S=db,n_mfcc=config['n_mfcc'],dct_type=config['dct_type'],norm=config['dct_norm'])
    return np.r_[mf.mean(axis=1),mf.std(axis=1,ddof=0)].astype(np.float32)


def load(row,config):
    p=ROOT/row['source_path']
    if sha(p)!=row['source_file_sha256']: raise ValueError(f'Changed source {p}')
    x,sr=sf.read(p,dtype='float64',always_2d=True)
    y=observation(x,sr,config['preprocessing'])
    return y,mfcc_features(y,config['mfcc']),__import__('hashlib').sha256(y.astype('<f4').tobytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--lock',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    started=time.perf_counter()
    lock=json.loads(args.lock.read_text()); folder=args.lock.parent
    for name,h in lock['artifacts_sha256'].items():
        if sha(folder/name)!=h: raise ValueError(f'Frozen file changed: {name}')
    config=json.loads((folder/'config.json').read_text())
    rows={r['file_id']:r for r in map(json.loads,(folder/'admitted.jsonl').open())}
    ids=lock['feature_ids']; out=args.output; out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'executed_extract.py')
    shutil.copy2(ROOT/'src/vehicle_audio/beats_adapter.py',out/'executed_beats_adapter.py')
    torch.set_num_threads(config['hardware']['torch_threads']); torch.set_num_interop_threads(1)
    np.random.seed(42); torch.manual_seed(42)
    encoder,provenance=load_encoder(config['encoder'])
    # Warm up only on a deterministic artificial signal, never on target scores.
    t=np.arange(32000)/16000
    toy=torch.from_numpy((.1*np.sin(2*np.pi*123*t)).astype(np.float32))[None]
    a=extract_embeddings(encoder,toy); b=extract_embeddings(encoder,toy)
    np.testing.assert_allclose(a.numpy(),b.numpy(),rtol=0,atol=0)
    trainable=sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    assert trainable==0
    beats=np.empty((len(ids),768),dtype=np.float32); mfcc=np.empty((len(ids),26),dtype=np.float32)
    audiohashes=[]; batch=config['hardware']['batch_size']
    timings=dict(load_and_mfcc_s=0.,beats_forward_s=0.)
    # Compile librosa outside pool before concurrent reads.
    mfcc_features(toy.numpy()[0],config['mfcc'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0,len(ids),batch):
            before=time.perf_counter()
            items=list(pool.map(lambda fid:load(rows[fid],config),ids[start:start+batch]))
            timings['load_and_mfcc_s']+=time.perf_counter()-before
            waves=np.stack([v[0] for v in items]); mfcc[start:start+len(items)]=np.stack([v[1] for v in items])
            audiohashes.extend(v[2] for v in items)
            before=time.perf_counter()
            beats[start:start+len(items)]=extract_embeddings(encoder,torch.from_numpy(waves)).numpy()
            timings['beats_forward_s']+=time.perf_counter()-before
            if (start//batch)%100==0 or start+batch>=len(ids):
                print(f'features {min(start+batch,len(ids))}/{len(ids)} elapsed {time.perf_counter()-started:.1f}s',flush=True)
    assert np.isfinite(beats).all() and np.isfinite(mfcc).all()
    np.savez_compressed(out/'features.npz',file_ids=np.array(ids),BEATs768=beats,MFCC26=mfcc,
                        waveform_sha256=np.array(audiohashes))
    save(out/'summary.json',dict(protocol_id=config['protocol_id'],lock_sha256=sha(args.lock),
        completed_utc=datetime.now(timezone.utc).isoformat(),files=len(ids),frozen_parameters=True,
        deterministic_toy_check=True,encoder_provenance=provenance,
        no_training_or_target_statistics=True,timings=timings,total_s=time.perf_counter()-started,
        artifacts_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print(f'Complete: {time.perf_counter()-started:.1f}s',flush=True)


if __name__=='__main__':main()

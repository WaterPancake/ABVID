"""Extract three fixed observations for IDMT only; no target cache is opened."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import shutil
import time
import sys

import numpy as np
import soundfile as sf
import torch
from common import ROOT,HERE,sha,save,read_lock,source_path
from extract import observation,mfcc_features
sys.path.insert(0,str(ROOT/'src'))
from vehicle_audio.beats_adapter import load_encoder,extract_embeddings,masked_mean


@torch.inference_mode()
def variable_embeddings(encoder,waves):
    if waves.ndim!=2 or waves.shape[1] not in (16000,32000) or not torch.isfinite(waves).all():
        raise ValueError('Expected finite true-length 1 s or 2 s input')
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):raise ValueError('Encoder not frozen')
    tokens,mask=encoder.extract_features(waves,padding_mask=torch.zeros_like(waves,dtype=torch.bool))
    z=masked_mean(tokens,mask)
    assert z.shape==(len(waves),768) and torch.isfinite(z).all()
    return z.float()


def load_one(row,cfg):
    path=source_path(row)
    if sha(path)!=row['source_file_sha256']:raise ValueError(f'Audio changed: {path}')
    x,sr=sf.read(path,dtype='float64',always_2d=True)
    output={}
    for name,pcfg in cfg['feature_profiles'].items():
        y=observation(x,sr,pcfg)
        output[name]=(y,mfcc_features(y,cfg['mfcc']),hashlib.sha256(y.astype('<f4').tobytes()).hexdigest())
    return output


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lock',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    lock,cfg,rows=read_lock(args.lock);out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(42);np.random.seed(42)
    encoder,provenance=load_encoder(cfg['encoder'])
    toy=torch.from_numpy((.1*np.sin(2*np.pi*123*np.arange(32000)/16000)).astype(np.float32))[None]
    np.testing.assert_array_equal(variable_embeddings(encoder,toy).numpy(),extract_embeddings(encoder,toy).numpy())
    s=variable_embeddings(encoder,toy[:,8000:24000]);assert s.shape==(1,768)
    np.testing.assert_array_equal(s.numpy(),variable_embeddings(encoder,toy[:,8000:24000]).numpy())
    mfcc_features(toy.numpy()[0],cfg['mfcc'])
    beats={name:np.empty((len(rows),768),dtype=np.float32) for name in cfg['feature_profiles']}
    mfcc={name:np.empty((len(rows),26),dtype=np.float32) for name in cfg['feature_profiles']}
    hashes={name:[] for name in cfg['feature_profiles']};batch=cfg['hardware']['batch_size'];timings={name:0. for name in cfg['feature_profiles']}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0,len(rows),batch):
            items=list(pool.map(lambda r:load_one(r,cfg),rows[start:start+batch]))
            for name in cfg['feature_profiles']:
                waves=np.stack([r[name][0] for r in items]);begin=time.perf_counter()
                beats[name][start:start+len(items)]=variable_embeddings(encoder,torch.from_numpy(waves)).numpy()
                timings[name]+=time.perf_counter()-begin
                mfcc[name][start:start+len(items)]=np.stack([r[name][1] for r in items]);hashes[name].extend(r[name][2] for r in items)
            if start%800==0 or start+batch>=len(rows):print(f'IDMT profiles {min(start+batch,len(rows))}/{len(rows)} elapsed {time.perf_counter()-started:.1f}s',flush=True)
    for name in cfg['feature_profiles']:
        assert np.isfinite(beats[name]).all() and np.isfinite(mfcc[name]).all()
        np.savez_compressed(out/(name+'.npz'),file_ids=np.array([r['file_id'] for r in rows]),BEATs768=beats[name],MFCC26=mfcc[name],waveform_sha256=np.array(hashes[name]))
    for path in [Path(__file__),HERE/'common.py',ROOT/'experiments/h1/extract.py',ROOT/'src/vehicle_audio/beats_adapter.py']:
        shutil.copy2(path,out/('executed_'+path.name))
    save(out/'summary.json',dict(protocol_id=cfg['protocol_id'],completed_utc=datetime.now(timezone.utc).isoformat(),
        lock_sha256=sha(args.lock),source_only=True,source_files=len(rows),observations=len(rows)*len(beats),
        total_s=time.perf_counter()-started,forward_s=timings,encoder_provenance=provenance,
        toy_2s_matches_existing_adapter_exactly=True,toy_1s_finite_and_repeatable=True,
        target_paths_accessed=[],artifacts_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print(f'Complete in {time.perf_counter()-started:.1f}s',flush=True)


if __name__=='__main__':main()

"""Validate candidate raw events against released P01 MFCC rows before replay."""
from pathlib import Path
import hashlib
import json
import os
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/reproduction'
os.environ.setdefault('NUMBA_CACHE_DIR',str(BASE/'work/numba_cache'))
import librosa
import numpy as np
import soundfile as sf

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    t0=time.perf_counter()
    candidate=BASE/'results/P01_balanced_row_candidates.json'
    manifest=json.loads(candidate.read_text())
    feature_root=BASE/'releases/P01/features'
    means=np.load(feature_root/'scaler_mean.npy',allow_pickle=False)
    scale=np.load(feature_root/'scaler_scale.npy',allow_pickle=False)
    features={d:np.load(feature_root/f'X_test_{d}.npy',allow_pickle=False,mmap_mode='r') for d in ['melaudis','ch34']}
    results=[]
    for number,row in enumerate(manifest['rows']):
        original=ROOT/row['original'] if 'original' in row else BASE/'work/raw_melaudis'/row['archive_member']
        output=BASE/'work/processed_selected'/row['domain']/row['class']/row['processed_name']
        output.parent.mkdir(parents=True,exist_ok=True)
        # Exactly the released preprocessData_p2.py::resample_and_save sequence.
        y,sr=librosa.load(str(original),sr=22050,mono=True)
        assert len(y)>=2205 and np.isfinite(y).all()
        peak=np.max(np.abs(y))
        if peak>0:y=y/peak*.95
        sf.write(str(output),y,22050,subtype='PCM_16')
        # Exactly the released corrected MFCC extractor and supplied scaler.
        y,sr=librosa.load(str(output),sr=22050,mono=True)
        y=np.pad(y,(0,max(0,44100-len(y))),mode='constant')[:44100]
        mfcc=librosa.feature.mfcc(y=y,sr=22050,n_mfcc=40,n_fft=2048,hop_length=512,n_mels=128)
        feature=np.vstack([mfcc,librosa.feature.delta(mfcc),librosa.feature.delta(mfcc,order=2)]).T
        # StandardScaler preserves float32 after each in-place operation.
        feature-=means
        feature/=scale
        reference=features[row['domain']][row['row_index'],:,:,0]
        assert feature.shape==reference.shape
        match=bool(np.allclose(feature,reference,rtol=1e-4,atol=1e-4))
        results.append(dict(row,original_sha256=sha256(original),processed_path=str(output.relative_to(ROOT)),processed_sha256=sha256(output),feature_match=match,max_abs_error=float(np.max(np.abs(feature-reference)))))
        if (number+1)%50==0:print(number+1,'rows checked;',sum(r['feature_match'] for r in results),'within fixed tolerance',flush=True)
    summary={'scope':'500 published seed-42 balanced evaluation rows only; not full-corpus row validation','method':'reconstruct released 22.05 kHz mono peak-0.95 PCM16 preprocessing and corrected MFCC extractor; apply supplied training scaler; compare to supplied row','rtol':1e-4,'atol':1e-4,'tolerance_chosen_before_predictions':True,'librosa':librosa.__version__,'numpy':np.__version__,'soundfile':sf.__version__,'candidate_manifest_sha256':sha256(candidate),'script_sha256':sha256(__file__),'all_rows_match':all(r['feature_match'] for r in results),'seconds':time.perf_counter()-t0,'rows':results}
    (BASE/'results/P01_row_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('All rows match:',summary['all_rows_match'], 'seconds',round(summary['seconds'],2),flush=True)

if __name__=='__main__':main()

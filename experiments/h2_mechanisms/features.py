"""Frozen representations and prespecified gain diagnostics; no classifier fitting."""
import argparse
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import welch
from common import *

ACOUSTIC_NAMES=['rms_dbfs','peak','crest_db','centroid_hz','rolloff85_hz',
    'power_0_200','power_200_500','power_500_1000','power_1000_2000','power_2000_4000']


def acoustic(x):
    x=np.asarray(x,dtype=np.float64);rms=np.sqrt(np.mean(x*x));peak=abs(x).max()
    if not np.isfinite(x).all() or rms<=1e-12:raise ValueError('Invalid diagnostic waveform')
    f,p=welch(x,fs=16000,window='hann',nperseg=1024,noverlap=512)
    keep=f<=4000;f=f[keep];p=p[keep];total=p.sum()
    if total<=0:raise ValueError('Zero common-band power')
    band=CONFIG['spectral_diagnostics']['bands_hz']
    values=[float(p[(f>=a)&((f<b) if b<4000 else (f<=b))].sum()/total) for a,b in zip(band[:-1],band[1:])]
    return np.array([20*np.log10(rms),peak,20*np.log10(peak/rms),np.sum(f*p)/total,
        f[np.searchsorted(np.cumsum(p),.85*total)],*values],dtype=np.float64)


def normalize_target(x):
    wave=np.asarray(x,dtype=np.float64);rms=np.sqrt(np.mean(wave*wave))
    if wave.shape!=(32000,) or not np.isfinite(wave).all() or rms<=1e-12:raise ValueError('Invalid target RMS')
    gain=10**(-26/20)/rms;y=(wave*gain).astype(np.float32)
    if abs(np.sqrt(np.mean(y.astype(float)**2))/10**(-26/20)-1)>1e-5:raise ValueError('Gain diagnostic RMS')
    return y,float(gain)


def encoder_tools():
    import torch
    sys.path.insert(0,str(ROOT/'experiments/h1'))
    from extract import mfcc_features,load
    sys.path.insert(0,str(ROOT/'src'))
    from vehicle_audio.beats_adapter import load_encoder,extract_embeddings
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.manual_seed(42);np.random.seed(42)
    encoder,provenance=load_encoder(BASE['encoder'])
    return torch,encoder,provenance,extract_embeddings,mfcc_features,load


def synthetic():
    check_execution();validation=read(CORPUS/'verification.json')
    if not validation['passed'] or validation['complete_waveform_replays']!=9600:raise ValueError('Full replay required')
    CACHE.mkdir(parents=True,exist_ok=False);begin=time.perf_counter()
    torch,encoder,prov,embed,mfcc,_=encoder_tools()
    obs=joined_observations();index={r['job_id']:r for r in obs};ids=sorted(index)
    with np.load(ROOT/CONFIG['parent_features']/'features.npz',allow_pickle=False) as old:
        old_ids={j:i for i,j in enumerate(old['job_ids'].tolist())}
        inherited={rep:old[rep].copy() for rep in REPS}
    matrices={'BEATs768':np.empty((len(ids),768),np.float32),'MFCC26':np.empty((len(ids),26),np.float32)}
    descriptors=np.empty((len(ids),len(ACOUSTIC_NAMES)),float);hashes=[];new_positions=[]
    for i,jid in enumerate(ids):
        row=index[jid];path=Path(row['resolved_observation']);wave=np.load(path,allow_pickle=False)
        if sha(path)!=row['observation_file_sha256']:raise ValueError('Observation changed')
        digest=hashlib.sha256(wave.tobytes()).hexdigest()
        if digest!=row['observation_sha256']:raise ValueError('Observation data changed')
        descriptors[i]=acoustic(wave);hashes.append(digest)
        if row['reused']:
            for rep in REPS:matrices[rep][i]=inherited[rep][old_ids[row['parent_job_id']]]
        else:new_positions.append(i)
    for lo in range(0,len(new_positions),8):
        positions=new_positions[lo:lo+8]
        waves=np.stack([np.load(index[ids[i]]['resolved_observation'],allow_pickle=False) for i in positions])
        encoded=embed(encoder,torch.from_numpy(waves)).numpy()
        if lo==0:np.testing.assert_array_equal(encoded,embed(encoder,torch.from_numpy(waves)).numpy())
        matrices['BEATs768'][positions]=encoded
        matrices['MFCC26'][positions]=np.stack([mfcc(w,BASE['mfcc']) for w in waves])
        if lo%800==0:print('mechanism features',lo+len(positions),'/9600',flush=True)
    np.savez_compressed(CACHE/'features.npz',job_ids=np.array(ids),waveform_sha256=np.array(hashes),
        acoustic_names=np.array(ACOUSTIC_NAMES),acoustic=descriptors,**matrices)
    # Every validation example gets both predeclared scalar interventions.
    vids=[jid for jid in ids if index[jid]['role']=='validation'];gains=CONFIG['validation_gain_diagnostic_db']
    stressed={'BEATs768':np.empty((2,len(vids),768),np.float32),'MFCC26':np.empty((2,len(vids),26),np.float32)}
    peak=0.
    for gi,gain_db in enumerate(gains):
        for lo in range(0,len(vids),8):
            waves=np.stack([(np.load(index[j]['resolved_observation']).astype(float)*10**(gain_db/20)).astype(np.float32)
                for j in vids[lo:lo+8]])
            peak=max(peak,float(abs(waves).max()))
            stressed['BEATs768'][gi,lo:lo+len(waves)]=embed(encoder,torch.from_numpy(waves)).numpy()
            stressed['MFCC26'][gi,lo:lo+len(waves)]=np.stack([mfcc(w,BASE['mfcc']) for w in waves])
            if lo%800==0:print('synthetic gain features',gain_db,lo+len(waves),'/4000',flush=True)
    for data in [*matrices.values(),*stressed.values()]:
        if not np.isfinite(data).all():raise ValueError('Nonfinite features')
    np.savez_compressed(CACHE/'validation_gain.npz',job_ids=np.array(vids),gains_db=np.array(gains),**stressed)
    save(CACHE/'summary.json',dict(passed=True,events=len(ids),new_events=9600,reused_events=9600,
        validation_gain_events=8000,validation_gain_max_peak=peak,encoder_frozen=True,
        encoder_provenance=prov,target_access=False,elapsed_s=time.perf_counter()-begin,
        execution_lock_sha256=sha(EXECUTION/'lock.json'),corpus_verification_sha256=sha(CORPUS/'verification.json'),
        artifacts_sha256={p.name:sha(p) for p in CACHE.iterdir() if p.is_file()}))


def target():
    check_execution();model_lock=read(MODELS/'lock.json')
    if model_lock['models']!=80:raise ValueError('All heads must be frozen before target diagnostics')
    out=CACHE/'target_gain';out.mkdir(parents=True,exist_ok=False);begin=time.perf_counter()
    torch,encoder,prov,embed,mfcc,load=encoder_tools()
    from learning import h1_cache_compatibility
    compatibility=h1_cache_compatibility();target=rows(FREEZE/'target_manifest.jsonl');ids=[r['file_id'] for r in target]
    with np.load(compatibility['feature_cache'],allow_pickle=False) as old:
        pos={j:i for i,j in enumerate(old['file_ids'].tolist())}
        old_hash=old['waveform_sha256'].copy();old_mfcc=old['MFCC26'].copy()
    matrices={'BEATs768':np.empty((len(ids),768),np.float32),'MFCC26':np.empty((len(ids),26),np.float32)}
    a_original=np.empty((len(ids),len(ACOUSTIC_NAMES)));a_normalized=np.empty_like(a_original)
    gain_array=np.empty(len(ids));normalized_hashes=[];original_hashes=[];peak_gt_one=0;max_peak=0.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for lo in range(0,len(ids),8):
            rr=target[lo:lo+8];items=list(pool.map(lambda r:load(r,BASE),rr));waves=[]
            for k,(row,(wave,old_m,old_h)) in enumerate(zip(rr,items)):
                i=lo+k
                if old_h!=old_hash[pos[row['file_id']]]:raise ValueError('Original target waveform mismatch')
                np.testing.assert_array_equal(old_m,old_mfcc[pos[row['file_id']]])
                norm,gain=normalize_target(wave);waves.append(norm);gain_array[i]=gain
                a_original[i]=acoustic(wave);a_normalized[i]=acoustic(norm)
                peak=float(abs(norm).max());max_peak=max(max_peak,peak);peak_gt_one+=int(peak>1.)
                normalized_hashes.append(hashlib.sha256(norm.tobytes()).hexdigest());original_hashes.append(old_h)
            waves=np.stack(waves)
            matrices['BEATs768'][lo:lo+len(waves)]=embed(encoder,torch.from_numpy(waves)).numpy()
            matrices['MFCC26'][lo:lo+len(waves)]=np.stack([mfcc(w,BASE['mfcc']) for w in waves])
            if lo%800==0:print('fixed target gain features',lo+len(waves),'/8066',flush=True)
    np.savez_compressed(out/'features.npz',file_ids=np.array(ids),original_waveform_sha256=np.array(original_hashes),
        normalized_waveform_sha256=np.array(normalized_hashes),gain=gain_array,
        acoustic_names=np.array(ACOUSTIC_NAMES),original_acoustic=a_original,normalized_acoustic=a_normalized,**matrices)
    save(out/'summary.json',dict(passed=True,events=len(ids),model_lock_sha256=sha(MODELS/'lock.json'),
        features_sha256=sha(out/'features.npz'),target_labels_used_to_choose_transform=False,
        normalized_rms_dbfs=-26,peak_gt_one_events=peak_gt_one,max_peak=max_peak,clipped_events=0,
        target_role='exposed_development_diagnostic',encoder_provenance=prov,
        original_cache_sha256=compatibility['feature_cache_sha256'],elapsed_s=time.perf_counter()-begin))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['synthetic','target']);a=ap.parse_args()
    {'synthetic':synthetic,'target':target}[a.stage]()

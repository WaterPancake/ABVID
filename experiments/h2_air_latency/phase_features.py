"""Extract only the new synthetic observations; reuse both endpoint caches."""
import hashlib
import time
from phase_common import *
from features import encoder_tools


def main():
    check_execution();check=read(CORPUS/'verification.json')
    if not check['passed'] or check['complete_waveform_replays']!=9600:raise ValueError('Full replay required')
    CACHE.mkdir(parents=True,exist_ok=False);begin=time.perf_counter()
    torch,encoder,provenance,embed,mfcc,_=encoder_tools()
    rows_by_id={r['job_id']:r for r in joined_observations()};ids=sorted(rows_by_id)
    if len(ids)!=19200:raise ValueError('Incomplete population')
    with np.load(parent.CACHE/'features.npz',allow_pickle=False) as old:
        old_pos={j:i for i,j in enumerate(old['job_ids'].tolist())}
        inherited={rep:old[rep].copy() for rep in REPS}
        inherited_hashes=old['waveform_sha256'].copy()
    matrices={'BEATs768':np.empty((19200,768),np.float32),'MFCC26':np.empty((19200,26),np.float32)}
    hashes=[];new=[]
    for i,jid in enumerate(ids):
        row=rows_by_id[jid];path=Path(row['resolved_observation'])
        if sha(path)!=row['observation_file_sha256']:raise ValueError('Observation file changed')
        wave=np.load(path,allow_pickle=False);digest=hashlib.sha256(wave.tobytes()).hexdigest()
        if wave.shape!=(32000,) or wave.dtype!=np.float32 or digest!=row['observation_sha256']:raise ValueError('Observation data changed')
        hashes.append(digest)
        if row['reused']:
            p=old_pos[row['parent_job_id']]
            if inherited_hashes[p]!=digest:raise ValueError('Parent feature waveform mismatch')
            for rep in REPS:matrices[rep][i]=inherited[rep][p]
        else:new.append(i)
    if len(new)!=9600:raise ValueError('New observation count')
    for lo in range(0,len(new),8):
        positions=new[lo:lo+8]
        wave=np.stack([np.load(rows_by_id[ids[j]]['resolved_observation'],allow_pickle=False) for j in positions])
        encoded=embed(encoder,torch.from_numpy(wave)).numpy()
        if lo==0:np.testing.assert_array_equal(encoded,embed(encoder,torch.from_numpy(wave)).numpy())
        matrices['BEATs768'][positions]=encoded
        matrices['MFCC26'][positions]=np.stack([mfcc(x,BASE['mfcc']) for x in wave])
        if lo%800==0:print('attenuation/latency features',lo+len(positions),'/9600',flush=True)
    if any(not np.isfinite(x).all() for x in matrices.values()):raise ValueError('Nonfinite features')
    np.savez_compressed(CACHE/'features.npz',job_ids=np.array(ids),waveform_sha256=np.array(hashes),**matrices)
    save(CACHE/'summary.json',dict(passed=True,events=19200,new_events=9600,reused_events=9600,
        encoder_frozen=True,encoder_provenance=provenance,target_access=False,
        execution_lock_sha256=sha(EXECUTION/'lock.json'),corpus_verification_sha256=sha(CORPUS/'verification.json'),
        features_sha256=sha(CACHE/'features.npz'),elapsed_s=time.perf_counter()-begin))


if __name__=='__main__':main()

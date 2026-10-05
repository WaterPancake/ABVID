"""Paired generation and exact full-waveform replay without duplicate masters."""
import argparse
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import multiprocessing
import resource
import shutil
import time
from phase_common import *
from phase_renderer import render_all
from renderer import exact_trajectory,observation
from source_adapter import array_sha


def group_ids(stage):
    table='timing_slots.jsonl' if stage=='pilot' else 'geometries.jsonl'
    return {r['geometry_id'] for r in rows(parent.PARENT_DESIGN/table)}


def worker(payload):
    gid,g,ids,waves=payload
    return gid,ids,render_all(waves,exact_trajectory(g),np.array(g['mic_xyz_m']))


def jobs(groups):
    source_records=rows(parent.PARENT/'source_manifest.jsonl')
    waves={r['dry_source_id']:np.load(parent.PARENT/r['path'],allow_pickle=False) for r in source_records}
    for r in source_records:
        if array_sha(waves[r['dry_source_id']])!=r['waveform_sha256']:raise ValueError('Source changed')
    geometry={r['geometry_id']:r for r in rows(parent.PARENT_DESIGN/'geometries.jsonl')}
    members=defaultdict(set)
    for row in rows(FREEZE/'observations.jsonl'):
        if row['geometry_id'] in groups:members[row['geometry_id']].add(row['dry_source_id'])
    keys=sorted(members)
    def payload(key):
        ids=sorted(members[key]);return key,geometry[key],ids,np.column_stack([waves[i] for i in ids])
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={i:pool.submit(worker,payload(key)) for i,key in enumerate(keys[:4])}
        for i,key in enumerate(keys):
            value=pending.pop(i).result()
            if i+4<len(keys):pending[i+4]=pool.submit(worker,payload(keys[i+4]))
            yield value


def generate(out,stage):
    check_design()
    if stage=='full':check_execution()
    if stage=='full' and shutil.disk_usage(ROOT).free<2_000_000_000:raise ValueError('Need 2 GB free')
    out.mkdir(parents=True,exist_ok=False);(out/'observations').mkdir()
    begin=time.perf_counter();groups=group_ids(stage);plans=defaultdict(list)
    for row in rows(FREEZE/'observations.jsonl'):
        if row['geometry_id'] in groups:plans[row['geometry_id']].append(row)
    original={r['job_id']:r for r in parent.joined_observations()}
    geometry={r['geometry_id']:r for r in rows(parent.PARENT_DESIGN/'geometries.jsonl')}
    records=[];endpoints=0;gid=None
    try:
        with (out/'observations.jsonl').open('w') as manifest:
            for i,(gid,ids,rendered) in enumerate(jobs(groups)):
                pending=[]
                for row in plans[gid]:
                    full=rendered[row['cell']][:,ids.index(row['dry_source_id'])]
                    wave,meta=observation(full,geometry[gid],BASE)
                    full_hash=array_sha(full);wave_hash=__import__('hashlib').sha256(wave.tobytes()).hexdigest()
                    if row['reused']:
                        ref=original[row['parent_job_id']]
                        if full_hash!=ref['rendered_waveform_sha256'] or wave_hash!=ref['observation_sha256']:
                            raise ValueError('Endpoint changed: '+row['job_id'])
                        endpoints+=1
                    else:pending.append((row,wave,meta,full_hash,wave_hash))
                if len(pending)!=8:raise ValueError('Incomplete new paired geometry')
                for row,wave,meta,full_hash,wave_hash in pending:
                    path=out/'observations'/(row['job_id']+'.npy');np.save(path,wave,allow_pickle=False)
                    record=dict(row,**meta,rendered_waveform_sha256=full_hash,
                        full_render_samples=80000,full_render_dtype='float64',full_render_array_retained=False,
                        observation_sha256=wave_hash,observation_path=str(path.relative_to(out)),
                        observation_file_sha256=sha(path))
                    records.append(record);manifest.write(jsonl([record]))
                manifest.flush()
                save(out/'progress.json',dict(status='running',geometries=i+1,total_geometries=len(groups),
                    observations=len(records),endpoints_identical=endpoints,elapsed_s=time.perf_counter()-begin))
                if i%20==0:print('attenuation/latency generation',i+1,'/',len(groups),flush=True)
    except Exception as error:
        save(out/'failure.json',dict(geometry=gid,error=str(error),redraws=0));raise
    summary=dict(passed=True,stage=stage,observations=len(records),endpoints_identical=endpoints,
        geometries=len(groups),failed_geometries=0,redraws=0,elapsed_s=time.perf_counter()-begin,
        completed_utc=datetime.now(timezone.utc).isoformat(),design_lock_sha256=sha(FREEZE/'lock.json'),
        execution_lock_sha256=sha(EXECUTION/'lock.json') if stage=='full' else None,
        manifest_sha256=sha(out/'observations.jsonl'),max_peak=max(r['final_peak'] for r in records),
        peak_parent_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        complete_new_render_arrays_retained=False,complete_new_render_hashes_retained=True)
    save(out/'summary.json',summary);save(out/'progress.json',dict(summary,status='completed'));print(summary,flush=True)


def verify(out):
    check_design();summary=read(out/'summary.json');begin=time.perf_counter()
    if not summary['passed'] or (out/'failure.json').exists():raise ValueError('Incomplete corpus')
    if (out/'verification.json').exists():raise ValueError('Preserve existing verification')
    if summary['manifest_sha256']!=sha(out/'observations.jsonl'):raise ValueError('Manifest changed')
    groups=group_ids(summary['stage']);records=rows(out/'observations.jsonl')
    planned=[r for r in rows(FREEZE/'observations.jsonl') if not r['reused'] and r['geometry_id'] in groups]
    byid={r['job_id']:r for r in records}
    if len(records)!=len(planned) or set(byid)!={r['job_id'] for r in planned}:raise ValueError('Population mismatch')
    grouped=defaultdict(list)
    for plan in planned:
        row=byid[plan['job_id']]
        for key,value in plan.items():
            if row[key]!=value:raise ValueError('Role/parameter changed')
        path=out/row['observation_path'];wave=np.load(path,allow_pickle=False)
        if sha(path)!=row['observation_file_sha256'] or wave.shape!=(32000,) or wave.dtype!=np.float32:
            raise ValueError('Observation changed')
        if __import__('hashlib').sha256(wave.tobytes()).hexdigest()!=row['observation_sha256']:
            raise ValueError('Observation data changed')
        grouped[row['geometry_id']].append(row)
    geometry={r['geometry_id']:r for r in rows(parent.PARENT_DESIGN/'geometries.jsonl')}
    count=0
    for i,(gid,ids,rendered) in enumerate(jobs(groups)):
        for row in grouped[gid]:
            full=rendered[row['cell']][:,ids.index(row['dry_source_id'])]
            if array_sha(full)!=row['rendered_waveform_sha256']:raise ValueError('Complete float64 replay mismatch')
            wave,meta=observation(full,geometry[gid],BASE)
            np.testing.assert_array_equal(wave,np.load(out/row['observation_path'],allow_pickle=False))
            if abs(meta['crop_float32_rms']/10**(-26/20)-1)>1e-5:raise ValueError('RMS replay failed')
            count+=1
        if i%20==0:print('attenuation/latency verified replay',i+1,'/',len(groups),flush=True)
    result=dict(passed=True,observations=count,complete_waveform_replays=count,
        exact_observation_replays=count,endpoint_equality_checks=summary['endpoints_identical'],
        roles=dict(Counter(r['role'] for r in records)),summary_sha256=sha(out/'summary.json'),
        verifier_sha256=sha(Path(__file__)),elapsed_s=time.perf_counter()-begin)
    save(out/'verification.json',result);print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['generate','verify'])
    ap.add_argument('--population',choices=['pilot','full'],default='full');ap.add_argument('--output',type=Path)
    a=ap.parse_args();out=a.output or CORPUS
    if a.stage=='generate':generate(out,a.population)
    else:verify(out)

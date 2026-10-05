"""Complete paired generation, endpoint equality and exact fresh-process replay."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
from collections import defaultdict,Counter
import multiprocessing
import hashlib
import resource
import time
from common import *
from render import render_all,exact_trajectory,observation
from source_adapter import array_sha


def worker(payload):
    gid,geometry,ids,waves=payload
    result=render_all(waves,exact_trajectory(geometry),np.array(geometry['mic_xyz_m']))
    return gid,ids,result


def jobs(selected,workers=4):
    source_rows=rows(PARENT/'source_manifest.jsonl')
    waves={r['dry_source_id']:np.load(PARENT/r['path'],allow_pickle=False) for r in source_rows}
    for r in source_rows:
        if array_sha(waves[r['dry_source_id']])!=r['waveform_sha256']:raise ValueError('Parent source changed')
    geometries={r['geometry_id']:r for r in rows(PARENT_DESIGN/'geometries.jsonl')}
    grouped=defaultdict(set)
    for r in rows(FREEZE/'observations.jsonl'):
        if r['geometry_id'] in selected:grouped[r['geometry_id']].add(r['dry_source_id'])
    keys=sorted(grouped)
    def payload(key):
        ids=sorted(grouped[key]);return key,geometries[key],ids,np.column_stack([waves[i] for i in ids])
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={i:pool.submit(worker,payload(key)) for i,key in enumerate(keys[:workers])}
        for i,key in enumerate(keys):
            result=pending.pop(i).result()
            if i+workers<len(keys):pending[i+workers]=pool.submit(worker,payload(keys[i+workers]))
            yield result


def selected_groups(stage):
    table='timing_slots.jsonl' if stage=='pilot' else 'geometries.jsonl'
    return {r['geometry_id'] for r in rows(PARENT_DESIGN/table)}


def generate(out,stage):
    check_design()
    if stage=='full':check_execution()
    if __import__('shutil').disk_usage(ROOT).free<8_500_000_000 and stage=='full':raise ValueError('Need 8.5 GB free')
    out.mkdir(parents=True,exist_ok=False)
    for name in ['renders','observations']:(out/name).mkdir()
    begin=time.perf_counter();groups=selected_groups(stage)
    plans=defaultdict(list)
    for row in rows(FREEZE/'observations.jsonl'):
        if row['geometry_id'] in groups:plans[row['geometry_id']].append(row)
    original={r['job_id']:r for r in rows(PARENT/'observation_manifest.jsonl')}
    geometries={r['geometry_id']:r for r in rows(PARENT_DESIGN/'geometries.jsonl')}
    all_rows=[];endpoints=0;gid=None
    try:
        with (out/'observation_manifest.jsonl').open('w') as manifest:
            for i,(gid,ids,waves) in enumerate(jobs(groups)):
                position={v:j for j,v in enumerate(ids)};pending=[]
                for row in plans[gid]:
                    full=waves[row['path_level']][:,position[row['dry_source_id']]]
                    y,meta=observation(full,geometries[gid],BASE)
                    h=array_sha(full);yh=hashlib.sha256(y.tobytes()).hexdigest()
                    if row['reused']:
                        ref=original[row['parent_job_id']]
                        if h!=ref['rendered_waveform_sha256'] or yh!=ref['observation_sha256']:
                            raise ValueError('Frozen endpoint mismatch: '+row['job_id'])
                        endpoints+=1
                    else:pending.append((row,full,y,meta,h,yh))
                if len(pending)!=8:raise ValueError('Incomplete new paired geometry')
                for row,full,y,meta,h,yh in pending:
                    rp=out/'renders'/(row['job_id']+'.npy');op=out/'observations'/(row['job_id']+'.npy')
                    np.save(rp,full,allow_pickle=False);np.save(op,y,allow_pickle=False)
                    record=dict(row,**meta,render_path=str(rp.relative_to(out)),observation_path=str(op.relative_to(out)),
                        rendered_waveform_sha256=h,observation_sha256=yh,render_file_sha256=sha(rp),observation_file_sha256=sha(op))
                    all_rows.append(record);manifest.write(jsonl([record]))
                manifest.flush()
                save(out/'progress.json',dict(status='running',geometries=i+1,total_geometries=len(groups),
                    observations=len(all_rows),endpoints_identical=endpoints,elapsed_s=time.perf_counter()-begin))
                if i%20==0:print('ground/air generation',i+1,'/',len(groups),flush=True)
    except Exception as error:
        save(out/'failure.json',dict(geometry=gid,error=str(error),redraws=0));raise
    summary=dict(passed=True,stage=stage,observations=len(all_rows),endpoints_identical=endpoints,
        geometries=len(groups),failed_geometries=0,redraws=0,elapsed_s=time.perf_counter()-begin,
        completed_utc=datetime.now(timezone.utc).isoformat(),design_lock_sha256=sha(FREEZE/'lock.json'),
        execution_lock_sha256=sha(EXECUTION/'lock.json') if stage=='full' else None,
        manifest_sha256=sha(out/'observation_manifest.jsonl'),
        max_peak=max(r['final_peak'] for r in all_rows),
        peak_parent_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    save(out/'summary.json',summary);save(out/'progress.json',dict(summary,status='completed'));print(summary,flush=True)


def verify(out):
    check_design();summary=read(out/'summary.json');begin=time.perf_counter()
    if not summary['passed'] or (out/'failure.json').exists():raise ValueError('Incomplete run')
    if sha(out/'observation_manifest.jsonl')!=summary['manifest_sha256']:raise ValueError('Manifest changed')
    new=rows(out/'observation_manifest.jsonl');byid={r['job_id']:r for r in new}
    groups=selected_groups(summary['stage']);expected=[r for r in rows(FREEZE/'observations.jsonl') if not r['reused'] and r['geometry_id'] in groups]
    if {r['job_id'] for r in expected}!=set(byid) or len(new)!=len(expected):raise ValueError('Population mismatch')
    geometries={r['geometry_id']:r for r in rows(PARENT_DESIGN/'geometries.jsonl')}
    for plan in expected:
        r=byid[plan['job_id']]
        for key,value in plan.items():
            if r[key]!=value:raise ValueError('Role/pairing mismatch')
        full=np.load(out/r['render_path'],allow_pickle=False);y=np.load(out/r['observation_path'],allow_pickle=False)
        if sha(out/r['render_path'])!=r['render_file_sha256'] or sha(out/r['observation_path'])!=r['observation_file_sha256']:
            raise ValueError('File bytes changed')
        if full.shape!=(80000,) or full.dtype!=np.float64 or y.shape!=(32000,) or y.dtype!=np.float32:
            raise ValueError('Shape/dtype mismatch')
        replay,meta=observation(full,geometries[r['geometry_id']],BASE)
        np.testing.assert_array_equal(y,replay)
        if abs(meta['crop_float32_rms']/10**(-26/20)-1)>1e-5:raise ValueError('RMS mismatch')
    group_rows=defaultdict(list)
    for r in new:group_rows[r['geometry_id']].append(r)
    count=0
    for i,(gid,ids,waves) in enumerate(jobs(groups)):
        for r in group_rows[gid]:
            if array_sha(waves[r['path_level']][:,ids.index(r['dry_source_id'])])!=r['rendered_waveform_sha256']:
                raise ValueError('Fresh-process replay failed')
            count+=1
        if i%20==0:print('ground/air verified replay',i+1,'/',len(groups),flush=True)
    report=dict(passed=True,observations=count,complete_waveform_replays=count,
        endpoint_equality_checks=summary['endpoints_identical'],roles=dict(Counter(r['role'] for r in new)),
        summary_sha256=sha(out/'summary.json'),elapsed_s=time.perf_counter()-begin,verifier_sha256=sha(Path(__file__)))
    if (out/'verification.json').exists():raise ValueError('Verification exists')
    save(out/'verification.json',report);print(report,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['generate','verify']);ap.add_argument('--stage',choices=['pilot','full'],default='full')
    ap.add_argument('--output',type=Path);args=ap.parse_args();out=args.output or CORPUS
    if args.action=='generate':generate(out,args.stage)
    else:verify(out)

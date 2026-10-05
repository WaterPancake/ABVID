"""Generate the frozen paired H2 bank; timing uses only prespecified validation slots."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
import platform
from pathlib import Path
import resource
import shutil
import time

import numpy as np

from metadata import ROOT, HERE, read, rows, save, sha, jsonl, validate
from renderer import exact_trajectory, build_plan, render_sources, observation
from source_adapter import synthesize, array_sha


def write_npy(path, value):
    np.save(path,value,allow_pickle=False)
    return sha(path)


def checked_execution(path):
    lock=read(path)
    if lock['ready_for_bulk_generation'] is not True: raise ValueError('Execution not admitted')
    for name,digest in lock['code_sha256'].items():
        if sha(ROOT/name)!=digest: raise ValueError('Executable changed: '+name)
    for name,digest in lock['evidence_sha256'].items():
        if sha(ROOT/name)!=digest: raise ValueError('Gate evidence changed: '+name)
    versions={name.lower():version for name,version in lock['environment'].items()}
    for package in ['numpy','scipy','numba','llvmlite','scikit-learn','torch','torchaudio','librosa','soundfile','matplotlib','joblib']:
        if importlib.metadata.version(package)!=versions[package]: raise ValueError('Pinned dependency changed: '+package)
    if platform.python_version()!=lock['python']: raise ValueError('Pinned Python changed')
    return lock


def render_geometry(payload):
    """Worker with no filesystem writes; return all paired outputs together."""
    g,rr,ids,sources=payload; start=time.perf_counter()
    column={sid:i for i,sid in enumerate(ids)}
    trajectory=exact_trajectory(g);plan=build_plan(trajectory,np.array(g['mic_xyz_m']))
    rendered=render_sources(sources,plan)
    return dict(trajectory=trajectory,rendered=rendered,column=column,setup_s=plan['setup_s'],worker_s=time.perf_counter()-start)


def ordered_render_jobs(grouped,geometries,cache,workers):
    """At most workers full scenes in flight; results remain in frozen ID order."""
    gids=sorted(grouped)
    def payload(gid):
        rr=grouped[gid]
        if len(rr)!=8: raise ValueError('Incomplete paired geometry')
        ids=sorted({r['dry_source_id'] for r in rr})
        return geometries[gid],rr,ids,np.column_stack([cache[sid] for sid in ids])
    if workers==1:
        for gid in gids: yield gid,render_geometry(payload(gid))
        return
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={i:pool.submit(render_geometry,payload(gids[i])) for i in range(min(workers,len(gids)))}
        for i,gid in enumerate(gids):
            try:
                result=pending.pop(i).result()
            except Exception as error:
                raise RuntimeError('Rendering failed for '+gid+': '+str(error)) from error
            if i+workers<len(gids): pending[i+workers]=pool.submit(render_geometry,payload(gids[i+workers]))
            yield gid,result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['timing','full'],required=True)
    ap.add_argument('--output',type=Path,required=True); ap.add_argument('--execution-lock',type=Path)
    ap.add_argument('--workers',type=int,choices=[1,4],default=4)
    args=ap.parse_args(); begin=time.perf_counter()
    if args.stage=='full':
        if args.execution_lock is None: raise ValueError('Bulk generation requires an execution lock')
        checked_execution(args.execution_lock)
    freeze=HERE/'frozen/source_path_20261004_v1'; cfg=read(freeze/'config.json')
    names=['templates','dry_source_plan','geometries','events','render_plan','fit_plan','timing_slots']
    tables={name:rows(freeze/(name+'.jsonl')) for name in names}
    validate(cfg,tables,rows(freeze/'target_manifest.jsonl'))
    for rel in ['source_validation_20261004_v2','renderer_validation_20261004_v2']:
        if not read(HERE/'results'/rel/'report.json')['passed']: raise ValueError('Physical/source gate failed')
    selected={r['geometry_id'] for r in tables['timing_slots']} if args.stage=='timing' else {g['geometry_id'] for g in tables['geometries']}
    jobs=[r for r in tables['render_plan'] if r['geometry_id'] in selected]
    template_ids={r['template_id'] for r in jobs}; templates=[t for t in tables['templates'] if t['template_id'] in template_ids]
    if args.stage=='timing' and any(t['role']!='validation' for t in templates): raise ValueError('Timing role violation')
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    for directory in ['sources','components','renders','observations']: (out/directory).mkdir()
    for name in ['generate_corpus.py','renderer.py','source_adapter.py','deterministic_filters.py']:
        shutil.copy2(HERE/name,out/('executed_'+name))
    source_records=[]; cache={}; times=[]; records=[]; geometries={g['geometry_id']:g for g in tables['geometries']}
    for t in templates:
        for level in ['S0','S1']:
            waveform,meta,components,diagnostics=synthesize(t,level,cfg,True)
            sid=t['template_id']+'.'+level; path=out/'sources'/(sid+'.npy')
            component_path=out/'components'/(sid+'.npz')
            payload=dict(components)
            for key in ['exhaust_broadband','exhaust_harmonics']:
                if key in diagnostics: payload[key]=diagnostics[key]
            np.savez_compressed(component_path,**payload)
            meta.update(dry_source_id=sid,source_parent_group=t['source_parent_group'],role=t['role'],
                class_id=t['class_id'],replicate_seed=t['replicate_seed'],
                path=str(path.relative_to(out)),file_sha256=write_npy(path,waveform),
                components_path=str(component_path.relative_to(out)),components_file_sha256=sha(component_path))
            source_records.append(meta); cache[sid]=waveform
    (out/'source_manifest.jsonl').write_text(jsonl(source_records))
    grouped=defaultdict(list)
    for job in jobs: grouped[job['geometry_id']].append(job)
    gid='before_first_geometry'
    try:
        with (out/'observation_manifest.jsonl').open('w') as manifest:
            rendering_started=time.perf_counter()
            for gi,(gid,bundle) in enumerate(ordered_render_jobs(grouped,geometries,cache,args.workers)):
                start=time.perf_counter(); g=geometries[gid]; rr=grouped[gid]
                column=bundle['column'];trajectory=bundle['trajectory'];rendered=bundle['rendered']
                # Validate all eight observations before writing any member.
                pending=[]
                for row in rr:
                    full=rendered[row['path_level']][:,column[row['dry_source_id']]]
                    try:
                        processed,meta=observation(full,g,cfg)
                        if abs(meta['crop_float32_rms']/10**(-26/20)-1)>1e-5:
                            raise ValueError('Float32 receiver RMS failed')
                    except Exception:
                        (out/'failures').mkdir(exist_ok=True)
                        np.savez(out/'failures'/(gid+'.npz'),P0=rendered['P0'],P1=rendered['P1'],trajectory=trajectory)
                        raise
                    pending.append((row,full,processed,meta))
                for row,full,processed,meta in pending:
                    rp=out/'renders'/(row['job_id']+'.npy'); op=out/'observations'/(row['job_id']+'.npy')
                    record=dict(row,**meta)
                    record.update(status='generated',source_waveform_sha256=array_sha(cache[row['dry_source_id']]),
                        rendered_waveform_sha256=array_sha(full),render_path=str(rp.relative_to(out)),
                        render_file_sha256=write_npy(rp,full),observation_path=str(op.relative_to(out)),
                        observation_file_sha256=write_npy(op,processed),
                        observation_sha256=hashlib.sha256(processed.astype('<f4').tobytes()).hexdigest(),
                        geometry_sha256=hashlib.sha256(trajectory.astype('<f8').tobytes()).hexdigest(),
                        sample_rate_hz=16000,samples=32000,full_render_sample_rate_hz=8000,full_render_samples=80000)
                    records.append(record); manifest.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
                manifest.flush()
                times.append(dict(geometry_id=gid,setup_s=bundle['setup_s'],**rendered['timing'],
                    worker_s=bundle['worker_s'],write_s=time.perf_counter()-start,
                    total_s=bundle['worker_s']+time.perf_counter()-start))
                save(out/'progress.json',dict(completed_geometries=gi+1,total_geometries=len(grouped),observations=len(records),
                    elapsed_s=time.perf_counter()-begin,target_access=False,status='running'))
                if gi%10==0 or gi+1==len(grouped): print(f'{args.stage}: {gi+1}/{len(grouped)} geometries, {len(records)} observations, {time.perf_counter()-begin:.1f}s',flush=True)
    except Exception as error:
        save(out/'failure.json',dict(geometry_id=gid,reason=str(error),status='stopped_complete_paired_bank_no_redraw',
            completed_observations=len(records),target_access=False))
        raise
    save(out/'timings.json',times)
    rendering_s=time.perf_counter()-rendering_started
    summary=dict(stage=args.stage,completed_utc=datetime.now(timezone.utc).isoformat(),passed=True,workers=args.workers,
        design_lock_sha256=sha(freeze/'lock.json'),execution_lock_sha256=sha(args.execution_lock) if args.execution_lock else None,
        templates=len(templates),dry_sources=len(source_records),geometries=len(grouped),observations=len(records),
        source_manifest_sha256=sha(out/'source_manifest.jsonl'),observation_manifest_sha256=sha(out/'observation_manifest.jsonl'),
        full_render_dtype='float64',observation_dtype='float32',target_access=False,failed_geometries=0,redraws=0,
        elapsed_s=time.perf_counter()-begin,rendering_wall_s=rendering_s,
        peak_parent_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        output_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
        mean_geometry_s=float(np.mean([t['total_s'] for t in times])),
        estimated_1200_geometry_s=float(rendering_s/len(grouped)*1200),
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [HERE/'generate_corpus.py',HERE/'renderer.py',HERE/'source_adapter.py',HERE/'deterministic_filters.py']})
    save(out/'summary.json',summary); save(out/'progress.json',dict(status='completed',**summary))
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()

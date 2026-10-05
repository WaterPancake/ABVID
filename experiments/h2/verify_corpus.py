"""Hash, role, pairing and fresh-process replay checks for H2 synthetic artifacts."""
import argparse
from collections import Counter,defaultdict
import hashlib
from pathlib import Path
import time

import numpy as np
from scipy.signal import resample_poly

from metadata import HERE,ROOT,rows,read,save,sha
from source_adapter import synthesize,array_sha
from renderer import exact_trajectory,build_plan,render_sources
from generate_corpus import ordered_render_jobs


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--corpus',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--replay-geometries',choices=['all','timing'],default='all')
    ap.add_argument('--workers',type=int,choices=[1,4],default=4)
    args=ap.parse_args()
    if args.output.exists(): raise ValueError('Verification output already exists')
    started=time.perf_counter(); corpus=args.corpus; summary=read(corpus/'summary.json')
    if not summary['passed'] or (corpus/'failure.json').exists(): raise ValueError('Incomplete or failed corpus')
    for name,digest in summary['code_sha256'].items():
        if sha(ROOT/name)!=digest: raise ValueError('Generation code changed: '+name)
    freeze=HERE/'frozen/source_path_20261004_v1'; cfg=read(freeze/'config.json')
    templates={t['template_id']:t for t in rows(freeze/'templates.jsonl')}
    geometries={g['geometry_id']:g for g in rows(freeze/'geometries.jsonl')}
    timing={r['geometry_id'] for r in rows(freeze/'timing_slots.jsonl')}
    expected_jobs=rows(freeze/'render_plan.jsonl')
    if summary['stage']=='timing': expected_jobs=[r for r in expected_jobs if r['geometry_id'] in timing]
    if sha(corpus/'source_manifest.jsonl')!=summary['source_manifest_sha256']: raise ValueError('Source manifest changed')
    if sha(corpus/'observation_manifest.jsonl')!=summary['observation_manifest_sha256']: raise ValueError('Observation manifest changed')
    sources=rows(corpus/'source_manifest.jsonl'); observations=rows(corpus/'observation_manifest.jsonl')
    if len(observations)!=len(expected_jobs) or {r['job_id'] for r in observations}!={r['job_id'] for r in expected_jobs}:
        raise ValueError('Generated job population does not match freeze')
    if len(sources)!=summary['dry_sources'] or len({r['dry_source_id'] for r in sources})!=len(sources): raise ValueError('Source population mismatch')
    by_job={r['job_id']:r for r in observations}; source_index={}; waves={}; component_ids={}
    for record in sources:
        sid=record['dry_source_id']; t=templates[record['template_id']]
        wave=np.load(corpus/record['path'],allow_pickle=False)
        if sha(corpus/record['path'])!=record['file_sha256'] or array_sha(wave)!=record['waveform_sha256']:
            raise ValueError('Source bytes changed: '+sid)
        replay,meta,components,diagnostics=synthesize(t,record['source_level'],cfg,True)
        if array_sha(replay)!=record['waveform_sha256']: raise ValueError('Source replay failed: '+sid)
        if record['role']!=t['role'] or wave.shape!=(80000,) or wave.dtype!=np.float64: raise ValueError('Source role/shape/dtype')
        if abs(np.sqrt(np.mean(wave[32000:48000]**2))/.1-1)>1e-6: raise ValueError('Source RMS')
        if sha(corpus/record['components_path'])!=record['components_file_sha256']: raise ValueError('Components changed')
        with np.load(corpus/record['components_path'],allow_pickle=False) as saved:
            for key,value in components.items():
                if not np.array_equal(saved[key],value): raise ValueError('Component replay: '+sid+' '+key)
            component_ids[sid]={'tire':array_sha(saved['tire'])}
            if 'exhaust_broadband' in saved: component_ids[sid]['exhaust_broadband']=array_sha(saved['exhaust_broadband'])
        source_index[sid]=record; waves[sid]=wave
    for tid in {r['template_id'] for r in sources}:
        if component_ids[tid+'.S0']!=component_ids[tid+'.S1']: raise ValueError('Noise realization changed with S')
    grouped=defaultdict(list);max_rms_error=0.;max_peak=0.; role_counts=Counter()
    for expected in expected_jobs:
        record=by_job[expected['job_id']]; sid=record['dry_source_id']; g=geometries[record['geometry_id']]
        for key in ['event_id','template_id','source_parent_group','geometry_id','cell','class_id','replicate_seed','role','source_level','path_level','dry_source_id']:
            if record[key]!=expected[key]: raise ValueError('Pairing/role changed: '+key)
        if record['source_waveform_sha256']!=source_index[sid]['waveform_sha256']: raise ValueError('Dry source differs across P/geometry')
        full=np.load(corpus/record['render_path'],allow_pickle=False)
        obs=np.load(corpus/record['observation_path'],allow_pickle=False)
        for kind in ['render','observation']:
            if sha(corpus/record[kind+'_path'])!=record[kind+'_file_sha256']: raise ValueError('Waveform file changed')
        if array_sha(full)!=record['rendered_waveform_sha256'] or hashlib.sha256(obs.astype('<f4').tobytes()).hexdigest()!=record['observation_sha256']:
            raise ValueError('Waveform data changed')
        if full.shape!=(80000,) or full.dtype!=np.float64 or obs.shape!=(32000,) or obs.dtype!=np.float32: raise ValueError('Waveform dimensions/dtype')
        if not np.isfinite(full).all() or not np.isfinite(obs).all(): raise ValueError('Nonfinite samples')
        raw=full[g['crop_start_sample_8k']:g['crop_end_sample_8k']]
        crop=raw*(10**(-26/20)/np.sqrt(np.mean(raw**2)))
        rms_error=abs(np.sqrt(np.mean(crop.astype(np.float32).astype(np.float64)**2))/10**(-26/20)-1)
        peak=float(abs(obs).max());max_rms_error=max(max_rms_error,float(rms_error));max_peak=max(max_peak,peak)
        if rms_error>1e-5 or peak>.98: raise ValueError('Observation gate')
        independent=resample_poly(crop,2,1,window=('kaiser',5.),padtype='constant').astype(np.float32)
        if not np.array_equal(independent,obs): raise ValueError('Observation preprocessing changed')
        grouped[g['geometry_id']].append(record);role_counts[record['role']]+=1
    replayed=0;replay_groups={}
    for gid,rr in sorted(grouped.items()):
        if len(rr)!=8 or {(r['class_id'],r['cell']) for r in rr}!={(label,cell) for label in [0,1] for cell in cfg['cells']}:
            raise ValueError('Incomplete class/cell geometry')
        trajectory=exact_trajectory(geometries[gid]); digest=array_sha(trajectory)
        if {r['geometry_sha256'] for r in rr}!={digest}: raise ValueError('Class/factor-dependent geometry')
        if args.replay_geometries=='timing' and gid not in timing: continue
        replay_groups[gid]=rr
    for gid,bundle in ordered_render_jobs(replay_groups,geometries,waves,args.workers):
        rr=replay_groups[gid];position=bundle['column'];result=bundle['rendered']
        for r in rr:
            if array_sha(result[r['path_level']][:,position[r['dry_source_id']]])!=r['rendered_waveform_sha256']:
                raise ValueError('Fresh-process full waveform replay failed: '+r['job_id'])
        replayed+=1
        if replayed%10==0: print('verified replay geometries',replayed,flush=True)
    report=dict(passed=True,stage=summary['stage'],corpus=str(corpus),corpus_summary_sha256=sha(corpus/'summary.json'),
        source_replays=len(sources),observations_checked=len(observations),geometry_groups=len(grouped),
        geometry_replays=replayed,complete_waveform_replays=replayed*8,replay_scope=args.replay_geometries,workers=args.workers,
        role_counts=dict(role_counts),max_float32_crop_rms_relative_error=max_rms_error,max_observation_peak=max_peak,
        target_access=False,elapsed_s=time.perf_counter()-started,checker_sha256=sha(Path(__file__)))
    save(args.output,report);print(report,flush=True)


if __name__=='__main__': main()

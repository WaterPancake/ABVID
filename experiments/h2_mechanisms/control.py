"""Freeze the metadata first, then admit code only after synthetic-only checks."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import subprocess
from collections import Counter

from common import *


def freeze():
    check_parent()
    if not read(ROOT/CONFIG['parent_results']/'verification.json')['passed']:
        raise ValueError('Verified parent H2 required')
    FREEZE.mkdir(parents=True,exist_ok=False)
    originals=rows(PARENT/'observation_manifest.jsonl')
    by_key={(r['event_id'],r['source_level'],r['path_level']):r for r in originals}
    plans=[]
    for row in originals:
        if row['path_level']!='P0':continue
        for path in PATHS:
            reused=path in CONFIG['endpoint_mapping']
            parent=by_key[(row['event_id'],row['source_level'],CONFIG['endpoint_mapping'][path])] if reused else None
            keep={k:row[k] for k in ['event_id','geometry_id','template_id','dry_source_id','source_parent_group',
                'source_level','role','class_id','replicate_seed']}
            plans.append(dict(keep,job_id=mapped_id(row,path),path_level=path,
                ground=path[1]=='1',air=path[3]=='1',reused=reused,
                parent_job_id=parent['job_id'] if parent else None))
    if len(plans)!=19200 or len({r['job_id'] for r in plans})!=19200:raise ValueError('Incomplete plan')
    source_roles={}
    for row in plans:source_roles.setdefault(row['source_parent_group'],set()).add(row['role'])
    if any(len(x)!=1 for x in source_roles.values()):raise ValueError('Parent role leakage')
    fits=[];plan_index={r['job_id']:r for r in plans};old_fits=read(ROOT/CONFIG['parent_models']/'fits.json')
    old_fit_index={r['fit_id']:r for r in old_fits}
    old_jobs={r['job_id']:r for r in originals}
    for seed in SEEDS:
        for source in SOURCES:
            for path in PATHS:
                old_cell=('F00' if source=='S0' else 'F10') if path!='G1A1' else ('F01' if source=='S0' else 'F11')
                for rep in REPS:
                    old=old_fit_index[f'h2.r{seed}.{old_cell}.{rep}']
                    plan=dict(fit_id=f'h2m.r{seed}.{source}.{path}.{rep}',replicate_seed=seed,
                        source_level=source,path_level=path,representation=rep,reused=path in CONFIG['endpoint_mapping'],
                        parent_fit_id=old['fit_id'] if path in CONFIG['endpoint_mapping'] else None,
                        train_job_ids=[mapped_id(old_jobs[j],path) for j in old['train_job_ids']],
                        validation_job_ids=[mapped_id(old_jobs[j],path) for j in old['validation_job_ids']])
                    if Counter(plan_index[j]['class_id'] for j in plan['train_job_ids'])!={0:190,1:190}:
                        raise ValueError('Class budget changed')
                    fits.append(plan)
    (FREEZE/'observations.jsonl').write_text(jsonl(plans))
    (FREEZE/'fits.jsonl').write_text(jsonl(fits))
    for name in ['config.json','PROTOCOL.md']:shutil.copy2(HERE/name,FREEZE/name)
    shutil.copy2(PARENT_DESIGN/'target_manifest.jsonl',FREEZE/'target_manifest.jsonl')
    parents=[p for p in H2.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    parents.append(ROOT/'reports/H2_source_path_results.md')
    records=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(parents)]
    (FREEZE/'parent_artifacts.jsonl').write_text(jsonl(records))
    save(FREEZE/'lock.json',dict(protocol_id=ID,completed_utc=datetime.now(timezone.utc).isoformat(),
        config_sha256=sha(HERE/'config.json'),protocol_sha256=sha(HERE/'PROTOCOL.md'),
        observations=19200,new_observations=9600,heads=80,new_heads=40,target_events=8066,
        parent_files=len(records),target_already_exposed=True,
        artifacts_sha256={p.name:sha(p) for p in FREEZE.iterdir() if p.is_file()}))
    print('Frozen complete design:',FREEZE,flush=True)


def admit():
    check_design();protected=verify_parent_files()
    evidence=[HERE/'results/renderer_validation_v1/report.json',HERE/'results/pilot_v1/verification.json']
    for path in evidence:
        if not read(path)['passed']:raise ValueError('Synthetic admission check failed')
    test=subprocess.run([__import__('sys').executable,'-m','unittest','discover','-s',str(HERE),'-p','test_controls.py','-v'],
        cwd=ROOT,text=True,capture_output=True)
    if test.returncode:raise ValueError(test.stdout+test.stderr)
    EXECUTION.mkdir(parents=True,exist_ok=False)
    (EXECUTION/'tests.txt').write_text(test.stdout+test.stderr)
    code=list(HERE.glob('*.py'))
    for path in code:shutil.copy2(path,EXECUTION/path.name)
    save(EXECUTION/'lock.json',dict(protocol_id=ID,completed_utc=datetime.now(timezone.utc).isoformat(),
        design_lock_sha256=sha(FREEZE/'lock.json'),parent_execution_sha256=sha(ROOT/CONFIG['parent_execution']),
        ready_for_generation=True,target_evaluation_started=False,**protected,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in code},
        evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in evidence},tests_sha256=sha(EXECUTION/'tests.txt')))
    print('Admitted frozen execution:',EXECUTION,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','admit']);args=ap.parse_args()
    {'freeze':freeze,'admit':admit}[args.stage]()

"""Versioned paths and immutable parent interfaces for the H2 follow-up."""
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
H2=ROOT/'experiments/h2'
sys.path.insert(0,str(H2))
from metadata import read,rows,save,sha,jsonl
from generate_corpus import checked_execution

CONFIG=read(HERE/'config.json')
ID=CONFIG['protocol_id']
FREEZE=HERE/'frozen'/ID
EXECUTION=HERE/'execution'/ID
CORPUS=HERE/'corpus'/ID
CACHE=HERE/'cache'/ID
MODELS=HERE/'models'/ID
RESULTS=HERE/'results'/ID
PARENT=ROOT/CONFIG['parent_corpus']
PARENT_DESIGN=ROOT/CONFIG['parent_design']
BASE=read(PARENT_DESIGN/'config.json')
PATHS=CONFIG['path_levels']
SOURCES=CONFIG['source_levels']
REPS=CONFIG['representations']
SEEDS=CONFIG['seeds']


def check_parent():
    return checked_execution(ROOT/CONFIG['parent_execution'])


def check_design():
    lock=read(FREEZE/'lock.json')
    for name,digest in lock['artifacts_sha256'].items():
        if sha(FREEZE/name)!=digest: raise ValueError('Design changed: '+name)
    if sha(HERE/'config.json')!=lock['config_sha256'] or sha(HERE/'PROTOCOL.md')!=lock['protocol_sha256']:
        raise ValueError('Declared follow-up changed')
    check_parent()
    return lock


def check_execution():
    check_design();lock=read(EXECUTION/'lock.json')
    for name,digest in lock['code_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen follow-up code changed: '+name)
    for name,digest in lock['evidence_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Follow-up admission changed: '+name)
    if not lock['ready_for_generation']:raise ValueError('Execution not admitted')
    return lock


def mapped_id(row,path):
    return row['event_id']+'.'+row['source_level']+'.'+path


def verify_parent_files():
    records=rows(FREEZE/'parent_artifacts.jsonl')
    for row in records:
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Parent changed: '+row['path'])
    protected=read(PARENT_DESIGN/'protected_artifacts.json')
    for row in protected:
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Historical artifact changed: '+row['path'])
    return dict(parent_files=len(records),historical_files=len(protected))


def joined_observations():
    old={r['job_id']:r for r in rows(PARENT/'observation_manifest.jsonl')}
    new={r['job_id']:r for r in rows(CORPUS/'observation_manifest.jsonl')}
    result=[]
    for plan in rows(FREEZE/'observations.jsonl'):
        data=old[plan['parent_job_id']] if plan['reused'] else new[plan['job_id']]
        home=PARENT if plan['reused'] else CORPUS
        result.append(dict(data,**{k:v for k,v in plan.items()},
            resolved_observation=str(home/data['observation_path']),resolved_render=str(home/data['render_path'])))
    return result


def percentile_summary(values):
    x=np.asarray(values,dtype=float)
    return dict(mean=float(x.mean()),std=float(x.std()),
        quantiles=dict(zip(['min','p05','p25','median','p75','p95','max'],
            np.quantile(x,[0,.05,.25,.5,.75,.95,1]).tolist())))

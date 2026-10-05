"""Isolated attenuation/latency controls, referencing immutable parent artifacts."""
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT_HERE=ROOT/'experiments/h2_mechanisms'
sys.path.insert(0,str(PARENT_HERE))
import common as parent
from common import read,rows,save,sha,jsonl

CONFIG=read(HERE/'config.json')
ID=CONFIG['protocol_id']
FREEZE=HERE/'frozen'/ID
EXECUTION=HERE/'execution'/ID
CORPUS=HERE/'corpus'/ID
CACHE=HERE/'cache'/ID
MODELS=HERE/'models'/ID
RESULTS=HERE/'results'/ID
CELLS=CONFIG['cells']
SOURCES=CONFIG['source_levels']
SEEDS=CONFIG['seeds']
REPS=CONFIG['representations']
BASE=parent.BASE


def check_parent():
    parent.check_execution()
    audit=read(parent.RESULTS/'verification.json')
    if not audit['passed'] or audit['result_summary_sha256']!=sha(parent.RESULTS/'summary.json'):
        raise ValueError('Verified ground/air parent required')
    return audit


def check_design():
    check_parent()
    lock=read(FREEZE/'lock.json')
    for name,digest in lock['artifacts_sha256'].items():
        if sha(FREEZE/name)!=digest:raise ValueError('Design changed: '+name)
    for name in ['config.json','PROTOCOL.md']:
        if sha(HERE/name)!=lock['artifacts_sha256'][name]:raise ValueError('Declaration changed: '+name)
    return lock


def check_execution():
    check_design();lock=read(EXECUTION/'lock.json')
    for path,digest in lock['code_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Execution changed: '+path)
    for path,digest in lock['evidence_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Admission changed: '+path)
    return lock


def check_preserved():
    records=rows(FREEZE/'parent_artifacts.jsonl')
    for row in records:
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Ground/air parent changed: '+row['path'])
    return dict(ground_air_parent_files=len(records),**parent.verify_parent_files())


def job_id(row,cell):
    return row['event_id']+'.'+row['source_level']+'.'+cell


def joined_observations():
    old={r['job_id']:r for r in parent.joined_observations()}
    new={r['job_id']:r for r in rows(CORPUS/'observations.jsonl')}
    records=[]
    for plan in rows(FREEZE/'observations.jsonl'):
        record=old[plan['parent_job_id']] if plan['reused'] else new[plan['job_id']]
        records.append(dict(record,**plan,resolved_observation=(record['resolved_observation'] if plan['reused']
            else str(CORPUS/record['observation_path']))))
    return records

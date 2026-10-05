"""Small source-only diagnostic helpers; no target data paths accepted."""
from pathlib import Path
import json
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/h1'))
from freeze import sha,save,select,assert_separation


def source_path(row):
    if row.get('dataset_id')!='IDMT' or row.get('class_id') not in (0,1):
        raise ValueError('Diagnostic accepts admitted IDMT car/truck only')
    path=(ROOT/row['source_path']).resolve()
    allowed=(ROOT/'dataset/IDMT_Traffic/audio').resolve()
    if not path.is_relative_to(allowed):raise ValueError('Non-IDMT source path')
    return path


def read_lock(path):
    path=Path(path);lock=json.loads(path.read_text())
    for name,h in lock['artifacts_sha256'].items():
        if sha(path.parent/name)!=h:raise ValueError(f'Frozen artifact changed: {name}')
    cfg=json.loads((path.parent/'config.json').read_text())
    rows=[json.loads(s) for s in (path.parent/'source_manifest.jsonl').open()]
    for row in rows:source_path(row)
    assert len(rows)==4413 and all(r['admitted_for_training'] for r in rows)
    return lock,cfg,rows


def train_counts(ids,index):
    return [sum(index[i]['class_id']==c for i in ids) for c in (0,1)]

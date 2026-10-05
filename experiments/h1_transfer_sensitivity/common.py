"""Read-only artifact checks and binary metrics for the frozen transfer sensitivity."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'experiments/h1'))
from freeze import assert_separation


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')


def read_json(path): return json.loads(Path(path).read_text())
def read_rows(path): return [json.loads(s) for s in Path(path).read_text().splitlines()]


def check_artifacts(folder, summary):
    for name, h in summary['artifacts_sha256'].items():
        if sha(Path(folder)/name) != h: raise ValueError(f'Artifact changed: {folder}/{name}')


def load_lock(path):
    path = Path(path); lock = read_json(path)
    check_artifacts(path.parent, lock)
    cfg = read_json(path.parent/'config.json')
    rows = read_rows(path.parent/'manifest.jsonl')
    assert {r['dataset_id'] for r in rows} == {'IDMT', 'MELAUDIS'}
    assert {r['class_id'] for r in rows} == {0, 1}
    assert lock['target_role'] == cfg['target_role']
    return lock, cfg, {r['file_id']: r for r in rows}


def confusion(y, p, threshold=.5):
    return np.bincount(np.asarray(y, dtype=int)*2+(np.asarray(p) > threshold), minlength=4).reshape(2, 2)


def cm_scores(cm):
    cm = np.asarray(cm, dtype=float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support = cm.sum(axis=-1); predicted = cm.sum(axis=-2)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f = np.divide(2*tp, support+predicted, out=np.zeros_like(tp), where=(support+predicted)>0)
    return dict(macro_f1=f.mean(axis=-1), balanced_accuracy=recall.mean(axis=-1),
        accuracy=tp.sum(axis=-1)/cm.sum(axis=(-1,-2)), car_recall=recall[...,0], truck_recall=recall[...,1],
        car_precision=precision[...,0], truck_precision=precision[...,1], car_f1=f[...,0], truck_f1=f[...,1])


def score(y, p, threshold=.5):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    assert y.shape == p.shape and set(y) == {0,1}
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    cm = confusion(y,p,threshold); result = {k:float(v) for k,v in cm_scores(cm).items()}
    bins = np.minimum((p*10).astype(int),9)
    result.update(confusion_matrix=cm.tolist(), count=len(y), class_counts=np.bincount(y,minlength=2).tolist(),
        truck_prevalence=float(y.mean()), roc_auc=float(roc_auc_score(y,p)),
        average_precision=float(average_precision_score(y,p)), brier_truck=float(np.mean((p-y)**2)),
        log_loss=float(-np.log(np.clip(np.where(y==1,p,1-p),1e-15,1)).mean()),
        ece10_truck=float(sum(abs(float((p[bins==b]-y[bins==b]).sum()))/len(y) for b in range(10))))
    return result


def protected_check(path):
    records = read_json(path)
    for r in records:
        if sha(ROOT/r['path']) != r['sha256']: raise ValueError(f'Protected artifact changed: {r["path"]}')
    return len(records)

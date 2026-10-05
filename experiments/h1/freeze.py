"""Resolve the metadata-only H1 largest-common-budget lock before feature extraction."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n')


def stream_seed(seed, stream):
    return int(hashlib.sha256(f'h1_common_budget_v1.3|{seed}|{stream}'.encode()).hexdigest()[:16], 16)


def select(rows, n, seed, stream):
    result = []
    for label in [0, 1]:
        pools = {}
        for g in sorted(set(r['provenance_group_id'] for r in rows)):
            pool = sorted(r['file_id'] for r in rows if r['provenance_group_id'] == g and r['class_id'] == label)
            rng = np.random.default_rng(stream_seed(seed, f'{stream}|{g}|{label}'))
            pools[g] = list(rng.permutation(pool))
        # Deterministic round-robin avoids allowing the largest source date to
        # supply the whole car class. Exhausted small pools simply drop out.
        selected = []
        while len(selected) < n:
            before = len(selected)
            for g in sorted(pools):
                if pools[g] and len(selected) < n:
                    selected.append(str(pools[g].pop()))
            if len(selected) == before:
                raise ValueError('Insufficient support')
        result.extend(selected)
    assert len(set(result)) == 2*n
    return result


def assert_separation(train_ids, test_ids, index):
    if set(train_ids) & set(test_ids):
        raise ValueError('File leakage')
    for field in ('provenance_group_id', 'source_file_sha256', 'duplicate_group_id'):
        if {index[i][field] for i in train_ids} & {index[i][field] for i in test_ids}:
            raise ValueError(f'{field} leakage')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'experiments/h1/config.json').read_text())
    manifest=ROOT/config['admission']['manifest']
    audit=json.loads((manifest.parent/'summary.json').read_text())
    assert sha(manifest)==audit['artifacts_sha256']['admitted.jsonl']
    rows=[json.loads(s) for s in manifest.open()]
    index={r['file_id']:r for r in rows}
    idmt=[r for r in rows if r['dataset_id']=='IDMT']
    target=sorted(r['file_id'] for r in rows if r['dataset_id']=='MELAUDIS')
    groups=sorted(set(r['provenance_group_id'] for r in idmt))
    banks={'procedural':[r for r in rows if r['dataset_id']=='AI4TEN_pyroadacoustics'],
           'audioldm':[r for r in rows if r['dataset_id']=='AI4TEN_audioldm']}
    supports=[sum(r['class_id']==c for r in b) for b in banks.values() for c in (0,1)]
    supports += [sum(r['class_id']==c and r['provenance_group_id']!=g for r in idmt)
                 for g in groups for c in (0,1)]
    n=min(supports)
    assert n==config['budget']['examples_per_class'], (n,config['budget'])
    assert len(groups)==6 and len(target)==8066
    synthetic={arm:{str(seed):select(bank,n,seed,arm) for seed in config['seeds']}
               for arm,bank in banks.items()}
    folds=[]
    for fold,g in enumerate(groups):
        test=sorted(r['file_id'] for r in idmt if r['provenance_group_id']==g)
        pool=[r for r in idmt if r['provenance_group_id']!=g]
        trains={str(seed):select(pool,n,seed,f'IDMT|{g}') for seed in config['seeds']}
        for train in trains.values():
            assert_separation(train,test,index); assert_separation(train,target,index)
        folds.append(dict(fold=fold,held_out_group=g,test_ids=test,train_ids=trains,
                          test_counts=dict(Counter(index[i]['canonical_class'] for i in test))))
    for selections in synthetic.values():
        for train in selections.values():
            assert_separation(train,target,index)
    selected=set(target)
    for f in folds:
        selected.update(f['test_ids'])
        for train in f['train_ids'].values(): selected.update(train)
    for selections in synthetic.values():
        for train in selections.values(): selected.update(train)
    shutil.copy2(manifest,out/'admitted.jsonl')
    config['admission']['sha256']=sha(manifest)
    save(out/'config.json',config)
    for path in ['experiments/EXPERIMENTAL_PROTOCOL.md','experiments/DATASET_PROTOCOL.md',
                 'experiments/EXPERIMENT_MATRIX.csv','reports/proposed_experiments.md']:
        shutil.copy2(ROOT/path,out/Path(path).name)
    shutil.copy2(__file__,out/'executed_freeze.py')
    lock=dict(protocol_id=config['protocol_id'], frozen_utc=datetime.now(timezone.utc).isoformat(),
              budget_per_class=n,validation=None,final_test=None,target_role='exposed_development',
              folds=folds,synthetic_train_ids=synthetic,target_ids=target,feature_ids=sorted(selected),
              metadata_only_freeze=True,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              dirty_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).splitlines(),
              versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','torch','torchaudio','librosa','soundfile']},
              artifacts_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    save(out/'lock.json',lock)
    print(json.dumps(dict(budget_per_class=n,folds=len(folds),target=len(target),features=len(selected),lock_sha256=sha(out/'lock.json')),indent=2))


if __name__=='__main__':main()

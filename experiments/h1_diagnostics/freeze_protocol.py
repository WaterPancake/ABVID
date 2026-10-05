"""Freeze exact IDMT-only selections before running diagnostic comparisons."""
from collections import Counter
from datetime import datetime,timezone
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

from common import ROOT,HERE,sha,save,select,assert_separation,source_path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    cfg=json.loads((HERE/'config.json').read_text());hp=ROOT/cfg['reference_h1_lock']
    assert sha(hp)==cfg['reference_h1_lock_sha256']
    old=json.loads(hp.read_text());hc=json.loads((hp.parent/'config.json').read_text())
    source_manifest=hp.parent/'admitted.jsonl'
    assert sha(source_manifest)==old['artifacts_sha256']['admitted.jsonl']
    rows=sorted([r for r in map(json.loads,source_manifest.open()) if r['dataset_id']=='IDMT'],key=lambda r:r['file_id'])
    assert len(rows)==4413
    for r in rows:source_path(r)
    index={r['file_id']:r for r in rows};groups=sorted(set(r['provenance_group_id'] for r in rows))
    max_n=min(sum(r['class_id']==c and r['provenance_group_id']!=g for r in rows) for c in (0,1) for g in groups)
    assert max_n==max(cfg['learning_curve_examples_per_class'])==359
    folds=[]
    for old_fold in old['folds']:
        g=old_fold['held_out_group'];test=old_fold['test_ids']
        pool=[r for r in rows if r['provenance_group_id']!=g]
        train={str(n):{str(seed):select(pool,n,seed,f'IDMT|{g}') for seed in cfg['seeds']} for n in cfg['learning_curve_examples_per_class']}
        assert train['190']==old_fold['train_ids']
        inner=[]
        for ig in groups:
            if ig==g:continue
            val=sorted(r['file_id'] for r in pool if r['provenance_group_id']==ig)
            ip=[r for r in pool if r['provenance_group_id']!=ig]
            it={str(seed):select(ip,190,seed,f'source_diagnostics|outer:{g}|inner:{ig}') for seed in cfg['seeds']}
            for ids in it.values():
                assert_separation(ids,val,index);assert_separation(ids,test,index);assert_separation(val,test,index)
            inner.append(dict(validation_group=ig,validation_ids=val,train_ids=it))
        for byseed in train.values():
            for ids in byseed.values():assert_separation(ids,test,index)
        folds.append(dict(fold=old_fold['fold'],held_out_group=g,test_ids=test,train_ids=train,inner=inner,
                          site=index[test[0]]['site_id'],class_counts=old_fold['test_counts']))
    sites=sorted(set(r['site_id'] for r in rows));locations=[]
    for si,site in enumerate(sites):
        test=sorted(r['file_id'] for r in rows if r['site_id']==site)
        pool=[r for r in rows if r['site_id']!=site]
        train={str(seed):select(pool,190,seed,f'source_diagnostics|held_site:{site}') for seed in cfg['seeds']}
        for ids in train.values():assert_separation(ids,test,index)
        locations.append(dict(site=site,site_index=si,test_ids=test,train_ids=train))
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cfg['encoder']=hc['encoder'];cfg['mfcc']=hc['mfcc'];cfg['classifier']=hc['classifier']
    save(out/'config.json',cfg)
    with (out/'source_manifest.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r,sort_keys=True)+'\n')
    shutil.copy2(HERE/'PROTOCOL.md',out/'PROTOCOL.md');shutil.copy2(__file__,out/'executed_freeze.py')
    # Full H1 artifact hashes are preserved for a before/after immutability check;
    # no H1 target feature/probability array is decoded here.
    protected=[]
    for directory in [ROOT/'experiments/h1/frozen/h1_common_budget_v1.3',ROOT/'experiments/h1/results/H1_20261002']:
        for p in sorted(directory.rglob('*')):
            if p.is_file():protected.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
    save(out/'protected_h1_hashes.json',protected)
    sklearn_logistic=ROOT/'experiments/reproduction/.venv/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py'
    lock=dict(protocol_id=cfg['protocol_id'],frozen_utc=datetime.now(timezone.utc).isoformat(),source_only=True,
        source_files=4413,classes=dict(Counter(r['canonical_class'] for r in rows)),folds=folds,locations=locations,
        target_roles=[],metadata_only_freeze=True,h1_lock_sha256=sha(hp),
        sklearn_objective_source_sha256=sha(sklearn_logistic),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dirty_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).splitlines(),
        versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','torch','torchaudio','librosa','soundfile']},
        artifacts_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    save(out/'lock.json',lock)
    print(json.dumps(dict(files=len(rows),outer_folds=len(folds),inner_folds=sum(len(f['inner']) for f in folds),sites=len(sites),max_balanced_n=max_n,lock_sha256=sha(out/'lock.json')),indent=2))


if __name__=='__main__':main()

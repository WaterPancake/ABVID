"""Freeze source-selected models and target roles before new target inference."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import shutil
import subprocess

from common import ROOT, HERE, sha, save, read_json, read_rows, check_artifacts, assert_separation


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    cfg=read_json(HERE/'config.json'); hp=ROOT/cfg['h1_lock']; dp=ROOT/cfg['diagnostic_lock']
    assert sha(hp)==cfg['h1_lock_sha256'] and sha(dp)==cfg['diagnostic_lock_sha256']
    hl=read_json(hp); dl=read_json(dp); check_artifacts(hp.parent,hl); check_artifacts(dp.parent,dl)
    hc=read_json(hp.parent/'config.json'); dr=ROOT/cfg['diagnostic_results']; ds=read_json(dr/'summary.json')
    check_artifacts(dr,ds); assert read_json(dr/'verification.json')['passed']
    assert ds['source_only'] and ds['target_paths_accessed']==[]
    cache=ROOT/cfg['feature_cache']; cs=read_json(cache/'summary.json'); check_artifacts(cache,cs)
    assert cs['lock_sha256']==sha(hp)
    rows=[r for r in read_rows(hp.parent/'admitted.jsonl') if r['dataset_id'] in ('IDMT','MELAUDIS')]
    index={r['file_id']:r for r in rows}; target=hl['target_ids']
    assert len(rows)==12479 and len(target)==8066
    assert all(index[i]['dataset_id']=='MELAUDIS' for i in target)
    assert Counter(index[i]['class_id'] for i in target)=={0:7810,1:256}
    groups=sorted({index[i]['provenance_group_id'] for i in target}); assert len(groups)==4
    fits={r['fit_id']:r for r in read_json(dr/'fit_index.json')}
    choices=read_json(dr/'selections.json'); choice={(r['representation'],r['fold'],r['seed']):r for r in choices}
    models=[]
    for e in read_json(dr/'evaluations.json'):
        if e['case'] not in cfg['arms']: continue
        fit=fits[e['fit_id']]; fold=hl['folds'][e['fold']]; diagnostic_fold=dl['folds'][e['fold']]
        expected=fold['train_ids'][str(e['seed'])]
        assert fit['train_ids']==expected and fit['train_class_counts']==[190,190]
        assert e['representation'] in cfg['representations'] and e['seed'] in cfg['seeds']
        assert e['profile']==cfg['profile'] and e['threshold']==cfg['threshold']
        assert fit['C']==e['C']==(1. if e['case']=='baseline' else choice[e['representation'],e['fold'],e['seed']]['chosen_C'])
        assert sha(ROOT/fit['model_path'])==fit['model_sha256']
        assert sha(ROOT/e['predictions_path'])==e['predictions_sha256']
        assert_separation(expected,target,index); assert_separation(expected,fold['test_ids'],index)
        for inner in diagnostic_fold['inner']:
            ids=inner['train_ids'][str(e['seed'])]; val=inner['validation_ids']
            assert_separation(ids,val,index); assert_separation(ids,target,index); assert_separation(val,target,index)
            assert_separation(ids,fold['test_ids'],index); assert_separation(val,fold['test_ids'],index)
        models.append(dict(representation=e['representation'],arm=e['case'],seed=e['seed'],fold=e['fold'],
            fit_id=e['fit_id'],C=fit['C'],train_ids=expected,source_test_ids=fold['test_ids'],
            source_group=fold['held_out_group'],model_path=fit['model_path'],model_sha256=fit['model_sha256'],
            source_predictions_path=e['predictions_path'],source_predictions_sha256=e['predictions_sha256']))
    assert len(models)==120 and len({r['fit_id'] for r in models})==107
    assert len({(r['representation'],r['arm'],r['seed'],r['fold']) for r in models})==120
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    cfg['fixed_model_config']={k:hc[k] for k in ['preprocessing','encoder','mfcc','classifier']}
    cfg['feature_dtype_for_heads']='float64'
    save(out/'config.json',cfg); save(out/'model_references.json',models); save(out/'source_selections.json',choices)
    (out/'manifest.jsonl').write_text(''.join(__import__('json').dumps(r,sort_keys=True)+'\n' for r in sorted(rows,key=lambda r:r['file_id'])))
    for p in [HERE/'PROTOCOL.md',Path(__file__),HERE/'common.py']:
        shutil.copy2(p,out/(p.name if p.suffix=='.md' else 'executed_'+p.name))
    protected=[]
    for directory in [hp.parent,ROOT/cfg['h1_results'],dp.parent,dr,cache,
                      ROOT/'experiments/h1_diagnostics/cache/source_diagnostics_20261003_v1']:
        for p in sorted(directory.rglob('*')):
            if p.is_file(): protected.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
    save(out/'protected_artifacts.json',protected)
    lock=dict(protocol_id=cfg['protocol_id'],frozen_utc=datetime.now(timezone.utc).isoformat(),metadata_only_freeze=True,
        new_target_inference_before_freeze=False,target_role=cfg['target_role'],target_ids=target,target_groups=groups,
        feature_ids=hl['feature_ids'],feature_cache_sha256=cs['artifacts_sha256']['features.npz'],
        feature_summary_sha256=sha(cache/'summary.json'),diagnostic_selection_sha256=sha(dr/'selections.json'),
        source_folds=hl['folds'],unique_models=107,model_evaluation_records=120,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dirty_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).splitlines(),
        versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','joblib','torch']},
        artifacts_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    save(out/'lock.json',lock)
    print(dict(lock_sha256=sha(out/'lock.json'),models=107,evaluations=120,target_files=8066,target_groups=4,protected_files=len(protected)))


if __name__=='__main__':main()

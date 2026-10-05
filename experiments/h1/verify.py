"""Independent artifact/leakage/metric checks; no refitting or target tuning."""
import argparse
import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix,f1_score,balanced_accuracy_score
from freeze import ROOT,sha,save,assert_separation


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--lock',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--results',type=Path,required=True);args=ap.parse_args()
    lock=json.loads(args.lock.read_text());folder=args.lock.parent
    summary=json.loads((args.results/'summary.json').read_text())
    assert summary['lock_sha256']==sha(args.lock)
    assert summary['models']==260 and not summary['target_used_for_fitting']
    for name,h in summary['artifacts_sha256'].items():assert sha(args.results/name)==h,name
    for name,h in lock['artifacts_sha256'].items():assert sha(folder/name)==h,name
    rows={r['file_id']:r for r in map(json.loads,(folder/'admitted.jsonl').open())}
    results=json.loads((args.results/'metrics.json').read_text())
    data=np.load(args.cache/'features.npz');ids=data['file_ids'].tolist();pos={fid:i for i,fid in enumerate(ids)}
    checked=0
    for model in results['models']:
        prediction=np.load(ROOT/model['predictions_path'],allow_pickle=False)
        target_ids=prediction['target_ids'].tolist();assert target_ids==lock['target_ids']
        assert_separation(model['train_ids'],target_ids,rows)
        for domain in ['target','source']:
            if domain not in model:continue
            y=prediction[domain+'_true'];p=prediction[domain+'_probabilities'];pred=p.argmax(axis=1)
            np.testing.assert_allclose(p.sum(axis=1),1.,atol=1e-12)
            np.testing.assert_array_equal(y,[rows[i]['class_id'] for i in prediction[domain+'_ids']])
            np.testing.assert_array_equal(confusion_matrix(y,pred,labels=[0,1]),model[domain]['confusion_matrix'])
            assert abs(f1_score(y,pred,average='macro',labels=[0,1],zero_division=0)-model[domain]['macro_f1'])<1e-12
            assert abs(balanced_accuracy_score(y,pred)-model[domain]['balanced_accuracy'])<1e-12
            if domain=='source':assert_separation(model['train_ids'],prediction['source_ids'].tolist(),rows)
        fitted=joblib.load(ROOT/model['model_path'])
        x=np.asarray(data[model['representation']],dtype=np.float64)
        train=np.array([pos[i] for i in model['train_ids']])
        np.testing.assert_allclose(fitted[0].mean_,x[train].mean(axis=0),rtol=1e-6,atol=1e-7)
        # Replayed probabilities on 3 fixed positions, selected without scores.
        ix=np.array([0,len(target_ids)//2,len(target_ids)-1])
        xp=x[[pos[target_ids[i]] for i in ix]]
        np.testing.assert_allclose(fitted.predict_proba(xp),prediction['target_probabilities'][ix],rtol=1e-10,atol=1e-12)
        checked+=1
    result=dict(models_verified=checked,hashes_verified=True,confusions_and_f1_independently_recomputed=True,
                known_group_file_duplicate_separation=True,all_scalers_verified_against_training_only=True,
                fixed_positions_replayed=True,lock_sha256=sha(args.lock),results_summary_sha256=sha(args.results/'summary.json'))
    save(args.results/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()

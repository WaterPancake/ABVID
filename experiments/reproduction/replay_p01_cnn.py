"""Inference-only replay of the released P01 seed-42 CNN and test features."""
from pathlib import Path
import ast
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time

import numpy as np
import torch
from torch import nn
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[2]

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    base=ROOT/'experiments/reproduction/releases/P01'
    output=ROOT/'experiments/reproduction/results/P01_CNN'; output.mkdir(parents=True,exist_ok=True)
    source=base/'code_and_results/scripts/revisionReRunProcessAll_p2.py'
    tree=ast.parse(source.read_text())
    # Load only the reviewed, unmodified published model declaration. Never run
    # the script's top-level file moves, preprocessing, training or downloads.
    declarations=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BaselineCNN']
    assert len(declarations)==1
    namespace={'nn':nn}
    exec(compile(ast.Module(body=declarations,type_ignores=[]),str(source),'exec'),namespace)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    checkpoint=base/'models/baseline_cnn_best.pt'
    model=namespace['BaselineCNN'](5)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    model.eval()
    archived_path=base/'code_and_results/results/baseline_results.json'
    archived=json.loads(archived_path.read_text())
    target=next(r for r in archived['runs'] if r['seed']==42)
    result={
        'scope':'released seed-42 checkpoint replay; original five classes and supplied feature order',
        'training_run_reproduction':False,
        'seed_source':'release README identifies supplied CNN as seed 42; no seed search',
        'balanced_sampling':'published corrected-script balanced_evaluate: MT19937 seed 42 reset independently per domain, 50/class in class order',
        'inference_adapter':'published BaselineCNN declaration only; CPU batched inference in eval mode; no training',
        'classes':['car','truck','motorcycle','bus','background'],
        'checkpoint_sha256':sha256(checkpoint),'source_sha256':sha256(source),
        'script_sha256':sha256(__file__),'archived_results_sha256':sha256(archived_path),
        'versions':{p:importlib.metadata.version(p) for p in ['torch','numpy','scikit-learn']},
        'hardware':platform.platform(),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'domains':{},
    }
    for domain,counts in [('melaudis',[8102,263,248,258,5792]),('ch34',[3902,511,251,53,4071])]:
        t0=time.perf_counter()
        xp=base/'features'/f'X_test_{domain}.npy';yp=base/'features'/f'y_test_{domain}.npy'
        x=np.load(xp,allow_pickle=False,mmap_mode='r'); y=np.load(yp,allow_pickle=False)
        assert x.shape==(sum(counts),87,120,1) and np.array_equal(np.bincount(y,minlength=5),counts)
        probs=[]
        with torch.inference_mode():
            for start in range(0,len(y),64):
                batch=np.array(x[start:start+64],copy=True)
                assert np.isfinite(batch).all()
                logits=model(torch.from_numpy(batch).permute(0,3,1,2))
                probs.append(torch.softmax(logits,dim=1).numpy())
        probs=np.concatenate(probs);pred=probs.argmax(axis=1)
        rng=np.random.RandomState(42)
        indices=np.concatenate([rng.choice(np.flatnonzero(y==c),50,replace=False) for c in range(5)])
        cm=confusion_matrix(y[indices],pred[indices],labels=list(range(5)))
        f1=float(f1_score(y[indices],pred[indices],average='macro',labels=list(range(5)),zero_division=0))
        npz=output/f'{domain}_predictions.npz'
        np.savez_compressed(npz,y_true=y,y_pred=pred,probabilities=probs,balanced_indices=indices)
        result['domains'][domain]={
            'n':len(y),'class_counts':counts,
            'input_sha256':{xp.name:sha256(xp),yp.name:sha256(yp)},
            'balanced_macro_f1':f1,'balanced_confusion_matrix':cm.tolist(),
            'archived_seed42_macro_f1':target[domain]['balanced_f1'],
            'matches_archived_seed42_confusion':bool(np.array_equal(cm,target[domain]['confusion_matrix'])),
            'matches_archived_seed42_f1_rounded4':round(f1,4)==target[domain]['balanced_f1'],
            'full_set_macro_f1':float(f1_score(y,pred,average='macro',labels=list(range(5)),zero_division=0)),
            'full_set_confusion_matrix':confusion_matrix(y,pred,labels=list(range(5))).tolist(),
            'mean_confidence':float(probs.max(axis=1).mean()),
            'archived_mean_confidence':target[domain]['confidence'],
            'predictions_sha256':sha256(npz),'seconds':time.perf_counter()-t0,
        }
        (output/'replay.json').write_text(json.dumps(result,indent=2)+'\n')
        print(domain, 'balanced F1',f1,'archived',target[domain]['balanced_f1'],'exact CM',result['domains'][domain]['matches_archived_seed42_confusion'],flush=True)

if __name__=='__main__':main()

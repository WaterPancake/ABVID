"""Replay published transformer heads on reconstructed published seed-42 rows.

This is a compatibility replay with measured preprocessing residuals, not new
training or a claim of bit-identical raw-input reconstruction.

For BEATs-family models this preserves the September 28 legacy sampling for
diagnostic history only. Use replay_p01_beats_rng.py for the corrected published
evaluation selection. AST does not consume the same NumPy evaluation draws.
"""
from pathlib import Path
import argparse
import ast
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.autograd import Function
import torchaudio
from sklearn.metrics import confusion_matrix,f1_score
from transformers import ASTConfig,ASTModel,ASTFeatureExtractor

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/reproduction'

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',nargs='+',choices=['BEATs','DANN','ArcFace','AST'],default=['BEATs','DANN','ArcFace','AST'])
    parser.add_argument('--output',type=Path,default=BASE/'results/P01_transformers')
    parser.add_argument('--allow-legacy-pre-inference-sampling',action='store_true',help='Explicitly reproduce the superseded September 28 BEATs-family selection; not the published selection')
    args=parser.parse_args()
    if any(name!='AST' for name in args.models) and not args.allow_legacy_pre_inference_sampling:
        parser.error('BEATs-family sampling in this legacy script is superseded. Use replay_p01_beats_rng.py; the legacy flag is for historical diagnostics only.')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    validation_path=BASE/'results/P01_row_validation.json'
    validation=json.loads(validation_path.read_text())
    assert len(validation['rows'])==500
    # Do not relabel the initial 1e-4 feature-check failures as passes.
    # All rows are replayed; there is no residual-based exclusion or score tuning.
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    notebook=BASE/'releases/P01/code_and_results/scripts/AI4TEN_p2_colab_notebook.ipynb'
    cells=json.loads(notebook.read_text())['cells']
    chosen={'BEATsClassifier','ASTClassifier','GradientReversalFunction','GradientReversal','BEATsDANN','ArcFaceHead','BEATsArcFace'}
    declarations=[]
    for cell in cells:
        if cell['cell_type']!='code':continue
        declarations.extend(n for n in ast.parse(''.join(cell['source'])).body if isinstance(n,ast.ClassDef) and n.name in chosen)
    assert {c.name for c in declarations}==chosen
    feature_config=json.loads((BASE/'releases/AST_preprocessor_config.json').read_text())
    namespace={'torch':torch,'nn':nn,'Function':Function,'math':math,'ast_feature_extractor':ASTFeatureExtractor(**feature_config)}
    exec(compile(ast.Module(body=declarations,type_ignores=[]),str(notebook),'exec'),namespace)
    pretrained=ROOT/'.artifacts/models/beats/BEATs_iter3_plus_AS2M.pt'
    assert sha256(pretrained)=='d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34'
    provenance=json.loads((pretrained.parent/'provenance.json').read_text())
    upstream=pretrained.parent/'upstream'
    for name,entry in provenance['code'].items():assert sha256(upstream/name)==entry['sha256']
    sys.path.insert(0,str(upstream))
    from BEATs import BEATs,BEATsConfig
    original=torch.load(pretrained,map_location='cpu',weights_only=True,mmap=True)
    waveforms={};labels={};rows_by_domain={}
    for domain in ['melaudis','ch34']:
        rows=[r for r in validation['rows'] if r['domain']==domain]
        waves=[]
        for row in rows:
            path=ROOT/row['processed_path']; assert sha256(path)==row['processed_sha256']
            # PCM16 WAV decoding with soundfile replaces torchaudio's optional
            # codec dependency; stored samples are decoded exactly to float32.
            raw,sr=sf.read(path,dtype='float32',always_2d=True)
            x=torch.from_numpy(raw.T.copy()).mean(dim=0,keepdim=True)
            if sr!=16000:x=torchaudio.transforms.Resample(sr,16000)(x)
            x=nn.functional.pad(x,(0,max(0,32000-x.shape[1])))[:,:32000]
            waves.append(x.squeeze(0))
        waveforms[domain]=torch.stack(waves);labels[domain]=np.array([r['label'] for r in rows]);rows_by_domain[domain]=rows
        assert np.array_equal(np.bincount(labels[domain],minlength=5),[50]*5)
    released=BASE/'releases/P01/code_and_results/results'
    specs=[('BEATs','beats_finetuned.pt','BEATsClassifier','beats_results.json',None),('DANN','beats_dann.pt','BEATsDANN','da_results_full.json','dann'),('ArcFace','beats_arcface.pt','BEATsArcFace','da_results_full.json','arcface'),('AST','ast_finetuned.pt','ASTClassifier','ast_results.json',None)]
    result={'scope':'inference-only released checkpoint compatibility replay on published seed-42 balanced rows','training':False,'selection':'all 500 published seed-42 candidate rows, no prediction-driven selection','preprocessing_check_sha256':sha256(validation_path),'preprocessing_initial_tolerance_all_pass':validation['all_rows_match'],'preprocessing_max_absolute_standardized_mfcc_error':max(r['max_abs_error'] for r in validation['rows']),'input_limit':'reconstructed waveforms; residuals measured against supplied MFCCs; not bit-identical input proof','paper_reference':'P01 Table 3 PDF p.10 and released notebook evaluate_model defaults','script_sha256':sha256(__file__),'notebook_sha256':sha256(notebook),'hardware':platform.platform(),'versions':{p:importlib.metadata.version(p) for p in ['torch','torchaudio','transformers','numpy','scikit-learn','soundfile']},'upstream_beats_commit':provenance['upstream_commit'],'ast_config_sha256':sha256(BASE/'releases/AST_config.json'),'ast_preprocessor_sha256':sha256(BASE/'releases/AST_preprocessor_config.json'),'models':[]}
    for name,filename,cls,result_file,key in specs:
        if name not in args.models:continue
        t0=time.perf_counter()
        checkpoint=BASE/'releases/P01/models'/filename
        state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        if name=='AST':
            cfg=ASTConfig.from_dict(json.loads((BASE/'releases/AST_config.json').read_text()))
            backbone=ASTModel(cfg)
        else:backbone=BEATs(BEATsConfig(original['cfg']))
        model=namespace[cls](backbone=backbone,n_classes=5)
        model.load_state_dict(state,strict=True);model.eval().requires_grad_(False)
        archived=json.loads((released/result_file).read_text())
        if key:archived=archived[key]
        entry={'model':name,'checkpoint_sha256':sha256(checkpoint),'state_loaded_strict':True,'domains':{}}
        if name!='AST':
            keys=[k for k in state if k.startswith('backbone.')]
            entry['backbone_tensors_compared_to_pretrained']=len(keys)
            entry['backbone_tensors_different_from_pretrained']=sum(not torch.equal(state[k],original['model'][k[9:]]) for k in keys)
        for domain in ['melaudis','ch34']:
            probs=[]
            with torch.inference_mode():
                for start in range(0,len(labels[domain]),8):
                    logits=model(waveforms[domain][start:start+8])
                    probs.append(torch.softmax(logits,dim=1).numpy())
            probs=np.concatenate(probs);pred=probs.argmax(axis=1);y=labels[domain]
            cm=confusion_matrix(y,pred,labels=list(range(5)))
            score=float(f1_score(y,pred,average='macro',labels=list(range(5)),zero_division=0))
            prediction_path=output/f'{name}_{domain}_predictions.npz'
            np.savez_compressed(prediction_path,probabilities=probs,y_true=y,y_pred=pred,released_row_indices=np.array([r['row_index'] for r in rows_by_domain[domain]]))
            entry['domains'][domain]={'balanced_macro_f1':score,'archived_macro_f1':archived[domain]['balanced_f1'],'confusion_matrix':cm.tolist(),'matches_archived_confusion':bool(np.array_equal(cm,archived[domain]['confusion_matrix'])),'matches_archived_f1_rounded4':round(score,4)==archived[domain]['balanced_f1'],'predictions_sha256':sha256(prediction_path)}
            print(name,domain,'F1',round(score,6),'archived',archived[domain]['balanced_f1'],'exact CM',entry['domains'][domain]['matches_archived_confusion'],flush=True)
        entry['seconds']=time.perf_counter()-t0;result['models'].append(entry)
        (output/'replay.json').write_text(json.dumps(result,indent=2)+'\n')
        del model,backbone,state;gc.collect()

if __name__=='__main__':main()

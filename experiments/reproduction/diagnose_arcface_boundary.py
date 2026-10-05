"""Check a residual P01 ArcFace discrepancy without altering the replay result.

The archived confusion matrix differs by one net truck count, but individual
archived predictions are absent. We examine the lowest-margin truck prediction
as a candidate, not as an identified historical disagreement. Batch contexts
are fixed from the released data loader and current replay, not score search.
"""
from pathlib import Path
import ast
import json
import math
import sys
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch import nn
import librosa
from replay_p01_beats_rng import BASE,ROOT,OUT,WORK,inventory,sha256

def wave_from_original(row):
    x,_=librosa.load(ROOT/row['original'],sr=22050,mono=True)
    peak=np.max(np.abs(x))
    if peak>0:x=x/peak*.95
    path=WORK/'arcface_batch'/row['processed_name'];path.parent.mkdir(parents=True,exist_ok=True)
    sf.write(path,x,22050,subtype='PCM_16')
    return wave_from_processed(path)

def wave_from_processed(path):
    x,sr=sf.read(path,dtype='float32')
    assert sr==22050
    x=torchaudio.transforms.Resample(sr,16000)(torch.from_numpy(x).unsqueeze(0))
    return nn.functional.pad(x,(0,max(0,32000-x.shape[1])))[:,:32000].squeeze(0)

def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    upstream=ROOT/'.artifacts/models/beats/upstream';sys.path.insert(0,str(upstream))
    from BEATs import BEATs,BEATsConfig
    nb=BASE/'releases/P01/code_and_results/scripts/AI4TEN_p2_colab_notebook.ipynb'
    nodes=[]
    for cell in json.loads(nb.read_text())['cells']:
        if cell['cell_type']=='code':nodes.extend(n for n in ast.parse(''.join(cell['source'])).body if isinstance(n,ast.ClassDef) and n.name in {'ArcFaceHead','BEATsArcFace'})
    ns={'torch':torch,'nn':nn,'math':math};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(nb),'exec'),ns)
    original=torch.load(upstream.parent/'BEATs_iter3_plus_AS2M.pt',weights_only=True,mmap=True,map_location='cpu')
    model=ns['BEATsArcFace'](BEATs(BEATsConfig(original['cfg'])),n_classes=5)
    checkpoint=BASE/'releases/P01/models/beats_arcface.pt'
    model.load_state_dict(torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu'),strict=True)
    model.eval().requires_grad_(False)
    validation=json.loads((OUT/'input_validation.json').read_text())
    rows=[r for r in validation['rows'] if r['domain']=='ch34']
    with np.load(OUT/'ArcFace_ch34_predictions.npz',allow_pickle=False) as p:
        probs=p['probabilities'];truth=p['y_true'];pred=p['y_pred']
        sorted_probs=np.sort(probs,axis=1);margins=sorted_probs[:,-1]-sorted_probs[:,-2]
        truck_rows=np.flatnonzero((truth==1)&(pred==1))
        index=int(truck_rows[np.argmin(margins[truck_rows])])
        old=probs[index].copy()
    row=rows[index];dataset_index=row['dataset_index']
    full=[r for c in ['car','truck','motorcycle','bus','background'] for r in sorted(inventory()['ch34'][c],key=lambda r:r['processed_name'])]
    start=dataset_index//16*16
    original_batch=torch.stack([wave_from_original(r) for r in full[start:start+16]])
    local_start=index//16*16
    selected_batch=torch.stack([wave_from_processed(ROOT/r['processed_path']) for r in rows[local_start:local_start+16]])
    wave=wave_from_processed(ROOT/row['processed_path'])
    assert torch.equal(wave,original_batch[dataset_index-start])
    result={'scope':'fixed batch-context diagnostic, no checkpoint/preprocessing change or target tuning','training':False,
        'candidate_only':'Archived per-event predictions are absent; the lowest-margin truck is not proven to be the historical mismatched event.',
        'candidate':row,'current_balanced_row':index,'original_probability_margin':float(margins[index]),
        'checkpoint_sha256':sha256(checkpoint),'script_sha256':sha256(__file__),'cases':{}}
    for name,x,offset in [('single',wave[None],0),('selected_subset_batch16',selected_batch,index-local_start),('original_full_dataset_batch16',original_batch,dataset_index-start)]:
        with torch.inference_mode():prob=torch.softmax(model(x),1)[offset].numpy()
        result['cases'][name]={'probabilities':prob.tolist(),'predicted_class':int(prob.argmax()),'max_abs_probability_difference_from_saved':float(np.max(np.abs(prob-old)))}
    result['all_contexts_predict_truck']=all(c['predicted_class']==1 for c in result['cases'].values())
    result['conclusion']='These CPU batch-context checks do not resolve the net confusion-matrix difference; no attribution to a specific historical event or numerical cause is established.'
    (OUT/'arcface_boundary_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='candidate'},indent=2))

if __name__=='__main__':main()

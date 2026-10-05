"""Reproduce P01's BEATs evaluation RNG consumption, then replay its subset.

The published evaluator seeds NumPy BEFORE inference. Upstream BEATs consumes
one NumPy uniform per layer even in eval mode. Sampling after ceil(N/16)
forwards therefore differs from sampling immediately after seeding. We derive
the state from code and verify it with an actual forward; no seed/score search.
Original September 28 results and provider artifacts are never overwritten.
"""
from pathlib import Path, PureWindowsPath
import argparse
import ast
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'experiments/reproduction'
OUT = BASE/'results/R0_20260929/P01_BEATs_rng'
WORK = BASE/'work/R0_20260929'
os.environ.setdefault('NUMBA_CACHE_DIR',str(WORK/'numba_cache'))
import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch import nn
from torch.autograd import Function
from sklearn.metrics import confusion_matrix, f1_score

CLASSES = ['car','truck','motorcycle','bus','background']

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def equal_rng(a,b):
    return a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]

def inventory():
    groups={d:{c:[] for c in CLASSES} for d in ['ch34','melaudis']}
    for name in (ROOT/'dataset/IDMT_Traffic/annotation/idmt_traffic_all.txt').read_text().splitlines():
        if '_SE_' not in name:continue
        cls='background' if '-BG' in name else {'C':'car','T':'truck','M':'motorcycle','B':'bus'}[name.split('_')[6][0]]
        groups['ch34'][cls].append({'original':str(Path('dataset/IDMT_Traffic/audio')/name),'processed_name':name.replace('_CH12','_CH34'),'class':cls})
    labels={'1V-Car':'car','1V-Truck':'truck','1V-MC':'motorcycle','1V-Bus':'bus'}
    for name in (BASE/'results/melaudis_vehicle_members.txt').read_text().splitlines():
        if not name.endswith('.wav'):continue
        hits=[labels[t] for t in Path(name).name.split('_') if t in labels]
        if len(hits)==1:
            cls=hits[0];groups['melaudis'][cls].append({'archive':'MELAUDIS_Vehicles.rar','archive_member':name,'processed_name':Path(name).name,'class':cls})
    for name in (BASE/'results/melaudis_background_members.txt').read_text().splitlines():
        if name.endswith('.wav'):groups['melaudis']['background'].append({'archive':'MELAUDIS_ BG.rar','archive_member':name,'processed_name':Path(name).name,'class':'background'})
    return groups

def main():
    global OUT, WORK
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT,help='Fresh result directory; an existing replay is never overwritten')
    parser.add_argument('--work-dir',type=Path,default=WORK,help='Separate derived-audio directory for a fresh replay')
    args=parser.parse_args()
    OUT=args.output.resolve();WORK=args.work_dir.resolve()
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    if (OUT/'replay.json').exists():raise RuntimeError('Result exists; preserve it rather than overwrite')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    upstream=ROOT/'.artifacts/models/beats/upstream'
    provenance=json.loads((upstream.parent/'provenance.json').read_text())
    for name,entry in provenance['code'].items():assert sha256(upstream/name)==entry['sha256']
    sys.path.insert(0,str(upstream))
    from BEATs import BEATs,BEATsConfig
    pretrained=upstream.parent/'BEATs_iter3_plus_AS2M.pt'
    assert sha256(pretrained)=='d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34'
    original=torch.load(pretrained,weights_only=True,mmap=True,map_location='cpu')
    notebook=BASE/'releases/P01/code_and_results/scripts/AI4TEN_p2_colab_notebook.ipynb'
    cells=json.loads(notebook.read_text())['cells']
    wanted={'BEATsClassifier','GradientReversalFunction','GradientReversal','BEATsDANN','ArcFaceHead','BEATsArcFace'}
    declarations=[]
    for cell in cells:
        if cell['cell_type']=='code':declarations.extend(n for n in ast.parse(''.join(cell['source'])).body if isinstance(n,ast.ClassDef) and n.name in wanted)
    assert {n.name for n in declarations}==wanted
    namespace={'torch':torch,'nn':nn,'Function':Function,'math':math}
    exec(compile(ast.Module(body=declarations,type_ignores=[]),str(notebook),'exec'),namespace)
    backbone=BEATs(BEATsConfig(original['cfg']));backbone.load_state_dict(original['model'],strict=True);backbone.eval()
    x=torch.zeros(2,32000);x[:,::40]=.1
    np.random.seed(42)
    with torch.inference_mode():backbone.extract_features(x,padding_mask=torch.zeros_like(x,dtype=torch.bool))
    layers=len(backbone.encoder.layers)
    reference=np.random.RandomState(42);reference.random_sample(layers)
    assert equal_rng(np.random.get_state(),reference.get_state())
    result={'scope':'P01 released BEATs-family checkpoint replay with code-derived post-inference NumPy state',
        'date':'2026-09-29','training':False,'seed_search':False,'classes':CLASSES,
        'published_evaluation_seed':42,'published_evaluation_batch_size':16,
        'path_order':'POSIX sorted Path basenames, as in the released Colab notebook',
        'rng_explanation':'backbone.py draws numpy.random.random once/layer before testing self.training; evaluate_model samples after all full-domain forwards',
        'actual_eval_forward_rng_check':{'layers':layers,'draws_per_forward':layers,'state_matches':True},
        'input_limit':'reconstructed PCM16 waveforms; supplied MFCC rows validate candidates approximately, not bit-identical original waveform identity',
        'script_sha256':sha256(__file__),'notebook_sha256':sha256(notebook),
        'upstream_commit':provenance['upstream_commit'],'backbone_code_sha256':sha256(upstream/'backbone.py'),
        'versions':{p:importlib.metadata.version(p) for p in ['torch','torchaudio','numpy','librosa','soundfile','scikit-learn']},
        'hardware':platform.platform(),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'sampling':{},'models':[]}
    del backbone;gc.collect()
    candidates=[]
    for domain,group in inventory().items():
        full=[r for cls in CLASSES for r in sorted(group[cls],key=lambda r:r['processed_name'])]
        windows=[r for cls in CLASSES for r in sorted(group[cls],key=lambda r:PureWindowsPath(r['processed_name']))]
        feature_indices={r['processed_name']:i for i,r in enumerate(windows)}
        y=np.load(BASE/'releases/P01/features'/f'y_test_{domain}.npy',allow_pickle=False)
        assert [CLASSES.index(r['class']) for r in full]==y.tolist()
        assert len(feature_indices)==len(full)
        batches=math.ceil(len(y)/16);draws=layers*batches
        rng=np.random.RandomState(42);rng.random_sample(draws)
        indices=np.concatenate([rng.choice(np.flatnonzero(y==i),50,replace=False) for i in range(5)])
        result['sampling'][domain]={'full_n':len(y),'full_batches':batches,'numpy_uniform_draws_before_selection':draws,'indices':indices.tolist()}
        for index in indices:
            row=full[index]
            candidates.append(dict(row,domain=domain,dataset_index=int(index),label=int(y[index]),feature_row_index=feature_indices[row['processed_name']]))
    # Freeze selections before predictions are produced or compared.
    (OUT/'sampling_manifest.json').write_text(json.dumps({'method':result['rng_explanation'],'selection':result['sampling'],'rows':candidates},indent=2)+'\n')
    raw_root=WORK/'raw_melaudis';raw_root.mkdir(exist_ok=True)
    for archive,label in [('MELAUDIS_Vehicles.rar','vehicle'),('MELAUDIS_ BG.rar','background')]:
        members=[r['archive_member'] for r in candidates if r.get('archive')==archive]
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in members)
        selection=WORK/f'{label}_rng_members.txt';selection.write_text('\n'.join(members)+'\n')
        subprocess.run(['tar','-xf',str(ROOT/'dataset/MELAUDIS'/archive),'-C',str(raw_root),'-T',str(selection)],check=True)
    feature_root=BASE/'releases/P01/features'
    means=np.load(feature_root/'scaler_mean.npy',allow_pickle=False);scale=np.load(feature_root/'scaler_scale.npy',allow_pickle=False)
    features={d:np.load(feature_root/f'X_test_{d}.npy',allow_pickle=False,mmap_mode='r') for d in ['melaudis','ch34']}
    waveforms={d:[] for d in features};rows={d:[] for d in features}
    resample=torchaudio.transforms.Resample(22050,16000)
    checked=[];t0=time.perf_counter()
    for row in candidates:
        raw=ROOT/row['original'] if 'original' in row else raw_root/row['archive_member']
        dest=WORK/'processed_rng'/row['domain']/row['class']/row['processed_name'];dest.parent.mkdir(parents=True,exist_ok=True)
        y,_=librosa.load(raw,sr=22050,mono=True)
        peak=np.max(np.abs(y))
        if peak>0:y=y/peak*.95
        sf.write(dest,y,22050,subtype='PCM_16')
        quantized,sr=sf.read(dest,dtype='float32');assert sr==22050 and quantized.ndim==1
        mfcc_input=np.pad(quantized,(0,max(0,44100-len(quantized))))[:44100]
        mfcc=librosa.feature.mfcc(y=mfcc_input,sr=22050,n_mfcc=40,n_fft=2048,hop_length=512,n_mels=128)
        feature=np.vstack([mfcc,librosa.feature.delta(mfcc),librosa.feature.delta(mfcc,order=2)]).T
        feature-=means;feature/=scale
        target=features[row['domain']][row['feature_row_index'],:,:,0]
        checked_row=dict(row,original_sha256=sha256(raw),processed_path=str(dest.relative_to(ROOT)),processed_sha256=sha256(dest),
            max_abs_mfcc_error=float(np.max(np.abs(feature-target))),initial_tolerance_pass=bool(np.allclose(feature,target,atol=1e-4,rtol=1e-4)))
        checked.append(checked_row);rows[row['domain']].append(checked_row)
        wave=resample(torch.from_numpy(quantized).unsqueeze(0))
        wave=nn.functional.pad(wave,(0,max(0,32000-wave.shape[1])))[:,:32000]
        waveforms[row['domain']].append(wave.squeeze(0))
    result['preprocessing']={'seconds':time.perf_counter()-t0,'rtol':1e-4,'atol':1e-4,'rows':len(checked),
        'initial_tolerance_passes':sum(r['initial_tolerance_pass'] for r in checked),'max_abs_mfcc_error':max(r['max_abs_mfcc_error'] for r in checked)}
    (OUT/'input_validation.json').write_text(json.dumps({'summary':result['preprocessing'],'rows':checked},indent=2)+'\n')
    result['input_validation_sha256']=sha256(OUT/'input_validation.json')
    print('Preprocessing',result['preprocessing'],flush=True)
    waveforms={d:torch.stack(w) for d,w in waveforms.items()}
    specifications=[('BEATs','beats_finetuned.pt','BEATsClassifier','beats_results.json',None),('DANN','beats_dann.pt','BEATsDANN','da_results_full.json','dann'),('ArcFace','beats_arcface.pt','BEATsArcFace','da_results_full.json','arcface')]
    for name,filename,cls,resultsfile,key in specifications:
        t0=time.perf_counter();backbone=BEATs(BEATsConfig(original['cfg']))
        model=namespace[cls](backbone=backbone,n_classes=5)
        checkpoint=BASE/'releases/P01/models'/filename
        state=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
        model.load_state_dict(state,strict=True);model.eval().requires_grad_(False)
        archived=json.loads((BASE/'releases/P01/code_and_results/results'/resultsfile).read_text())
        if key:archived=archived[key]
        entry={'model':name,'checkpoint_sha256':sha256(checkpoint),'strict_state_load':True,'domains':{}}
        for domain in ['melaudis','ch34']:
            probs=[]
            with torch.inference_mode():
                for start in range(0,len(waveforms[domain]),16):
                    probs.append(torch.softmax(model(waveforms[domain][start:start+16]),1).numpy())
            probs=np.concatenate(probs);pred=probs.argmax(1);y=np.array([r['label'] for r in rows[domain]])
            cm=confusion_matrix(y,pred,labels=list(range(5)));score=float(f1_score(y,pred,average='macro',zero_division=0))
            prediction=OUT/f'{name}_{domain}_predictions.npz'
            np.savez_compressed(prediction,y_true=y,y_pred=pred,probabilities=probs,dataset_indices=np.array([r['dataset_index'] for r in rows[domain]]))
            entry['domains'][domain]={'balanced_macro_f1':score,'archived_macro_f1':archived[domain]['balanced_f1'],
                'confusion_matrix':cm.tolist(),'matches_archived_confusion':bool(np.array_equal(cm,archived[domain]['confusion_matrix'])),
                'matches_archived_f1_rounded4':round(score,4)==archived[domain]['balanced_f1'],'predictions_sha256':sha256(prediction)}
            print(name,domain,'F1',round(score,6),'archived',archived[domain]['balanced_f1'],'exact CM',entry['domains'][domain]['matches_archived_confusion'],flush=True)
        entry['seconds']=time.perf_counter()-t0;result['models'].append(entry)
        (OUT/'replay.json').write_text(json.dumps(result,indent=2)+'\n')
        del model,backbone,state;gc.collect()

if __name__=='__main__':main()

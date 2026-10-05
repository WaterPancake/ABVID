"""Replay released P02 CNN snapshots on supplied standardized test arrays.

No training, resampling, scaler fitting, subset search, or raw-audio mutation.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import time
import zipfile

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np
import h5py
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = [
    ('01_real_only.keras', 'exp1_real_only', .25, .06),
    ('02_real_specaugment.keras', 'exp1_real_specaugment', .32, .05),
    ('03_aldm_only.keras', 'exp1_aldm_only', .22, .11),
    ('04_pyroad_only.keras', 'exp1_pyroad_only', .19, .03),
    ('05_both_synthetic.keras', 'exp1_both_synthetic', .35, .02),
    ('06_real_plus_aldm.keras', 'exp2_real_plus_aldm', .27, .03),
    ('07_real_plus_pyroad.keras', 'exp2_real_plus_pyroad', .36, .02),
    ('08_real_plus_both_BEST.keras', 'exp2_real_plus_both', .39, .03),
]

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def normalize_hdf5_paths(original, destination):
    """Copy Windows HDF5 group separators to portable paths; preserve all tensors."""
    with zipfile.ZipFile(original) as archive:
        source_bytes = archive.read('model.weights.h5')
        target_bytes = io.BytesIO()
        tensors = []
        with h5py.File(io.BytesIO(source_bytes),'r') as source, h5py.File(target_bytes,'w') as target:
            for key,value in source.attrs.items(): target.attrs[key]=value
            def copy(name,obj):
                portable=name.replace('\\','/')
                if isinstance(obj,h5py.Group):
                    new=target.require_group(portable)
                else:
                    assert portable not in target
                    new=target.create_dataset(portable,data=obj[()],dtype=obj.dtype)
                    before=np.asarray(obj[()]);after=np.asarray(new[()])
                    assert before.dtype==after.dtype and before.shape==after.shape and before.tobytes()==after.tobytes()
                    tensors.append({'original':name,'portable':portable,'sha256':hashlib.sha256(before.tobytes()).hexdigest(),'shape':list(before.shape),'dtype':str(before.dtype)})
                for key,value in obj.attrs.items(): new.attrs[key]=value
            source.visititems(copy)
        assert any('\\' in item['original'] for item in tensors)
        destination.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as output:
            for name in archive.namelist():
                output.writestr(name,target_bytes.getvalue() if name=='model.weights.h5' else archive.read(name))
    return {'change':'HDF5 path separators only; configuration and every tensor byte preserved','derived_checkpoint_sha256':sha256(destination),'tensors':tensors}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/reproduction/results/P02')
    parser.add_argument('--normalize-hdf5-paths', action='store_true', help='Explicit compatibility conversion for Windows backslashes in released HDF5 group names')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    release = ROOT/'experiments/reproduction/releases/P02/code_and_results'
    feature_dir = ROOT/'dataset/AI4TEN/data/features'
    x = np.load(feature_dir/'X_test.npy', allow_pickle=False)
    y = np.load(feature_dir/'y_test.npy', allow_pickle=False)
    assert x.shape == (1019,216,120,1) and y.shape == (1019,)
    assert np.isfinite(x).all() and np.array_equal(np.bincount(y, minlength=3), [703,75,241])
    metadata = json.loads((feature_dir/'metadata.json').read_text())
    assert metadata['classes'] == ['car','truck','motorcycle']
    split = json.loads((release/'configs/real_data_split.json').read_text())
    assert len(split['test_files']) == len(y)
    listed_labels = np.array([metadata['classes'].index(Path(p.replace('//','/')).parent.name) for p in split['test_files']])
    label_sequence_agrees = bool(np.array_equal(y, listed_labels))
    result = {
        'scope': 'inference-only snapshot replay; author arrays in their supplied order; no refitting',
        'paper_reference': 'P02, Table 4, PDF p.8; release results/*.json',
        'classes': metadata['classes'], 'test_n': len(y),
        'label_sequence_matches_released_file_list': label_sequence_agrees,
        'row_identity_limit': 'Class sequence agreement is not independent proof of raw-event-to-array row identity.',
        'training_run_reproduction': False,
        'balanced_subset_replay': 'unknown: exact original indices and soundclass_v1 sampling implementation absent',
        'hardware': platform.platform(),
        'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'script_sha256': sha256(__file__),
        'versions': {p: importlib.metadata.version(p) for p in ['tensorflow','keras','numpy','scikit-learn']},
        'inputs': {str(p.relative_to(ROOT)):sha256(p) for p in [feature_dir/'X_test.npy',feature_dir/'y_test.npy',feature_dir/'metadata.json',release/'configs/real_data_split.json']},
        'snapshots': [],
    }
    for filename, config, paper_mean, paper_sd in CONFIGS:
        t0 = time.perf_counter()
        checkpoint = release/'models'/filename
        archived_path = release/'results'/f'{config}_experiment_results.json'
        archived = json.loads(archived_path.read_text())
        compatibility = None
        load_path = checkpoint
        if args.normalize_hdf5_paths:
            load_path = args.output/'portable_checkpoints'/filename
            compatibility = normalize_hdf5_paths(checkpoint,load_path)
        model = tf.keras.models.load_model(load_path, compile=False, safe_mode=True)
        assert model.input_shape == (None,216,120,1) and model.output_shape == (None,3)
        probabilities = np.concatenate([model(x[start:start+32], training=False).numpy() for start in range(0,len(x),32)])
        assert np.isfinite(probabilities).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-5)
        predicted = probabilities.argmax(axis=1)
        cm = confusion_matrix(y, predicted, labels=[0,1,2])
        matches = [r for r in archived['all_runs'] if np.array_equal(cm,np.array(r['confusion_matrix']))]
        output = args.output/f'{config}_predictions.npz'
        np.savez_compressed(output, y_true=y, y_pred=predicted, probabilities=probabilities, row_index=np.arange(len(y)))
        balanced = np.array([r['balanced_test_f1_macro'] for r in archived['all_runs']])
        entry = {
            'config': config, 'checkpoint':filename, 'checkpoint_sha256':sha256(checkpoint),
            'compatibility':compatibility,
            'archived_results_sha256':sha256(archived_path),
            'full_set_confusion_matrix':cm.tolist(),
            'full_set_accuracy':float(accuracy_score(y,predicted)),
            'full_set_balanced_accuracy':float(balanced_accuracy_score(y,predicted)),
            'full_set_macro_f1':float(f1_score(y,predicted,average='macro',labels=[0,1,2],zero_division=0)),
            'full_set_per_class_f1':f1_score(y,predicted,average=None,labels=[0,1,2],zero_division=0).tolist(),
            'archived_best_run':archived['best_run']['run_id'],
            'matches_archived_best_full_confusion':bool(np.array_equal(cm,np.array(archived['best_run']['confusion_matrix']))),
            'matching_archived_runs':[{'run_id':r['run_id'],'seed':r['seed']} for r in matches],
            'archived_table4_recalculation':{'source':'archived five runs, not new model runs','mean':float(balanced.mean()),'std_ddof0':float(balanced.std()),'paper_mean':paper_mean,'paper_std':paper_sd,'rounds_to_paper':round(float(balanced.mean()),2)==paper_mean and round(float(balanced.std()),2)==paper_sd},
            'seconds':time.perf_counter()-t0,
            'predictions_sha256':sha256(output),
        }
        result['snapshots'].append(entry)
        (args.output/'replay.json').write_text(json.dumps(result,indent=2)+'\n')
        print(config, 'full F1', round(entry['full_set_macro_f1'],6),'matches best',entry['matches_archived_best_full_confusion'],'matches',entry['matching_archived_runs'],flush=True)
        tf.keras.backend.clear_session()
    print('Saved',args.output/'replay.json',flush=True)

if __name__ == '__main__':
    main()

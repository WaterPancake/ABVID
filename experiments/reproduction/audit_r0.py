"""Create R0's result ledger from immutable releases and saved replays.

Checks actual prediction artifacts separately from arithmetic on author-stored
results. Retains mismatches, missing inputs and unexecuted training explicitly.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile
import numpy as np
from audit_released_results import BASE,ROOT,read,digest,f1,prediction_check

OUT=BASE/'results/R0_20260929'

def stats_from_cms(runs,cm_key='balanced_confusion_matrix'):
    scores=[f1(r[cm_key]) for r in runs]
    return float(np.mean(scores)),float(np.std(scores))

def main():
    result={'date':'2026-09-29','scope':'R0 current-release audit: checkpoint inference and archived-result arithmetic; no new training',
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'script_sha256':digest(__file__),'training_runs_reproduced':0,'custom_experiments_run':False,
        'release_checks':[],'checkpoint_results':[],'arithmetic_results':[],
        'method_findings':[],'unexecuted_results':[]}
    for paper in ['P01','P02']:
        previous=read(BASE/'releases'/f'{paper}_zenodo_record.json');current=read(OUT/f'{paper}_record.json')
        old={f['key']:f['checksum'] for f in previous['files']};new={f['key']:f['checksum'] for f in current['files']}
        assert old==new
        archive=BASE/'releases'/f'{paper}_code_and_results.zip'
        assert 'md5:'+digest(archive,'md5')==new['code_and_results.zip']
        with zipfile.ZipFile(archive) as z:names=z.namelist()
        result['release_checks'].append({'paper':paper,'live_record_url':f'https://zenodo.org/api/records/{current["id"]}',
            'live_record_sha256':digest(OUT/f'{paper}_record.json'),'file_checksums_unchanged':True,
            'code_archive_sha256':digest(archive),'code_archive_members':names,
            'soundclass_v1_member_present':any(Path(n).name=='soundclass_v1.py' for n in names),
            'csv_members':[n for n in names if n.lower().endswith('.csv')]})
    cnn=read(BASE/'results/P01_CNN/replay.json')
    for domain,entry in cnn['domains'].items():
        audit=prediction_check(BASE/'results/P01_CNN'/f'{domain}_predictions.npz',5,entry['balanced_confusion_matrix'],entry['predictions_sha256'],'balanced_indices')
        result['checkpoint_results'].append(dict(audit,paper='P01',model='CNN',domain=domain,
            status='checkpoint_replay_exact',archived_f1=entry['archived_seed42_macro_f1'],exact_confusion=entry['matches_archived_seed42_confusion'],limitations='seed-42 snapshot only; five-run mean is archived arithmetic'))
    for folder in [BASE/'results/P01_AST_compatibility',OUT/'P01_BEATs_rng']:
        replay=read(folder/'replay.json')
        for model in replay['models']:
            for domain,entry in model['domains'].items():
                audit=prediction_check(folder/f"{model['model']}_{domain}_predictions.npz",5,entry['confusion_matrix'],entry['predictions_sha256'])
                result['checkpoint_results'].append(dict(audit,paper='P01',model=model['model'],domain=domain,
                    status='compatibility_replay_exact' if entry['matches_archived_confusion'] else 'mismatch_unresolved',
                    archived_f1=entry['archived_macro_f1'],exact_confusion=entry['matches_archived_confusion'],
                    limitations='reconstructed waveforms; numerical input residuals remain; no retraining'))
    p02=read(BASE/'results/P02/replay.json')
    for entry in p02['snapshots']:
        audit=prediction_check(BASE/'results/P02'/f"{entry['config']}_predictions.npz",3,entry['full_set_confusion_matrix'],entry['predictions_sha256'])
        result['checkpoint_results'].append(dict(audit,paper='P02',model=entry['config'],domain='DATASEC+MAVD full supplied test',
            status='compatibility_replay_exact',exact_confusion=True,matching_runs=entry['matching_archived_runs'],
            matches_best_run=entry['matches_archived_best_full_confusion'],limitations='full 1019 rows; exact balanced selections unavailable; weight-preserving HDF5 path conversion'))
    prior=read(BASE/'results/reproduction_audit.json')
    result['arithmetic_results'].extend(prior['arithmetic_checks'])
    p02r=BASE/'releases/P02/code_and_results/results'
    expected_balanced={0:(.25,.06),250:(.29,.04),500:(.35,.05),1000:(.34,.05),2000:(.36,.04),5000:(.36,.02)}
    expected_unbalanced={0:(.20,.04),250:(.29,.02),500:(.22,.06),1000:(.17,.00),2000:(.26,.11),5000:(.39,.04)}
    for n,expected in expected_balanced.items():
        stem='exp1_real_only' if n==0 else 'exp2_real_plus_pyroad' if n==5000 else f'exp2_ratio_pyroad_{n}'
        saved=read(p02r/f'{stem}_experiment_results.json');mean,sd=stats_from_cms(saved['all_runs'])
        result['arithmetic_results'].append({'paper':'P02','reference':'Table 6, PDF p.13','source':f'{stem}_experiment_results.json',
            'condition':'balanced-real','synthetic_per_class':n,'mean':mean,'std_ddof0':sd,'rounds_to_paper':(round(mean,2),round(sd,2))==expected,'inference_reproduced':False})
    revised=read(p02r/'revision_ratio_sweep/revised_ratio_sweep_results.json')
    for name,saved in revised['results'].items():
        mean,sd=stats_from_cms(saved['runs'],'confusion_matrix');n=saved['pyroad_per_class']
        result['arithmetic_results'].append({'paper':'P02','reference':'Table 6, PDF p.13','source':'revised_ratio_sweep_results.json',
            'condition':'original-imbalanced-real','synthetic_per_class':n,'mean':mean,'std_ddof0':sd,'rounds_to_paper':(round(mean,2),round(sd,2))==expected_unbalanced[n],'inference_reproduced':False})
    congestion=read(p02r/'congestion_results.json')
    expected={1:(.52,.47,.86),2:(.34,.27,.88),3:(.39,.31,.89),5:(.46,.37,.89)}
    for density,saved in congestion['results'].items():
        cm=np.array(saved['confusion_matrix']);score=f1(cm);accuracy=float(np.trace(cm)/cm.sum())
        assert abs(score-saved['f1_macro'])<1e-12
        assert abs(accuracy-saved['accuracy'])<1e-12
        a,f,c=expected[int(density)]
        result['arithmetic_results'].append({'paper':'P02','reference':'Table 7, PDF p.17','density':int(density),
            'f1':score,'accuracy':accuracy,'n_windows':int(cm.sum()),'metric_rounds_to_paper':round(score,2)==f and round(accuracy,2)==a,
            'saved_mean_confidence':saved['mean_confidence'],'printed_confidence':c,
            'confidence_rounds_to_paper':round(saved['mean_confidence'],2)==c,
            'confidence_recomputed':False,'inference_reproduced':False})
    figure_cm=[[362,311,30],[25,43,7],[119,54,68]]
    figure_entry=next(e for e in p02['snapshots'] if e['config']=='exp2_real_plus_both')
    assert figure_entry['full_set_confusion_matrix']==figure_cm
    result['method_findings']=[
        {'paper':'P01','finding':'BEATs eval advances NumPy RNG before balanced sampling; prior immediate-seed replay selected different events.',
         'evidence':'P01_BEATs_rng/replay.json and pinned backbone.py TransformerEncoder.extract_features; notebook evaluate_model',
         'draws':{'melaudis':11004,'ch34':6600},'seed':42,'published_batch_size':16,
         'implication':'Same seed does not guarantee same balanced events for CNN/AST and BEATs-family comparisons.'},
        {'paper':'P01','finding':'Nine of ten model/domain checkpoint matrices match; ArcFace CH34 has one net truck-count shift.',
         'arcface_ch34_delta':[[0,0,0,0,0],[-1,1,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],
         'individual_historical_disagreement_known':False,'boundary_diagnostic':'P01_BEATs_rng/arcface_boundary_diagnostic.json'},
        {'paper':'P02','finding':'Figure 5 caption says balanced, but counts sum to 1019 and match full-test replay.',
         'reference':'Figure 5, PDF p.13','counts':figure_cm,'full_test_replay_match':True},
        {'paper':'P02','finding':'Table 7 density-1 caption says 150 windows; saved confusion matrix and n_windows_evaluated contain 146.',
         'reference':'Table 7, PDF p.17; congestion_results.json/results/1'},
    ]
    result['unexecuted_results']=[
        {'paper':'P01','result':'five-seed CNN retraining and transformer training trajectories','status':'not_run_exact_inputs_unverified',
         'missing':'original preprocessing CSVs/processed waveforms and original train-row identity/environment; public raw reconstruction is possible but has not been certified exact',
         'available':'raw IDMT/MELAUDIS and training code','no_claim':'Checkpoint agreement does not reproduce training or validate the claimed fine-tuning gradients.'},
        {'paper':'P01','result':'five-evaluation-seed transformer uncertainty','status':'not_reproduced',
         'missing':'released corrected JSONs contain the single seed-42 endpoint; figure code uses zero transformer SD; no complete original five-seed prediction archive'},
        {'paper':'P02','result':'exact balanced inference for supplied snapshots','status':'not_executable_from_current_release',
         'missing':'exact balanced-subset indices or original sampling implementation inside absent soundclass_v1; no seed/subset search performed'},
        {'paper':'P02','result':'five-seed main, matched-count and ratio-sweep retraining','status':'not_executable_from_current_release',
         'missing':'soundclass_v1 training engine, complete derivative-to-raw source membership and all run checkpoints'},
        {'paper':'P02','result':'spectral fidelity and congestion waveform reproduction','status':'not_executable_from_current_local_inputs',
         'missing':'original real derivative audio/order and generated-scene/selection identities; saved arithmetic is separately audited'},
        {'paper':'P02','result':'synthetic generation identity','status':'not_reproduced',
         'missing':'complete per-file source/render lineage; generated-archive names do not certify backend or parameters'},
    ]
    result['summary']={'p01_exact_matrices':sum(e['paper']=='P01' and e['exact_confusion'] for e in result['checkpoint_results']),
        'p01_matrix_comparisons':10,'p02_exact_full_test_matrices':8,'p02_matches_json_best':3,
        'verified_prediction_artifacts':len(result['checkpoint_results']),
        'arithmetic_rows_including_historical_checks':len(result['arithmetic_results']),
        'verdict':'partial reproduction; current-release findings bounded and documented; no training reproduced'}
    result['result_file_hashes']={str(p.relative_to(ROOT)):digest(p) for p in [OUT/'P01_BEATs_rng/replay.json',OUT/'P01_BEATs_rng/input_validation.json',OUT/'P01_BEATs_rng/sampling_manifest.json',OUT/'P01_BEATs_rng/arcface_boundary_diagnostic.json',BASE/'results/P01_CNN/replay.json',BASE/'results/P01_AST_compatibility/replay.json',BASE/'results/P02/replay.json']}
    result['reproduction_scripts']={str(p.relative_to(ROOT)):digest(p) for p in sorted(BASE.glob('*.py'))}
    (OUT/'R0_result_ledger.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))
    print('Table arithmetic differences:',[e for e in result['arithmetic_results'] if e.get('rounds_to_paper') is False or e.get('metric_rounds_to_paper') is False or e.get('confidence_rounds_to_paper') is False])

if __name__=='__main__':main()

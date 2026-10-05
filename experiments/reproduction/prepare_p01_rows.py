"""Recover published balanced-row candidates with explicit pathlib sort semantics."""
from pathlib import Path, PureWindowsPath
from collections import Counter
import argparse
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/reproduction'
CLASSES=['car','truck','motorcycle','bus','background']

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--path-order',choices=['posix','windows'],required=True)
    args=parser.parse_args()
    idmt={c:[] for c in CLASSES}; labelmap={'C':'car','T':'truck','M':'motorcycle','B':'bus'}
    for name in (ROOT/'dataset/IDMT_Traffic/annotation/idmt_traffic_all.txt').read_text().splitlines():
        if '_SE_' not in name:continue
        cls='background' if '-BG' in name else labelmap[name.split('_')[6][0]]
        idmt[cls].append({'original':str(Path('dataset/IDMT_Traffic/audio')/name),'processed_name':name.replace('_CH12','_CH34'),'class':cls})
    mel={c:[] for c in CLASSES}; mmap={'1V-Car':'car','1V-Truck':'truck','1V-MC':'motorcycle','1V-Bus':'bus'}
    for name in (BASE/'results/melaudis_vehicle_members.txt').read_text().splitlines():
        if not name.endswith('.wav'):continue
        labels=[mmap[t] for t in Path(name).name.split('_') if t in mmap]
        if len(labels)==1:
            cls=labels[0];mel[cls].append({'archive':'MELAUDIS_Vehicles.rar','archive_member':name,'processed_name':Path(name).name,'class':cls})
    for name in (BASE/'results/melaudis_background_members.txt').read_text().splitlines():
        if name.endswith('.wav'):mel['background'].append({'archive':'MELAUDIS_ BG.rar','archive_member':name,'processed_name':Path(name).name,'class':'background'})
    selected=[]
    for domain,groups in [('ch34',idmt),('melaudis',mel)]:
        full=[]
        key=lambda r:PureWindowsPath(r['processed_name']) if args.path_order=='windows' else r['processed_name']
        for cls in CLASSES:full.extend(sorted(groups[cls],key=key))
        y=np.load(BASE/'releases/P01/features'/f'y_test_{domain}.npy',allow_pickle=False)
        assert len(full)==len(y) and [CLASSES.index(r['class']) for r in full]==y.tolist()
        assert len({r['processed_name'] for r in full})==len(full)
        rng=np.random.RandomState(42)
        indices=np.concatenate([rng.choice(np.flatnonzero(y==i),50,replace=False) for i in range(5)])
        for i in indices:selected.append(dict(full[i],domain=domain,row_index=int(i),label=int(y[i])))
    result={'status':'candidate mapping; compare reconstructed MFCC rows before interpretation','path_order':args.path_order,'mapping_rule':'released class order and sorted Path basenames; IDMT SE reassignment; MELAUDIS exact single-vehicle label across all traffic states plus all background','sampling_seed':42,'n_per_class':50,'rows':selected}
    (BASE/'results/P01_balanced_row_candidates.json').write_text(json.dumps(result,indent=2)+'\n')
    for archive in ['MELAUDIS_Vehicles.rar','MELAUDIS_ BG.rar']:
        names=[r['archive_member'] for r in selected if r.get('archive')==archive]
        target=BASE/'work'/('vehicle_selected_members.txt' if archive=='MELAUDIS_Vehicles.rar' else 'background_selected_members.txt')
        target.write_text('\n'.join(names)+'\n')
    print('Prepared 500 candidates, path order:',args.path_order)

if __name__=='__main__':main()

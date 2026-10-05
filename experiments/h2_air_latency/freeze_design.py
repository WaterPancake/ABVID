"""Freeze all populations before either new attenuation/latency arm is rendered."""
from collections import Counter
from datetime import datetime,timezone
import shutil
from phase_common import *


def main():
    check_parent()
    FREEZE.mkdir(parents=True,exist_ok=False)
    old=parent.joined_observations()
    index={(r['event_id'],r['source_level'],r['path_level']):r for r in old}
    plans=[]
    for row in old:
        if row['path_level']!='G1A0':continue
        for cell in CELLS:
            prior=CONFIG['parent_cell_mapping'].get(cell)
            keep={key:row[key] for key in ['event_id','geometry_id','template_id','dry_source_id',
                'source_parent_group','source_level','role','class_id','replicate_seed']}
            plans.append(dict(keep,job_id=job_id(row,cell),cell=cell,
                amplitude=cell[1]=='1',latency=cell[3]=='1',ground=True,reused=prior is not None,
                parent_job_id=index[row['event_id'],row['source_level'],prior]['job_id'] if prior else None))
    assert len(plans)==len({p['job_id'] for p in plans})==19200
    assert Counter(p['role'] for p in plans)=={'train':15200,'validation':4000}
    obs={r['job_id']:r for r in old}
    heads={(r['replicate_seed'],r['source_level'],r['path_level'],r['representation']):r
           for r in read(parent.MODELS/'fits.json')}
    fits=[]
    for seed in SEEDS:
        for source in SOURCES:
            for cell in CELLS:
                prior=CONFIG['parent_cell_mapping'].get(cell)
                for rep in REPS:
                    ref=heads[seed,source,prior or 'G1A0',rep]
                    fits.append(dict(fit_id=f'h2l.r{seed}.{source}.{cell}.{rep}',replicate_seed=seed,
                        source_level=source,cell=cell,representation=rep,reused=prior is not None,
                        parent_fit_id=ref['fit_id'] if prior else None,
                        train_job_ids=[job_id(obs[j],cell) for j in ref['train_job_ids']],
                        validation_job_ids=[job_id(obs[j],cell) for j in ref['validation_job_ids']]))
    assert len(fits)==80 and sum(r['reused'] for r in fits)==40
    (FREEZE/'observations.jsonl').write_text(jsonl(plans))
    (FREEZE/'fits.jsonl').write_text(jsonl(fits))
    for name in ['config.json','PROTOCOL.md']:shutil.copy2(HERE/name,FREEZE/name)
    shutil.copy2(parent.FREEZE/'target_manifest.jsonl',FREEZE/'target_manifest.jsonl')
    files=[p for p in PARENT_HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files.append(ROOT/'reports/H2_ground_air_collapse.md')
    files.extend(p for p in (ROOT/'reports/H2_ground_air_collapse_figures').iterdir() if p.is_file())
    inventory=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(files)]
    (FREEZE/'parent_artifacts.jsonl').write_text(jsonl(inventory))
    protected=parent.verify_parent_files()
    save(FREEZE/'lock.json',dict(protocol_id=ID,completed_utc=datetime.now(timezone.utc).isoformat(),
        observations=19200,new_observations=9600,heads=80,new_heads=40,target_events=8066,
        ground_air_parent_files=len(inventory),**protected,target_previously_exposed=True,
        new_arm_results_accessed=False,parent_execution_sha256=sha(parent.EXECUTION/'lock.json'),
        parent_verification_sha256=sha(parent.RESULTS/'verification.json'),
        artifacts_sha256={p.name:sha(p) for p in FREEZE.iterdir() if p.is_file()}))
    print('Frozen attenuation/latency design:',FREEZE,flush=True)


if __name__=='__main__':main()

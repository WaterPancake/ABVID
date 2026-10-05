"""Admit a complete frozen executable only after renderer and pilot replay pass."""
from datetime import datetime,timezone
import shutil
import subprocess
from phase_common import *


def main():
    check_design()
    evidence=[HERE/'results/renderer_validation_v1/report.json',HERE/'results/pilot_v1/verification.json']
    validation=read(evidence[0]);pilot=read(evidence[1])
    if not validation['passed'] or not pilot['passed'] or pilot['complete_waveform_replays']!=240:
        raise ValueError('Admission evidence incomplete')
    for path,digest in validation['code_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Validated renderer changed')
    if sha(HERE/'corpus.py')!=pilot['verifier_sha256']:raise ValueError('Replayed corpus code changed')
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_controls.py','-v'],
        cwd=ROOT,capture_output=True,text=True)
    if tests.returncode:raise ValueError(tests.stdout+tests.stderr)
    preserved=check_preserved()
    EXECUTION.mkdir(parents=True,exist_ok=False)
    (EXECUTION/'tests.txt').write_text(tests.stdout+tests.stderr)
    code=sorted(HERE.glob('*.py'))
    for path in code:shutil.copy2(path,EXECUTION/path.name)
    save(EXECUTION/'lock.json',dict(protocol_id=ID,completed_utc=datetime.now(timezone.utc).isoformat(),
        ready_for_generation=True,new_target_evaluation_started=False,**preserved,
        design_lock_sha256=sha(FREEZE/'lock.json'),parent_execution_sha256=sha(parent.EXECUTION/'lock.json'),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in code},
        evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in evidence},tests_sha256=sha(EXECUTION/'tests.txt')))
    print('Admitted attenuation/latency execution:',EXECUTION,flush=True)


if __name__=='__main__':main()

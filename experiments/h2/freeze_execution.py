"""Admit executable H2 only after independent pre-target gates pass."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
from datetime import datetime,timezone
import difflib
import importlib.metadata
import io
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest

from metadata import HERE,ROOT,read,save,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=read(HERE/'results/source_validation_20261004_v2/report.json')
    physical=read(HERE/'results/renderer_validation_20261004_v2/report.json')
    timing=read(HERE/'results/paired_timing_20261004_v3/summary.json')
    replay=read(HERE/'results/paired_timing_20261004_v3/verification_parallel.json')
    if not all(r['passed'] for r in [source,physical,timing,replay]): raise ValueError('Pre-target gate failure')
    if source['waveform_cases']!=480 or replay['complete_waveform_replays']!=240 or timing['workers']!=4:
        raise ValueError('Gate scope incomplete')
    if source['source_adapter_sha256']!=sha(HERE/'source_adapter.py'): raise ValueError('Source changed after validation')
    for name,digest in physical['code_sha256'].items():
        if sha(HERE/name)!=digest: raise ValueError('Renderer changed after validation: '+name)
    for name,digest in timing['code_sha256'].items():
        if sha(ROOT/name)!=digest: raise ValueError('Generator changed after timing: '+name)
    if replay['corpus_summary_sha256']!=sha(HERE/'results/paired_timing_20261004_v3/summary.json'):
        raise ValueError('Timing replay does not match its corpus')
    if shutil.disk_usage(ROOT).free<11*1024**3: raise ValueError('Less than 11 GiB free for full corpus')
    from learning import h1_cache_compatibility
    compatibility=h1_cache_compatibility();save(out/'target_cache_compatibility.json',compatibility)
    suite=unittest.TestSuite()
    for pattern in ['test_design.py','test_ground_indexing.py','test_execution.py']:
        suite.addTests(unittest.defaultTestLoader.discover(str(HERE),pattern=pattern))
    stream=io.StringIO()
    with redirect_stdout(stream),redirect_stderr(stream):
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (out/'tests.txt').write_text(stream.getvalue())
    if not result.wasSuccessful(): raise ValueError('Execution tests failed; see saved log')
    checked=subprocess.run([sys.executable,str(HERE/'verify_design.py'),'--freeze',str(HERE/'frozen/source_path_20261004_v1'),
        '--output',str(out/'design_verification.json')],cwd=ROOT,text=True,capture_output=True)
    (out/'design_verification.log').write_text(checked.stdout+checked.stderr)
    if checked.returncode: raise ValueError('Immutable design or protected artifact changed')
    # Reconstruct and check the complete scalar-backend patch, keeping upstream intact.
    backend=HERE/'backend';ref=HERE/'evidence/pyroadacoustics';chunks=[]
    for path in sorted((backend/'pyroadacoustics').glob('*.py')):
        relative=path.relative_to(backend); original=ref/relative
        if path.read_bytes()==original.read_bytes():continue
        diff=difflib.unified_diff(original.read_text().splitlines(keepends=True),path.read_text().splitlines(keepends=True),
            fromfile='a/'+str(relative),tofile='b/'+str(relative))
        chunks.extend(line if line.endswith('\n') else line+'\n\\ No newline at end of file\n' for line in diff)
    patch=out/'renderer.patch';patch.write_text(''.join(chunks))
    with tempfile.TemporaryDirectory(prefix='h2-renderer-patch-') as temporary:
        shutil.copytree(ref/'pyroadacoustics',Path(temporary)/'pyroadacoustics',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        applied=subprocess.run(['git','apply',str(patch)],cwd=temporary,text=True,capture_output=True)
        if applied.returncode: raise ValueError(applied.stderr)
        for path in (backend/'pyroadacoustics').glob('*.py'):
            if sha(path)!=sha(Path(temporary)/path.relative_to(backend)): raise ValueError('Patch roundtrip mismatch')
    versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    (out/'environment.freeze.txt').write_text(''.join(f'{k}=={v}\n' for k,v in sorted(versions.items(),key=lambda kv:kv[0].lower())))
    paths=list(HERE.glob('*.py'))+list((backend/'pyroadacoustics').glob('*.py'))
    paths += [HERE/'config.json',backend/'pyroadacoustics/materials.json',
        ROOT/'experiments/h1/extract.py',ROOT/'experiments/h1/freeze.py',ROOT/'src/vehicle_audio/beats_adapter.py']
    code={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
    evidence=[HERE/'IMPLEMENTATION_AMENDMENTS.md',HERE/'frozen/source_path_20261004_v1/lock.json',
        HERE/'results/source_validation_20261004_v2/report.json',HERE/'results/renderer_validation_20261004_v2/report.json',
        HERE/'results/physics_probe_20261004_v2/report.json',HERE/'results/paired_timing_20261004_v3/summary.json',
        HERE/'results/paired_timing_20261004_v3/verification_parallel.json']
    for name in code:
        destination=out/'code'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,destination)
    save(out/'lock.json',dict(execution_id=out.name,completed_utc=datetime.now(timezone.utc).isoformat(),
        ready_for_bulk_generation=True,ready_for_target_evaluation=False,
        design_lock_sha256=sha(HERE/'frozen/source_path_20261004_v1/lock.json'),
        code_sha256=code,evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in evidence},
        environment=versions,python=platform.python_version(),platform=platform.platform(),render_workers=4,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),git_dirty=True,
        tests=result.testsRun,renderer_patch_sha256=sha(patch),target_scores_computed=False,
        artifacts_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
        remaining_gates=['complete_verified_corpus','feature_lock','all_40_fixed_heads_lock_before_target_scores']))
    print('Execution locked:',out,'tests:',result.testsRun,flush=True)


if __name__=='__main__':main()

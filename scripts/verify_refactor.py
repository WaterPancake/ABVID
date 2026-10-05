"""Compare archived and refactored kernels without training or target evaluation."""
import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time

from abvid.paths import archive_root, repository_root
from abvid.provenance.hashing import sha256


def imported(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def symbol_check(repo, archive):
    checked = 0
    class IgnoreImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return ast.Pass()
        def visit_Import(self, node):
            return ast.Pass()
    for row in json.loads((repo/'reports/refactor_symbol_map.json').read_text()):
        before, after = archive/row['original'], repo/row['current']
        assert sha256(before) == row['original_sha256'], str(before)
        original = {n.name:n for n in ast.parse(before.read_text()).body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for node in ast.parse(after.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assert node.name in original, (after,node.name)
                assert ast.dump(IgnoreImports().visit(node), include_attributes=False) == ast.dump(
                    IgnoreImports().visit(original[node.name]), include_attributes=False), (after,node.name)
                checked += 1
    return checked


def cast_check(archive):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    for path in ['CAST/src','CAST/generalization/src','CAST/improvement/src','CAST/improvement/tests']:
        sys.path.insert(0,str(archive/path))
    from cast.config import configuration as old_config
    from cast.renderer import Renderer as OldRenderer, tensors as old_tensors
    from cast.synthetic import fixtures
    from cast_improvement.engine import SmoothRenderer as OldSmooth, model_config as old_smooth_config
    from cast_improvement.resolution_model import ResolutionRenderer as OldResolution
    from cast_improvement.width_prior import WidthPrior as OldWidth
    from test_group_prior import fixture
    from abvid.cast.config import configuration
    from abvid.cast.renderer import Renderer, tensors
    from abvid.cast.population.engine import SmoothRenderer, model_config
    from abvid.cast.population.resolution_model import ResolutionRenderer
    from abvid.cast.population.width_prior import WidthPrior
    assert configuration() == old_config()
    assert model_config() == old_smooth_config()
    renders=0
    for key, params in fixtures().items():
        for old, new, cfg in [(OldRenderer, Renderer, configuration()),
                              (OldSmooth, SmoothRenderer, model_config())]:
            with torch.no_grad():
                a=old(cfg,'refactor_'+key).render(old_tensors(params),'check').numpy()
                b=new(cfg,'refactor_'+key).render(tensors(params),'check').numpy()
            np.testing.assert_array_equal(a,b);renders+=1
    cfg,metric,bank=fixture()
    draws=0
    for temperature in (1.,1.1,1.25,1.5):
        old=OldWidth(bank,cfg,metric,'spectrotemporal_context','held',temperature)
        new=WidthPrior(bank,cfg,metric,'spectrotemporal_context','held',temperature)
        assert old.statistics == new.statistics
        for label in ('car','truck'):
            for arm in ('joint','marginals','prototype'):
                for index in (0,7):
                    a=old.draw(label,42,index,arm);b=new.draw(label,42,index,arm)
                    assert a==b;draws+=1
        with torch.no_grad():
            a=OldResolution(cfg,'refactor_v15',True).render(old_tensors(a['parameters']),'sampling').numpy()
            b=ResolutionRenderer(cfg,'refactor_v15',True).render(tensors(b['parameters']),'sampling').numpy()
        np.testing.assert_array_equal(a,b);renders+=1
    return dict(exact_waveform_pairs=renders,exact_v15_population_draws=draws,exact_population_statistics=True)


def baseline_check(repo, archive):
    import numpy as np
    sys.path[:0]=[str(archive/'experiments/h1'),str(archive/'src')]
    old=imported(archive/'experiments/h1/extract.py','archived_h1_extract')
    from abvid.data.audio import observation
    from abvid.representations.mfcc import mfcc_features
    from abvid.evaluation import factorial
    cfg=json.loads((repo/'configs/reference/h1.json').read_text())
    t=np.arange(3*48000)/48000
    audio=np.c_[np.sin(2*np.pi*123*t),.4*np.sin(2*np.pi*237*t)]
    a=old.observation(audio,48000,cfg['preprocessing']);b=observation(audio,48000,cfg['preprocessing'])
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(old.mfcc_features(a,cfg['mfcc']),mfcc_features(b,cfg['mfcc']))
    sys.path.insert(0,str(archive/'experiments/h2'))
    old_source=imported(archive/'experiments/h2/source_adapter.py','archived_h2_sources')
    old_path=imported(archive/'experiments/h2/renderer.py','archived_h2_renderer')
    old_metrics=imported(archive/'experiments/h2/metrics.py','archived_h2_metrics')
    from abvid.simulation import sources,propagation
    cfg=json.loads((repo/'configs/reference/h2.json').read_text())
    template=json.loads((archive/'experiments/h2/frozen/source_path_20261004_v1/templates.jsonl').read_text().splitlines()[0])
    waveforms=[]
    for level in ('S0','S1'):
        a,ma=old_source.synthesize(template,level,cfg);b,mb=sources.synthesize(template,level,cfg)
        np.testing.assert_array_equal(a,b);assert ma==mb;waveforms.append(b[:2048])
    trajectory=np.tile([0.,10.,.5],(2048,1))
    x=np.stack(waveforms,axis=1)
    a=old_path.render_sources(x,old_path.build_plan(trajectory),components=True)
    b=propagation.render_sources(x,propagation.build_plan(trajectory),components=True)
    for key in ('P0','P1','direct_filtered','reflected'):np.testing.assert_array_equal(a[key],b[key])
    y=np.array([0,1,0,1,1,0]);p=np.array([.1,.7,.6,.8,.3,.2])
    assert old_metrics.score(y,p)==factorial.score(y,p)
    return dict(exact_observation=True,exact_mfcc=True,exact_source_waveforms=2,
                exact_path_component_arrays=4,exact_metric_dictionary=True,
                target_access=False,training_runs=0)


def storage_check(archive):
    original=json.loads((archive.parent/'migration-20261005/archive.json').read_text())
    import subprocess
    assert subprocess.check_output(['git','diff','--binary'],cwd=archive) == (archive.parent/'migration-20261005/tracked.diff').read_bytes()
    sys.path.insert(0,str(archive/'CAST/src'))
    from cast.config import configuration
    from cast.provenance import read_frozen, checked_path
    cfg, permitted, lock = read_frozen(archive/'CAST/runs/cast_pilot_v0_20261004')
    for row in permitted:
        checked_path(row,cfg,permitted)
    for row in original['source_hashes']:
        assert sha256(archive/row['path'])==row['sha256'],row['path']
    manifest=archive/'experiments/h1/frozen/h1_common_budget_v1.3/admitted.jsonl'
    rows=[json.loads(line) for line in manifest.read_text().splitlines()]
    missing=[r['source_path'] for r in rows if not (archive/r['source_path']).is_file()]
    assert not missing,missing[:5]
    return dict(unchanged_archived_source_files=len(original['source_hashes']),
                original_git_tracked_diff_unchanged=True,
                frozen_CAST_allowlist_hash_checks=len(permitted),
                resolved_admitted_audio_paths=len(rows),missing_paths=0)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['cast','baseline','storage'],required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    start=time.perf_counter();repo=repository_root();archive=archive_root()
    report={'profile':args.profile,'comparison':'working_refactor_vs_preserved_original','numerical_definitions_unchanged':symbol_check(repo,archive)}
    if args.profile=='cast':report.update(cast_check(archive))
    elif args.profile=='baseline':report.update(baseline_check(repo,archive))
    else:report.update(storage_check(archive))
    report.update(passed=True,elapsed_s=time.perf_counter()-start)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()

"""Synthetic-only source/reference/manipulation checks before H2 generation."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.signal import butter, lfilter, sosfiltfilt, hilbert, welch

from metadata import read, rows, save, sha
from source_adapter import synthesize, array_sha, rms, gaussian, peak_normalize

HERE=Path(__file__).resolve().parent


class SuppliedDraws:
    """Feed frozen draws through the unchanged released dry functions."""
    def __init__(self, uniform, normal):
        self.uniforms=iter(uniform); self.normals=iter(normal)
    def uniform(self, low, high):
        value=next(self.uniforms)
        if not low<=value<=high: raise ValueError('Frozen draw outside released call bounds')
        return value
    def normal(self, loc, scale, count):
        assert loc==0 and scale==1
        return gaussian(next(self.normals),count)
    def exhausted(self):
        sentinel=object()
        return next(self.uniforms,sentinel) is sentinel and next(self.normals,sentinel) is sentinel


def released_functions():
    path=HERE/'evidence/p02/_2_genSyntheticData.py'
    tree=ast.parse(path.read_text())
    names={'make_engine_harmonics','make_tire_noise','make_exhaust_noise','generate_car_source','generate_truck_source'}
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if len(functions)!=5: raise ValueError('Reference functions changed')
    scope=dict(np=np,butter=butter,lfilter=lfilter)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),scope)
    return scope


def reference_source(t, functions):
    b=t['base']; values=[b['rpm'],*b['engine_phases_rad'],b['engine_am_depth'],b['tire_center_hz'],b['tire_width_hz']]
    noises=[b['tire_noise_seed']]
    if t['canonical_class']=='truck':
        values.extend(b['exhaust_phases_rad']); noises.append(b['exhaust_noise_seed'])
        values.extend(b['weights'][k] for k in ['engine','tire','exhaust'])
    rng=SuppliedDraws(values,noises)
    x,rpm=functions['generate_'+t['canonical_class']+'_source'](10.,8000,rng)
    assert rng.exhausted() and rpm==b['rpm']
    return x*(.1/rms(x[32000:48000]))


def measure_component(engine, template, level):
    """Measure amplitudes by least squares and frequency by filtered Hilbert phase.

    Phase bases come from the declared law, not adapter diagnostic outputs.
    The separate waveform-based frequency estimate detects incorrect FM synthesis.
    """
    fs=8000; n=len(engine); b=template['base']; s=template['S1']; time=np.arange(n)/fs
    f=np.full(n,b['firing_hz'])
    if level=='S1': f*=1+s['frequency_fractional_amplitude']*np.sin(2*np.pi*s['frequency_modulation_hz']*time+s['phase_rad'])
    phase=np.zeros(n); phase[1:]=np.add.accumulate(f[:-1])*(2*np.pi/fs)
    if level=='S0': phase=2*np.pi*b['firing_hz']*time
    central=slice(32000,48000)
    unmodulated=engine[central]/(1+b['engine_am_depth']*np.sin(2*np.pi*(b['firing_hz']/2)*time[central]))
    bases=np.column_stack([func(h*phase[central]) for h in range(1,b['harmonics']+1) for func in [np.sin,np.cos]])
    amplitudes=np.linalg.lstsq(bases,unmodulated,rcond=None)[0].reshape(-1,2)
    amplitudes=np.hypot(amplitudes[:,0],amplitudes[:,1])
    rolloff=b['engine_rolloff_db_per_order']+(s['engine_rolloff_offset_db_per_order'] if level=='S1' else 0.)
    expected=10**(-rolloff/20)
    ratio_error=float(np.max(np.abs((amplitudes[1:]/amplitudes[:-1])/expected-1)))
    # Use a full-recording, zero-phase analysis filter; this is never preprocessing
    # for the classifier or an operation in the source generator.
    sos=butter(6,[b['firing_hz']*.8,b['firing_hz']*1.2],btype='band',fs=fs,output='sos')
    analytic=hilbert(sosfiltfilt(sos,engine))
    measured=np.diff(np.unwrap(np.angle(analytic)))*fs/(2*np.pi)
    frequency_error=float(np.max(np.abs(measured[32000:48000]/f[32000:48000]-1)))
    return dict(max_harmonic_ratio_relative_error=ratio_error,
        max_instantaneous_firing_relative_error=frequency_error,
        measured_firing_min_hz=float(measured[32000:48000].min()),
        measured_firing_max_hz=float(measured[32000:48000].max()),
        passed=bool(ratio_error<=.01 and frequency_error<=.01))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False); start=time.perf_counter()
    freeze=HERE/'frozen/source_path_20261004_v1'; cfg=read(freeze/'config.json')
    templates=rows(freeze/'templates.jsonl'); functions=released_functions(); records=[]
    for index,t in enumerate(templates):
        generated={}
        for level in ['S0','S1']:
            x,meta,components,diagnostics=synthesize(t,level,cfg,True)
            replay,_=synthesize(t,level,cfg)
            checks=measure_component(components['engine'],t,level)
            record=dict(template_id=t['template_id'],source_level=level,**checks,
                waveform_sha256=array_sha(x),replay_identical=array_sha(x)==array_sha(replay),
                rms_relative_error=abs(rms(x[32000:48000])/.1-1),shape=list(x.shape),finite=bool(np.isfinite(x).all()))
            if level=='S0':
                ref=reference_source(t,functions)
                error=float(np.max(np.abs(x-ref)))
                record['released_source_max_absolute_error']=error
                record['passed'] &= error<=1e-9
            record['passed'] &= record['replay_identical'] and record['rms_relative_error']<=1e-6 and record['shape']==[80000] and record['finite']
            records.append(record); generated[level]=(x,components,diagnostics)
        x0,c0,d0=generated['S0']; x1,c1,d1=generated['S1']
        unchanged=np.array_equal(c0['tire'],c1['tire'])
        if 'exhaust' in c0: unchanged &= np.array_equal(d0['exhaust_broadband'],d1['exhaust_broadband'])
        frequency,power0=welch(x0[32000:48000],fs=8000,nperseg=2048)
        _,power1=welch(x1[32000:48000],fs=8000,nperseg=2048)
        records[-1].update(unchanged_noise_components=bool(unchanged),
            aggregate_source_relative_l2=float(np.linalg.norm(x1-x0)/np.linalg.norm(x0)),
            S0_spectral_centroid_hz=float(np.sum(frequency*power0)/power0.sum()),
            S1_spectral_centroid_hz=float(np.sum(frequency*power1)/power1.sum()))
        records[-1]['passed'] &= bool(unchanged) and not np.array_equal(x0,x1)
        if index%40==0: print(f'source checks {index+1}/{len(templates)}',flush=True)
    report=dict(passed=all(r['passed'] for r in records),templates=len(templates),waveform_cases=len(records),
        records=records,target_access=False,elapsed_s=time.perf_counter()-start,
        source_adapter_sha256=sha(HERE/'source_adapter.py'),checker_sha256=sha(Path(__file__)),
        reference_sha256=sha(HERE/'evidence/p02/_2_genSyntheticData.py'),design_lock_sha256=sha(freeze/'lock.json'))
    save(out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
    failures=[r for r in records if not r['passed']]
    if failures:
        print('Failed:',json.dumps(failures[:10])); raise SystemExit(1)


if __name__=='__main__': main()

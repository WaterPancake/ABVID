"""Synthetic-only analytic tests of separate air and ground conditions."""
import argparse
import time
from common import *
from render import render_all
from renderer import build_plan,render_sources,reference_manager
from physics_probe import scalar_render,static_position,tone_amplitude
from validate_renderer import complex_response,observed_frequency
from source_adapter import array_sha


def main(out):
    check_design();out.mkdir(parents=True,exist_ok=False);begin=time.perf_counter();records=[]
    def add(kind,passed,**values):
        records.append(dict(check=kind,passed=bool(passed),**values))
        save(out/'progress.json',dict(records=records,target_access=False))
    n=52000;t=np.arange(n)/8000
    x=.1*np.sin(2*np.pi*731*t)+.03*np.random.default_rng(314159).normal(size=n)
    trajectory=np.column_stack([np.full(n,5.),25*(t-3.25),np.full(n,.5)])
    result=render_all(x,trajectory)
    for path,g,a in [('G0A1',False,True),('G1A0',True,False)]:
        scalar=scalar_render(x,trajectory,g,a)
        error=float(np.abs(result[path][:,0]-scalar).max())
        relative=float(np.linalg.norm(result[path][:,0]-scalar)/np.linalg.norm(scalar))
        add('moving_scalar_buffer_wrap',error<=1e-10 and relative<=1e-9,path=path,
            samples=n,max_absolute_error=error,relative_l2_error=relative)
    old=render_sources(x,build_plan(trajectory))
    add('P0_endpoint_exact',np.array_equal(result['G0A0'],old['P0']))
    add('P1_endpoint_exact',np.array_equal(result['G1A1'],old['P1']))
    repeat=render_all(x,trajectory)
    for path in PATHS:add('exact_repeat',array_sha(repeat[path])==array_sha(result[path]),path=path)
    frequencies=[125.,500.,1500.,3000.];n=16000;t=np.arange(n)/8000;obj=reference_manager()
    tones=np.column_stack([np.sin(2*np.pi*f*t) for f in frequencies])
    for distance,height,mic_h in [(5.,.5,1.2),(20.,.5,1.2),(50.,.5,1.2),(5.,1.2,.5)]:
        tr=static_position(distance,height,n);mic=np.array([0.,0.,mic_h]);p=build_plan(tr,mic)
        rendered=render_all(tones,tr,mic)
        for fi,f in enumerate(frequencies):
            direct=np.exp(-2j*np.pi*f*p['d'][0]/obj.c)/p['d'][0]
            reflection=complex_response(p['ground'][0],f)*np.exp(-2j*np.pi*f*(p['a'][0]+p['b'][0])/obj.c)/(p['a'][0]+p['b'][0])
            air_direct=complex_response(p['air'][0][0],f)
            air_reflected=complex_response(p['air'][1][0],f)*complex_response(p['air'][2][0],f)
            expected=dict(G0A0=direct,G0A1=direct*air_direct,G1A0=direct+reflection,
                G1A1=direct*air_direct+reflection*air_reflected)
            for path in PATHS:
                wave=rendered[path][:,fi]
                error=float(20*np.log10(tone_amplitude(wave[8000:],f)/abs(expected[path])))
                frequency_error=float(abs(observed_frequency(wave)[8000:14000]/f-1).max())
                add('static_transfer',abs(error)<=.5 and frequency_error<=.01,path=path,
                    distance_m=distance,source_height_m=height,mic_height_m=mic_h,frequency_hz=f,
                    magnitude_error_db=error,frequency_relative_error=frequency_error)
        print('toggle static checks',distance,height,mic_h,flush=True)
    report=dict(passed=all(r['passed'] for r in records),records=records,target_access=False,
        elapsed_s=time.perf_counter()-begin,code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),HERE/'render.py']})
    save(out/'report.json',report);print('Toggle validation',len(records),'cases:',report['passed'],flush=True)
    if not report['passed']:raise ValueError([r for r in records if not r['passed']])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();main(a.output)

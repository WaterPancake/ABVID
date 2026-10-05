"""Synthetic-only amplitude/latency admission: FIR algebra, phase and endpoints."""
import argparse
import time
from phase_common import *
from phase_renderer import centered_fir,render_all
from renderer import air_filters,build_plan,render_sources,reference_manager
from physics_probe import static_position
from source_adapter import array_sha


def response(h,f):
    return np.dot(h,np.exp(-2j*np.pi*f*np.arange(len(h))/8000))


def measured_response(wave,f,lo=8000,hi=14000):
    t=np.arange(lo,hi)/8000
    basis=np.column_stack([np.sin(2*np.pi*f*t),np.cos(2*np.pi*f*t),np.ones(len(t))])
    values=np.linalg.lstsq(basis,wave[lo:hi],rcond=None)[0]
    return values[0]+1j*values[1]


def main(out):
    check_design();out.mkdir(parents=True,exist_ok=False);begin=time.perf_counter();checks=[]
    def add(name,passed,**kw):
        checks.append(dict(check=name,passed=bool(passed),**kw))
        save(out/'progress.json',dict(records=checks,target_access=False))
    rng=np.random.default_rng(314159);x=rng.normal(size=(503,3))
    h=rng.normal(size=(503,11));h=(h+h[:,::-1])/2
    scalar=np.zeros_like(x)
    for n in range(len(x)):
        for k in range(11):
            j=n+5-k
            if 0<=j<len(x):
                for c in range(3):scalar[n,c]+=h[n,k]*x[j,c]
    actual=centered_fir(x,h)
    error=float(abs(actual-scalar).max())
    add('moving_coefficients_scalar_stencil',error<1e-12,max_error=error)
    identity=np.zeros((len(x),11));identity[:,5]=1
    add('centered_delayed_identity_is_identity',np.array_equal(centered_fir(x,identity),x))
    obj=reference_manager();geometry=rows(parent.PARENT_DESIGN/'geometries.jsonl')
    maximum=max(float(np.sqrt(g['range_m']**2+(g['speed_kmh']/3.6*5)**2+
        (g['source_height_m']+g['mic_xyz_m'][2])**2)) for g in geometry)
    distances=np.linspace(0,maximum,1025);coeff=air_filters(distances,obj)
    w=np.linspace(0,np.pi,2049)
    amplitude=coeff[:,5,None]+2*coeff[:,6:]@np.cos(np.arange(1,6)[:,None]*w)
    add('centered_air_real_positive_grid',np.all(amplitude>0),minimum_response=float(amplitude.min()),
        distance_max_m=maximum,distance_points=1025,frequency_points=2049)
    add('air_filter_symmetry',np.array_equal(coeff,coeff[:,::-1]))
    n=52000;t=np.arange(n)/8000;wave=.1*np.sin(2*np.pi*731*t)+.03*rng.normal(size=n)
    trajectory=np.column_stack([np.full(n,5.),25*(t-3.25),np.full(n,.5)])
    result=render_all(wave,trajectory);plan=build_plan(trajectory)
    add('ground_only_endpoint_exact',np.array_equal(result['A0L0'],render_sources(wave,dict(plan,air=None))['P1']))
    add('joint_endpoint_exact',np.array_equal(result['A1L1'],render_sources(wave,plan)['P1']))
    repeated=render_all(wave,trajectory)
    for cell in CELLS:add('exact_repeat',array_sha(result[cell])==array_sha(repeated[cell]),cell=cell)
    n=16000;t=np.arange(n)/8000;frequencies=[125.,500.,1500.,3000.]
    tones=np.column_stack([np.sin(2*np.pi*f*t) for f in frequencies])
    for distance,height,mic_height in [(5.,.5,1.2),(20.,.5,1.2),(50.,.5,1.2),(5.,1.2,.5)]:
        trajectory=static_position(distance,height,n);mic=np.array([0.,0.,mic_height])
        plan=build_plan(trajectory,mic);rendered=render_all(tones,trajectory,mic)
        for fi,f in enumerate(frequencies):
            omega=2*np.pi*f/8000
            direct=np.exp(-2j*np.pi*f*plan['d'][0]/obj.c)/plan['d'][0]
            reflected=response(plan['ground'][0],f)*np.exp(-2j*np.pi*f*(plan['a'][0]+plan['b'][0])/obj.c)/(plan['a'][0]+plan['b'][0])
            hd,ha,hb=[response(v[0],f) for v in plan['air']]
            expected=dict(A0L0=direct+reflected,A1L0=direct*abs(hd)+reflected*abs(ha*hb),
                A0L1=direct*np.exp(-5j*omega)+reflected*np.exp(-10j*omega),
                A1L1=direct*hd+reflected*ha*hb)
            for cell in CELLS:
                measured=measured_response(rendered[cell][:,fi],f)
                ratio=measured/expected[cell]
                magnitude_error=float(20*np.log10(abs(ratio)));phase_error=float(np.angle(ratio))
                add('static_complex_transfer',abs(magnitude_error)<=.5 and abs(phase_error)<=.05,
                    cell=cell,distance_m=distance,source_height_m=height,mic_height_m=mic_height,
                    frequency_hz=f,magnitude_error_db=magnitude_error,phase_error_rad=phase_error)
        print('phase-control static checks',distance,height,mic_height,flush=True)
    result=dict(passed=all(v['passed'] for v in checks),records=checks,target_access=False,
        elapsed_s=time.perf_counter()-begin,code_sha256={str(p.relative_to(ROOT)):sha(p)
            for p in [Path(__file__),HERE/'phase_renderer.py']})
    save(out/'report.json',result);print('Phase-control renderer checks',len(checks),result['passed'],flush=True)
    if not result['passed']:raise ValueError([v for v in checks if not v['passed']])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    main(ap.parse_args().output)

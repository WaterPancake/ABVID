"""Pre-target numerical diagnostics for the repaired scalar reference backend."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy import signal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/"backend"))
from pyroadacoustics import Environment, SimulatorManager, DelayLine


def make_manager(reflection=False, air=False, unit_ground=False):
    env = Environment(fs=8000, temperature=20, pressure=1, rel_humidity=50, road_material=20000)
    params = {"interp_method":"Sinc", "include_reflected_path":reflection, "include_air_absorption":air}
    obj = SimulatorManager(env.c, env.fs, env.Z0, env.road_material, env.air_absorption_coefficients, params)
    if unit_ground:
        obj._get_asphalt_reflection_filter = lambda theta, dist: np.r_[1., np.zeros(39)]
    return obj


def scalar_render(x, trajectory, reflection=False, air=False, unit_ground=False, mic=None):
    if mic is None: mic = np.array([0.,0.,1.2])
    obj = make_manager(reflection, air, unit_ground)
    obj.initialize(trajectory, mic, 0., "omnidirectional", 0., "omnidirectional")
    return np.array([obj.update(position, mic, sample) for position, sample in zip(trajectory,x)])


def static_position(distance, height, n):
    return np.tile(np.array([distance,0.,height]), (n,1))


def tone_amplitude(x, f, fs=8000):
    t = np.arange(len(x))/fs
    return float(np.hypot(*(np.linalg.lstsq(np.c_[np.sin(2*np.pi*f*t), np.cos(2*np.pi*f*t), np.ones(len(t))], x, rcond=None)[0][:2])))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,required=True); args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter(); fs=8000; records=[]
    obj=make_manager()
    for distance in [5.,20.,50.]:
        n=4000; trajectory=static_position(distance,.5,n); x=np.zeros(n); x[100]=1
        direct=scalar_render(x,trajectory)
        total=scalar_render(x,trajectory,True,False,True); reflected=total-direct
        d=float(np.hypot(distance,.7)); R=float(np.hypot(distance,1.7))
        for name,y,r in [("direct",direct,d),("unity_reflected",reflected,R)]:
            expected=100+fs*r/obj.c
            energy=np.abs(y)
            centroid=float(np.sum(np.arange(n)*energy)/energy.sum())
            row=dict(check="static_impulse",component=name,distance_m=distance,
                amplitude_relative_error=abs(y.sum()*r-1),delay_error_samples=centroid-expected,
                expected_delay_samples=expected-100)
            row['passed']=bool(row['amplitude_relative_error']<=.01 and abs(row['delay_error_samples'])<=2)
            records.append(row)
        for f in [125.,500.,1500.,3000.]:
            x=np.sin(2*np.pi*f*np.arange(n)/fs)
            direct=scalar_render(x,trajectory)
            total=scalar_render(x,trajectory,True,False,True)
            reflected=total-direct
            for name,y,r in [("direct",direct,d),("unity_reflected",reflected,R)]:
                err=20*np.log10(tone_amplitude(y[2000:],f)*r)
                records.append(dict(check="static_tone",component=name,distance_m=distance,frequency_hz=f,
                    amplitude_error_db=float(err),passed=bool(abs(err)<=.5)))
        print(f"static checks at {distance}m complete",flush=True)
    for distance in [5.,20.,50.,135.]:
        h=obj._compute_air_absorption_filter(distance,11)
        f,H=signal.freqz(h, worN=1025,fs=fs)
        expected=10**(-np.interp(f,np.linspace(0,fs/2,20),obj.airAbsorptionCoefficients)*distance/20)
        difference=20*np.log10(np.maximum(np.abs(H),1e-15)/expected)
        records.append(dict(check="air_fir",distance_m=distance,max_gain_db=float(20*np.log10(abs(H).max())),
            max_magnitude_error_db=float(abs(difference).max()),passed=bool(abs(difference).max()<=.5 and abs(H).max()<=10**(.5/20))))
    code_files=[Path(__file__), *sorted((HERE/'backend/pyroadacoustics').glob('*.py'))]
    report=dict(records=records,passed=all(r['passed'] for r in records),elapsed_s=time.perf_counter()-start,
                code_sha256={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files},
                target_access=False,scope="static_reference_and_unity_reflection;moving_validation_pending")
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
    if not report['passed']:
        print('Failed cases:',json.dumps([r for r in records if not r['passed']]))
        raise SystemExit(1)


if __name__=='__main__': main()

"""Analytic and scalar-equivalence gates; never reads real data or model scores."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.signal import hilbert, savgol_filter, freqz

from metadata import read, rows, save, sha
from source_adapter import array_sha
from physics_probe import scalar_render, tone_amplitude, static_position
from renderer import build_plan, render_sources, exact_trajectory, reference_manager, fractional_delay

HERE=Path(__file__).resolve().parent


def retarded_frequencies(receiver_time, lateral, speed, height_delta, frequency, c):
    """Independent fixed-point solution t_r=t_e+R(t_e)/c, then analytic derivative."""
    emission=receiver_time.copy()
    for _ in range(20):
        r=np.sqrt(lateral*lateral+(speed*(emission-5.))**2+height_delta**2)
        emission=receiver_time-r/c
    r=np.sqrt(lateral*lateral+(speed*(emission-5.))**2+height_delta**2)
    residual=np.max(np.abs(emission+r/c-receiver_time))
    if residual>1e-12: raise ValueError('Independent retarded-time solve did not converge')
    radial=speed**2*(emission-5.)/r
    return frequency/(1+radial/c)


def observed_frequency(waveform,fs=8000):
    phase=np.unwrap(np.angle(hilbert(waveform)))
    # Fixed 41-sample Savitzky-Golay derivative, used for all paths/frequencies.
    return savgol_filter(phase,41,3,deriv=1,delta=1/fs)/(2*np.pi)


def complex_response(coefficients,frequency,fs=8000):
    return np.sum(coefficients*np.exp(-2j*np.pi*frequency*np.arange(len(coefficients))/fs))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False); started=time.perf_counter(); records=[]
    obj=reference_manager(); fs=8000; frequencies=[125.,500.,1500.,3000.]
    def add(check,passed,**data):
        records.append(dict(check=check,passed=bool(passed),**data))
        save(out/'progress.json',dict(records=records,target_access=False))
    # Finite artificial waveform; compare complete outputs, including startup and
    # a circular-buffer wrap, against independent scalar execution.
    n=52000; tt=np.arange(n)/fs
    x=.1*np.sin(2*np.pi*731*tt)+.03*np.random.default_rng(314159).normal(size=n)
    trajectory=np.column_stack([np.full(n,5.),25*(tt-3.25),np.full(n,.5)])
    plan=build_plan(trajectory); fast=render_sources(x,plan,True)
    for level,reflection,air in [('P0',False,False),('P1',True,True)]:
        begin=time.perf_counter(); scalar=scalar_render(x,trajectory,reflection,air)
        error=float(np.max(np.abs(fast[level][:,0]-scalar)))
        relative=float(np.linalg.norm(fast[level][:,0]-scalar)/np.linalg.norm(scalar))
        add('scalar_equivalence',error<=1e-10 and relative<=1e-9,path=level,
            max_absolute_error=error,relative_l2_error=relative,samples=n,scalar_s=time.perf_counter()-begin)
        print('scalar equivalence',level,error,relative,flush=True)
    replay=render_sources(x,build_plan(trajectory))
    for level in ['P0','P1']:
        identical=array_sha(fast[level])==array_sha(replay[level])
        add('full_render_replay',identical,path=level,first_sha256=array_sha(fast[level]),second_sha256=array_sha(replay[level]))
    # Static individual and combined transfer, with actual ground/air FIRs.
    n=16000; time_axis=np.arange(n)/fs
    tones=np.column_stack([np.sin(2*np.pi*f*time_axis) for f in frequencies])
    for distance,source_height,mic_height in [(5.,.5,1.2),(20.,.5,1.2),(50.,.5,1.2),(5.,1.2,.5),(20.,2.,.2)]:
        trajectory=static_position(distance,source_height,n); mic=np.array([0.,0.,mic_height])
        plan=build_plan(trajectory,mic); rendered=render_sources(tones,plan,True)
        d=plan['d'][0]; a=plan['a'][0]; b=plan['b'][0]; R=a+b
        for fi,f in enumerate(frequencies):
            omega=2*np.pi*f
            direct_expected=complex_response(plan['air'][0][0],f)*np.exp(-1j*omega*d/obj.c)/d
            reflected_expected=(complex_response(plan['air'][1][0],f)*complex_response(plan['ground'][0],f)
                *complex_response(plan['air'][2][0],f)*np.exp(-1j*omega*R/obj.c)/R)
            for component,expected in [('direct_filtered',direct_expected),('reflected',reflected_expected),('P1',direct_expected+reflected_expected)]:
                amplitude=tone_amplitude(rendered[component][8000:,fi],f)
                error=float(20*np.log10(amplitude/abs(expected)))
                measured=observed_frequency(rendered[component][:,fi])[8000:14000]
                frequency_error=float(np.max(np.abs(measured/f-1)))
                add('static_actual_transfer',abs(error)<=.5 and frequency_error<=.01,
                    distance_m=distance,source_height_m=source_height,mic_height_m=mic_height,
                    frequency_hz=f,component=component,magnitude_error_db=error,
                    max_frequency_relative_error=frequency_error,expected_amplitude=float(abs(expected)))
        # Independent unity-reflection impulse check, with unequal heights too.
        unity_plan=build_plan(trajectory,mic,air=False,unit_ground=True)
        impulse=np.zeros(n); impulse[100]=1.
        unity=render_sources(impulse,unity_plan,True)
        for component,r in [('P0',d),('reflected',R)]:
            wave=unity[component][:,0]; centroid=np.dot(np.arange(n),abs(wave))/abs(wave).sum()
            delay_error=float(centroid-100-fs*r/obj.c); amplitude_error=float(abs(wave.sum()*r-1))
            add('static_unity_impulse',abs(delay_error)<=2 and amplitude_error<=.01,
                component=component,distance_m=distance,source_height_m=source_height,mic_height_m=mic_height,
                delay_error_samples=delay_error,amplitude_relative_error=amplitude_error)
        print('static transfer',distance,source_height,mic_height,'complete',flush=True)
    # Exact frozen geometry corners and independent moving-source frequency law.
    n=80000; time_axis=np.arange(n)/fs
    tones=np.column_stack([np.sin(2*np.pi*f*time_axis) for f in frequencies])
    for distance in [5.,50.]:
        for speed in [30.,90.]:
            for direction in [-1,1]:
                geometry=dict(sample_count=n,range_m=distance,speed_kmh=speed,direction=direction,source_height_m=.5)
                trajectory=exact_trajectory(geometry); plan=build_plan(trajectory,air=False,unit_ground=True)
                waves=render_sources(tones,plan,True)
                center=5+np.hypot(distance,.7)/obj.c
                lo=int(np.floor(fs*(center-1)+.5)); hi=lo+16000
                max_speed_error=float(np.max(np.abs(np.diff(trajectory[:,1])*fs-direction*speed/3.6)))
                add('exact_sample_grid',max_speed_error<=1e-8,distance_m=distance,speed_kmh=speed,direction=direction,max_speed_error_m_s=max_speed_error)
                for component,height_delta in [('P0',.7),('reflected',1.7)]:
                    for fi,f in enumerate(frequencies):
                        expected=retarded_frequencies(time_axis,distance,speed/3.6,height_delta,f,obj.c)
                        measured=observed_frequency(waves[component][:,fi])
                        errors=np.abs(measured[lo:hi]/expected[lo:hi]-1)
                        add('moving_retarded_frequency',float(errors.max())<=.01,
                            distance_m=distance,speed_kmh=speed,direction=direction,component=component,frequency_hz=f,
                            max_relative_error=float(errors.max()),rms_relative_error=float(np.sqrt(np.mean(errors**2))))
                print('moving checks',distance,speed,direction,'complete',flush=True)
    # Passive local plane-wave coefficient and air filters. Q includes a curved-
    # wave/ground-wave correction and is not a local energy reflection ratio.
    plane_max=float(np.abs(obj.Rp).max()); add('plane_wave_passivity',plane_max<=10**(.5/20),max_amplitude=plane_max)
    field_gain=[]
    for distance in [5.,20.,50.,135.]:
        h=obj._compute_air_absorption_filter(distance,11); f,H=freqz(h,worN=1025,fs=fs)
        expected=10**(-np.interp(f,np.linspace(0,fs/2,20),obj.airAbsorptionCoefficients)*distance/20)
        error=float(np.max(np.abs(20*np.log10(np.abs(H)/expected))))
        add('air_response_passivity',error<=.5 and np.abs(H).max()<=10**(.5/20),distance_m=distance,
            max_magnitude_error_db=error,max_gain_db=float(20*np.log10(abs(H).max())),latency_samples=5)
        for angle in [0.,30.,60.,85.,89.]:
            q=obj._get_asphalt_reflection_filter(angle,distance); _,Q=freqz(q,worN=1025,fs=fs)
            field_gain.append(dict(distance_m=distance,angle_deg=angle,max_gain_db=float(20*np.log10(abs(Q).max()))))
    code=[Path(__file__),HERE/'renderer.py',HERE/'physics_probe.py',*sorted((HERE/'backend/pyroadacoustics').glob('*.py'))]
    report=dict(passed=all(r['passed'] for r in records),records=records,field_reflection_gains=field_gain,
        field_gain_note='Spherical field coefficient Q is not capped to a local plane-wave energy coefficient; see implementation amendment.',
        elapsed_s=time.perf_counter()-started,target_access=False,code_sha256={str(p.relative_to(HERE)):sha(p) for p in code},
        ground_fir_taps=40,air_fir_taps=11,sinc_taps=31,real_road_calibration=False)
    save(out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['records','code_sha256','field_reflection_gains']}),flush=True)
    if not report['passed']:
        print('Failed:',json.dumps([r for r in records if not r['passed']])); raise SystemExit(1)


if __name__=='__main__': main()

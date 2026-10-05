"""Batched execution of the repaired scalar H2 renderer's operations.

Only the declared omnidirectional point-source, stationary mono-microphone case
is supported. Geometry-dependent filters are shared across source columns. No
distance/angle interpolation, new physics, target access or fallback is used.
The independent scalar implementation remains the equivalence reference.
"""
from functools import lru_cache
import cmath
import math
from pathlib import Path
import sys
import time

import numpy as np
from numba import njit
from scipy.special import erfcx
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'backend'))
from pyroadacoustics import Environment, SimulatorManager


@lru_cache(maxsize=1)
def reference_manager():
    env = Environment(fs=8000, temperature=20., pressure=1., rel_humidity=50., road_material=20000)
    obj = SimulatorManager(env.c, env.fs, env.Z0, env.road_material, env.air_absorption_coefficients,
        dict(interp_method='Sinc', include_reflected_path=True, include_air_absorption=True))
    # Evaluate the same complex equations in a fixed scalar order. Vector complex
    # arithmetic in this environment exhibited allocation-dependent last bits.
    # Do not quantize waveforms or relax the exact replay requirement.
    impedance=[complex(1+9.08*(1000*float(f)/20000)**(-.75),
        -11.9*(1000*float(f)/20000)**(-.73)) for f in obj.f_tmp]
    for i,angle in enumerate(obj._theta_vector):
        cosine=math.cos(math.radians(int(angle))); sine=math.sin(math.radians(int(angle)))
        for j,z in enumerate(impedance):
            inverse=1/z
            obj.w_tmp[i,j]=1+inverse*cosine-cmath.sqrt(1-inverse*inverse)*sine
            obj.Rp[i,j]=(z*cosine-1)/(z*cosine+1)
    return obj


def exact_trajectory(geometry):
    if geometry['sample_count'] != 80000:
        raise ValueError('H2 requires the complete ten-second source grid')
    t = np.arange(80000, dtype=np.float64)/8000
    return np.column_stack([np.full(80000, geometry['range_m']),
        geometry['direction']*(geometry['speed_kmh']/3.6)*(t-5.),
        np.full(80000, geometry['source_height_m'])])


@njit(cache=False, fastmath=False)
def fractional_delay(x, delay, table, delta):
    """Same pointer recurrence and ordered sinc sum as the scalar delay line."""
    count, columns = x.shape
    out = np.zeros_like(x)
    size = 48000
    pointer = -delay[0]
    if pointer < 0: pointer += size
    write_pointer = 0
    taps = table.shape[1]
    half = taps//2
    for n in range(count):
        write_pointer += 1
        integer = int(pointer)
        frac = pointer-integer
        coordinate = frac/delta
        position = min(int(np.floor(coordinate)), len(table)-2)
        weight = coordinate-position
        for k in range(taps):
            index = (integer+k-half) % size
            age = ((write_pointer-1)-index) % size
            j = n-age
            coefficient = (1-weight)*table[position,k]+weight*table[position+1,k]
            if j >= 0:
                for column in range(columns):
                    out[n,column] += coefficient*x[j,column]
        pointer = write_pointer-delay[n]
        while pointer < 0: pointer += size
        while write_pointer >= size: write_pointer -= size
        while pointer >= size: pointer -= size
    return out


@njit(cache=False, fastmath=False)
def time_varying_fir(x, coefficients):
    out = np.zeros_like(x)
    for n in range(len(x)):
        for k in range(min(n+1, coefficients.shape[1])):
            for column in range(x.shape[1]):
                out[n,column] += coefficients[n,k]*x[n-k,column]
    return out


def air_filters(distances, obj):
    alpha = 10**(-distances[:,None]*obj.airAbsorptionCoefficients[None,:]/20)
    right = alpha @ obj._pseud_A.T
    return np.ascontiguousarray(np.column_stack([right[:,1:][:,::-1], right]))


@njit(cache=False, fastmath=False)
def numerical_distance(frequency, speed_of_sound, distance, indices, table):
    out=np.empty((len(distance),len(frequency)),dtype=np.complex128)
    for i in range(len(distance)):
        for j in range(len(frequency)):
            out[i,j]=np.sqrt(complex(0.,-2*np.pi*frequency[j]/speed_of_sound)
                *distance[i]*table[indices[i],j])
    return out


@njit(cache=False, fastmath=False)
def boundary_spectrum(w, erfc, reflection):
    out=np.empty_like(w)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            boundary=1-complex(0.,np.sqrt(np.pi))*w[i,j]*erfc[i,j]
            out[i,j]=reflection[i,j]+(1-reflection[i,j])*boundary
    return out


def ground_filters(angle, distance, obj):
    index = np.clip(np.rint(angle), -89, 89).astype(np.int64)+89
    out = np.empty((len(distance),40), dtype=np.float64)
    # Bound complex working memory; evaluate every sample's original formula.
    for start in range(0,len(distance),2048):
        stop = min(start+2048,len(distance)); ids=index[start:stop]
        w=numerical_distance(obj.f_tmp,obj.c,distance[start:stop],ids,obj.w_tmp)
        spectrum=boundary_spectrum(w,erfcx(1j*w),obj.Rp[ids])
        out[start:stop] = np.fft.irfft(spectrum, axis=1)[:,:40]
    return out


def build_plan(trajectory, mic=None, air=True, unit_ground=False):
    started=time.perf_counter(); obj=reference_manager()
    if mic is None: mic=np.array([0.,0.,1.2])
    trajectory=np.asarray(trajectory,dtype=np.float64); mic=np.asarray(mic,dtype=np.float64)
    if trajectory.ndim!=2 or trajectory.shape[1]!=3 or mic.shape!=(3,):
        raise ValueError('Invalid geometry shape')
    if not np.isfinite(trajectory).all() or not np.isfinite(mic).all():
        raise ValueError('Nonfinite geometry')
    if (trajectory[:,2]<=0).any() or mic[2]<=0:
        raise ValueError('Sources and microphone must be above the ground')
    d=np.sqrt(np.sum((trajectory-mic)**2,axis=1))
    mirror=trajectory.copy(); mirror[:,2]*=-1
    image_distance=np.sqrt(np.sum((mirror-mic)**2,axis=1))
    theta=np.arcsin((trajectory[:,2]+mic[2])/image_distance)
    a=trajectory[:,2]/np.sin(theta); b=mic[2]/np.sin(theta)
    delays=np.column_stack([d,a,b])/obj.c*obj.fs
    # Symmetric interpolation must read previously emitted samples only.
    if (delays <= obj.primaryDelLine._SINC_SMP//2+1).any() or (delays>=47000).any():
        raise ValueError('Delay outside the validated causal interpolation support')
    air_coeffs = [air_filters(v,obj) for v in [d,a,b]] if air else None
    if unit_ground:
        ground=np.zeros((len(d),40)); ground[:,0]=1
    else:
        ground=ground_filters(90-np.rad2deg(theta),a+b,obj)
    if not np.isfinite(ground).all(): raise ValueError('Nonfinite reflection filter')
    return dict(d=d,a=a,b=b,delays=delays,air=air_coeffs,ground=ground,
        trajectory=trajectory,mic=mic,setup_s=time.perf_counter()-started)


def render_sources(sources, plan, components=False):
    """Render all source columns identically; returns P0/P1, before any gain fit."""
    x=np.asarray(sources,dtype=np.float64)
    if x.ndim==1: x=x[:,None]
    if x.ndim!=2 or len(x)!=len(plan['d']) or not np.isfinite(x).all() or not np.any(x):
        raise ValueError('Finite explicit length-matched nonzero source required')
    obj=reference_manager(); dl=obj.primaryDelLine
    started=time.perf_counter()
    direct=fractional_delay(x,plan['delays'][:,0],dl._sinc_table,dl._delta)
    p0=direct/plan['d'][:,None]
    direct_s=time.perf_counter()-started
    started=time.perf_counter()
    direct_filtered=time_varying_fir(direct,plan['air'][0]) if plan['air'] is not None else direct
    first=fractional_delay(x,plan['delays'][:,1],dl._sinc_table,dl._delta)
    if plan['air'] is not None: first=time_varying_fir(first,plan['air'][1])
    reflected=time_varying_fir(first,plan['ground'])
    reflected=fractional_delay(reflected,plan['delays'][:,2],dl._sinc_table,dl._delta)
    if plan['air'] is not None: reflected=time_varying_fir(reflected,plan['air'][2])
    reflected=reflected/(plan['a']+plan['b'])[:,None]
    direct_filtered=direct_filtered/plan['d'][:,None]
    p1=direct_filtered+reflected
    if not np.isfinite(p0).all() or not np.isfinite(p1).all(): raise ValueError('Nonfinite renderer output')
    timing=dict(P0_s=direct_s,P1_extra_s=time.perf_counter()-started)
    result=dict(P0=p0,P1=p1,timing=timing)
    if components: result.update(direct_filtered=direct_filtered,reflected=reflected)
    return result


def observation(full_render, geometry, config):
    lo=geometry['crop_start_sample_8k']; hi=geometry['crop_end_sample_8k']
    if np.shape(full_render)!=(80000,) or hi-lo!=16000 or not 0<=lo<hi<=80000:
        raise ValueError('Invalid complete render or observation interval')
    raw=np.asarray(full_render[lo:hi],dtype=np.float64)
    before=float(np.sqrt(np.mean(raw**2))); rc=config['receiver_normalization']
    if not np.isfinite(raw).all() or before<=rc['floor_rms']:
        raise ValueError('Invalid or silent receiver crop')
    gain=10**(rc['rms_dbfs']/20)/before
    normalized=raw*gain
    if abs(np.sqrt(np.mean(normalized**2))/10**(rc['rms_dbfs']/20)-1)>1e-6:
        raise ValueError('Receiver RMS mismatch')
    y=resample_poly(normalized,2,1,window=('kaiser',5.),padtype='constant').astype(np.float32)
    if y.shape!=(32000,) or not np.isfinite(y).all() or not np.any(y):
        raise ValueError('Invalid final observation')
    if float(np.max(np.abs(y)))>rc['peak_limit_after_resampling']:
        raise ValueError('Receiver peak exceeds the frozen limit; stop the complete paired bank')
    return y,dict(crop_rms_before=before,normalization_gain=gain,
        crop_rms_after=float(np.sqrt(np.mean(normalized**2))),
        crop_float32_rms=float(np.sqrt(np.mean(normalized.astype(np.float32).astype(np.float64)**2))),
        final_rms=float(np.sqrt(np.mean(y.astype(np.float64)**2))),final_peak=float(np.max(np.abs(y))))

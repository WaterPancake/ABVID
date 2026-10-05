"""Isolated numerical controls for air amplitude shaping and FIR latency."""
from phase_common import *
from numba import njit
from renderer import build_plan,render_sources,fractional_delay,time_varying_fir,reference_manager


@njit(cache=False,fastmath=False)
def centered_fir(x,coefficients):
    out=np.zeros_like(x)
    center=coefficients.shape[1]//2
    for n in range(len(x)):
        for k in range(coefficients.shape[1]):
            j=n+center-k
            if 0<=j<len(x):
                for column in range(x.shape[1]):
                    out[n,column]+=coefficients[n,k]*x[j,column]
    return out


def amplitude_only(sources,plan):
    x=np.asarray(sources,dtype=np.float64)
    if x.ndim==1:x=x[:,None]
    obj=reference_manager();dl=obj.primaryDelLine
    direct=fractional_delay(x,plan['delays'][:,0],dl._sinc_table,dl._delta)
    direct=centered_fir(direct,plan['air'][0])/plan['d'][:,None]
    first=fractional_delay(x,plan['delays'][:,1],dl._sinc_table,dl._delta)
    first=centered_fir(first,plan['air'][1])
    reflected=time_varying_fir(first,plan['ground'])
    reflected=fractional_delay(reflected,plan['delays'][:,2],dl._sinc_table,dl._delta)
    reflected=centered_fir(reflected,plan['air'][2])/(plan['a']+plan['b'])[:,None]
    return direct+reflected


def render_all(sources,trajectory,mic=None):
    plan=build_plan(trajectory,mic)
    delay=np.zeros_like(plan['air'][0]);delay[:,5]=1.
    result=dict(
        A0L0=render_sources(sources,dict(plan,air=None))['P1'],
        A1L0=amplitude_only(sources,plan),
        A0L1=render_sources(sources,dict(plan,air=[delay,delay,delay]))['P1'],
        A1L1=render_sources(sources,plan)['P1'])
    for value in result.values():
        if not np.isfinite(value).all():raise ValueError('Nonfinite phase-control render')
    return result

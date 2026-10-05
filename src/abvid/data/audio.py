"""Shared numeric functions from experiments/h1/extract.py."""
from math import gcd
import numpy as np
from scipy.signal import resample_poly

def observation(x,sr,config):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim==2: x=x.mean(axis=1)
    if x.ndim!=1 or not np.isfinite(x).all(): raise ValueError('Invalid waveform')
    for rate in [config['intermediate_rate_hz'],config['sample_rate_hz']]:
        if sr!=rate:
            g=gcd(int(sr),int(rate))
            x=resample_poly(x,rate//g,sr//g,window=('kaiser',config['window_beta']),padtype='constant')
        sr=rate
    n=config['samples']
    if len(x)<n: raise ValueError('Short waveform; padding is forbidden')
    start=(len(x)-n)//2
    x=np.asarray(x[start:start+n],dtype=np.float32)
    if not np.isfinite(x).all() or not np.any(x): raise ValueError('Invalid processed waveform')
    return x

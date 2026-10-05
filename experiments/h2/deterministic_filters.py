"""Fixed-order Butterworth algebra for exact local replay without quantization.

Same analog Butterworth prototype, prewarping, band transform and bilinear
transform as scipy.signal.butter. Python scalar complex operations prevent the
observed allocation-dependent last-bit differences in vector complex arithmetic.
Validated against the released SciPy-based sources; this adds no filter effect.
"""
import cmath
import math
import numpy as np


def product(values):
    answer=1+0j
    for value in values: answer*=value
    return answer


def polynomial(roots):
    coefficients=[1+0j]
    for root in roots:
        following=[0j]*(len(coefficients)+1)
        for j,value in enumerate(coefficients):
            following[j]+=value
            following[j+1]-=value*root
        coefficients=following
    if max(abs(value.imag) for value in coefficients)>1e-10:
        raise ValueError('Butterworth polynomial is not numerically real')
    return np.array([value.real for value in coefficients],dtype=np.float64)


def butterworth(order, cutoff, band=False):
    poles=[-cmath.exp(1j*math.pi*m/(2*order)) for m in range(-order+1,order,2)]
    if band:
        low,high=[4*math.tan(math.pi*float(w)/2) for w in cutoff]
        width=high-low; center=math.sqrt(low*high)
        scaled=[p*width/2 for p in poles]
        poles=([p+cmath.sqrt(p*p-center*center) for p in scaled]
             +[p-cmath.sqrt(p*p-center*center) for p in scaled])
        zeros=[0j]*order; gain=width**order
    else:
        warped=4*math.tan(math.pi*float(cutoff)/2)
        poles=[p*warped for p in poles]; zeros=[]; gain=warped**order
    degree=len(poles)-len(zeros)
    digital_gain=gain*(product([4-z for z in zeros])/product([4-p for p in poles])).real
    zeros=[(4+z)/(4-z) for z in zeros]+[-1+0j]*degree
    poles=[(4+p)/(4-p) for p in poles]
    return digital_gain*polynomial(zeros),polynomial(poles)

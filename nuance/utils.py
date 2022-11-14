import jax
from jax import numpy as jnp
import numpy as np

@jax.jit
def transit(t, t0=None, D=None, d=None, c=12):
    a = 0.5*c
    b = c*t/D
    b0 = c*t0/D

    return - 0.5 * (d*jnp.tanh(a-b+b0) + d*jnp.tanh(a+b-b0))

def periodic_transit(t, t0, D, d, P=1, c=12):
    _t = P * np.sin(np.pi * (t - t0) / P) / (np.pi * D)
    return - d + (d / 2) * (2 - np.tanh(c * (_t + 1 / 2)) + np.tanh(c * (_t - 1 / 2)))

def interp_split_times(time, p):
    dt = np.median(np.diff(time))
    tmax, tmin = np.max(time), np.min(time)
    total = tmax - tmin
    phase = np.arange(0, 1, dt/p)
    n = np.arange(np.ceil(total/p)) # number of 'folds'
    pt0s = np.array([tmin + phase*p + j*p for j in n]) # corresponding t0s
    has_time = np.any([np.abs(time - _t0) < p/2 for _t0 in pt0s.mean(1)], 1)
    pt0s = pt0s[has_time]

    return pt0s
import jax
from jax import numpy as jnp
import numpy as np



def transit(t, t0=None, D=None, d=None, c=12, P=None):
    if P is None:
        return single_transit(t, t0=t0, D=D, d=d, c=c).__array__()
    else:
        return periodic_transit(t, t0, D, d, P=P, c=c)

@jax.jit
def single_transit(t, t0=None, D=None, d=None, c=12):
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
    # since for very small periods we might fold on few points only, it's better to impose
    # at least 200 phases, to compare period folds more fairly
    phase = np.arange(0, 1, np.min([dt, dt/p]))
    n = np.arange(np.ceil(total/p)) # number of 'folds'
    pt0s = np.array([tmin + phase*p + j*p for j in n]) # corresponding t0s
    has_time = np.any([np.abs(time - _t0) < p/2 for _t0 in pt0s.mean(1)], 1)
    pt0s = pt0s[has_time]

    return pt0s

def phase(t, t0, p):
    return (t - t0 + 0.5 * p) % p - 0.5 * p
    
def tv_dv(duration, depth, omega, sigma):
    return np.pi/(omega*duration), 2*sigma/depth

def binn(x, y, n):
    N = int(len(x)/n)
    ns = np.histogram(x, N)[0]
    bx = np.histogram(x, N, weights=x)[0]/ns
    by = np.histogram(x, N, weights=y)[0]/ns
    return bx, by
import jax
from jax import numpy as jnp

@jax.jit
def transit(t, t0=None, D=None, d=None, c=12):
    a = 0.5*c
    b = c*t/D
    b0 = c*t0/D

    return - 0.5 * (d*jnp.tanh(a-b+b0) + d*jnp.tanh(a+b-b0))
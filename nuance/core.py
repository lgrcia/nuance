from functools import partial

import jax
import jax.numpy as jnp


def eval_model(flux, X, gp):
    Liy = gp.solver.solve_triangular(flux)
    LiX = gp.solver.solve_triangular(X.T)

    @jax.jit
    def function(m):
        Xm = jnp.vstack([X, m])
        Lim = gp.solver.solve_triangular(m)
        LiXm = jnp.hstack([LiX, Lim[:, None]])
        LiXmT = LiXm.T
        LimX2 = LiXmT @ LiXm
        w = jnp.linalg.lstsq(LimX2, LiXmT @ Liy)[0]
        v = jnp.linalg.inv(LimX2)
        return gp.log_probability(flux - w @ Xm), w, v

    return function


@jax.jit
def transit_protopapas(t, t0, D, P=1e15, c=12):
    _t = P * jnp.sin(jnp.pi * (t - t0) / P) / (jnp.pi * D)
    return -0.5 * jnp.tanh(c * (_t + 1 / 2)) + 0.5 * jnp.tanh(c * (_t - 1 / 2))


@jax.jit
def transit_box(time, t0, D, P=1e15):
    return -((jnp.abs(time - t0) % P) < D / 2).astype(float)


def map_function(eval_function, model, time, backend, map_t0, map_D):
    jitted_eval = jax.jit(eval_function, backend=backend)

    @jax.jit
    def single_eval(t0, D):
        m = model(time, t0, D)
        ll, w, v = jitted_eval(m)
        return w[-1], v[-1, -1], ll

    t0s_eval = map_t0(single_eval, in_axes=(0, None))
    ds_t0s_eval = map_D(t0s_eval, in_axes=(None, 0))

    return ds_t0s_eval


pmap_cpus = partial(map_function, backend="cpu", map_t0=jax.pmap, map_D=jax.vmap)
vmap_gpu = partial(map_function, backend="gpu", map_t0=jax.vmap, map_D=jax.vmap)

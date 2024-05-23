from functools import partial

import jax
import jax.numpy as jnp
import jaxopt


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


def model(x, y, build_gp, X=None):

    if X is None:
        X = jnp.atleast_2d(jnp.ones_like(x))

    @jax.jit
    def nll_w(params):
        gp = build_gp(params, x)
        Liy = gp.solver.solve_triangular(y)
        LiX = gp.solver.solve_triangular(X.T)
        LiXT = LiX.T
        LiX2 = LiXT @ LiX
        w = jnp.linalg.lstsq(LiX2, LiXT @ Liy)[0]
        nll = -gp.log_probability(y - w @ X)
        return nll, w

    @jax.jit
    def nll(params):
        return nll_w(params)[0]

    @jax.jit
    def mu(params):
        gp = build_gp(params, x)
        _, w = nll_w(params)
        cond_gp = gp.condition(y - w @ X, x).gp
        return cond_gp.loc + w @ X

    return mu, nll


def optimize(fun, init_params, param_names=None):
    def inner(theta, *args, **kwargs):
        params = dict(init_params, **theta)
        return fun(params, *args, **kwargs)

    param_names = list(init_params.keys()) if param_names is None else param_names
    start = {k: init_params[k] for k in param_names}

    solver = jaxopt.ScipyMinimize(fun=inner)
    soln = solver.run(start)

    return dict(init_params, **soln.params)


def transit_protopapas(t, t0, D, P=1e15, c=12):
    _t = P * jnp.sin(jnp.pi * (t - t0) / P) / (jnp.pi * D)
    return -0.5 * jnp.tanh(c * (_t + 1 / 2)) + 0.5 * jnp.tanh(c * (_t - 1 / 2))


def transit_box(time, t0, D, P=1e15):
    return -((jnp.abs(time - t0) % P) < D / 2).astype(float)


def transit_exocomet(time, t0, duration, P=None, n=3):
    flat = jnp.zeros_like(time)
    left = -(time - (t0 - duration / n)) / (duration / n)
    right = -jnp.exp(-2 / duration * (time - t0 - duration / n)) ** 2
    triangle = jnp.maximum(left, right)
    mask = time >= t0 - duration / n
    signal = jnp.where(mask, triangle, flat)
    return signal / jnp.max(jnp.array([-jnp.min(signal), 1]))


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

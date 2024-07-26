import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from tinygp import GaussianProcess, kernels


def solve_triangular(*gps_y):
    Ls = [gp.solver.solve_triangular(y) for gp, y in gps_y]
    if Ls[0].ndim == 1:
        return jnp.hstack(Ls)
    else:
        return block_diag(*Ls)


DEFAULT_X = lambda time: jnp.atleast_2d(jnp.ones_like(time))
DEFAULT_GP = lambda time: GaussianProcess(kernels.quasisep.Exp(1e12), time)


def solve_model(flux, X, gp):

    # multiple gps and datasets
    if isinstance(flux, (list, tuple)):
        assert (
            len(gp) == len(flux) == len(X)
        ), "gp, flux, and datasets must have the same length"
        Liy = solve_triangular(*[(_gp, _flux) for _gp, _flux in zip(gp, flux)])
        LiX = solve_triangular(*[(_gp, _X.T) for _gp, _X in zip(gp, X)])

        @jax.jit
        def function(ms):
            Lim = solve_triangular(*[(_gp, m) for _gp, m in zip(gp, ms)])
            LiXm = jnp.hstack([LiX, Lim[:, None]])
            LiXmT = LiXm.T
            LimX2 = LiXmT @ LiXm
            w = jnp.linalg.lstsq(LimX2, LiXmT @ Liy)[0]
            v = jnp.linalg.inv(LimX2)
            return 0.0, w, v

        return function

    # single gp and dataset
    else:
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


def solve(time, flux, gp=None, X=None, model=None):
    if X is None:
        X = DEFAULT_X(time)
    if gp is None:
        gp = DEFAULT_GP(time)
    if model is None:
        model = transit

    solve_m = solve_model(flux, X, gp)

    # multiple datasets
    if isinstance(time, (list, tuple)):

        def _model(time, epoch, duration, period=None):
            return [model(t, epoch, duration, period=period) for t in time]

    else:
        _model = model

    def function(epoch, duration, period=None):
        m = _model(time, epoch, duration, period=period)
        ll, w, v = solve_m(m)
        return jnp.array([w[-1], v[-1, -1], ll])

    return function


def gp_model(x, y, build_gp, X=None):

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


def transit(t, epoch, duration, period=None, c=12):
    if period is None:
        period = 1e15
    _t = period * jnp.sin(jnp.pi * (t - epoch) / period) / (jnp.pi * duration)
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


def separate_models(time, flux, X=None, gp=None, model=None):
    if X is None:
        X = DEFAULT_X(time)
    if gp is None:
        gp = DEFAULT_GP(time)
    if model is None:
        model = transit

    solver = solve_model(flux, X, gp)

    def function(epoch, duration, period=None):
        m = model(time, epoch, duration, period=period)
        w = solver(m)[1]
        mean = w[0:-1] @ X
        signal = m * w[-1]

        @jax.jit
        def gp_mean():
            return gp.condition(flux - mean - signal).gp.mean

        noise = gp_mean()

        return mean, signal, noise

    return function


def snr(time, flux, X=None, gp=None, model=None):
    if X is None:
        X = DEFAULT_X(time)
    if gp is None:
        gp = DEFAULT_GP(time)
    if model is None:
        model = transit

    solver = solve(time, flux, gp=gp, X=X, model=model)

    def function(epoch, duration, period=None):
        z, vz, _ = solver(epoch, duration, period)
        return jnp.max(jnp.array([0, z / jnp.sqrt(vz)]))

    return function


def interpolate(xp, fp):
    import numpy as np

    xp = np.asarray(xp)
    fp = np.asarray(fp.T)

    def fun(x):
        x = np.asarray(x)
        j = np.searchsorted(xp, x) - 1
        d = np.where(j + 1 < len(xp), (x - xp[j]) / (xp[j + 1] - xp[j]), 0)
        return ((1 - d) * fp[:, j] + fp[:, j + 1] * d).T

    return fun


def nearest_neighbors(xp, fp):
    import numpy as np

    xp = np.asarray(xp)
    fp = np.asarray(fp)

    def fun(x):
        x = np.asarray(x)
        j = np.searchsorted(xp, x) - 1
        return fp[j]

    return fun

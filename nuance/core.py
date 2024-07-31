import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from tinygp import GaussianProcess, kernels

INPUTS_DOCS = """
    time : np.ndarray
        array of times
    flux : np.ndarray
        array of fluxes
    X : np.ndarray, optional
        linear model design matrix, by default a constant model
    gp : tinygp.GaussianProcess, optional
        Gaussian process object, by default a very long scale exponential kernel
    model : callable, optional
        model function with signature :code:`model(time, epoch, duration, period=None) -> Array`,
        by default an empirical :py:func:`~nuance.core.transit` model
"""

DEFAULT_X = lambda time: jnp.atleast_2d(jnp.ones_like(time))
DEFAULT_GP = lambda time: GaussianProcess(kernels.quasisep.Exp(1e12), time)


def check_default(time, X=None, gp=None, model=None):
    if X is None:
        X = DEFAULT_X(time)
    if gp is None:
        gp = DEFAULT_GP(time)
    if model is None:
        model = transit
    return X, gp, model


def solve_triangular(*gps_y):
    Ls = [gp.solver.solve_triangular(y) for gp, y in gps_y]
    if Ls[0].ndim == 1:
        return jnp.hstack(Ls)
    else:
        return block_diag(*Ls)


def solve_model(flux, X, gp):

    # multiple gps and datasets
    if isinstance(flux, (list, tuple)):
        assert (
            len(gp) == len(flux) == len(X)
        ), "gp, flux, and datasets must have the same length"
        base_Liy = solve_triangular(*[(_gp, _flux) for _gp, _flux in zip(gp, flux)])
        LiX = solve_triangular(*[(_gp, _X.T) for _gp, _X in zip(gp, X)])

        @jax.jit
        def function(ms, depth=None):
            if depth is not None:
                raise NotImplementedError(
                    "depth is not implemented for multiple datasets, open an issue"
                )
            _Liy = base_Liy
            Lim = solve_triangular(*[(_gp, m) for _gp, m in zip(gp, ms)])
            LiXm = jnp.hstack([LiX, Lim[:, None]])
            LiXmT = LiXm.T
            LimX2 = LiXmT @ LiXm
            w = jnp.linalg.lstsq(LimX2, LiXmT @ _Liy)[0]
            v = jnp.linalg.inv(LimX2)
            return 0.0, w, v

        return function

    # single gp and dataset
    else:
        base_Liy = gp.solver.solve_triangular(flux)
        LiX = gp.solver.solve_triangular(X.T)

        @jax.jit
        def function(m, depth=None):
            if depth is not None:
                Liy = base_Liy + gp.solver.solve_triangular(depth * m)
            else:
                Liy = base_Liy
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
    """Returns a function to compute the log likelihood of data assuming it is drawn
    from a Gaussian Process with a mean linear model.

    `X` is a design matrix for the linear model whose last column is the searched signal
    :code:`model`.

    Parameters
    ----------
    time : np.ndarray
        array of times
    flux : np.ndarray
        array of fluxes
    X : np.ndarray, optional
        linear model design matrix, by default a constant model
    gp : tinygp.GaussianProcess, optional
        Gaussian process object, by default a very long scale exponential kernel
    model : callable, optional
        model function with signature :code:`model(time, epoch, duration, period=None) -> Array`,
        by default an empirical :py:func:`~nuance.core.transit` model

    Returns
    -------
    callable
        function that computes the log likelihood of data assuming it is drawn from a
        Gaussian Process with a mean linear model. Signature is:
        :code:`function(epoch, duration, period=None) -> (log_likelihood, weights, variance)`
    """

    X, gp, model = check_default(time, X, gp, model)

    if model is None:
        model = transit

    solve_m = solve_model(flux, X, gp)

    # multiple datasets
    if isinstance(time, (list, tuple)):

        def _model(time, epoch, duration, period=None):
            return [model(t, epoch, duration, period=period) for t in time]

    else:
        _model = model

    def function(epoch, duration, period=None, depth=None):
        m = _model(time, epoch, duration, period=period)
        ll, w, v = solve_m(m, depth=depth)
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


def transit(time, epoch, duration, period=None, c=12):
    """Empirical transit model from Protopapas et al. 2005.

    Parameters
    ----------
    time : np.ndarray
        array of times
    epoch : float
        signal epoch
    duration : float
        signal duration
    period : float, optional
        signal period, by default None for a non-periodic signal
    c : int, optional
        controls the 'roundness' of transit shape, by default 12

    Returns
    -------
    np.ndarray
        array of transit model values
    """
    if period is None:
        period = 1e15
    _t = period * jnp.sin(jnp.pi * (time - epoch) / period) / (jnp.pi * duration)
    return -0.5 * jnp.tanh(c * (_t + 1 / 2)) + 0.5 * jnp.tanh(c * (_t - 1 / 2))


def transit_box(time, epoch, duration, period=1e15):
    """Box-shaped transit model.

    Parameters
    ----------
    time : np.ndarray
        array of times
    epoch : float
        signal epoch
    duration : float
        signal duration
    period : float, optional
        signal period, by default None for a non-periodic signal

    Returns
    -------
    np.ndarray
        array of transit model values
    """
    return -((jnp.abs(time - epoch) % period) < duration / 2).astype(float)


def transit_exocomet(time, epoch, duration, period=None, n=3):
    """Empirical exocomet transit model.

    Parameters
    ----------
    time : np.ndarray
        array of times
    epoch : float
        signal epoch
    duration : float
        signal duration
    period : float, optional
        dummy parameter for compatibility with other models, by default None
    n : int, optional
        TBD, by default 3

    Returns
    -------
    np.ndarray
        array of transit model values
    """
    flat = jnp.zeros_like(time)
    left = -(time - (epoch - duration / n)) / (duration / n)
    right = -jnp.exp(-2 / duration * (time - epoch - duration / n)) ** 2
    triangle = jnp.maximum(left, right)
    mask = time >= epoch - duration / n
    signal = jnp.where(mask, triangle, flat)
    return signal / jnp.max(jnp.array([-jnp.min(signal), 1]))


def separate_models(time, flux, X=None, gp=None, model=None):
    """Returns a function to compute the mean, signal, and noise model of a light curve
    given the epoch, duration, and period of the signal model.

    Parameters
    ----------
    time : np.ndarray
        array of times
    flux : np.ndarray
        array of fluxes
    X : np.ndarray, optional
        linear model design matrix, by default a constant model
    gp : tinygp.GaussianProcess, optional
        Gaussian process object, by default a very long scale exponential kernel
    model : callable, optional
        model function with signature :code:`model(time, epoch, duration, period=None) -> Array`,
        by default an empirical :py:func:`~nuance.core.transit` model

    Returns
    -------
    callable
        function that computes the mean, signal, and noise model of a light curve given
        the epoch, duration, and period of the signal model. Signature is:
        :code:`function(epoch, duration, period=None) -> (mean, signal, noise)`
    """

    X, gp, model = check_default(time, X, gp, model)

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
    """Returns a function to compute the signal-to-noise ratio of a signal model given
    its epoch, duration, and period.

    Parameters
    ----------
    time : np.ndarray
        array of times
    flux : np.ndarray
        array of fluxes
    X : np.ndarray, optional
        linear model design matrix, by default a constant model
    gp : tinygp.GaussianProcess, optional
        Gaussian process object, by default a very long scale exponential kernel
    model : callable, optional
        model function with signature :code:`model(time, epoch, duration, period=None) -> Array`,
        by default an empirical :py:func:`~nuance.core.transit` model

    Returns
    -------
    callable
        function that computes the signal-to-noise ratio of a signal model given its
        epoch, duration, and period. Signature is:
        :code:`function(epoch, duration, period=None) -> float`
    """

    X, gp, model = check_default(time, X, gp, model)

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

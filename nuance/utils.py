import jax
import jaxopt
import numpy as np
import tinygp
from scipy.ndimage import minimum_filter1d

from nuance import core


def interp_split_times(time, p, dphi=0.01):
    tmax, tmin = np.max(time), np.min(time)
    total = tmax - tmin
    # since for very small periods we might fold on few points only, it's better to impose
    # at least 200 phases, to compare period folds more fairly
    phase = np.arange(0, 1, np.min([1 / 200, dphi / p]))
    n = np.arange(np.ceil(total / p) + 1)  # number of 'folds'
    # this line is important so that different time-series have the same phase 0
    tmin -= tmin % p
    pt0s = np.array([tmin + phase * p + j * p for j in n])  # corresponding t0s
    has_time = np.any([np.abs(time - _t0) < p / 2 for _t0 in pt0s.mean(1)], 1)
    pt0s = pt0s[has_time]

    return pt0s


def time_phase_resample(time, phase, period):
    t0 = np.min(time)
    duration = np.ptp(time)
    n = np.arange(np.ceil(duration / period) + 1)  # number of 'folds'
    t0 -= t0 % period  # phase 0
    t0s = np.array([t0 + (phase + j) * period for j in n])  # corresponding t0s
    has_time = np.any([np.abs(time - _t0) < period / 2 for _t0 in t0s.mean(1)], 1)
    return t0s[has_time]


def phase(t, t0, p):
    return (t - t0 + 0.5 * p) % p - 0.5 * p


def log_gauss_product_integral(a, va, b, vb):
    """Log of the product ot two Normal distributions (normalized)

    Parameters
    ----------
    a : float
        mean of distribution A
    va : float
        variance of distribution A
    b : float
        mean of distribution B
    vb : float
        variance of distribution B

    Returns
    -------
    float
        log of the integral of A*B
    """
    return (
        -0.5 * np.square(a - b) / (va + vb)
        - 0.5 * np.log(va + vb)
        - 0.5 * np.log(np.pi)
        - np.log(2) / 2
    )


def simulated(
    t0=0.2, D=0.05, depth=0.02, P=0.7, time=None, kernel=None, error=0.001, w=None
):
    if time is None:
        time = np.arange(0, 4, 2 / 60 / 24)

    X = np.vander(time, N=4, increasing=True).T
    if w is None:
        w = [1.0, 5e-4, -2e-4, -5e-4]

    true_transit = depth * core.transit(time, t0, D, period=P)

    if kernel is None:
        kernel = tinygp.kernels.quasisep.SHO(np.pi / (6 * D), 45.0, depth)

    gp = tinygp.GaussianProcess(kernel, time, diag=error**2, mean=0.0)
    flux = gp.sample(jax.random.PRNGKey(40)) + true_transit + w @ X

    return (time, flux, error), X, gp


def clean_periods(periods, period, tol=0.02):
    close = periods / period
    return periods[np.abs(close - np.rint(close)) > tol]


def index_binning(x, size):
    if isinstance(size, float):
        bins = np.arange(np.min(x), np.max(x), size)
    else:
        x = np.arange(0, len(x))
        bins = np.arange(0.0, len(x), size)

    d = np.digitize(x, bins)
    n = np.max(d) + 2
    indexes = []

    for i in range(0, n):
        s = np.where(d == i)
        if len(s[0]) > 0:
            s = s[0]
            indexes.append(s)

    return indexes


def binn_time(time, flux, bins=7 / 24 / 60):
    indexes = index_binning(time, bins)
    binned_time = np.array([np.mean(time[i]) for i in indexes])
    binned_flux = np.array([np.mean(flux[i]) for i in indexes])
    binned_error = np.array([np.std(flux[i]) / np.sqrt(len(i)) for i in indexes])
    return binned_time, binned_flux, binned_error


def minimize(fun, init_params, param_names=None):
    """Minimize a function using jaxopt.ScipyMinimize

    Parameters
    ----------
    fun : callable
        the function to minimize of the form fun(params: dict) -> float
    init_params : dict
        initial parameters
    param_names : list, optional
        list of parameters to optimize, all others being fixed,
        by default None

    Returns
    -------
    dict
        optimized parameters
    """

    def inner(theta, *args, **kwargs):
        params = dict(init_params, **theta)
        return fun(params, *args, **kwargs)

    param_names = list(init_params.keys()) if param_names is None else param_names
    start = {k: init_params[k] for k in param_names}

    solver = jaxopt.ScipyMinimize(fun=inner)
    soln = solver.run(start)

    return dict(init_params, **soln.params)


def sigma_clip_mask(residuals, sigma=5, window=20):
    """Returns a masked of the sigma clipped data. The mask is set
        to false on the +/-{windows} data points around the sigma
        clipped residuals, i.e. code:`residuals > sigma * std(residuals)`.

    Parameters
    ----------
    residuals : array
        Data to sigma clip.
    sigma : int, optional
        Sigma clipping standard deviation, by default 5.
    window : int, optional
        The window of points around which mask is set to False if
        a data point is sigma clipped, by default 20

    Returns
    -------
    array
        boolean mask of sigma clipped values (i.e :code:`<std(residuals)`).
    """
    mask = np.ones_like(residuals, dtype=bool)
    mask[residuals > sigma * np.std(residuals)] = False
    mask = np.roll(minimum_filter1d(mask, window), shift=window // 3)
    return mask

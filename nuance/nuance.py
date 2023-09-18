import multiprocessing as mp
import pickle
from copy import deepcopy
from dataclasses import asdict, dataclass
from multiprocessing import set_start_method

import jax
import jax.numpy as jnp
import jaxopt
import multiprocess as mp
import numpy as np
from tinygp import GaussianProcess, kernels
from tqdm import tqdm
from tqdm.autonotebook import tqdm

from functools import partial

from . import CPU_counts, utils
from .search_data import SearchData

# set_start_method("spawn")


@dataclass
class Nuance:
    """
    An object for nuanced transit search
    """

    time: np.ndarray
    """Time"""
    flux: np.ndarray
    """Flux time series"""
    error: np.ndarray = None
    """Flux error time series"""
    gp: GaussianProcess = None
    """Gaussian process instance"""
    X: np.ndarray = None
    """Design matrix"""
    compute: bool = True
    """Whether to pre-compute the Cholesky decomposition"""
    mean: float = 0.0
    """Mean of the GP"""
    search_data: SearchData = None
    """Search data instance"""
    c: float = 12

    def __post_init__(self):
        assert (self.error is None) ^ (
            self.gp is None
        ), "Either error or gp must be defined"

        if self.X is None:
            self.X = np.atleast_2d(np.ones_like(self.time))

        if self.gp is None:
            kernel = kernels.quasisep.Exp(1e12)

            self.gp = GaussianProcess(
                kernel, self.time, diag=self.error**2, mean=self.mean
            )

        if self.compute:
            self._compute_L()

        self.search_data = None

    def _compute_L(self):
        Liy = self.gp.solver.solve_triangular(self.flux)
        LiX = self.gp.solver.solve_triangular(self.X.T)

        @jax.jit
        def eval_m(m):
            Xm = jnp.vstack([self.X, m])
            Lim = self.gp.solver.solve_triangular(m)
            LiXm = jnp.hstack([LiX, Lim[:, None]])
            LiXmT = LiXm.T
            LimX2 = LiXmT @ LiXm
            w = jnp.linalg.lstsq(LimX2, LiXmT @ Liy)[0]
            v = jnp.linalg.inv(LimX2)
            return self.gp.log_probability(self.flux - w @ Xm), w, v

        self.eval_m = eval_m

    @property
    def ll0(self) -> float:
        """log-likelihood of data to a model without transit

        Returns
        -------
        float
        """
        return self.eval_m(np.zeros_like(self.time))[0].__array__()

    def linear_search(
        self,
        t0s: np.ndarray,
        Ds: np.ndarray,
        positive: bool = True,
        progress: bool = True,
    ):
        """Performs the linear search. Saves the linear search `Nuance.search_data` as a :py:class:`nuance.SearchData` object

        Parameters
        ----------
        t0s : np.ndarray
            array of transit epochs
        Ds : np.ndarray
            array of transit durations
        positive : bool, optional
            wether to force depth to be positive, by default True
        progress : bool, optional
            wether to show progress bar, by default True

        Returns
        -------
        None
        """

        n = len(self.X)

        @jax.jit
        def eval_transit(t0, D):
            m = utils.single_transit(self.time, t0, D, c=self.c)
            _ll, w, v = self.eval_m(m)
            return w[n], v[n, n], _ll

        chunk_size = CPU_counts
        chunks = int(np.ceil(len(t0s) / chunk_size))
        padded_t0s = np.pad(t0s, pad_width=[0, chunks * chunk_size - len(t0s)])
        splitted_t0s = np.array(np.array_split(padded_t0s, chunks))

        ll = np.zeros((len(padded_t0s), len(Ds)))
        depths = ll.copy()
        vars = ll.copy()
        depths = ll.copy()

        _progress = lambda x: tqdm(x) if progress else x

        f = jax.pmap(eval_transit, in_axes=(0, None))
        g = jax.vmap(f, in_axes=(None, 0))
        for i, t0 in enumerate(_progress(splitted_t0s)):
            _depths, _vars, _ll = g(t0, Ds)
            depths[i * CPU_counts : (i + 1) * CPU_counts, :] = _depths.T
            vars[i * CPU_counts : (i + 1) * CPU_counts, :] = _vars.T
            ll[i * CPU_counts : (i + 1) * CPU_counts, :] = _ll.T

        depths = np.array(depths[0 : len(t0s), :])
        vars = np.array(vars[0 : len(t0s), :])
        ll = np.array(ll[0 : len(t0s), :])

        if positive:
            ll0 = self.eval_m(np.zeros_like(self.time))[0]
            ll[depths < 0] = ll0

        vars[~np.isfinite(vars)] = 1e25

        self.search_data = SearchData(
            t0s=t0s, Ds=Ds, ll=ll, z=depths, vz=vars, ll0=self.ll0
        )

    def periodic_search(self, periods: np.ndarray, progress=True, dphi=0.01):
        """Performs the periodic search

        Parameters
        ----------
        periods : np.ndarray
            array of periods to search
        progress : bool, optional
            wether to show progress bar, by default True

        Returns
        -------
        :py:class:`nuance.SearchData`
            search results
        """
        new_search_data = self.search_data.copy()
        n = len(periods)
        snr = np.zeros(n)
        params = np.zeros((n, 3))

        def _progress(x, **kwargs):
            return tqdm(x, **kwargs) if progress else x

        global SEARCH
        SEARCH = partial(self.search_data.fold_ll, dphi=dphi)

        with mp.Pool() as pool:
            for p, (Ti, j, P) in enumerate(
                _progress(pool.imap(_search, periods), total=len(periods))
            ):
                Dj = new_search_data.Ds[j]
                pass
                snr[p], params[p] = float(self.snr(Ti, Dj, P)), (Ti, Dj, P)

        new_search_data.periods = periods
        new_search_data.Q_snr = snr
        new_search_data.Q_params = params

        return new_search_data

    def _models(self, m):
        _, w, _ = self.eval_m(m)
        mean = w[0:-1] @ self.X
        signal = m * w[-1]
        _, cond = self.gp.condition(self.flux - mean - signal)
        noise = cond.mean

        return mean, signal, noise

    def mu(self, mask=None):
        """
        Computes the mean model of the GP.

        Parameters
        ----------
        mask : np.ndarray, optional
            A boolean mask to apply to the data, by default None.

        Returns
        -------
        np.ndarray
            The mean model of the GP.

        Example
        -------
        >>> mu = model.mu()
        """
        if mask is None:
            mask = mask = np.ones_like(self.time).astype(bool)

        masked_y = self.flux[mask]
        masked_X = self.X[:, mask]

        @jax.jit
        def _mu():
            gp = self.gp
            _, w, _ = self.eval_m(np.zeros_like(self.time))
            w = w[0:-1]
            cond_gp = gp.condition(masked_y - w @ masked_X, self.time).gp
            return cond_gp.loc + w @ self.X

        return _mu()

    def models(self, t0: float, D: float, P: float = None, c=None):
        """Return the models corresponding the transit of epoch `t0` and duration `D`(and period `P` for a periodic transit)

        Parameters
        ----------
        t0 : float
            epoch, same unit as time
        D : float
            duration, same unit as time
        P : float, optional
            period, same unit as time, by default None

        Returns
        -------
        list np.ndarray
            a list of three np.ndarray:

            - linear: linear model
            - astro: signal being searched (transit)
            - noise: noise model

        Example
        -------

        .. code-block::

            from nuance import Nuance, utils

            time, flux, error = utils.simulation()[0]

            nu = Nuance(time, flux, error)
            linear, astro, noise = nu.models(0.2, 0.05, 1.3)

        """
        if c is None:
            c = self.c
        m = utils.transit(self.time, t0, D, 1, P=P, c=c)
        return self._models(m)

    def solve(self, t0: float, D: float, P: float = None, c: float = None):
        """solve linear model (design matrix `Nuance.X`)

        Parameters
        ----------
        t0 : float
            epoch, same unit as time
        D : float
            duration, same unit as time
        P : float, optional
            period, same unit as time, by default None

        Returns
        -------
        list
            (w, v): linear coefficients and their covariance matrix
        """
        if c is None:
            c = self.c
        m = utils.transit(self.time, t0, D, 1, P=P, c=c)
        _, w, v = self.eval_m(m)
        return w, v

    def depth(self, t0: float, D: float, P: float = None):
        """depth linearly solved for epoch `t0` and duration `D` (and period `P` for a periodic transit)

        Parameters
        ----------
        t0 : float
            epoch, same unit as time
        D : float
            duration, same unit as time
        P : float, optional
            period, same unit as time, by default None

        Returns
        -------
        float, float
            transit depth, depth error
        """
        w, v = self.solve(t0, D, P)
        return w[-1], np.sqrt(v[-1, -1])

    def snr(self, t0: float, D: float, P: float = None):
        """SNR of transit linearly solved for epoch `t0` and duration `D` (and period `P` for a periodic transit)

        Parameters
        ----------
        t0 : float
            epoch, same unit as time
        D : float
            duration, same unit as time
        P : float, optional
            period, same unit as time, by default None

        Returns
        -------
        float
            transit snr
        """
        w, dw = self.depth(t0, D, P=P)
        return w / dw

    def gp_optimization(self, build_gp, mask=None):
        """
        Optimize the Gaussian Process (GP) model using the given build_gp function.

        Parameters
        ----------
        build_gp : function
            A function that returns a GP object.
        mask : array-like, optional
            A boolean array to mask the data, by default None.

        Returns
        -------
        tuple
            A tuple containing three functions:
            - optimize: a function that optimizes the GP model.
            - mu: a function that returns the mean of the GP model.
            - nll: a function that returns the negative log-likelihood of the GP model.
        """
        if mask is None:
            mask = mask = np.ones_like(self.time).astype(bool)

        masked_x = self.time[mask]
        masked_y = self.flux[mask]
        masked_X = self.X[:, mask]

        @jax.jit
        def nll_w(params):
            gp = build_gp(params, masked_x)
            Liy = gp.solver.solve_triangular(masked_y)
            LiX = gp.solver.solve_triangular(masked_X.T)
            LiXT = LiX.T
            LiX2 = LiXT @ LiX
            w = jnp.linalg.lstsq(LiX2, LiXT @ Liy)[0]
            nll = -gp.log_probability(masked_y - w @ masked_X)
            return nll, w

        @jax.jit
        def nll(params):
            return nll_w(params)[0]

        @jax.jit
        def mu(params):
            gp = build_gp(params, masked_x)
            _, w = nll_w(params)
            cond_gp = gp.condition(masked_y - w @ masked_X, self.time).gp
            return cond_gp.loc + w @ self.X

        def optimize(init_params, param_names=None):
            def inner(theta, *args, **kwargs):
                params = dict(init_params, **theta)
                return nll(params, *args, **kwargs)

            param_names = (
                list(init_params.keys()) if param_names is None else param_names
            )
            start = {k: init_params[k] for k in param_names}

            solver = jaxopt.ScipyMinimize(fun=inner)
            soln = solver.run(start)
            print(soln.state)

            return dict(init_params, **soln.params)

        return optimize, mu, nll

    def mask(self, t0: float, D: float, P: float):
        """Return a `Nuance` object where the transit of epoch `t0` and duration `D` and period `P` is masked.

        Parameters
        ----------
        t0 : float
            transit epoch
        D : float
            transit duration
        P : float
            transit period

        Returns
        -------
        `Nuance`
            A `Nuance` instance with transit masked.
        """
        # search data
        search_data = self.search_data.copy()
        ph = utils.phase(search_data.t0s, t0, P)
        t0_mask = np.abs(ph) > 2 * D

        if np.count_nonzero(t0_mask) == len(search_data.t0s):
            raise ValueError("Mask covers all data points")
        elif len(t0_mask) == 0:
            raise ValueError("No transit to mask")

        search_data.llv = None
        search_data.llc = None
        search_data.periods = None
        search_data.t0s = search_data.t0s[t0_mask]
        search_data.ll = search_data.ll[t0_mask]
        search_data.z = search_data.z[t0_mask]
        search_data.vz = search_data.vz[t0_mask]

        # nu
        ph = utils.phase(self.time, t0, P)
        t0_mask = np.abs(ph) > 2 * D

        ph = utils.phase(self.time, t0, P)
        time_mask = np.abs(ph) > 2 * D
        gp = GaussianProcess(
            self.gp.kernel,
            self.time[time_mask],
            mean=0.0,
            diag=self.gp.variance[time_mask],
        )

        nu = Nuance(
            self.time[time_mask],
            self.flux[time_mask],
            gp=gp,
            X=self.X[:, time_mask],
            mean=0.0,
        )
        nu.search_data = search_data
        return nu

    # The all thing needs to be rewritten
    def flares_mask(self, window=30, sigma=4, iterations=3):
        """Return a mask where flares are masked.

        Parameters
        ----------
        window : int, optional
            The sliding mask window (typical number of points of a flare duration),
            by default 30
        sigma : int, optional
            Flare sigma-clipping factor, by default 4
        iterations : int, optional
            Number of iterations, by default 3

        Returns
        -------
        np.ndarray
            Flare mask
        """
        raise NotImplementedError
        mask = np.ones_like(self.time).astype(bool)
        window = 30

        def build_gp(params, x):
            return GaussianProcess(
                self.kernel, x, diag=np.mean(self.error) ** 2, mean=0.0
            )

        _, mu, _ = self.gp_optimization(build_gp)

        for _ in range(iterations):
            m = mu(None)
            r = self.flux - m
            mask_up = r < np.std(r[mask]) * sigma

            # mask around flares
            ups = np.flatnonzero(~mask_up)
            if len(ups) > 0:
                mask[
                    np.hstack(
                        [
                            np.arange(
                                max(u - window, 0), min(u + window, len(self.time))
                            )
                            for u in ups
                        ]
                    )
                ] = False

            _, mu, _ = self.gp_optimization(build_gp, mask=mask)

        return ~mask

    def save(self, filename):
        """Save the current state of the object to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the object to.
        """
        pickle.dump(asdict(self), open(filename, "wb"))

    def copy(self):
        """Return a deep copy of the object."""
        return deepcopy(self)

    @classmethod
    def load(cls, filename):
        """Load an object from a file.

        Parameters
        ----------
        filename : str
            The name of the file to load the object from.

        Returns
        -------
        Nuance
            The loaded object.
        """
        return cls(**pickle.load(open(filename, "rb")))


def _search(p):
    phase, _, P2 = SEARCH(p)
    i, j = np.unravel_index(np.argmax(P2), P2.shape)
    Ti = phase[i] * p
    return Ti, j, p

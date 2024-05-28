import multiprocessing as mp
import pickle
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jaxopt
import multiprocess as mp
import numpy as np
from scipy.ndimage import minimum_filter1d
from tinygp import GaussianProcess, kernels
from tqdm.auto import tqdm

from nuance import DEVICES_COUNT, core, utils
from nuance.search_data import SearchData


@dataclass
class Nuance:
    """
    `nuance` object containing the data and the linear and periodic search functions.

    .. note::

        `nuance` relies on `tinygp <https://github.com/dfm/tinygp>`_ to instantiate and manipulate Gaussian processes. See the
        `tinygp documentation <https://tinygp.readthedocs.io>`_ for more details.

    Parameters
    ----------
    time : np.ndarray
        Time
    flux : np.ndarray
        Flux time series
    error : np.ndarray, optional
        Flux error time series, by default None
    gp : tinygp GaussianProcess, optional
        `tinygp.GaussianProcess <https://tinygp.readthedocs.io/en/latest/api/summary/tinygp.GaussianProcess.html>`_ instance, by default None,
        which creates a GP with a squared exponential kernel of lengthscale :math:`10^{12}` and sigma of  :math:`1`.
    X : np.ndarray, optional
        Design matrix, by default None, which create a design matrix with a single column of ones.
    compute : bool, optional
        Whether to pre-compute the Cholesky decomposition used by the GP, by default True.
    mean : float, optional
        Mean of the GP, by default 0.0
    search_data : SearchData, optional
        Search data instance, by default None.
    model : callable, optional
        Model function with signature `model(time, t0, D, P=None)`, by default None, which set the model to
        :code:`partial(nuance.core.transit_protopapas, c=12)`
    """

    time: np.ndarray
    """Time"""
    flux: np.ndarray
    """Flux time series"""
    error: np.ndarray = None
    """Flux error time series"""
    gp: GaussianProcess = None
    """tinygp.GaussianProcess instance"""
    X: np.ndarray = None
    """Design matrix"""
    compute: bool = True
    """Whether to pre-compute the Cholesky decomposition"""
    mean: float = 0.0
    """Mean of the GP"""
    search_data: SearchData = None
    """Search data instance"""
    model: callable = None
    """Model function with signature :code:`model(time, t0, D, P=None)`"""

    def __post_init__(self):
        if self.model is None:
            self.model = partial(core.transit_protopapas, c=12)

        self.model = jax.jit(self.model)

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
            self._solve = jax.jit(core.solve(self.flux, self.X, self.gp))

        self.search_data = None

    def __repr__(self):
        noise = (
            f"kernel={self.gp.kernel}"
            if self.error is None
            else f"error={self.error:.3e}"
        )
        return f"Nuance(N={len(self.time)}, M={self.X.shape[0]}, {noise}, searched={self.searched})"

    @property
    def searched(self):
        """Whether the linear search has been performed."""
        return self.search_data is not None

    @property
    def time_span(self):
        """Time span"""
        return np.ptp(self.time)

    @property
    def ll0(self) -> float:
        """Log-likelihood of data without model.

        Returns
        -------
        float
        """
        return self._solve(np.zeros_like(self.time))[0].__array__()

    def _models(self, m):
        _, w, _ = self._solve(m)
        mean = w[0:-1] @ self.X
        signal = m * w[-1]

        @jax.jit
        def gp_mean():
            return self.gp.condition(self.flux - mean - signal).gp.mean

        noise = gp_mean()

        return mean, signal, noise

    def gp_optimization(self, build_gp, mask=None):
        """
        Optimization functions to fit a Gaussian Process given a building function.

        Parameters
        ----------
        build_gp : function
            A function that returns a tinygp.GaussianProcess object.
        mask : array-like, optional
            A boolean array to mask the data, by default None.

        Returns
        -------
        tuple
            A tuple containing three functions:

            - :code:`optimize`: a function that optimizes the GP model.
            - :code:`mu`: a function that returns the mean of the GP model (jax.jit-compiled).
            - :code:`nll`: a function that returns the negative log-likelihood of the GP model (jax.jit-compiled).
        """
        mu, nll = core.gp_model(
            self.time[mask], self.flux[mask], build_gp, X=self.X[:, None]
        )
        optimize = partial(utils.minimize, nll)

        return optimize, mu, nll

    def mu(self, time=None):
        """
        Computes the mean model of the GP.

        Parameters
        ----------
        time : np.ndarray, optional
            The time at which to compute the mean model, by default None (uses `self.time`).
        Returns
        -------
        np.ndarray
            The mean model of the GP.
        """
        if time is None:
            time = self.time

        @jax.jit
        def _mu():
            gp = self.gp
            _, w, _ = self._solve(np.zeros_like(time))
            w = w[0:-1]
            cond_gp = gp.condition(self.flux - w @ self.X, time).gp
            return cond_gp.loc + w @ self.X

        return _mu()

    def models(self, t0: float = None, D: float = None, P: float = 1e15):
        """Return the models corresponding to epoch `t0` and duration `D`(and period `P` for a periodic model).

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

            - linear: linear model (using `X`)
            - model: model being searched
            - noise: noise model (using `GP`)

        Example
        -------

        .. code-block::

            from nuance import Nuance, utils

            time, flux, error = utils.simulation()[0]

            nu = Nuance(time, flux, error)
            linear, astro, noise = nu.models(0.2, 0.05, 1.3)

        """
        if t0 is not None:
            m = self.model(self.time, t0, D, P)
        else:
            m = np.zeros_like(self.time)
        return self._models(m)

    def solve(self, t0: float, D: float, P: float = None):
        """Solve linear model (suing `X`).

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
        m = self.model(self.time, t0, D, P)
        _, w, v = self._solve(m)
        return w, v

    def depth(self, t0: float, D: float, P: float = None):
        """Depth linearly solved for epoch `t0` and duration `D` (and period `P` for a periodic model).

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
            model depth, depth error
        """
        w, v = self.solve(t0, D, P)
        return w[-1], jnp.sqrt(v[-1, -1])

    def snr(self, t0: float, D: float, P: float = None):
        """SNR of the model linearly solved for epoch `t0` and duration `D` (and period `P` for a periodic model).

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
            model snr
        """
        w, dw = self.depth(t0, D, P=P)
        return jnp.max(jnp.array([0, w / dw]))

    def linear_search(
        self,
        t0s: np.ndarray,
        Ds: np.ndarray,
        positive: bool = True,
        progress: bool = True,
        backend: str = None,
        batch_size: int = None,
    ):
        """Performs the linear search. Saves the linear search `Nuance.search_data` as a :py:class:`nuance.SearchData` object.

        Parameters
        ----------
        t0s : np.ndarray
            array of model epochs
        Ds : np.ndarray
            array of model durations
        positive : bool, optional
            wether to force depth to be positive, by default True
        progress : bool, optional
            wether to show progress bar, by default True
        backend : str, optional
            backend to use, by default jax.default_backend() (options: "cpu", "gpu").
            This affects the linear search function jax-mapping strategy. For more details, see
            :py:func:`nuance.core.map_function`
        batch_size : int, optional
            batch size for parallel evaluation, by default None
        Returns
        -------
        None
        """
        assert backend in [None, "cpu", "gpu"], "backend must be 'cpu' or 'gpu'"

        if backend is None:
            backend = jax.default_backend()

        if batch_size is None:
            batch_size = {"cpu": DEVICES_COUNT, "gpu": 1000}[backend]

        @jax.jit
        def solve(t0, D):
            m = self.model(self.time, t0, D)
            ll, w, v = self._solve(m)
            return jnp.array([w[-1], v[-1, -1], ll])

        if backend == "cpu":
            solve_batch = jax.pmap(
                jax.vmap(solve, in_axes=(None, 0)), in_axes=(0, None)
            )
        else:
            solve_batch = jax.vmap(
                jax.vmap(solve, in_axes=(None, 0)), in_axes=(0, None)
            )

        t0s_padded = np.pad(t0s, [0, batch_size - (len(t0s) % batch_size) % batch_size])
        t0s_batches = np.reshape(
            t0s_padded, (len(t0s_padded) // batch_size, batch_size)
        )

        _progress = lambda x: (tqdm(x, unit_scale=batch_size) if progress else x)

        results = []

        for t0_batch in _progress(t0s_batches):
            results.append(solve_batch(t0_batch, Ds))

        depths, vars, ll = np.transpose(results, axes=[3, 0, 1, 2]).reshape(
            (3, len(t0s_padded), len(Ds))
        )[:, 0 : len(t0s), :]

        if positive:
            ll0 = self._solve(np.zeros_like(self.time))[0]
            ll[depths < 0] = ll0

        vars[~np.isfinite(vars)] = 1e25

        self.search_data = SearchData(
            t0s=t0s, Ds=Ds, ll=ll, z=depths, vz=vars, ll0=self.ll0
        )

    def periodic_search(self, periods: np.ndarray, progress=True, dphi=0.01):
        """Performs the periodic search.

        Parameters
        ----------
        periods : np.ndarray
            array of periods to search
        progress : bool, optional
            wether to show progress bar, by default True
        dphi: float, optional
            the relative step size of the phase grid. For each period, all likelihood quantities along time are
            interpolated along a phase grid of resolution `min(1/200, dphi/P))`. The smaller dphi
            the finer the grid, and the more resolved the model epoch and period (but the more computationally expensive the
            periodic search). The default is 0.01.

        Returns
        -------
        :py:class:`~nuance.SearchData`
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

    def mask_model(self, t0: float, D: float, P: float):
        """Return a `Solver` object where the model of epoch `t0` and duration `D` and period `P` is masked.

        Parameters
        ----------
        t0 : float
            model epoch
        D : float
            model duration
        P : float
            model period

        Returns
        -------
        `Nuance`
            A `Nuance` instance with model masked.
        """
        # search data
        search_data = self.search_data.copy()
        ph = utils.phase(search_data.t0s, t0, P)
        t0_mask = np.abs(ph) > 2 * D

        # if np.count_nonzero(t0_mask) == len(search_data.t0s):
        #     raise ValueError("Mask covers all data points")
        # elif len(t0_mask) == 0:
        #     raise ValueError("No data to mask")

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

    def mask_flares(self, build_gp=None, init=None, window=20, sigma=5, iterations=3):
        # for now
        assert build_gp is not None and init is not None

        flare_mask = np.ones_like(self.time).astype(bool)
        mu, nll = utils.minimize(self.time, self.flux, build_gp, X=self.X)

        for _ in range(iterations):
            residuals = self.flux - mu(gp_params)
            flare_mask = flare_mask & utils.sigma_clip_mask(
                residuals, sigma=sigma, window=window
            )
            gp_params = utils.minimize(nll, gp_params)

        new_nu = Nuance(
            time=self.time[flare_mask],
            flux=self.flux[flare_mask],
            X=self.X[:, flare_mask],
            gp=build_gp(gp_params, self.time[flare_mask]),
            compute=True,
        )

        return new_nu


def _search(p):
    phase, _, P2 = SEARCH(p)
    i, j = np.unravel_index(np.argmax(P2), P2.shape)
    Ti = phase[i] * p
    return Ti, j, p

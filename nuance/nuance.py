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
from tqdm import tqdm
from tqdm.autonotebook import tqdm

from nuance import DEVICES_COUNT, core, utils
from nuance.search_data import SearchData


@dataclass
class Nuance:
    """
    nuance search model
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
    model: callable = None
    """Model function with signature `model(time, t0, D, P=None)`"""

    def __post_init__(self):
        if self.model is None:
            self.model = partial(core.transit_protopapas, c=12)

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
            self.eval_model = jax.jit(core.eval_model(self.flux, self.X, self.gp))

        self.search_data = None

    @property
    def ll0(self) -> float:
        """log-likelihood of data without model

        Returns
        -------
        float
        """
        return self.eval_model(np.zeros_like(self.time))[0].__array__()

    def _models(self, m):
        _, w, _ = self.eval_model(m)
        mean = w[0:-1] @ self.X
        signal = m * w[-1]

        @jax.jit
        def gp_mean():
            return self.gp.condition(self.flux - mean - signal).gp.mean

        noise = gp_mean()

        return mean, signal, noise

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
            mask = np.ones_like(self.time).astype(bool)

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

    def mu(self, time=None):
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
        if time is None:
            time = self.time

        @jax.jit
        def _mu():
            gp = self.gp
            _, w, _ = self.eval_model(np.zeros_like(self.time))
            w = w[0:-1]
            cond_gp = gp.condition(self.flux - w @ self.X, time).gp
            return cond_gp.loc + w @ self.X

        return _mu()

    def models(self, t0: float = None, D: float = None, P: float = 1e15):
        """Return the models corresponding to epoch `t0` and duration `D`(and period `P` for a periodic model)

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
            - astro: signal being searched
            - noise: noise model

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
        m = self.model(self.time, t0, D, P)
        _, w, v = self.eval_model(m)
        return w, v

    def depth(self, t0: float, D: float, P: float = None):
        """depth linearly solved for epoch `t0` and duration `D` (and period `P` for a periodic model)

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
        return w[-1], np.sqrt(v[-1, -1])

    def snr(self, t0: float, D: float, P: float = None):
        """SNR of the model linearly solved for epoch `t0` and duration `D` (and period `P` for a periodic model)

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
        return np.max([0, w / dw])

    def linear_search(
        self,
        t0s: np.ndarray,
        Ds: np.ndarray,
        positive: bool = True,
        progress: bool = True,
        backend: str = None,
        batch_size: int = None,
    ):
        """Performs the linear search. Saves the linear search `Nuance.search_data` as a :py:class:`nuance.SearchData` object

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
            This affect the linear search function jax mapping strategy. For more details, see
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

        if backend == "cpu":
            eval_t0_Ds_function = core.pmap_cpus
            if batch_size is None:
                batch_size = DEVICES_COUNT

        elif backend == "gpu":
            eval_t0_Ds_function = core.vmap_gpu
            if batch_size is None:
                batch_size = 1000

        eval_t0s_Ds = eval_t0_Ds_function(self.eval_model, self.model, self.time)

        batches_n = int(np.ceil(len(t0s) / batch_size))
        padded_t0s = np.pad(t0s, pad_width=[0, batches_n * batch_size - len(t0s)])
        batched_t0s = np.array(np.array_split(padded_t0s, batches_n))

        ll = np.zeros((len(padded_t0s), len(Ds)))
        depths = ll.copy()
        vars = ll.copy()
        depths = ll.copy()

        _progress = lambda x: (tqdm(x) if progress else x)

        for i, t0 in enumerate(_progress(batched_t0s)):
            _depths, _vars, _ll = eval_t0s_Ds(t0, Ds)
            depths[i * batch_size : (i + 1) * batch_size, :] = _depths.T
            vars[i * batch_size : (i + 1) * batch_size, :] = _vars.T
            ll[i * batch_size : (i + 1) * batch_size, :] = _ll.T

        depths = np.array(depths[0 : len(t0s), :])
        vars = np.array(vars[0 : len(t0s), :])
        ll = np.array(ll[0 : len(t0s), :])

        if positive:
            ll0 = self.eval_model(np.zeros_like(self.time))[0]
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
        dphi: float, optional
            the relative step size of the phase grid. For each period, all likelihood quantities along time are
            interpolated along a phase grid of resolution `min(1/200, dphi/P))`. The smaller dphi
            the finer the grid, and the more resolved the model epoch and period (the the more computationally expensive the
            periodic search). The default is 0.01.

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

        if np.count_nonzero(t0_mask) == len(search_data.t0s):
            raise ValueError("Mask covers all data points")
        elif len(t0_mask) == 0:
            raise ValueError("No data to mask")

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
        mask = np.ones_like(self.time).astype(bool)

        if build_gp is not None:
            optimize, mu, nll = self.gp_optimization(build_gp)
            opt = init.copy()

        for _ in range(iterations):
            residuals = self.flux - mu(opt)
            mask[residuals > sigma * np.std(residuals[mask])] = False
            mask = np.roll(minimum_filter1d(mask, window), shift=window // 3)

            if build_gp is not None:
                optimize, mu, _ = self.gp_optimization(build_gp, mask)
                opt = optimize(init)

        new_nu = Nuance(
            time=self.time[mask],
            flux=self.flux[mask],
            X=self.X[:, mask],
            gp=build_gp(opt, self.time[mask]),
            compute=True,
        )

        return new_nu


def _search(p):
    phase, _, P2 = SEARCH(p)
    i, j = np.unravel_index(np.argmax(P2), P2.shape)
    Ti = phase[i] * p
    return Ti, j, p

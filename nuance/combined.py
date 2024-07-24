from dataclasses import dataclass
from functools import partial
from typing import List

import jax.numpy as jnp
import multiprocess as mp
import numpy as np
from jax.scipy.linalg import block_diag
from tqdm.auto import tqdm

from nuance import utils
from nuance.nuance import Nuance
from nuance.search_data import SearchData


def solve_triangular(*gps_y):
    Ls = [gp.solver.solve_triangular(y) for gp, y in gps_y]
    if Ls[0].ndim == 1:
        return jnp.hstack(Ls)
    else:
        return block_diag(*Ls)


@dataclass
class CombinedNuance:
    """
    An object to combine `nuance` searches from multiple datasets.

    Parameters
    ----------
    datasets : List[Nuance]
        list of :py:class:`~nuance.Nuance` instances
    """

    datasets: List[Nuance]
    """Nuance instance of each dataset, where the linear search must be already ran."""
    search_data: SearchData = None
    """SearchData instance of the combined dataset."""

    def __post_init__(self):
        self._fill_search_data()
        self._compute_L()

    def __getitem__(self, i):
        return self.datasets[i]

    @property
    def model(self):
        """The model"""
        return self.datasets[0].model

    def _fill_search_data(self):
        if all([d.search_data is not None for d in self.datasets]):
            t0s = np.hstack([d.search_data.t0s for d in self.datasets])
            all_Ds = [n.search_data.Ds for n in self.datasets]
            all_equal = (
                np.diff(np.vstack(all_Ds).reshape(len(all_Ds), -1), axis=0) == 0
            ).all()
            assert (
                all_equal
            ), "All datasets linear searches must have same duration grids"
            Ds = self.datasets[0].search_data.Ds
            ll = np.vstack([d.search_data.ll for d in self.datasets])
            z = np.vstack([d.search_data.z for d in self.datasets])
            vz = np.vstack([d.search_data.vz for d in self.datasets])
            ll0 = np.sum([d.search_data.ll0 for d in self.datasets])
            ll = (
                ll
                - np.hstack(
                    [
                        np.ones_like(dataset.search_data.t0s) * dataset.search_data.ll0
                        for dataset in self.datasets
                    ]
                )[:, None]
            ) + ll0

            self.search_data = SearchData(t0s, Ds, ll, z, vz, ll0)
        else:
            self.search_data = None

    @property
    def n(self):
        """Number of datasets"""
        return len(self.datasets)

    @property
    def time(self):
        """Time of all datasets"""
        return np.hstack([d.time for d in self.datasets])

    @property
    def flux(self):
        """Flux of all datasets"""
        return np.hstack([d.flux for d in self.datasets])

    def _compute_L(self):
        Liy = solve_triangular(*[(d.gp, d.flux) for d in self.datasets])
        LiX = solve_triangular(*[(d.gp, d.X.T) for d in self.datasets])

        def solve(ms):
            Lim = solve_triangular(*[(d.gp, m) for d, m in zip(self.datasets, ms)])
            LiXm = jnp.hstack([LiX, Lim[:, None]])
            LiXmT = LiXm.T
            LimX2 = LiXmT @ LiXm
            w = jnp.linalg.lstsq(LimX2, LiXmT @ Liy)[0]
            v = jnp.linalg.inv(LimX2)
            return w, v

        self._solve = solve

    def linear_search(
        self,
        t0s: np.ndarray,
        Ds: np.ndarray,
        positive: bool = True,
        progress: bool = True,
        backend: str = None,
        batch_size: int = None,
    ):
        """Performs the linear search for each dataset. Linear searches are saved as :py:class:`~nuance.SearchData`
        within each :py:class:`~nuance.Nuance` dataset.

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
        for d in self.datasets:
            d.linear_search(
                t0s,
                Ds,
                progress=progress,
                backend=backend,
                batch_size=batch_size,
                positive=positive,
            )

    def solve(self, t0: float, D: float, P: float):
        """Solve the combined model for a given set of parameters.

        Parameters
        ----------
        t0 : float
            epoch, same unit as time
        D : float
            duration, same unit as time
        P : float, optionale
            period, same unit as time, by default None
        c : float, optional
            c parameter of the transit model, by default None

        Returns
        -------
        list
            (w, v): linear coefficients and their covariance matrix
        """
        models = [self.model(d.search_data.t0s, t0, D, P) for d in self.datasets]
        w, v = self._solve(models)
        return w, v

    def snr(self, t0: float, D: float, P: float):
        """SNR of transit linearly solved for epoch `t0` and duration `D` (and period `P` for a periodic transit)

        Parameters
        ----------
        t0 : float
            epoch, same unit as time
        D : float
            duration, same unit as time
        P : float, optional
            period, same unit as time, by default None
        c : float, optional
            c parameter of the transit model, by default None

        Returns
        -------
        float
            transit snr
        """
        w, v = self.solve(t0, D, P)
        return jnp.max(jnp.array([0, w[-1] / jnp.sqrt(v[-1, -1])]))

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
        SEARCH = partial(self.search_data.fold)

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

    def models(self, t0: float, D: float, P: float, split=False):
        """Solve the combined model for a given set of parameters.

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
        ms = [d.model(d.time, t0, D, P) for d in self.datasets]
        w, _ = self._solve(ms)

        # means
        w_idxs = [0, *np.cumsum([d.X.shape[0] for d in self.datasets])]
        means = []
        for i in range(len(w_idxs) - 1):
            means.append(np.array(w[w_idxs[i] : w_idxs[i + 1]]) @ self.datasets[i].X)

        # signals
        signals = [self.model(d.time, t0, D, P=P) * w[-1] for d in self.datasets]

        # noises
        noises = []
        for i, d in enumerate(self.datasets):
            _, cond = d.gp.condition(d.flux - means[i] - signals[i])
            noises.append(cond.mean)
        if split:
            return means, signals, noises
        else:
            return np.hstack(means), np.hstack(signals), np.hstack(noises)

    def mask_model(self, t0: float, D: float, P: float):
        new_self = self.__class__(
            datasets=[d.mask_model(t0, D, P) for d in self.datasets], c=self.c
        )
        new_self._fill_search_data()
        new_self._compute_L()
        return new_self


def _search(p):
    phase, P2 = SEARCH(p)
    i, j = np.unravel_index(np.argmax(P2), P2.shape)
    Ti = phase[i] * p
    return Ti, j, p

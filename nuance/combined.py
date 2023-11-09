from dataclasses import dataclass
from typing import List
from nuance.nuance import Nuance
from nuance.search_data import SearchData
import numpy as np
from scipy.linalg import block_diag
from nuance.utils import periodic_transit
from tqdm.autonotebook import tqdm
import jax.numpy as jnp
from nuance import utils


def solve_triangular(*gps_y):
    Ls = [gp.solver.solve_triangular(y) for gp, y in gps_y]
    if Ls[0].ndim == 1:
        return np.hstack(Ls)
    else:
        return block_diag(*Ls)


@dataclass
class CombinedNuance:
    """
    An object for nuanced transit search in multiple datasets.
    """

    datasets: List[Nuance]
    """Nuance instance of each dataset, where the linear search must be already ran."""
    search_data: SearchData = None
    """SearchData instance of the combined dataset."""
    c: float = 12.0
    """The c parameter of the transit model."""

    def __post_init__(self):
        self._fill_search_data()
        self._compute_L()

    def _fill_search_data(self):
        if all([d.search_data is not None for d in self.datasets]):
            t0s = np.hstack([d.search_data.t0s for d in self.datasets])
            Ds = np.hstack([d.search_data.Ds for d in self.datasets])
            ll = np.vstack([d.search_data.ll for d in self.datasets])
            z = np.vstack([d.search_data.z for d in self.datasets])
            vz = np.vstack([d.search_data.vz for d in self.datasets])
            ll0 = np.sum([d.search_data.ll0 for d in self.datasets])

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

        def eval_m(ms):
            Lim = solve_triangular(*[(d.gp, m) for d, m in zip(self.datasets, ms)])
            LiXm = jnp.hstack([LiX, Lim[:, None]])
            LiXmT = LiXm.T
            LimX2 = LiXmT @ LiXm
            w = jnp.linalg.lstsq(LimX2, LiXmT @ Liy)[0]
            v = jnp.linalg.inv(LimX2)
            return w, v

        self.eval_m = eval_m

    def linear_search(self, t0s, Ds, progress=True):
        for d in self.datasets:
            d.linear_search(t0s, Ds, progress=progress)

    def periodic_transits(self, t0, D, P, c=None):
        if c is None:
            c = self.c
        return [periodic_transit(d.search_data.t0s, t0, D, P) for d in self.datasets]

    def solve(self, t0, D, P, c=None):
        """Solve the combined model for a given set of parameters.

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
        list
            (w, v): linear coefficients and their covariance matrix
        """
        if c is None:
            c = self.c
        w, v = self.eval_m(self.periodic_transits(t0, D, P, c))
        return w, v

    def snr(self, t0, D, P, c=None):
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
        if c is None:
            c = self.c
        w, v = self.solve(t0, D, P, c)
        return w[-1] / jnp.sqrt(v[-1, -1])

    def periodic_search(self, periods, dphi=0.01):
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
            the finer the grid, and the more resolved the transit epoch and period (the the more computationally expensive the
            periodic search). The default is 0.01.

        Returns
        -------
        :py:class:`nuance.SearchData`
            search results
        """
        n = len(periods)
        fold_functions = [d.search_data.fold_ll for d in self.datasets]
        Ds = self.search_data.Ds

        def _search(p):
            phase, _, p0 = fold_functions[0](p, dphi)
            P2 = p0 + np.sum([f(p)[2] for f in fold_functions[1:]], axis=0)
            i, j = np.unravel_index(np.argmax(P2), P2.shape)
            Ti = phase[i] * p
            return float(self.snr(Ti, Ds[j], p)), (
                Ti,
                Ds[j],
                p,
            )

        snr = np.zeros(n)
        params = np.zeros((n, 3))

        for i, p in enumerate(tqdm(periods)):
            snr[i], params[i] = _search(p)

        new_search_data = self.search_data.copy()

        new_search_data.periods = periods
        new_search_data.Q_snr = snr
        new_search_data.Q_params = params

        return new_search_data

    def models(self, t0, D, P, c=None):
        """Solve the combined model for a given set of parameters.

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
        list
            (w, v): linear coefficients and their covariance matrix
        """
        if c is None:
            c = self.c
        m = self.periodic_transits(t0, D, P, c)
        w, _ = self.eval_m(m)

        # means
        w_idxs = [0, *np.cumsum([d.X.shape[0] for d in self.datasets])]
        means = []
        for i in range(len(w_idxs) - 1):
            means.append(np.array(w[w_idxs[i] : w_idxs[i + 1]]) @ self.datasets[i].X)

        # signals
        signals = [
            utils.transit(d.time, t0, D, P=P, c=c) * w[-1] for d in self.datasets
        ]

        # noises
        noises = []
        for i, d in enumerate(self.datasets):
            _, cond = d.gp.condition(d.flux - means[i] - signals[i])
            noises.append(cond.mean)

        return np.hstack(means), np.hstack(signals), np.hstack(noises)

    def mask(self):
        new_self = self.__class__(datasets=[d.mask() for d in self.datasets], c=self.c)
        new_self.datasets = [d.mask() for d in self.datasets]
        new_self._fill_search_data()

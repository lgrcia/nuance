from dataclasses import dataclass
from typing import List
from nuance.nuance import Nuance
from nuance.search_data import SearchData
import numpy as np
from scipy.linalg import block_diag
from nuance.utils import periodic_transit
from tqdm.autonotebook import tqdm
import jax.numpy as jnp


def solve_triangular(*gps_y):
    Ls = [gp.solver.solve_triangular(y) for gp, y in gps_y]
    if Ls[0].ndim == 1:
        return np.hstack(Ls)
    else:
        return block_diag(*Ls)


@dataclass
class CombinedNuance:
    datasets: List[Nuance]
    search_data: SearchData = None
    c: float = 12.0

    def __post_init__(self):
        for d in self.datasets:
            assert (
                d.search_data is not None
            ), "Linear search missing for at least one dataset. Run `linear_search` on all datasets."

        t0s = np.hstack([d.search_data.t0s for d in self.datasets])
        Ds = np.hstack([d.search_data.Ds for d in self.datasets])
        ll = np.vstack([d.search_data.ll for d in self.datasets])
        z = np.vstack([d.search_data.z for d in self.datasets])
        vz = np.vstack([d.search_data.vz for d in self.datasets])
        ll0 = np.sum([d.search_data.ll0 for d in self.datasets])

        self.search_data = SearchData(t0s, Ds, ll, z, vz, ll0)
        self._compute_L()

    @property
    def n(self):
        """Number of datasets"""
        return len(self.datasets)

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

    def periodic_transits(self, t0, D, P, c=None):
        if c is None:
            c = self.c
        return [periodic_transit(d.search_data.t0s, t0, D, P) for d in self.datasets]

    def solve(self, t0, D, P, c=None):
        if c is None:
            c = self.c
        w, v = self.eval_m(self.periodic_transits(t0, D, P, c))
        return w, v

    def snr(self, t0, D, P, c=None):
        if c is None:
            c = self.c
        w, v = self.solve(t0, D, P, c)
        return w[-1] / jnp.sqrt(v[-1, -1])

    def models(self, t0, D, P, c=None):
        if c is None:
            c = self.c
        w, v = self.solve(t0, D, P, c)
        return [d.gp.predict(w[:-1], v[:-1, :-1]) for d in self.datasets]

    def periodic_search(self, periods, dphi=0.01):
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

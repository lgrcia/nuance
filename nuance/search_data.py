import pickle
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d

from nuance.utils import interp_split_times


@dataclass
class SearchData:
    """
    An object that holds the results of the transit search.
    """

    # linear search
    t0s: np.ndarray
    Ds: np.ndarray
    ll: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    vz: Optional[np.ndarray] = None
    ll0: Optional[float] = None

    # periodic search, Q is periodogram
    periods: Optional[np.ndarray] = None
    Q_snr: Optional[np.ndarray] = None
    Q_ll: Optional[np.ndarray] = None
    Q_params: Optional[np.ndarray] = None

    @property
    def folds(self):
        f_ll = interp2d(self.Ds, self.t0s, self.ll)
        f_z = interp2d(self.Ds, self.t0s, self.z)
        f_dz2 = interp2d(self.Ds, self.t0s, self.vz)

        def interpolate_all(pt0s):
            # computing the likelihood folds by interpolating in phase
            folds_ll = np.array([f_ll(self.Ds, t) for t in pt0s])
            folds_z = np.array([f_z(self.Ds, t) for t in pt0s])
            folds_vz = np.array([f_dz2(self.Ds, t) for t in pt0s])

            return folds_ll, folds_z, folds_vz

        def _folds(p, dphi=0.01):
            pt0s = interp_split_times(self.t0s, p, dphi=dphi)
            return pt0s, interpolate_all(pt0s)

        return _folds

    @property
    def fold_ll(self):
        """
        Returns a function that folds the likelihoods of all periods and t0s into a single period-folded likelihood.

        Returns:
            function: A function that takes a period and returns the folded likelihoods.
        """

        folds = self.folds

        def _fold(p, dphi=0.01):
            pt0s, (lls, zs, vzs) = folds(p, dphi=dphi)
            P1 = np.sum(lls, 0)
            vZ = 1 / np.sum(1 / vzs, 0)
            Z = vZ * np.sum(zs / vzs, 0)
            P1 = np.sum(lls, 0)
            P2 = 0.5 * np.sum(
                np.log(vzs) - np.log(vzs + vZ) + (zs - Z) ** 2 / (vzs + vZ), 0
            )

            return pt0s[0] / p, P1, P1 - P2

        return _fold

    @property
    def best(self):
        """
        Returns the best-fit parameters for the search object.

        Returns:
            tuple: A tuple containing the best-fit t0, D, and period (if available).
        """
        assert (
            self.ll is not None and self.Q_params is not None and self.Q_snr is not None
        ), "No search performed"
        if self.periods is not None:
            i = np.argmax(self.Q_snr)
            return self.Q_params[i]
        else:
            i, j = np.unravel_index(np.argmax(self.ll), self.ll.shape)
            t0, D = self.t0s[i], self.Ds[j]
            period = None
        return t0, D, period

    @property
    def shape(self):
        """
        Returns the shape of the likelihood array.

        Returns:
            tuple: A tuple containing the number of t0s and Ds in the search objects.
        """
        return len(self.t0s), len(self.Ds)

    def show_ll(self, **kwargs):
        """
        Plots the likelihood array.

        Args:
            **kwargs: Additional keyword arguments to pass to `plt.imshow`.
        """
        assert self.ll is not None, "No search performed"
        extent = np.min(self.t0s), np.max(self.t0s), np.min(self.Ds), np.max(self.Ds)
        plt.imshow(self.ll.T, aspect="auto", origin="lower", extent=extent, **kwargs)

    def asdict(self):
        """
        Returns a dictionary representation of the search object.

        Returns:
            dict: A dictionary containing the search object parameters.
        """
        return asdict(self)

    def save(self, filename):
        """
        Saves the search to a file.

        Args:
            filename (str): The name of the file to save the search object to.
        """

        pickle.dump(self.asdict(), open(filename, "wb"))

    def copy(self):
        """
        Returns a deep copy of the search object.

        Returns:
            Search: A deep copy of the search object.
        """
        return deepcopy(self)

    @classmethod
    def load(cls, filename):
        """
        Loads a search object from a file.

        Args:
            filename (str): The name of the file to load the search object from.

        Returns:
            Search: The loaded search object.
        """
        return cls(**pickle.load(open(filename, "rb")))

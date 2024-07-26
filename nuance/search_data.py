import pickle
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from nuance.core import nearest_neighbors as interp2d
from nuance.utils import interp_split_times


@dataclass
class SearchData:
    """
    An object that holds the results of the search.
    """

    # linear search grid
    t0s: np.ndarray
    """Array of searched signal trial epochs."""
    Ds: np.ndarray
    """Array of searched signal trial durations."""

    # linear search results
    ll: Optional[np.ndarray] = None
    """Non-periodic signal likelihoods."""
    z: Optional[np.ndarray] = None
    """Non-periodic signal depths."""
    vz: Optional[np.ndarray] = None
    """Non-periodic signal depth variance."""
    ll0: Optional[float] = None
    """No-signal likelihood."""

    # periodic search, Q is periodogram
    periods: Optional[np.ndarray] = None
    """Array of signal trial periods."""
    Q_snr: Optional[np.ndarray] = None
    """Periodogram SNR."""
    Q_ll: Optional[np.ndarray] = None
    """Periodic signal likelihoods."""
    Q_params: Optional[np.ndarray] = None
    """Periodogram best-fit parameters."""

    @property
    def fold(self):
        """
        Returns a function that folds the signal likelihoods of all periods and t0s into a
        single period-folded likelihood.

        Returns:
            function: A function that takes a period and returns the folded likelihoods.
        """

        f_ll = interp2d(self.t0s, self.ll)
        f_z = interp2d(self.t0s, self.z)
        f_dz2 = interp2d(self.t0s, self.vz)

        def _fold(times):
            lls = np.array([f_ll(time) for time in times])
            zs = np.array([f_z(time) for time in times])
            vzs = np.array([f_dz2(time) for time in times])

            P1 = np.sum(lls, 0)
            vZ = 1 / np.sum(1 / vzs, 0)
            Z = vZ * np.sum(zs / vzs, 0)
            P1 = np.sum(lls, 0)
            P2 = 0.5 * np.sum(
                np.log(vzs) - np.log(vzs + vZ) + (zs - Z) ** 2 / (vzs + vZ), 0
            )

            return P1 - P2

        def fun(period):
            times = interp_split_times(self.t0s, period)
            return times[0] / period, _fold(times)

        return fun

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

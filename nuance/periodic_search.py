"""
The periodic search module provides functions to compute the probability of 
a periodic signal to be present in the data, using quantities computed from single 
events statistics.
"""

import multiprocess as mp
import numpy as np
from tqdm.auto import tqdm

from nuance import core
from nuance.utils import interp_split_times


def periodic_search(epochs, durations, ls, snr_f, progress=True):
    """Returns a function that performs the periodic search given an array of periods.

    Parameters
    ----------
    epochs : np.ndarray
        Values of epochs for with the likelihoods have been computed.
    durations : np.ndarray
        Values of durations for with the likelihoods have been computed.
    ls : list, tuple, np.ndarray
        Tuple of (likelihoods, depth, depths variance).
    snr_f : callable
        Function that computes the SNR given the epoch, duration and period.
    progress : bool, optional
        wether to show progress bar, by default True

    Returns
    -------
    callable
        Function that computes the SNR and parameters for each period.
    """
    global fold_f
    fold_f = _fold_ll(epochs, *ls)

    def _progress(x, **kwargs):
        return tqdm(x, **kwargs) if progress else x

    def function(periods):
        snr = np.zeros(len(periods))
        params = np.zeros((len(periods), 3))

        with mp.Pool() as pool:
            for p, (epoch, duration_i, period) in enumerate(
                _progress(pool.imap(_solve, periods), total=len(periods))
            ):
                Dj = durations[duration_i]
                snr[p], params[p] = float(snr_f(epoch, Dj, period)), (epoch, Dj, period)

        return snr, params

    return function


def _fold_ll(epochs, lls, z, vz):
    f_ll = core.nearest_neighbors(epochs, lls)
    f_z = core.nearest_neighbors(epochs, z)
    f_dz2 = core.nearest_neighbors(epochs, vz)

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
        times = interp_split_times(epochs, period)
        phase = times[0] / period
        lls = _fold(times)
        return phase, lls

    return fun


def _solve(period):
    phase, lls = fold_f(period)
    epoch_i, duration_i = np.unravel_index(np.argmax(lls), lls.shape)
    epoch = phase[epoch_i] * period
    return epoch, duration_i, period

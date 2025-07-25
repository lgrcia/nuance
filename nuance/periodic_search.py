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

    fold_f = _fold_ll(epochs, *ls)

    def _progress(x, **kwargs):
        return tqdm(x, **kwargs) if progress else x

    def function(periods, processes=None, chunksize=500):
        snr = np.zeros(len(periods))
        params = np.zeros((len(periods), 3))

        ctx = mp.get_context('spawn')  # Can't use fork with jax.
        with ctx.Pool(processes=processes) as pool:
            for p, (epoch, duration_i, period) in enumerate(
                _progress(pool.starmap(_solve, [(period, fold_f) for period in periods], chunksize=chunksize), total=len(periods))
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
        lls = f_ll(times)
        zs = f_z(times)
        vzs = f_dz2(times)

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


def _solve(period, fold_f):
    phase, lls = fold_f(period)
    epoch_i, duration_i = np.unravel_index(np.argmax(lls), lls.shape)
    epoch = phase[epoch_i] * period
    return epoch, duration_i, period


def main():
    return


if __name__ == '__main__':
    main()

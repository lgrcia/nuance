"""
The periodic search module provides functions to compute the probability of 
a periodic signal to be present in the data, using quantities computed from single 
events statistics.
"""

import multiprocess as mp
from functools import partial
import jax
import numpy as np
from tqdm.auto import tqdm

from nuance import DEVICES_COUNT, core
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

    def function(periods, processes=DEVICES_COUNT, batch_size=DEVICES_COUNT):
        snr = np.zeros(len(periods))
        params = np.zeros((len(periods), 3))

        # Use multiprocessing to get the optimal epoch and duration at each period.
        solve_f = partial(_solve, fold_f)
        ctx = mp.get_context('spawn')  # Can't use fork with jax.
        with ctx.Pool(processes=processes) as pool:
            period_chunks = [periods[i::processes] for i in range(processes)]

            for i, result in enumerate(_progress(pool.imap(solve_f, period_chunks), total=processes)):
                epochs_chunk, duration_idx_chunk, periods_chunk = result
                params[i::processes, 0] = epochs_chunk
                params[i::processes, 1] = durations[duration_idx_chunk]
                params[i::processes, 2] = periods_chunk

        # Use jax.vmap to get the SNR at each period.
        snr_vmap = jax.vmap(snr_f, in_axes=(0, 0, 0))
        for i in _progress(range(0, len(periods), batch_size), unit_scale=batch_size):
            imin = i
            imax = i + batch_size
            snr[imin:imax] = snr_vmap(params[imin:imax, 0], params[imin:imax, 1], params[imin:imax, 2])

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


def _solve(fold_f, periods):

    epochs = np.zeros_like(periods)
    duration_idx = np.zeros_like(periods, dtype='int')

    for i, period in enumerate(periods):
        phase, lls = fold_f(period)
        epoch_i, duration_i = np.unravel_index(np.argmax(lls), lls.shape)
        epoch = phase[epoch_i] * period

        epochs[i] = epoch
        duration_idx[i] = duration_i

    return epochs, duration_idx, periods


def main():
    return


if __name__ == '__main__':
    main()

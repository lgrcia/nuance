import jax
import numpy as np
from tqdm import tqdm

from nuance.experimental.utils import phase_coverage
from nuance.periodic_search import _fold_ll


def periodic_search_with_events(
    epochs, durations, ls, snr_f, progress=True, min_events=2
):
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

    global coverage_function
    coverage_function = jax.jit(phase_coverage(epochs))

    def _progress(x, **kwargs):
        return tqdm(x, **kwargs) if progress else x

    def single_snr(period):
        _phase, lls = fold_f(period)

        # where the check for multiple event happens
        has_less_than_min_events = coverage_function(period, _phase) < min_events
        dphi = np.median(np.diff(_phase))
        phi0 = int(0.5 / dphi)
        has_less_than_min_events = np.roll(has_less_than_min_events, phi0).astype(float)
        lls = lls - 1e16 * has_less_than_min_events[:, None]

        i, j = np.unravel_index(np.argmax(lls), lls.shape)
        Ti = _phase[i] * period
        snr_value = snr_f(Ti, durations[j], period) if np.any(lls > 0) else 0.0
        return snr_value, (Ti, durations[j], period)

    def function(periods):
        snr = np.zeros(len(periods))
        params = np.zeros((len(periods), 3))

        for p, period in enumerate(_progress(periods)):
            snr[p], params[p] = single_snr(period)

        return snr, params

    return function

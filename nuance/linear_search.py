"""
The linear search module provides functions to compute single events statistics.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import tinygp
from tqdm.auto import tqdm

from nuance import DEVICES_COUNT, core


def linear_search(
    time: np.ndarray,
    flux: np.ndarray,
    gp: tinygp.GaussianProcess | None = None,
    X: np.ndarray | None = None,
    model: Callable | None = None,
    positive: bool = True,
    progress: bool = True,
    backend: str | None = None,
    batch_size: int | None = None,
):
    """Returns a function that computes the log likelihood of a transit model at
    different epochs and durations (linear search)

    Parameters
    ----------
    time : np.ndarray
        array of times
    flux : np.ndarray
        flux time-series
    gp : tinygp.GaussianProcess, optional
        tinygp GaussianProcess model, by default None
    X : np.ndarray, optional
        linear model design matrix, by default None
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
    """
    X, gp, model = core.check_default(time, X, gp, model)

    solver = jax.jit(core.solve(time, flux, gp=gp, X=X))

    if backend is None:
        backend = jax.default_backend()

    if batch_size is None:
        batch_size = {"cpu": DEVICES_COUNT, "gpu": 1000}[backend]

    if backend == "cpu":
        solve_batch = jax.pmap(jax.vmap(solver, in_axes=(None, 0)), in_axes=(0, None))
    else:
        solve_batch = jax.jit(
            jax.vmap(jax.vmap(solver, in_axes=(None, 0)), in_axes=(0, None))
        )

    def function(epochs: np.ndarray, durations: np.ndarray):
        """Compute the log likelihood of a transit model at different epochs and durations

        Parameters
        ----------
        epochs : np.ndarray
            array of epochs
        durations : np.ndarray
            array of durations

        Returns
        -------
        tuple
            (log likelihoods, model depths, depths variances)
        """
        epochs_padded = np.pad(
            epochs, [0, batch_size - (len(epochs) % batch_size) % batch_size]
        )
        epochs_batches = np.reshape(
            epochs_padded, (len(epochs_padded) // batch_size, batch_size)
        )

        _progress = lambda x: (tqdm(x, unit_scale=batch_size) if progress else x)

        results = []

        for epoch_batch in _progress(epochs_batches):
            results.append(solve_batch(epoch_batch, durations))

        depths, vars, log_likelihoods = np.transpose(
            results, axes=[3, 0, 1, 2]
        ).reshape((3, len(epochs_padded), len(durations)))[:, 0 : len(epochs), :]

        if positive:
            ll0 = core.solve_model(flux, X, gp)(jnp.zeros_like(time))[0]
            log_likelihoods[depths < 0] = ll0

        vars[~np.isfinite(vars)] = 1e25

        # trick to store ll0
        log_likelihoods[0, 0] = ll0
        depths[0, 0] = 0.0

        return log_likelihoods, depths, vars

    return function


def combine_linear_searches(*linear_searches):
    """Combine the results of multiple linear searches

    linear_searches : list
        lists of (log likelihoods, model depths, depths variances)

    Returns
    -------
    tuple
        (log likelihoods, model depths, depths variances)

    Example
    -------
    .. code-block:: python

        ls, z, vz = combine_linear_searches((ls0, z0, vz0), (ls1, z1, vz1), (ls2, z2, vz2))

    """
    ll = np.vstack([ls[0] for ls in linear_searches])
    z = np.vstack([ls[1] for ls in linear_searches])
    vz = np.vstack([ls[2] for ls in linear_searches])
    ll0s = [ls[0][0, 0] for ls in linear_searches]
    ll = (
        ll
        - np.hstack(
            [
                np.ones(linear_searches[i][0].shape[0]) * ll0s[i]
                for i in range(len(linear_searches))
            ]
        )[:, None]
    ) + np.sum(ll0s)

    return np.array([ll, z, vz])

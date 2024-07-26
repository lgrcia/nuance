import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from nuance import DEVICES_COUNT, core


def linear_search(
    time,
    flux,
    gp=None,
    X=None,
    model=None,
    positive: bool = True,
    progress: bool = True,
    backend: str | None = None,
    batch_size: int | None = None,
):
    if X is None:
        X = core.DEFAULT_X(time)
    if gp is None:
        gp = core.DEFAULT_GP(time)
    if model is None:
        model = core.transit

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

    def function(epochs, durations):
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

        depths, vars, loglikelihoods = np.transpose(results, axes=[3, 0, 1, 2]).reshape(
            (3, len(epochs_padded), len(durations))
        )[:, 0 : len(epochs), :]

        if positive:
            ll0 = core.solve_model(flux, X, gp)(jnp.zeros_like(time))[0]
            loglikelihoods[depths < 0] = ll0

        vars[~np.isfinite(vars)] = 1e25

        return loglikelihoods, depths, vars

    return function

import os

import jax

jax.config.update("jax_enable_x64", True)
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"

import numpy as np
from tinygp import GaussianProcess, kernels

from nuance import core, linear_search, periodic_search

depth = 2.3e-3
error = 1e-3
time = np.linspace(0, 3.0, 4000)
transit_params = {"epoch": 0.2, "duration": 0.05, "period": 0.7}
transit_model = depth * core.transit(time, **transit_params)
kernel = kernels.quasisep.SHO(10.0, 10.0, 0.002)
gp = GaussianProcess(kernel, time, diag=error**2)
flux = transit_model + gp.sample(jax.random.PRNGKey(0)) + 1.0


def test_full():
    # linear search
    epochs = time.copy()
    durations = np.linspace(0.01, 0.2, 15)
    ls = linear_search(time, flux, gp=gp)(epochs, durations)

    # periodic search
    periods = np.linspace(0.3, 5, 2000)
    snr_function = jax.jit(core.snr(time, flux, gp=gp))
    ps_function = periodic_search(epochs, durations, ls, snr_function)
    snr, params = ps_function(periods)

    period = params[np.argmax(snr)][-1]
    np.testing.assert_allclose(period, transit_params["period"], atol=0.001)

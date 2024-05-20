import jax.numpy as jnp
from tinygp import GaussianProcess, kernels


def Rotation(sigma, period, Q0, dQ, f):
    """
    A kernel for a rotating star with a single mode of oscillation.
    """
    Q1 = 1 / 2 + Q0 + dQ
    w1 = (4 * jnp.pi * Q1) / (period * jnp.sqrt(4 * Q1**2 - 1))
    s1 = sigma**2 / ((1 + f) * w1 * Q1)

    Q2 = 1 / 2 + Q0
    w2 = (8 * jnp.pi * Q1) / (period * jnp.sqrt(4 * Q1**2 - 1))
    s2 = f * sigma**2 / ((1 + f) * w2 * Q2)
    kernel = kernels.quasisep.SHO(w1, Q1, s1) + kernels.quasisep.SHO(w2, Q2, s2)
    return kernel


def rotation(period=None, error=None, long_scale=0.5):
    initial_params = {
        "log_period": jnp.log(period) if period is not None else jnp.log(1.0),
        "log_Q": jnp.log(100),
        "log_sigma": jnp.log(1e-1),
        "log_dQ": jnp.log(100),
        "log_f": jnp.log(2.0),
        "log_short_scale": jnp.log(1e-2),
        "log_short_sigma": jnp.log(1e-2),
        "log_long_sigma": jnp.log(1e-2),
        "log_jitter": jnp.log(1.0) if error is None else jnp.log(error),
    }

    def build_gp(params, time):
        jitter2 = jnp.exp(2 * params["log_jitter"])
        short_scale = jnp.exp(params["log_short_scale"])
        short_sigma = jnp.exp(params["log_short_sigma"])
        long_sigma = jnp.exp(params["log_long_sigma"])

        kernel = (
            Rotation(
                jnp.exp(params["log_sigma"]),
                jnp.exp(params["log_period"]),
                jnp.exp(params["log_Q"]),
                jnp.exp(params["log_dQ"]),
                jnp.exp(params["log_f"]),
            )
            + kernels.quasisep.Exp(short_scale, short_sigma)
            + kernels.quasisep.Exp(long_scale, long_sigma)
        )

        return GaussianProcess(kernel, time, diag=jitter2, mean=0.0)

    return build_gp, initial_params

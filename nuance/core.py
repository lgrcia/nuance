import jax
import jax.numpy as jnp


def eval_model(flux, X, gp):
    Liy = gp.solver.solve_triangular(flux)
    LiX = gp.solver.solve_triangular(X.T)

    @jax.jit
    def function(m):
        Xm = jnp.vstack([X, m])
        Lim = gp.solver.solve_triangular(m)
        LiXm = jnp.hstack([LiX, Lim[:, None]])
        LiXmT = LiXm.T
        LimX2 = LiXmT @ LiXm
        w = jnp.linalg.lstsq(LimX2, LiXmT @ Liy)[0]
        v = jnp.linalg.inv(LimX2)
        return gp.log_probability(flux - w @ Xm), w, v

    return function


@jax.jit
def transit_protopapas(t, t0, D, P=1e15, c=12, d=1.0):
    _t = P * jnp.sin(jnp.pi * (t - t0) / P) / (jnp.pi * D)
    return -d * 0.5 * jnp.tanh(c * (_t + 1 / 2)) + 0.5 * jnp.tanh(c * (_t - 1 / 2))


@jax.jit
def transit_box(time, t0, D, P=1e15):
    return -(jnp.abs((time - t0) % P) < D / 2).astype(float)

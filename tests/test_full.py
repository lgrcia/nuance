import numpy as np

from nuance import Nuance, utils

true_P = 0.7
(time, flux, error), X, gp = utils.simulated(P=true_P)


def test_full():
    nu = Nuance(time, flux, gp=gp, X=X)

    # linear search
    t0s = time.copy()
    Ds = np.linspace(0.01, 0.2, 15)
    nu.linear_search(t0s, Ds)

    # periodic search
    periods = np.linspace(0.3, 5, 2000)
    search = nu.periodic_search(periods)

    t0, D, P = search.best
    np.testing.assert_allclose(P, true_P, atol=0.001)

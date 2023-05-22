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


def test_mask_t0s_equal_time():
    nu = Nuance(time, flux, gp=gp, X=X)

    # linear search
    t0s = time.copy()
    Ds = np.linspace(0.01, 0.2, 15)
    nu.linear_search(t0s, Ds)

    # mask
    nu.mask(0.1, 0.01, 2.0)


def test_mask_t0s_not_equal_time():
    nu = Nuance(time, flux, gp=gp, X=X)

    # linear search
    t0s = np.random.choice(time, size=100, replace=False)
    Ds = np.linspace(0.01, 0.2, 15)
    nu.linear_search(t0s, Ds)

    # mask
    nu.mask(0.1, 0.01, 2.0)

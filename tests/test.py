from nuance import Nuance
from nuance import utils
import numpy as np
import unittest

np.random.seed(42)
time = np.linspace(0, 6, 500)
diff_error = 0.001
X = np.vander(time, N=4, increasing=True)
w = [1.0, 0.05, -0.2, -0.5]
true_transit = 0.01 * utils.periodic_transit(time, 0.2 / 4, 0.05, P=1.3)
diff_flux = true_transit + np.random.normal(0.0, diff_error, size=len(time))


class TestTransitSearch(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_full_search(self):
        from nuance import Nuance
        from tinygp import kernels, GaussianProcess

        gp = GaussianProcess(
            kernels.quasisep.Exp(1e15), time, diag=diff_error**2, mean=1.0
        )
        nu = Nuance(time, diff_flux, gp=gp, X=X.T)

        t0s = time.copy()
        Ds = np.linspace(0.01, 0.1, 10)
        _ = nu.linear_search(t0s, Ds)

        periods = np.linspace(1, 5, 5000)
        _ = nu.periodic_search(periods)


if __name__ == "__main__":
    unittest.main()

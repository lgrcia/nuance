import jax
from tinygp import kernels, GaussianProcess
import numpy as np
import jax.numpy as jnp
from .utils import transit
from tqdm.autonotebook import tqdm

class Nuance:
    
    def __init__(self, x, y, e, X=None, kernel=None):
        self.x = x
        self.y = y
        self.e = e
        
        if kernel is None:
            kernel = kernels.Constant(0.)
        
        if X is None:
            X = np.atleast_2d(np.ones_like(x))
        
        self.kernel = kernel
        self.X = X
        
        self.gp = GaussianProcess(kernel, x, diag=e**2, mean=0.)
        Liy = self.gp.solver.solve_triangular(y)
        LiX = self.gp.solver.solve_triangular(X.T)
            
        @jax.jit
        def eval_m(m):
            Xm = jnp.vstack([X, m])
            Lim = self.gp.solver.solve_triangular(m)
            LiXm = jnp.hstack([LiX, Lim[:, None]])
            LiXmT = LiXm.T
            LimX2 = LiXmT@LiXm
            w = jnp.linalg.lstsq(LimX2, LiXmT@Liy)[0]
            v = jnp.linalg.inv(LimX2)
            return self.gp.log_probability(y - w@Xm), w, v

        self.eval_m = eval_m
        
    def linear_search(self, t0s, durations, positive=True):
        self.t0s = t0s
        self.durations = durations
        ls = np.zeros((len(t0s), len(durations)))
        depths = ls.copy()
        n = len(self.X)

        for i, t0 in enumerate(tqdm(t0s)):
            for j, D in enumerate(durations):
                m = transit(self.x, t0, D, 1)
                ll, w, _ = self.eval_m(m)
                depths[i, j] = w[n]
                ls[i, j] = ll.copy()

        if positive:
            ll0 = self.eval_m(np.zeros_like(self.x))[0]
            ls[depths<0] = ll0

        return ls
    
    def models(self, t0, D):
        astro = transit(self.x, t0, D, 1)
        _, w, v = self.eval_m(astro)
        mean = w[0:-1]@self.X
        _, cond = self.gp.condition(self.y)
        noise = cond.mean

        return mean, astro*w[-1], noise

    def solve(self, t0, D):
        m = transit(self.x, t0, D, 1)
        _, w, v = self.eval_m(m)
        return w, v
    
    def depth(self, t0, D):
        m = transit(self.x, t0, D, 1)
        _, w, v = self.eval_m(m)
        return w[-1], v[-1, -1]
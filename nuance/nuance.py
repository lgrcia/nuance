import jax
from tinygp import kernels, GaussianProcess
import numpy as np
import jax.numpy as jnp
from .utils import transit, interp_split_times, periodic_transit
from tqdm.autonotebook import tqdm
from scipy.interpolate import interp2d

class Nuance:
    
    def __init__(self, x, y, gp, X=None):
        """Nuance

        Parameters
        ----------
        x : array
            dimension
        y : array
            observed
        gp : array or tinygp.gp.GaussianProcess
            error or tinygp.GaussianProcess instance 
        X : ndarray, optional
            design matrix, by default None
        """
        self.x = x
        self.y = y
        
        if not isinstance(gp, GaussianProcess):
            kernel = kernels.Constant(0.)
            self.gp = GaussianProcess(kernel, x, diag=gp**2, mean=0.)
        else:
            self.gp = gp
        
        if X is None:
            X = np.atleast_2d(np.ones_like(x))
        
        self.X = X        
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
        
    def linear_search(self, t0s, Ds, positive=True):
        self.t0s = t0s
        self.Ds = Ds
        ll = np.zeros((len(t0s), len(Ds)))
        depths = np.zeros((len(t0s), len(Ds)))
        vars = ll.copy()
        depths = ll.copy()
        n = len(self.X)

        for i, t0 in enumerate(tqdm(t0s)):
            for j, D in enumerate(Ds):
                m = transit(self.x, t0, D, 1)
                _ll, w, v = self.eval_m(m)
                depths[i, j] = w[n]
                vars[i, j] = v[n, n]
                ll[i, j] = _ll.copy()

        if positive:
            ll0 = self.eval_m(np.zeros_like(self.x))[0]
            ll[depths<0] = ll0

        self.ll = ll
        self.z = depths
        self.vz = vars

        return ll, depths, vars

    def models_m(self, m):
        _, w, v = self.eval_m(m)
        mean = w[0:-1]@self.X
        astro = m*w[-1]
        _, cond = self.gp.condition(self.y-mean-astro)
        noise = cond.mean

        return mean, astro, noise
    
    def models(self, t0, D):
        m = transit(self.x, t0, D, 1)
        return self.models_m(m)

    def solve(self, t0, D):
        m = transit(self.x, t0, D, 1)
        _, w, v = self.eval_m(m)
        return w, v
    
    def depth(self, t0, D):
        m = transit(self.x, t0, D, 1)
        _, w, v = self.eval_m(m)
        return w[-1], v[-1, -1]

    def _f_fold(self):
        t0s, Ds = self.t0s, self.Ds
        f_ll = interp2d(Ds, t0s, self.ll)
        f_z = interp2d(Ds, t0s, self.z)
        f_dz2 = interp2d(Ds, t0s, self.vz)
        ll0 = float(self.eval_m(np.zeros_like(self.x))[0]) # from DeviceArray

        # TODO: Investigate why JAX is slowing everything down here
        #@jax.jit
        def log_gauss_product_integral(a, va, b, vb):
            return -0.5 * np.square(a-b)/(va +vb) - 0.5*np.log(va+vb) - 0.5*np.log(np.pi) - np.log(2)/2

        #@jax.jit
        def A(z, vz):
            vZ = 1/np.sum(1/vz, 0)
            Z = vZ * np.sum(z/vz, 0)
            return np.sum(log_gauss_product_integral(z, vz, np.expand_dims(Z, 0), np.expand_dims(vZ, 0)), 0)

        #@jax.jit
        def LL(ll, z, vz):
            s1 = np.sum(ll, 0)
            s2 = np.sum(A(z, vz), 0)
            S = s1 + np.expand_dims(s2, 0)

            return S

        def interpolate_all(pt0s):
            # computing the likelihood folds by interpolating in phase
            folds_ll = np.array([f_ll(Ds, t) for t in pt0s])
            folds_z = np.array([f_z(Ds, t) for t in pt0s])
            folds_vz = np.array([f_dz2(Ds, t) for t in pt0s])

            return folds_ll, folds_z, folds_vz
        
        def fold(p):
            pt0s = interp_split_times(self.x, p)
            n = len(pt0s)
            folds_ll, folds_z, folds_vz = interpolate_all(pt0s)
            lc = np.sum(folds_ll, 0) - n*ll0
            lv = LL(folds_ll, folds_z, folds_vz)
            
            return pt0s[0]/p, lc, lv
            
        return fold

    def periodic_search(self, periods):

        self.periods = periods
        
        ll0, _, _ = self.eval_m(np.zeros_like(self.x))
        llv = np.zeros((len(periods), len(self.Ds)))
        llc = llv.copy()
        ll0 = float(ll0) # from DeviceArray
        fold = self._f_fold()

        for i, p in enumerate(tqdm(periods)):
            _, lc, lv = fold(p)
            llc[i] = np.max(lc, 0)
            llv[i] = np.max(lv - np.min(lv, 0), 0)

        return llc, llv

    def best_periodic_transit(self, period):
        fold = self._f_fold()
        phase, lv, _ = fold(period)
        i, j = np.unravel_index(np.argmax(lv), lv.shape)
        t0 = phase[i]*period
        D = self.Ds[j]
        return t0, D
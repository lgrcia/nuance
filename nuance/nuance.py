import jax
from tinygp import kernels, GaussianProcess
import numpy as np
import jax.numpy as jnp
from . import utils
from tqdm.autonotebook import tqdm
from .search_data import SearchData

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

    @property
    def ll0(self) -> float:
        return self.eval_m(np.zeros_like(self.x))[0].__array__()
        
    def linear_search(self, t0s, Ds, positive=True, progress=True):
        ll = np.zeros((len(t0s), len(Ds)))
        depths = np.zeros((len(t0s), len(Ds)))
        vars = ll.copy()
        depths = ll.copy()
        n = len(self.X)

        _progress = lambda x: tqdm(x) if progress else x

        for i, t0 in enumerate(_progress(t0s)):
            for j, D in enumerate(Ds):
                m = utils.single_transit(self.x, t0, D, 1)
                _ll, w, v = self.eval_m(m)
                depths[i, j] = w[n]
                vars[i, j] = v[n, n]
                ll[i, j] = _ll.copy()

        if positive:
            ll0 = self.eval_m(np.zeros_like(self.x))[0]
            ll[depths<0] = ll0

        return SearchData(t0s=t0s, Ds=Ds, ll=ll, z=depths, vz=vars, ll0=self.ll0)

    def periodic_search(self, search_data : SearchData, periods, progress=True):
        ll0, _, _ = self.eval_m(np.zeros_like(self.x))
        llv = np.zeros((len(periods), search_data.shape[1]))
        llc = llv.copy()
        ll0 = float(ll0) # from DeviceArray
        fold = search_data.fold

        _progress = lambda x: tqdm(x) if progress else x

        for i, p in enumerate(_progress(periods)):
            _, lc, lv = fold(p)
            llc[i] = np.max(lc, 0)
            llv[i] = np.max(lv - np.mean(lv, 0), 0)
        
        search_data.periods = periods
        search_data.llc = llc
        search_data.llv = llv

    def _models(self, m):
        _, w, _ = self.eval_m(m)
        mean = w[0:-1]@self.X
        signal = m*w[-1]
        _, cond = self.gp.condition(self.y-mean-signal)
        noise = cond.mean

        return mean, signal, noise
    
    def models(self, t0, D, P=None):
        m = utils.transit(self.x, t0, D, 1, P=P)
        return self._models(m)

    def solve(self, t0, D, P=None):
        m = utils.transit(self.x, t0, D, 1, P=P)
        _, w, v = self.eval_m(m)
        return w, v
    
    def depth(self, t0, D, P=None):
        w, v = self.solve(t0, D, P)
        return w[-1], np.sqrt(v[-1, -1])

    def snr(self, t0, D, P=None):
        w, dw = self.depth(t0, D, P=P)
        return w/dw
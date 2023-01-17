from scipy.interpolate import interp2d
import numpy as np
from dataclasses import dataclass, replace
from .utils import interp_split_times, phase
import matplotlib.pyplot as plt
import copy

@dataclass
class SearchData:
    t0s: np.ndarray
    Ds: np.ndarray
    ll: np.ndarray = None
    z: np.ndarray = None
    vz: np.ndarray = None
    ll0: float = None
    llv: np.ndarray = None
    llp: np.ndarray = None
    periods: np.ndarray = None
    snr: np.ndarray = None

    @property
    def folds(self):

        f_ll = interp2d(self.Ds, self.t0s, self.ll)
        f_z = interp2d(self.Ds, self.t0s, self.z)
        f_dz2 = interp2d(self.Ds, self.t0s, self.vz)
        
        def new_interpolate_all(pt0s):
            # computing the likelihood folds by interpolating in phase
            folds_ll = np.array([f_ll(self.Ds, t) for t in pt0s])
            # folds_z = np.array([f_z(self.Ds, t) for t in pt0s])
            # folds_vz = np.array([f_dz2(self.Ds, t) for t in pt0s])

            #return folds_ll, folds_z, folds_vz
            return folds_ll, 0, 0

        def interpolate_all(pt0s):
            # computing the likelihood folds by interpolating in phase
            folds_ll = np.array([f_ll(self.Ds, t) for t in pt0s])
            folds_z = np.array([f_z(self.Ds, t) for t in pt0s])
            folds_vz = np.array([f_dz2(self.Ds, t) for t in pt0s])

            return folds_ll, folds_z, folds_vz
        
        def _folds(p):
            pt0s = interp_split_times(self.t0s, p)
            return pt0s, new_interpolate_all(pt0s)

        return _folds

    @property
    def fold(self):
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

        folds = self.folds
        
        def _fold(p):
            pt0s, (folds_ll, folds_z, folds_vz) = folds(p)
            n = len(pt0s)
            lc = np.sum(folds_ll, 0) - n*self.ll0
            #lv = LL(folds_ll, folds_z, folds_vz)
            
            return pt0s[0]/p, lc, lc
            
        return _fold

    @property
    def _best(self):
        if self.periods is not None:
            i, j = np.unravel_index(np.argmax(self.llv), self.llv.shape)
            period = self.periods[i]
            fold = self.fold
            phase, _, fll = fold(period)
            i, j = np.unravel_index(np.argmax(fll), fll.shape)
            t0 = phase[i]*period
            D = self.Ds[j]
        else:
            i, j = np.unravel_index(np.argmax(self.ll), self.ll.shape)
            t0, D = self.t0s[i], self.Ds[j]
            period = None
        return t0, D, period

    @property
    def best(self):
        if self.periods is not None:
            i = np.argmax(self.snr[:, 0])
            return self.snr[i, 1::]
        else:
            i, j = np.unravel_index(np.argmax(self.ll), self.ll.shape)
            t0, D = self.t0s[i], self.Ds[j]
            period = None
        return t0, D, period

    @property
    def shape(self):
        return len(self.t0s), len(self.Ds)

    def show_ll(self, **kwargs):
        extent = np.min(self.t0s), np.max(self.t0s), np.min(self.Ds), np.max(self.Ds)
        plt.imshow(self.ll.T, aspect='auto', origin="lower", extent=extent, **kwargs)

    def show_llv(self, **kwargs):
        extent = np.min(self.periods), np.max(self.periods), np.min(self.Ds), np.max(self.Ds)
        plt.imshow(self.llv.T, aspect='auto', origin="lower", extent=extent, **kwargs)
        
    def _periodogram(self, D=None):
        assert self.periods is not None
        if D is None:
            _, D, _ = self.best
        
        if isinstance(D, int):
            i = D
        else:
            i = np.flatnonzero(self.Ds == D)[0]
        
        return self.periods, self.llv.T[i]

    
    def periodogram(self, D=None):
        return self.periods, self.snr[:, 0]

    def copy(self):
        return copy.deepcopy(self)

    def mask(self, t0, D, P):
        new_search_data = self.copy()
        new_search_data.llv = None
        new_search_data.llc = None
        new_search_data.periods = None

        ph = phase(self.t0s, t0, P)
        mask = np.abs(ph) > 2*D
        new_search_data.t0s = new_search_data.t0s[mask]
        new_search_data.ll = new_search_data.ll[mask]
        new_search_data.z = new_search_data.z[mask]
        new_search_data.vz = new_search_data.vz[mask]

        return new_search_data
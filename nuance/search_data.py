from scipy.interpolate import interp2d
import numpy as np
from dataclasses import dataclass
from . import utils
import matplotlib.pyplot as plt
import copy

@dataclass
class SearchData:
    # linear search
    t0s: np.ndarray
    Ds: np.ndarray
    ll: np.ndarray = None
    z: np.ndarray = None
    vz: np.ndarray = None
    ll0: float = None

    # periodic search, Q is periodogram
    periods: np.ndarray = None
    Q_snr: np.ndarray = None
    Q_ll: np.ndarray = None
    Q_params: np.ndarray = None
    

    @property
    def folds(self):

        f_ll = interp2d(self.Ds, self.t0s, self.ll)
        f_z = interp2d(self.Ds, self.t0s, self.z)
        f_dz2 = interp2d(self.Ds, self.t0s, self.vz)
        
        def interpolate_all(pt0s):
            # computing the likelihood folds by interpolating in phase
            folds_ll = np.array([f_ll(self.Ds, t) for t in pt0s])
            folds_z = np.array([f_z(self.Ds, t) for t in pt0s])
            folds_vz = np.array([f_dz2(self.Ds, t) for t in pt0s])

            return folds_ll, folds_z, folds_vz
        
        def _folds(p):
            pt0s = utils.interp_split_times(self.t0s, p)
            return pt0s, interpolate_all(pt0s)

        return _folds

    @property
    def fold_ll(self):
        folds = self.folds
        
        def _fold(p):
            pt0s, (lls, zs, vzs) = folds(p)
            P1, P2 = utils.Ps(lls, zs, vzs)   
            
            return pt0s[0]/p, P1, P2
            
        return _fold

    @property
    def best(self):
        if self.periods is not None:
            i = np.argmax(self.Q_snr)
            return self.Q_params[i]
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
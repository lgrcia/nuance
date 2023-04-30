from dataclasses import dataclass

import astropy.units as u
import numpy as np

G = 2942.2062175044193
# R_sun^3 . M_sun^-1 . day^-2
# astropy: c.G.to(u.R_sun**3/u.M_sun/(u.day**2)).value

R_EARTH = 0.009167888457668534
# R_sun
# astropy: c.R_earth.to(u.R_sun).value


@dataclass
class Star:
    """A class to hold stellar parameters and variability characteristics."""

    radius: float = 1.0
    """Stellar radius in solar radius"""
    mass: float = 1.0
    """Steallar mass in solar mass"""
    amplitude: float = 1.0  # peak to peak
    """Stellat variability amplitude (peak to peak)"""
    period: float = 1.0
    """Stellar variability period in days"""

    def __post_init__(self):
        pass

    @property
    def omega(self) -> float:
        """Stellar variability angular frequency.

        Returns
        -------
        float
            Angular frequency in rad.s-1
        """
        return 2 * np.pi / self.period

    def transit_duration(self, orbital_period: float) -> float:
        """Transit duration from orbital period.

        Parameters
        ----------
        orbital_period : float
            planet orbital period in days

        Returns
        -------
        float
            duration in days
        """
        a = (G * self.mass * (orbital_period**2) / (4 * (np.pi**2))) ** (1 / 3)
        return (orbital_period * self.radius) / (np.pi * a)

    def transit_depth(self, radius: float) -> float:
        """Transit depth from planet's radius.

        Parameters
        ----------
        radius : float
            planet's radius in Earth radius

        Returns
        -------
        float
            transit depth
        """
        _radius = radius * R_EARTH
        return (_radius / self.radius) ** 2

    # period - tau

    def period2tau(self, orbital_period: float) -> float:
        """Relative duration given orbital period.

        Parameters
        ----------
        orbital_period : float
            planet orbital period in days

        Returns
        -------
        float
            relative duration tau
        """
        duration = self.transit_duration(orbital_period)
        return np.pi / (self.omega * duration)

    def tau2period(self, tau: float) -> float:
        """Period given relative duration.

        Parameters
        ----------
        tau : float
            relative duration

        Returns
        -------
        float
            orbital period in days
        """
        duration = np.pi / (self.omega * tau)
        a = (G * self.mass * duration**2) / (4 * (self.radius**2))
        return 2 * np.pi * a ** (3 / 2) / np.sqrt(G * self.mass)

    # radius - delta

    def radius2delta(self, radius: float) -> float:
        """Relative amplitude given planet's radius

        Parameters
        ----------
        radius : float
            planet's radius in Earth radius

        Returns
        -------
        float
            relative amplitude
        """
        depth = self.transit_depth(radius)
        return self.amplitude / depth

    def delta2radius(self, delta: float) -> float:
        """Planet's radius from relative amplitude

        Parameters
        ----------
        delta : float
            relative amplitude

        Returns
        -------
        float
            planet's radius in Earth radius
        """
        depth = self.delta2depth(delta)
        return np.sqrt(depth) * self.radius / R_EARTH

    def min_radius(self, period: float, SNR: float, N: int, sigma: float) -> float:
        """Given a target SNR and periods, returns the minimum planetary radius

        Parameters
        ----------
        period : float
            planet orbital period in days
        SNR : float
            signal-to-noise-ratio
        N : int
            number of points observed
        sigma : float
            observation error

        Returns
        -------
        float
            radius is earth radius
        """
        D = self.transit_duration(period)
        n = D * N / period
        return (np.sqrt(SNR * sigma) * self.radius / n ** (1 / 4)) / R_EARTH

    def snr(self, orbital_period: float, radius: float, N: int, sigma: float) -> float:
        """Transit signal-to-noise-ratio

        Parameters
        ----------
        orbital_period : float
            planet orbital period in days
        radius : float
            planet's radius in Earth radius
        N : int
            number of observation points
        sigma : float
            observation error

        Returns
        -------
        float
            signal-to-noise
        """
        depth = self.transit_depth(radius)
        duration = self.transit_duration(orbital_period)
        n_tr = N * duration / orbital_period  # points in transit
        return (depth / sigma) * np.sqrt(n_tr)

import numpy as np
from astropy import constants as c
from astropy import units as u

from nuance.star import Star

R = 1.0 * u.R_sun
M = 1.0 * u.M_sun
star = Star(R.value, M.value, 0, 0)


def test_duration():
    period = 3.0 * u.day

    a = (c.G * M * (period**2) / (4 * (np.pi**2))) ** (1 / 3)
    duration = (period * R) / (np.pi * a)
    expected = duration.to(u.day).value

    star = Star(R.value, M.value, 0, 0)
    computed = star.transit_duration(period.value)

    np.testing.assert_allclose(expected, computed)


def test_depth():
    radius = 1.0 * u.R_earth
    expected = ((radius / R) ** 2).decompose().value
    computed = star.transit_depth(radius.value)
    np.testing.assert_allclose(expected, computed)

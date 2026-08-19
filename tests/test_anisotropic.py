"""T4 anisotropic (wind-driven) spread tests."""
from __future__ import annotations

import numpy as np

from vhagar.models.spread import (
    anisotropic_arrival,
    eccentricity_from_lb,
    front_length_breadth,
    length_to_breadth,
)


def test_length_to_breadth_and_eccentricity():
    assert length_to_breadth(0.0) == 1.0
    assert length_to_breadth(1.0, lb_max=4.0) == 4.0
    assert length_to_breadth(0.5) > length_to_breadth(0.2)   # monotone in wind
    assert abs(eccentricity_from_lb(1.0)) < 1e-9              # circle has e = 0
    assert 0 < eccentricity_from_lb(4.0) < 1


def _grow(wind_speed, grid=121, q=0.06):
    c = grid // 2
    seed = np.zeros((grid, grid), dtype=bool)
    seed[c, c] = True
    T = anisotropic_arrival(np.ones((grid, grid)), wind_speed=wind_speed, wind_dir=0.0, seeds=seed)
    m = float(np.quantile(T[np.isfinite(T)], q)) >= T
    return m, c


def test_zero_wind_is_a_circle():
    m, c = _grow(0.0)
    assert front_length_breadth(m) < 1.15                    # ~circular
    xs = np.where(m)[1]
    assert abs((xs.max() - c) - (c - xs.min())) <= 1         # symmetric up/downwind


def test_wind_elongates_downwind():
    m, c = _grow(0.6)
    ys, xs = np.where(m)
    downwind, upwind = xs.max() - c, c - xs.min()
    assert downwind > 3 * upwind                             # head far outruns the back
    lb = front_length_breadth(m)
    assert lb > 2.0                                          # clearly elongated
    assert lb > front_length_breadth(_grow(0.2)[0])          # more wind, more elongation
    # tracks the prescribed length-to-breadth within tolerance
    assert abs(lb - float(length_to_breadth(0.6))) < 1.0

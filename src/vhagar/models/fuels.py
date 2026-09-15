"""LANDFIRE Scott & Burgan (FBFM40) fuel models -> a relative rate-of-spread factor.

The WUI faithful calibration (docs/17) drives the front with a *uniform* rate of
spread, so its main error is over-prediction (false alarms): the wind-driven front
ignites structures in places the real fire could not carry, across roads, irrigated
land, water, and other non-fuel gaps. Real fuels fix exactly that: fire does not
spread through non-burnable cells, so a fuel-aware ROS field stops the wildland front
at fuel breaks and lets structure-to-structure + spotting carry it into the built
area, which is what cuts the false-alarm rate.

This module maps the **FBFM40** (Scott & Burgan 2005) fuel-model codes that LANDFIRE
distributes onto a relative spread factor in [0, 1], to multiply a base ROS:

* **non-burnable** (NB1 urban 91, NB2 snow 92, NB3 agriculture 93, NB8 water 98,
  NB9 barren 99) -> **0** (the front cannot cross);
* grass (GR, 101-109) fastest, grass-shrub (GS) and slash (SB) high, shrub (SH)
  moderate-high, timber-understorey (TU) moderate, timber-litter (TL) slow.

These are documented *relative* surrogates (true ROS depends on wind, slope, and fuel
moisture via Rothermel; this is the fuel term only), chosen so the front respects fuel
continuity. Note the WUI subtlety: urban (NB1) is non-burnable to the *wildland* front
on purpose, intra-town spread is the structure-to-structure model's job, so the front
reaches the town edge and the graph carries it in.
"""

from __future__ import annotations

import numpy as np

__all__ = ["FBFM40_NONBURNABLE", "fbfm40_spread_factor", "ros_from_fuel_codes"]

#: FBFM40 non-burnable codes (NB1-NB3, NB8, NB9): no wildland spread.
FBFM40_NONBURNABLE = frozenset({91, 92, 93, 98, 99})

#: Relative spread factor by FBFM40 fuel-type band (leading code range).
_BANDS = (
    (101, 109, 1.00),   # GR  grass
    (121, 124, 0.80),   # GS  grass-shrub
    (141, 149, 0.70),   # SH  shrub
    (161, 165, 0.55),   # TU  timber-understorey
    (181, 189, 0.35),   # TL  timber-litter
    (201, 204, 0.85),   # SB  slash-blowdown
)


def fbfm40_spread_factor(codes, default: float = 0.4) -> np.ndarray:
    """Relative spread factor in [0, 1] for FBFM40 fuel-model codes.

    Non-burnable codes map to 0; each burnable band to a documented relative factor;
    anything unrecognised to ``default``. Vectorised over an array of integer codes.

    >>> import numpy as np
    >>> f = fbfm40_spread_factor(np.array([101, 121, 165, 91, 99, 183]))
    >>> [round(float(v), 2) for v in f]
    [1.0, 0.8, 0.55, 0.0, 0.0, 0.35]
    """
    codes = np.asarray(codes)
    out = np.full(codes.shape, float(default), dtype=np.float64)
    for lo, hi, fac in _BANDS:
        out[(codes >= lo) & (codes <= hi)] = fac
    for nb in FBFM40_NONBURNABLE:
        out[codes == nb] = 0.0
    return out


def ros_from_fuel_codes(codes, base: float = 0.2, floor: float = 1e-3) -> np.ndarray:
    """Rate-of-spread field from an FBFM40 fuel-code grid.

    ``ROS = base * spread_factor(codes)``, floored at ``floor`` on burnable cells so the
    solver always advances, and hard 0 on non-burnable cells so the front stops at fuel
    breaks. Drop this in for the uniform ROS in
    :func:`vhagar.eval.wui_calibrate.fire_from_points` (pass ``fuel_codes``) to make the
    front fuel-aware.

    >>> import numpy as np
    >>> ros_from_fuel_codes(np.array([101, 91]), base=1.0).tolist()
    [1.0, 0.0]
    """
    codes = np.asarray(codes)
    fac = fbfm40_spread_factor(codes)
    ros = base * fac
    burnable = fac > 0
    return np.where(burnable, np.maximum(ros, floor), 0.0)

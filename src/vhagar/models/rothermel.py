"""Rothermel (1972) surface fire rate-of-spread, the physics-standard ROS field.

VHAGAR's T4 front has until now used a fuel-and-wind *surrogate* rate of spread
(``models.fuels``). This module replaces that surrogate, when you want it, with the
canonical Rothermel surface fire spread model, the same equation that underlies
BehavePlus, FlamMap, and FARSITE. It computes a head rate of spread from fuel-model
parameters, dead-fuel moisture, midflame wind, and slope, in real units (m/min).

Reference: Rothermel (1972); Andrews (2018), "The Rothermel surface fire spread model
and associated developments" (USDA RMRS-GTR-371), whose notation this follows. The
implementation is the standard **single characteristic surface-area-to-volume**
formulation (one weighted dead-fuel class). What it does not yet do is the full
multi-size-class dead-and-live weighting of the 40 Scott and Burgan models; that is the
remaining refinement. So this closes the physics ROS core honestly, and the per-model
fuel parameters here are representative per fuel-type group, documented as such.

Wind is the **midflame** wind. Open (10 m) winds must be reduced by a midflame wind
adjustment factor before use; ``wind_adj`` does that for the field helper.

Coupling to T4: Rothermel gives the head (maximum, downwind) ROS *magnitude* including
wind and slope. The anisotropic arrival-time solver then distributes that magnitude
directionally through the Richards ellipse. So Rothermel sets the rate, the ellipse sets
the shape, and wind is not double-counted (the ellipse uses wind only for eccentricity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vhagar.models.fuels import FBFM40_NONBURNABLE

__all__ = ["FuelParams", "STANDARD_FUELS", "rothermel_ros", "params_for_codes",
           "rothermel_head_ros_field"]

_MS_TO_FTMIN = 196.850394        # m/s -> ft/min
_FTMIN_TO_MMIN = 0.3048          # ft/min -> m/min
_RHO_P = 32.0                    # oven-dry particle density, lb/ft^3
_S_T = 0.0555                    # total mineral content
_S_E = 0.010                     # effective mineral content
_HEAT = 8000.0                   # low heat content, Btu/lb


@dataclass(frozen=True)
class FuelParams:
    """One fuel model's Rothermel inputs (single characteristic dead-fuel class).

    w_o oven-dry fuel load (lb/ft^2); depth fuel bed depth (ft); sav characteristic
    surface-area-to-volume ratio (1/ft); m_x dead-fuel moisture of extinction (fraction).
    """

    w_o: float
    depth: float
    sav: float
    m_x: float


#: Representative Rothermel parameters per FBFM40 fuel-type group (documented surrogates
#: for the full per-model tables; grass fast and flashy, timber litter slow).
STANDARD_FUELS: dict[str, FuelParams] = {
    "GR": FuelParams(0.10, 1.0, 2000.0, 0.15),   # grass
    "GS": FuelParams(0.15, 1.5, 1800.0, 0.20),   # grass-shrub
    "SH": FuelParams(0.30, 2.5, 1600.0, 0.30),   # shrub
    "TU": FuelParams(0.35, 1.0, 1600.0, 0.25),   # timber-understory
    "TL": FuelParams(0.50, 0.3, 1800.0, 0.30),   # timber-litter
    "SB": FuelParams(0.60, 1.5, 1200.0, 0.25),   # slash-blowdown
}

#: FBFM40 leading-code ranges -> fuel-type group (matches models.fuels bands).
_BANDS = ((101, 109, "GR"), (121, 124, "GS"), (141, 149, "SH"),
          (161, 165, "TU"), (181, 189, "TL"), (201, 204, "SB"))


def _ros_core(w_o, depth, sav, m_x, m_f, wind_ms, slope_tan):
    """Rothermel head ROS in m/min for per-cell (broadcastable) parameter arrays."""
    m_f = np.asarray(m_f, dtype=np.float64)
    U = np.maximum(np.asarray(wind_ms, dtype=np.float64) * _MS_TO_FTMIN, 0.0)
    slope = np.maximum(np.asarray(slope_tan, dtype=np.float64), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_b = w_o / depth
        beta = rho_b / _RHO_P
        beta_op = 3.348 * sav ** -0.8189
        ratio = beta / beta_op
        a = 133.0 * sav ** -0.7913
        gamma_max = sav ** 1.5 / (495.0 + 0.0594 * sav ** 1.5)
        gamma = gamma_max * ratio ** a * np.exp(a * (1.0 - ratio))
        w_n = w_o * (1.0 - _S_T)
        rm = np.clip(m_f / m_x, 0.0, 1.0)
        eta_M = np.clip(1.0 - 2.59 * rm + 5.11 * rm ** 2 - 3.52 * rm ** 3, 0.0, 1.0)
        eta_s = min(1.0, 0.174 * _S_E ** -0.19)
        I_R = gamma * w_n * _HEAT * eta_M * eta_s
        xi = np.exp((0.792 + 0.681 * sav ** 0.5) * (beta + 0.1)) / (192.0 + 0.2595 * sav)
        c = 7.47 * np.exp(-0.133 * sav ** 0.55)
        b = 0.02526 * sav ** 0.54
        e = 0.715 * np.exp(-3.59e-4 * sav)
        phi_w = np.where(U > 0.0, c * U ** b * ratio ** -e, 0.0)
        phi_s = 5.275 * beta ** -0.3 * slope ** 2
        eps = np.exp(-138.0 / sav)
        Q_ig = 250.0 + 1116.0 * m_f
        R = I_R * xi * (1.0 + phi_w + phi_s) / (rho_b * eps * Q_ig) * _FTMIN_TO_MMIN
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    return np.where(np.asarray(m_f) >= m_x, 0.0, np.maximum(R, 0.0))


def rothermel_ros(fp: FuelParams, m_f, wind_ms, slope_tan=0.0) -> np.ndarray:
    """Head rate of spread (m/min) for one fuel model. Arrays broadcast.

    ``m_f`` dead-fuel moisture (fraction), ``wind_ms`` midflame wind (m/s), ``slope_tan``
    rise over run. Returns 0 at or above the moisture of extinction.

    >>> fp = STANDARD_FUELS["GR"]
    >>> bool(float(rothermel_ros(fp, 0.06, 5.0)) > float(rothermel_ros(fp, 0.06, 0.0)) > 0)
    True
    >>> float(rothermel_ros(fp, 0.20, 5.0))
    0.0
    """
    return _ros_core(fp.w_o, fp.depth, fp.sav, fp.m_x, m_f, wind_ms, slope_tan)


def params_for_codes(codes):
    """FBFM40 code array -> per-cell (w_o, depth, sav, m_x, burnable) arrays."""
    codes = np.asarray(codes)
    w_o = np.zeros(codes.shape)
    depth = np.ones(codes.shape)
    sav = np.full(codes.shape, 1600.0)
    m_x = np.full(codes.shape, 0.25)
    burn = np.zeros(codes.shape, bool)
    known = np.zeros(codes.shape, bool)
    for lo, hi, g in _BANDS:
        fp = STANDARD_FUELS[g]
        m = (codes >= lo) & (codes <= hi)
        w_o[m], depth[m], sav[m], m_x[m], burn[m] = fp.w_o, fp.depth, fp.sav, fp.m_x, True
        known |= m
    nb = np.isin(codes, list(FBFM40_NONBURNABLE))
    other = (~known) & (~nb)                          # unrecognised burnable -> TU default
    fp = STANDARD_FUELS["TU"]
    w_o[other], depth[other], sav[other], m_x[other], burn[other] = fp.w_o, fp.depth, fp.sav, fp.m_x, True
    return w_o, depth, sav, m_x, burn


def rothermel_head_ros_field(codes, m_f=0.08, wind_ms=0.0, slope_tan=0.0,
                             wind_adj=1.0) -> np.ndarray:
    """Head ROS field (m/min) over an FBFM40 code grid; non-burnable cells are 0.

    The physics drop-in for ``fuels.ros_from_fuel_codes``: feed the returned field to
    ``spread.anisotropic_arrival`` as ``head_ros`` and pass the same wind to the solver
    for the ellipse shape. ``wind_adj`` scales an open (10 m) wind down to midflame
    (a typical factor is 0.1 to 0.4); leave it 1.0 if ``wind_ms`` is already midflame.
    """
    w_o, depth, sav, m_x, burn = params_for_codes(np.asarray(codes))
    wind = np.broadcast_to(np.asarray(wind_ms, dtype=np.float64) * wind_adj, np.asarray(codes).shape)
    R = _ros_core(w_o, depth, sav, m_x, m_f, wind, slope_tan)
    return np.where(burn, R, 0.0)

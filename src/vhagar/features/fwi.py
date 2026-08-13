"""Canadian Forest Fire Weather Index (FWI) System. Van Wagner & Pickett 1985/1987.

A vectorised NumPy implementation of the classic (1987) FWI System, suitable
for gridded daily fields. Inputs are noon local standard time observations,
which is what the system was calibrated against.

    temp   : 2 m air temperature            [degC]
    rh     : relative humidity              [%]  (clipped to <= 100)
    wind   : 10 m wind speed                [km/h]
    rain   : 24 h accumulated precipitation [mm]

Outputs: FFMC, DMC, DC (moisture codes, carry state day to day) and
ISI, BUI, FWI, DSR (diagnostic, computed from the codes).

Why this module exists even though FWI2025 is out
-------------------------------------------------
VHAGAR runs FWI1987 **and** FWI2025 in parallel. FWI1987 is the index every
existing climatology, threshold and operational habit is calibrated against;
it is the continuity baseline and it must be bit-reproducible. FWI2025 (hourly
codes, reformulated ISI, grassland components) is ingested from the NRCan
reference implementation rather than reimplemented here.

.. warning::
   Do **not** feed FWI2025 outputs into the FBP1992 rate-of-spread equations.
   NRCan's reference repository carries a February 2026 notice that the two
   are not yet compatible; an interim ``iFBP2025`` module is in development.

.. note::
   Average **DSR**, never raw FWI. FWI is not a linear quantity and its
   arithmetic mean over time or space is not meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "FWIState",
    "bui",
    "day_length_dc",
    "day_length_dmc",
    "dc",
    "dmc",
    "dsr",
    "ffmc",
    "fwi",
    "fwi_system",
    "isi",
]

# Default starting values used at the start of a fire season (Van Wagner 1987).
DEFAULT_FFMC = 85.0
DEFAULT_DMC = 6.0
DEFAULT_DC = 15.0

# Effective day-length factors, DMC (Le), months Jan..Dec, northern mid-latitudes.
_DMC_DAY_LENGTH_NORTH = np.array(
    [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
)
# Southern hemisphere: the same curve, phase-shifted six months.
_DMC_DAY_LENGTH_SOUTH = np.roll(_DMC_DAY_LENGTH_NORTH, 6)
# Equatorial (|lat| <= 10): near-constant.
_DMC_DAY_LENGTH_EQUATORIAL = np.full(12, 9.0)

# Day-length adjustment, DC (Lf), months Jan..Dec.
_DC_DAY_LENGTH_NORTH = np.array(
    [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
)
_DC_DAY_LENGTH_SOUTH = np.roll(_DC_DAY_LENGTH_NORTH, 6)
_DC_DAY_LENGTH_EQUATORIAL = np.full(12, 1.4)


def day_length_dmc(month: int, lat: float = 46.0) -> float:
    """DMC effective day length ``Le`` for a month (1-12) and latitude."""
    i = int(month) - 1
    if abs(lat) <= 10.0:
        return float(_DMC_DAY_LENGTH_EQUATORIAL[i])
    table = _DMC_DAY_LENGTH_NORTH if lat > 0 else _DMC_DAY_LENGTH_SOUTH
    return float(table[i])


def day_length_dc(month: int, lat: float = 46.0) -> float:
    """DC day-length adjustment ``Lf`` for a month (1-12) and latitude."""
    i = int(month) - 1
    if abs(lat) <= 10.0:
        return float(_DC_DAY_LENGTH_EQUATORIAL[i])
    table = _DC_DAY_LENGTH_NORTH if lat > 0 else _DC_DAY_LENGTH_SOUTH
    return float(table[i])


def _asarray(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def ffmc(temp, rh, wind, rain, ffmc_prev) -> np.ndarray:
    """Fine Fuel Moisture Code. Litter and cured fine fuels, ~16 h time lag."""
    t = _asarray(temp)
    h = np.clip(_asarray(rh), 0.0, 100.0)
    w = np.maximum(_asarray(wind), 0.0)
    r = np.maximum(_asarray(rain), 0.0)
    f0 = _asarray(ffmc_prev)

    # Code -> moisture content (%)
    mo = 147.2 * (101.0 - f0) / (59.5 + f0)

    # --- rainfall phase ---------------------------------------------------
    wet = r > 0.5
    rf = np.where(wet, r - 0.5, 0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        base = np.where(
            rf > 0,
            42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / np.where(rf > 0, rf, 1.0))),
            0.0,
        )
        extra = np.where(mo > 150.0, 0.0015 * (mo - 150.0) ** 2 * np.sqrt(rf), 0.0)
    mo = np.where(wet, np.minimum(mo + base + extra, 250.0), mo)

    # --- drying / wetting equilibrium ------------------------------------
    ed = (
        0.942 * h**0.679
        + 11.0 * np.exp((h - 100.0) / 10.0)
        + 0.18 * (21.1 - t) * (1.0 - np.exp(-0.115 * h))
    )
    ew = (
        0.618 * h**0.753
        + 10.0 * np.exp((h - 100.0) / 10.0)
        + 0.18 * (21.1 - t) * (1.0 - np.exp(-0.115 * h))
    )

    # Drying (mo > Ed)
    ko = 0.424 * (1.0 - (h / 100.0) ** 1.7) + 0.0694 * np.sqrt(w) * (1.0 - (h / 100.0) ** 8)
    kd = ko * 0.581 * np.exp(0.0365 * t)
    m_dry = ed + (mo - ed) * 10.0 ** (-kd)

    # Wetting (mo < Ew)
    kl = 0.424 * (1.0 - ((100.0 - h) / 100.0) ** 1.7) + 0.0694 * np.sqrt(w) * (
        1.0 - ((100.0 - h) / 100.0) ** 8
    )
    kw = kl * 0.581 * np.exp(0.0365 * t)
    m_wet = ew - (ew - mo) * 10.0 ** (-kw)

    m = np.where(mo > ed, m_dry, np.where(mo < ew, m_wet, mo))
    m = np.clip(m, 0.0, 250.0)

    out = 59.5 * (250.0 - m) / (147.2 + m)
    return np.clip(out, 0.0, 101.0)


def dmc(temp, rh, rain, dmc_prev, month: int, lat: float = 46.0) -> np.ndarray:
    """Duff Moisture Code. Loosely compacted organic layers, ~12 day time lag."""
    t = np.maximum(_asarray(temp), -1.1)
    h = np.clip(_asarray(rh), 0.0, 100.0)
    r = np.maximum(_asarray(rain), 0.0)
    p0 = np.maximum(_asarray(dmc_prev), 0.0)
    le = day_length_dmc(month, lat)

    # --- rainfall phase ---------------------------------------------------
    wet = r > 1.5
    rw = 0.92 * r - 1.27
    wmi = 20.0 + 280.0 / np.exp(0.023 * p0)
    b = np.where(
        p0 <= 33.0,
        100.0 / (0.5 + 0.3 * p0),
        np.where(p0 <= 65.0, 14.0 - 1.3 * np.log(np.maximum(p0, 1e-6)), 6.2 * np.log(np.maximum(p0, 1e-6)) - 17.2),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        wmr = wmi + 1000.0 * rw / (48.77 + b * rw)
        pr = 43.43 * (5.6348 - np.log(np.maximum(wmr - 20.0, 1e-6)))
    pr = np.maximum(np.where(wet, pr, p0), 0.0)

    # --- drying phase -----------------------------------------------------
    rk = 1.894 * (t + 1.1) * (100.0 - h) * le * 1e-4
    return np.maximum(pr + rk, 0.0)


def dc(temp, rain, dc_prev, month: int, lat: float = 46.0) -> np.ndarray:
    """Drought Code. Deep compact organic layers, ~52 day time lag."""
    t = np.maximum(_asarray(temp), -2.8)
    r = np.maximum(_asarray(rain), 0.0)
    d0 = np.maximum(_asarray(dc_prev), 0.0)
    lf = day_length_dc(month, lat)

    # --- rainfall phase ---------------------------------------------------
    wet = r > 2.8
    rw = 0.83 * r - 1.27
    smi = 800.0 * np.exp(-d0 / 400.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dr = d0 - 400.0 * np.log(1.0 + 3.937 * rw / np.maximum(smi, 1e-9))
    dr = np.maximum(np.where(wet, dr, d0), 0.0)

    # --- drying phase -----------------------------------------------------
    pe = np.maximum((0.36 * (t + 2.8) + lf) / 2.0, 0.0)
    return np.maximum(dr + pe, 0.0)


def isi(ffmc_val, wind) -> np.ndarray:
    """Initial Spread Index. FFMC combined with wind."""
    f = _asarray(ffmc_val)
    w = np.maximum(_asarray(wind), 0.0)
    m = 147.2 * (101.0 - f) / (59.5 + f)
    f_wind = np.exp(0.05039 * w)
    f_fine = 91.9 * np.exp(-0.1386 * m) * (1.0 + m**5.31 / 4.93e7)
    return 0.208 * f_wind * f_fine


def bui(dmc_val, dc_val) -> np.ndarray:
    """Build Up Index, total fuel available to the spreading fire."""
    p = np.maximum(_asarray(dmc_val), 0.0)
    d = np.maximum(_asarray(dc_val), 0.0)
    denom = np.where((p + 0.4 * d) > 0, p + 0.4 * d, 1e-9)
    low = 0.8 * p * d / denom
    high = p - (1.0 - 0.8 * d / denom) * (0.92 + (0.0114 * p) ** 1.7)
    out = np.where(p <= 0.4 * d, low, high)
    return np.maximum(out, 0.0)


def fwi(isi_val, bui_val) -> np.ndarray:
    """Fire Weather Index, a proxy for frontal fire intensity."""
    i = np.maximum(_asarray(isi_val), 0.0)
    u = np.maximum(_asarray(bui_val), 0.0)
    f_bui = np.where(
        u <= 80.0,
        0.626 * np.maximum(u, 1e-9) ** 0.809 + 2.0,
        1000.0 / (25.0 + 108.64 * np.exp(-0.023 * u)),
    )
    b = 0.1 * i * f_bui
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(b > 1.0, np.exp(2.72 * (0.434 * np.log(np.maximum(b, 1e-9))) ** 0.647), b)
    return np.maximum(np.nan_to_num(s, nan=0.0), 0.0)


def dsr(fwi_val) -> np.ndarray:
    """Daily Severity Rating, ``0.0272 * FWI**1.77``.

    This is the quantity to average over time or space. Averaging raw FWI is a
    common and consequential error.
    """
    return 0.0272 * np.maximum(_asarray(fwi_val), 0.0) ** 1.77


@dataclass(slots=True)
class FWIState:
    """Carry-over moisture codes between days.

    Arrays may be scalars or grids; they must broadcast against the forcing.
    """

    ffmc: np.ndarray
    dmc: np.ndarray
    dc: np.ndarray

    @classmethod
    def season_start(cls, shape: tuple[int, ...] = ()) -> FWIState:
        """Van Wagner's standard fire-season startup values."""
        return cls(
            ffmc=np.full(shape, DEFAULT_FFMC),
            dmc=np.full(shape, DEFAULT_DMC),
            dc=np.full(shape, DEFAULT_DC),
        )


def fwi_system(
    temp,
    rh,
    wind,
    rain,
    state: FWIState,
    month: int,
    lat: float = 46.0,
) -> tuple[dict[str, np.ndarray], FWIState]:
    """Advance the FWI System one day.

    Returns ``(outputs, new_state)`` where ``outputs`` has keys
    ``ffmc dmc dc isi bui fwi dsr``.

    >>> import numpy as np
    >>> st = FWIState.season_start()
    >>> out, st = fwi_system(17.0, 42.0, 25.0, 0.0, st, month=4)
    >>> bool(out["fwi"] > 0)
    True
    """
    f = ffmc(temp, rh, wind, rain, state.ffmc)
    p = dmc(temp, rh, rain, state.dmc, month, lat)
    d = dc(temp, rain, state.dc, month, lat)
    i = isi(f, wind)
    u = bui(p, d)
    s = fwi(i, u)
    outputs = {
        "ffmc": f,
        "dmc": p,
        "dc": d,
        "isi": i,
        "bui": u,
        "fwi": s,
        "dsr": dsr(s),
    }
    return outputs, FWIState(ffmc=f, dmc=p, dc=d)


#: EFFIS European fire-danger classes (FWI thresholds).
EFFIS_DANGER_CLASSES: tuple[tuple[str, float], ...] = (
    ("very_low", 0.0),
    ("low", 5.2),
    ("moderate", 11.2),
    ("high", 21.3),
    ("very_high", 38.0),
    ("extreme", 50.0),
    ("very_extreme", 70.0),
)


def effis_class(fwi_val) -> np.ndarray:
    """Map FWI to the EFFIS danger class index (0..6)."""
    edges = np.array([c[1] for c in EFFIS_DANGER_CLASSES[1:]])
    return np.digitize(_asarray(fwi_val), edges, right=False).astype(np.int16)

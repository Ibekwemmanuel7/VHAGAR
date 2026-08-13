"""Shared synthetic granules. Kept out of any test module so both the
reader tests and the backfill tests build the same structurally faithful
granule rather than drifting apart."""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = ["_synthetic_cmip", "_synthetic_fdc"]


def _synthetic_fdc(n: int = 40, fire_codes=(10, 13, 11, 30, 15)) -> xr.Dataset:
    """A minimal but structurally faithful ABI L2 FDC granule.

    Grid is centred a little west of nadir so the CONUS bbox tests are
    meaningful, and a handful of fire pixels are planted with known codes.
    """
    x = np.linspace(-0.04, 0.02, n)
    y = np.linspace(0.06, 0.12, n)
    mask = np.zeros((n, n), dtype=np.int16)
    power = np.full((n, n), np.nan)
    temp = np.full((n, n), np.nan)
    area = np.full((n, n), np.nan)

    if fire_codes and n < 10 + 3 * len(fire_codes):
        raise ValueError(f"n={n} too small to place {len(fire_codes)} fire pixels")
    for i, code in enumerate(fire_codes):
        r, c = 10 + i * 3, 12 + i * 2
        mask[r, c] = code
        power[r, c] = 50.0 + 25.0 * i
        temp[r, c] = 400.0 + 10.0 * i
        area[r, c] = 4.0e6

    proj = xr.DataArray(
        0,
        attrs={
            "longitude_of_projection_origin": -75.0,
            "perspective_point_height": 35786023.0,
            "semi_major_axis": 6378137.0,
            "semi_minor_axis": 6356752.31414,
        },
    )
    return xr.Dataset(
        {
            "Mask": (("y", "x"), mask),
            "Power": (("y", "x"), power),
            "Temp": (("y", "x"), temp),
            "Area": (("y", "x"), area),
            "goes_imager_projection": proj,
            "t": ((), np.float64(838_000_000.0)),
        },
        coords={"x": x, "y": y},
    )


def _synthetic_cmip(n: int = 40, channel: str = "C07") -> xr.Dataset:
    """A minimal but structurally faithful ABI L2 CMIP channel granule.

    Deliberately on the **same grid** as :func:`_synthetic_fdc` (identical x and
    y), so tests can prove the navigation cache is shared across products. The
    ``CMI`` field is a 300 K background with a few planted features: a warm
    pixel, a saturated pixel (above the C07 threshold), a fill NaN, and one
    pixel flagged out of range via ``DQF``.
    """
    x = np.linspace(-0.04, 0.02, n)
    y = np.linspace(0.06, 0.12, n)
    cmi = np.full((n, n), 300.0)
    dqf = np.zeros((n, n), dtype=np.int16)

    cmi[10, 12] = 330.0           # a warm but usable pixel
    cmi[14, 16] = 450.0           # saturated for C07 (threshold 400 K)
    cmi[18, 20] = np.nan          # fill / no value
    cmi[22, 24] = 315.0           # good value but flagged unusable by DQF
    dqf[22, 24] = 2               # out of range

    proj = xr.DataArray(
        0,
        attrs={
            "longitude_of_projection_origin": -75.0,
            "perspective_point_height": 35786023.0,
            "semi_major_axis": 6378137.0,
            "semi_minor_axis": 6356752.31414,
        },
    )
    return xr.Dataset(
        {
            "CMI": (("y", "x"), cmi),
            "DQF": (("y", "x"), dqf),
            "goes_imager_projection": proj,
            "t": ((), np.float64(838_000_000.0)),
        },
        coords={"x": x, "y": y},
        attrs={"band_id": channel},
    )

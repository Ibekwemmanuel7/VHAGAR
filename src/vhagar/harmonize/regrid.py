"""Resampling onto the VHAGAR analysis grid.

The policy in this module is not negotiable per-script. Getting it wrong
silently corrupts fire energy accounting -- summing FRP after a bilinear
regrid does not conserve radiative power, and nobody notices until the
emissions numbers are wrong by 20%.

    quantity type          method            backend
    ---------------------  ----------------  --------------------------
    swath radiance / BT    nearest (swath)   pyresample
    gridded continuous     bilinear/cubic    odc-geo / rasterio.warp
    flux-like (FRP, precip) conservative     xesmf (cached weights)
    categorical            mode / nearest    odc-geo
    polygon -> grid stats  exact area-weight exactextract

Heavy dependencies are imported lazily so that ``import vhagar.harmonize``
works without GDAL installed.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

__all__ = ["Quantity", "RESAMPLING_POLICY", "conservative_regrid_2d", "check_mass_conservation"]


class Quantity(str, Enum):
    """What kind of physical quantity a band holds."""

    RADIANCE = "radiance"          # swath brightness temperature / radiance
    CONTINUOUS = "continuous"      # reflectance, LST, elevation
    FLUX = "flux"                  # FRP, precipitation, burned area fraction
    CATEGORICAL = "categorical"    # fuel model, land cover
    FRACTION = "fraction"          # 0-1 cover fractions


RESAMPLING_POLICY: dict[Quantity, str] = {
    Quantity.RADIANCE: "nearest",
    Quantity.CONTINUOUS: "bilinear",
    Quantity.FLUX: "conservative",
    Quantity.CATEGORICAL: "mode",
    Quantity.FRACTION: "average",
}


def resample_to_grid(source: Any, tile: Any, quantity: Quantity, **kwargs: Any) -> Any:
    """Reproject a raster onto a :class:`vhagar.grid.Tile`.

    Thin dispatcher over ``odc-geo``; raises with an actionable message if the
    geo extra is not installed.
    """
    try:
        from odc.geo.geobox import GeoBox
        from odc.geo.xr import xr_reproject
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "resample_to_grid requires the 'geo' extra: pip install 'vhagar[geo]'"
        ) from exc

    if quantity is Quantity.FLUX:
        raise NotImplementedError(
            "Flux quantities (FRP, precipitation) must use conservative regridding. "
            "Use conservative_regrid_2d() or an xesmf conservative weight set; "
            "bilinear/nearest resampling of a flux does not conserve the integral."
        )

    geobox = GeoBox.from_bbox(tile.haloed_bounds, crs=tile.crs, resolution=375.0)
    return xr_reproject(source, geobox, resampling=RESAMPLING_POLICY[quantity], **kwargs)


def conservative_regrid_2d(
    values: np.ndarray,
    src_edges_x: np.ndarray,
    src_edges_y: np.ndarray,
    dst_edges_x: np.ndarray,
    dst_edges_y: np.ndarray,
) -> np.ndarray:
    """First-order conservative regridding between two rectilinear grids.

    Redistributes ``values`` (an *extensive* quantity such as total FRP per
    cell) onto the destination grid by area overlap, so that the global sum is
    preserved exactly up to floating point.

    This is a reference implementation used for testing and for small grids.
    Production paths should use ``xesmf`` with weights computed once and
    cached to disk -- recomputing weights per tile is the single most common
    performance mistake in this pipeline.

    Parameters
    ----------
    values
        ``(ny, nx)`` source cell totals.
    src_edges_x, src_edges_y
        Monotonically increasing cell edge coordinates, length ``nx+1``/``ny+1``.
    dst_edges_x, dst_edges_y
        Destination edges.

    >>> import numpy as np
    >>> v = np.array([[1.0, 3.0], [5.0, 7.0]])
    >>> e = np.array([0.0, 1.0, 2.0])
    >>> out = conservative_regrid_2d(v, e, e, np.array([0.0, 2.0]), np.array([0.0, 2.0]))
    >>> float(out.sum())
    16.0
    """
    v = np.asarray(values, dtype=np.float64)
    sx = np.asarray(src_edges_x, dtype=np.float64)
    sy = np.asarray(src_edges_y, dtype=np.float64)
    dx = np.asarray(dst_edges_x, dtype=np.float64)
    dy = np.asarray(dst_edges_y, dtype=np.float64)

    if v.shape != (sy.size - 1, sx.size - 1):
        raise ValueError(f"values shape {v.shape} inconsistent with edges")
    for name, e in (("src_x", sx), ("src_y", sy), ("dst_x", dx), ("dst_y", dy)):
        if np.any(np.diff(e) <= 0):
            raise ValueError(f"{name} edges must be strictly increasing")

    # Fractional overlap matrices along each axis.
    def overlap(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        lo = np.maximum(src[:-1, None], dst[None, :-1])
        hi = np.minimum(src[1:, None], dst[None, 1:])
        inter = np.clip(hi - lo, 0.0, None)
        width = (src[1:] - src[:-1])[:, None]
        return inter / width  # fraction of each source cell landing in each dst cell

    fx = overlap(sx, dx)  # (nx_src, nx_dst)
    fy = overlap(sy, dy)  # (ny_src, ny_dst)
    # Extensive quantity: distribute the cell total by area fraction.
    return fy.T @ v @ fx


def check_mass_conservation(before: np.ndarray, after: np.ndarray, rtol: float = 1e-9) -> None:
    """Assert that a regrid conserved the integral. Call this in CI."""
    b = float(np.nansum(before))
    a = float(np.nansum(after))
    if b == 0.0:
        if a != 0.0:
            raise AssertionError(f"mass created from nothing: {a}")
        return
    rel = abs(a - b) / abs(b)
    if rel > rtol:
        raise AssertionError(
            f"regrid did not conserve mass: before={b!r} after={a!r} rel_err={rel:.3e} > {rtol:.1e}"
        )

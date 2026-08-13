"""Digital elevation sampling, for the terrain-parallax term of the GEO/LEO
matching tolerance.

ABI navigation solves for the ellipsoid, so ground at elevation ``h`` appears
displaced by ``h * tan(view_zenith)``. Over the Sierra that is well over a
kilometre, which is most of the observed GOES-to-VIIRS separation. The matching
tolerance in :func:`vhagar.harmonize.fusion.geo_leo_tolerance_m` accounts for it,
but only if it is given a real per-pixel elevation instead of the flat 1000 m
placeholder. This module provides that elevation.

Coordinate convention
---------------------
The DEM is assumed to be in the **same CRS as the detections**, i.e. the region
analysis CRS. Detections already carry projected ``x``/``y`` in that CRS, so
sampling is a direct lookup with no reprojection at match time. Resample a
source DEM into the region CRS once, up front, rather than reprojecting per
detection. Points off the edge of the DEM sample as NaN, and the tolerance
function treats a NaN elevation as "unknown" and falls back to the placeholder,
so a partial DEM never poisons a match.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["DEM", "attach_elevation"]


@dataclass(frozen=True, slots=True)
class DEM:
    """A regularly gridded elevation raster in a projected CRS.

    ``x0``/``y0`` are the coordinates of the centre of pixel ``[0, 0]``, and
    ``dx``/``dy`` are the pixel spacings. ``dy`` is negative for a north-up
    raster (rows running from high y to low y), which is the common case.
    """

    elevation: np.ndarray   # (ny, nx), metres, NaN for nodata
    x0: float
    y0: float
    dx: float
    dy: float

    def __post_init__(self) -> None:
        if self.elevation.ndim != 2:
            raise ValueError(f"elevation must be 2D, got shape {self.elevation.shape}")
        if self.dx == 0 or self.dy == 0:
            raise ValueError("dx and dy must be non-zero")

    @classmethod
    def from_rasterio(cls, path, band: int = 1) -> DEM:
        """Load a DEM from any raster rasterio can read (GeoTIFF, VRT, ...).

        The raster must already be in the region CRS; this does not reproject.
        Nodata is converted to NaN.
        """
        try:
            import rasterio
        except ImportError as exc:  # pragma: no cover
            raise ImportError("DEM.from_rasterio requires rasterio: pip install rasterio") from exc

        with rasterio.open(path) as ds:
            arr = ds.read(band).astype(np.float64)
            if ds.nodata is not None:
                arr = np.where(arr == ds.nodata, np.nan, arr)
            t = ds.transform
            # Affine: x = c + a*col + b*row; pixel [0,0] centre at col=row=0.5.
            x0 = t.c + t.a * 0.5 + t.b * 0.5
            y0 = t.f + t.d * 0.5 + t.e * 0.5
            return cls(elevation=arr, x0=x0, y0=y0, dx=t.a, dy=t.e)

    def sample(self, x, y) -> np.ndarray:
        """Bilinear elevation at projected coordinates. NaN outside the grid.

        Exact for a locally linear surface, which is what bilinear promises.
        Any of the four surrounding cells being NaN makes the sample NaN, so
        nodata does not bleed into a real value.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        ny, nx = self.elevation.shape

        col = (x - self.x0) / self.dx
        row = (y - self.y0) / self.dy

        c0 = np.floor(col).astype(np.int64)
        r0 = np.floor(row).astype(np.int64)
        fc = col - c0
        fr = row - r0

        inside = (c0 >= 0) & (c0 < nx - 1) & (r0 >= 0) & (r0 < ny - 1)
        # Clamp indices so the gather is always valid; masked back to NaN after.
        c0c = np.clip(c0, 0, nx - 2)
        r0c = np.clip(r0, 0, ny - 2)

        e = self.elevation
        v00 = e[r0c, c0c]
        v01 = e[r0c, c0c + 1]
        v10 = e[r0c + 1, c0c]
        v11 = e[r0c + 1, c0c + 1]
        top = v00 * (1 - fc) + v01 * fc
        bot = v10 * (1 - fc) + v11 * fc
        out = top * (1 - fr) + bot * fr

        return np.where(inside, out, np.nan)


def attach_elevation(detections: Sequence, dem: DEM) -> list:
    """Return copies of ``detections`` with ``elevation_m`` filled from the DEM.

    Samples at each detection's projected ``x``/``y``. Detections off the DEM
    get ``elevation_m = None``, so the tolerance falls back to the placeholder
    rather than to a fabricated height.
    """
    import dataclasses

    if not detections:
        return []
    xs = np.array([d.x for d in detections], dtype=np.float64)
    ys = np.array([d.y for d in detections], dtype=np.float64)
    elev = dem.sample(xs, ys)
    out = []
    for d, e in zip(detections, elev, strict=True):
        out.append(dataclasses.replace(d, elevation_m=None if np.isnan(e) else float(e)))
    return out

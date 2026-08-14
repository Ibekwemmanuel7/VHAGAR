"""Assemble independent-predictor T2 samples: Sentinel-2 RBR vs MTBS severity.

Ties the pieces together, per fire: build the fire's analysis window on the MTBS
grid, compute an independent Sentinel-2 RBR predictor on that window, read the
MTBS thematic severity as the reference on the same window, and make a
:class:`~vhagar.datasets.burned_area.T2Sample`. The result feeds the existing
Stage-0 driver, which calibrates a threshold on training fires and reports the
Olofsson error-adjusted burned area with a CI.

Only two functions touch the network or rasterio, and both are lazily imported:
:func:`read_mtbs_reference_on_grid` (warps the thematic mosaic onto the window)
and the Sentinel-2 pull inside :func:`build_optical_sample`. The window geometry
is pure and tested.
"""

from __future__ import annotations

import math

from vhagar.datasets.burned_area import make_sample, mtbs_burned_mask
from vhagar.grid import REGION_CRS
from vhagar.io.optical import TargetGrid

__all__ = [
    "build_optical_sample",
    "build_optical_samples",
    "read_mtbs_reference_on_grid",
    "target_grid_for_fire",
]


def target_grid_for_fire(
    record,
    res_m: float = 30.0,
    buffer_factor: float = 1.6,
    min_half_m: float = 5_000.0,
):
    """A fire's analysis window on the region's equal-area grid, plus a lon/lat bbox.

    The window is centred on the fire's representative point and sized from its
    area (a circle-equivalent radius, buffered), floored to ``min_half_m`` so
    small fires still get a usable window. Returns ``(TargetGrid, bbox_4326)``:
    the grid for warping rasters, and the lon/lat bbox for the Sentinel-2 search.
    Pure: needs only pyproj and arithmetic.
    """
    from pyproj import Transformer

    crs = REGION_CRS[record.region]
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    cx, cy = fwd.transform(record.lon, record.lat)
    radius = math.sqrt(record.area_ha * 1e4 / math.pi) if record.area_ha else 0.0
    half = max(min_half_m, radius * buffer_factor)

    x0 = math.floor((cx - half) / res_m) * res_m
    y0 = math.floor((cy - half) / res_m) * res_m
    x1 = math.ceil((cx + half) / res_m) * res_m
    y1 = math.ceil((cy + half) / res_m) * res_m
    width = int(round((x1 - x0) / res_m))
    height = int(round((y1 - y0) / res_m))
    # rasterio/GDAL affine: (a, b, c, d, e, f) with top-left origin, north-up.
    transform = (res_m, 0.0, x0, 0.0, -res_m, y1)
    grid = TargetGrid(crs=crs, transform=transform, width=width, height=height)

    # lon/lat bbox from the projected corners, for the STAC search.
    lons, lats = inv.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    bbox_4326 = (min(lons), min(lats), max(lons), max(lats))
    return grid, bbox_4326


def read_mtbs_reference_on_grid(mosaic_path, grid: TargetGrid):
    """Warp the MTBS thematic mosaic onto a window and return ``(burned, valid)``.

    Nearest resampling, because the thematic classes are categorical. Needs
    rasterio.
    """
    import numpy as np  # noqa: F401  (kept explicit; masks are numpy)
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.vrt import WarpedVRT

    with rasterio.open(mosaic_path) as src, WarpedVRT(
        src,
        crs=grid.crs,
        transform=Affine(*grid.transform),
        width=grid.width,
        height=grid.height,
        resampling=Resampling.nearest,
    ) as vrt:
        severity = vrt.read(1)
    return mtbs_burned_mask(severity)


def build_optical_sample(record, mosaic_path, max_cloud: float = 60.0, **grid_kw):
    """Build one fire's T2 sample: Sentinel-2 RBR predictor, MTBS reference.

    Needs network (Sentinel-2) and rasterio (MTBS warp). Raises on missing
    imagery, which the batch builder catches and skips.
    """
    from vhagar.io.optical import sentinel2_rbr

    if record.ignition_date is None:
        raise ValueError(f"{record.event_id} has no ignition date")
    grid, bbox = target_grid_for_fire(record, **grid_kw)
    predictor = sentinel2_rbr(
        bbox, record.ignition_date.isoformat(), grid, max_cloud=max_cloud
    )
    burned, valid = read_mtbs_reference_on_grid(mosaic_path, grid)
    return make_sample(
        record.event_id, predictor, burned, reference_valid=valid,
        tile_id=record.tile_ids[0] if record.tile_ids else None,
    )


def build_optical_samples(records, mosaic_path, on_error=None, **kw):
    """Build samples for many fires, skipping those without usable imagery.

    Returns ``{event_id: T2Sample}``. ``on_error(record, exc)`` is called for each
    skipped fire, so a run can report coverage rather than fail on one cloudy fire.
    """
    samples = {}
    for r in records:
        try:
            samples[r.event_id] = build_optical_sample(r, mosaic_path, **kw)
        except Exception as exc:  # noqa: BLE001
            if on_error is not None:
                on_error(r, exc)
    return samples

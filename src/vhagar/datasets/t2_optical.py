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
    max_half_m: float = 30_000.0,
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
    # Clamp the half-window: below by a floor so small fires get a usable scene,
    # above by a cap so a Dixie-scale fire does not allocate a 100+ km array. A
    # capped window clips the very largest fires' extent, which is a per-fire
    # area caveat, not a threshold-calibration problem.
    half = min(max_half_m, max(min_half_m, radius * buffer_factor))

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


def rasterize_burned_on_grid(geometries, src_crs, grid: TargetGrid):
    """Rasterise burned polygons (in ``src_crs``) onto a window. Needs rasterio, shapely.

    ``geometries`` is an iterable of shapely geometries. Each is reprojected to
    the grid CRS and burned in. Returns ``(burned, valid)`` with valid all-True:
    a delineation labels every window pixel as inside-or-outside the perimeter.
    Pure geometry-to-raster, so it is testable without any file.
    """
    import numpy as np
    from pyproj import Transformer
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from shapely.ops import transform as shp_transform

    tf = Transformer.from_crs(src_crs, grid.crs, always_xy=True)

    def _project(x, y, z=None):
        return tf.transform(x, y)

    shapes = [shp_transform(_project, g) for g in geometries if g is not None]
    if shapes:
        burned = rasterize(
            [(s, 1) for s in shapes],
            out_shape=grid.shape,
            transform=Affine(*grid.transform),
            fill=0,
            dtype="uint8",
        ).astype(bool)
    else:
        burned = np.zeros(grid.shape, dtype=bool)
    return burned, np.ones(grid.shape, dtype=bool)


def read_emsr_reference_on_grid(delineation_path, grid: TargetGrid):
    """Read a Copernicus EMS burnt-area delineation shapefile and rasterise it.

    The pyogrio read is the IO edge; the rasterising is
    :func:`rasterize_burned_on_grid`. Needs pyogrio and shapely.
    """
    from pyogrio.raw import read as _raw_read
    from shapely import wkb

    result = _raw_read(delineation_path, read_geometry=True)
    meta, geom_wkb = result[0], result[2]
    geoms = [wkb.loads(bytes(g)) for g in geom_wkb if g is not None]
    return rasterize_burned_on_grid(geoms, meta["crs"], grid)


def build_optical_sample(
    record,
    mosaic_path,
    max_cloud: float = 60.0,
    max_scenes: int = 6,
    res_m: float = 100.0,
    cache_dir=None,
    **grid_kw,
):
    """Build one fire's T2 sample: Sentinel-2 RBR predictor, MTBS reference.

    ``res_m`` is the analysis resolution. Coarser than native (default 100 m) is
    deliberate for a Stage-0 burned/unburned threshold: it shrinks the arrays and,
    crucially, lets GDAL read the Sentinel-2 COGs from their overviews rather than
    full resolution, which is the difference between seconds and minutes per fire.

    ``cache_dir`` persists each sample to ``.npz`` keyed by event and resolution,
    so a re-run (after a code fix, or to widen the fire set) reuses the expensive
    imagery pull instead of fetching it again. Needs network + rasterio only on a
    cache miss.
    """
    from pathlib import Path

    from vhagar.datasets.burned_area import T2Sample
    from vhagar.io.optical import sentinel2_rbr

    if record.ignition_date is None:
        raise ValueError(f"{record.event_id} has no ignition date")

    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe = record.event_id.replace(":", "_").replace("/", "_")
        cache_path = cache_dir / f"{safe}_r{int(res_m)}.npz"
        if cache_path.exists():
            return T2Sample.load(cache_path)

    grid, bbox = target_grid_for_fire(record, res_m=res_m, **grid_kw)
    predictor = sentinel2_rbr(
        bbox, record.ignition_date.isoformat(), grid,
        max_cloud=max_cloud, max_scenes=max_scenes,
    )
    # ``mosaic_path`` is either an MTBS thematic mosaic (path) or a callable
    # reference reader, e.g. an EMS delineation. This is what lets the same
    # optical predictor pipeline serve both the CONUS and the European side of
    # the leave-one-continent-out test.
    if callable(mosaic_path):
        burned, valid = mosaic_path(grid)
    else:
        burned, valid = read_mtbs_reference_on_grid(mosaic_path, grid)
    sample = make_sample(
        record.event_id, predictor, burned, reference_valid=valid,
        tile_id=record.tile_ids[0] if record.tile_ids else None,
    )
    if cache_path is not None:
        sample.save(cache_path)
    return sample


def build_optical_samples(
    records, mosaic_path, on_error=None, on_start=None, on_done=None, **kw
):
    """Build samples for many fires, skipping those without usable imagery.

    Returns ``{event_id: T2Sample}``. ``on_start(record)`` fires before each fire,
    ``on_done(record, sample)`` after a success, and ``on_error(record, exc)`` on a
    skip, so a long run shows progress and reports coverage rather than looking
    frozen or dying on one cloudy fire.
    """
    samples = {}
    for r in records:
        if on_start is not None:
            on_start(r)
        try:
            sample = build_optical_sample(r, mosaic_path, **kw)
        except Exception as exc:  # noqa: BLE001
            if on_error is not None:
                on_error(r, exc)
            continue
        samples[r.event_id] = sample
        if on_done is not None:
            on_done(r, sample)
    return samples

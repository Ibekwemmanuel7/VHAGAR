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
    "rasterize_burned_on_grid",
    "read_emsr_reference_on_grid",
    "read_mtbs_reference_on_grid",
    "select_fires",
    "target_grid_for_fire",
]


def select_fires(records, n: int, strategy: str = "largest"):
    """Pick ``n`` fires from ``records`` by a strategy.

    ``"largest"`` takes the ``n`` biggest fires (fast to image, but biases every
    downstream number toward megafires). ``"size"`` instead samples ``n`` fires
    spread evenly across the area-sorted list, from small to large, so the
    evaluation is representative of the fire-size distribution rather than its
    tail.
    """
    ranked = sorted(records, key=lambda r: r.area_ha or 0.0)
    if n >= len(ranked):
        return list(ranked)
    if strategy == "largest":
        return ranked[-n:]
    if strategy == "size":
        import numpy as np

        idx = np.linspace(0, len(ranked) - 1, n).round().astype(int)
        seen, out = set(), []
        for i in idx:
            if int(i) not in seen:
                seen.add(int(i))
                out.append(ranked[int(i)])
        return out
    raise ValueError(f"strategy must be 'largest' or 'size', got {strategy!r}")


def target_grid_for_fire(
    record,
    res_m: float = 30.0,
    buffer_factor: float = 2.5,
    min_half_m: float = 15_000.0,
    max_half_m: float = 30_000.0,
):
    """A fire's analysis window on the region's equal-area grid, plus a lon/lat bbox.

    The window is centred on the fire's representative point and sized from its
    area (a circle-equivalent radius, buffered by ``buffer_factor``), floored to
    ``min_half_m`` and capped at ``max_half_m``. Returns ``(TargetGrid, bbox_4326)``:
    the grid for warping rasters, and the lon/lat bbox for the Sentinel-2 search.
    Pure: needs only pyproj and arithmetic.

    Window sizing note (docs/11). The half-width scales with the fire radius
    (``radius * 2.5``) with a 15 km floor, so a small fire still sees a wide ring
    of unburned land around it. Earlier defaults (1.6x, 5 km floor) filled a small
    fire's window ~90% burned, which let an F1-tuned threshold collapse to
    "predict everything burned" and destroyed cross-continent transfer. A wider
    window restores a realistic burned/unburned class balance so the threshold
    calibration and the Olofsson area estimate are both meaningful. Widening the
    window changes what each sample measures, so samples built under the old and
    new sizing must not be pooled or compared; re-pull the cache after changing it.
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


def read_mtbs_reference_on_grid(mosaic_path, grid: TargetGrid, include_background: bool = True):
    """Warp the MTBS thematic mosaic onto a window and return ``(burned, valid)``.

    Nearest resampling, because the thematic classes are categorical. With
    ``include_background=True`` (the default) the unburned background around the
    fire is counted as valid unburned, so the sample is a burned-area detection
    task at a realistic base rate rather than a within-perimeter severity task
    (docs/11). Needs rasterio.
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
    return mtbs_burned_mask(severity, include_background=include_background)


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

    Uses the same burnt-area filter as the record builder, so the "Not
    applicable" and other non-burnt polygons in an observed-event layer do not
    leak into the reference. Needs pyogrio and shapely.
    """
    from vhagar.labels.ingest import read_emsr_burned_geometries

    geoms, crs = read_emsr_burned_geometries(delineation_path)
    return rasterize_burned_on_grid(geoms, crs, grid)


def build_optical_sample(
    record,
    mosaic_path,
    max_cloud: float = 60.0,
    max_scenes: int = 6,
    res_m: float = 100.0,
    cache_dir=None,
    include_background: bool = True,
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
        # The window floor is part of what the sample measures, so it is part of
        # the cache key: widening the window (docs/11) writes new files instead of
        # silently reusing narrow-window samples, which the "never compare across
        # code paths" rule forbids. Default floor 15 km -> ``w15``.
        win_km = int(grid_kw.get("min_half_m", 15_000.0) / 1000)
        # The reference framing is also part of what the sample measures. The
        # background-as-unburned detection reference (docs/11) gets a ``bg`` tag so
        # it never collides with the old perimeter-only samples of the same window.
        bg_tag = "bg" if (include_background and not callable(mosaic_path)) else ""
        cache_path = cache_dir / f"{safe}_r{int(res_m)}_w{win_km}{bg_tag}.npz"
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
        burned, valid = read_mtbs_reference_on_grid(
            mosaic_path, grid, include_background=include_background
        )
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

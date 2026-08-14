"""Assign each fire a stratum by sampling a global class raster at its location.

The cross-continent transfer gap is driven by fuel and climate: a threshold
learned on Californian conifer does not fit Greek Mediterranean scrub. But
California and Greece share a Köppen climate class (Csa, hot-summer
Mediterranean), so a **global** stratification, unlike US-only EPA ecoregions,
lets a threshold be calibrated on US Mediterranean fires and applied to European
Mediterranean fires, like for like.

This samples any categorical global raster (Köppen-Geiger is the intended one, but
a biome or land-cover map works too) at each fire's representative point and
returns the class. Pure sampling; the rasterio read is the lazily-imported edge.
"""

from __future__ import annotations

__all__ = ["assign_strata", "sample_class_raster"]


def sample_class_raster(lons, lats, raster_path):
    """Sample a categorical raster at lon/lat points. Returns an int per point.

    Points are transformed from EPSG:4326 into the raster CRS before sampling, so
    a Köppen raster in any projection works. Needs rasterio.
    """
    import numpy as np
    import rasterio
    from pyproj import Transformer

    with rasterio.open(raster_path) as src:
        tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = tf.transform(np.asarray(lons), np.asarray(lats))
        vals = [v[0] for v in src.sample(zip(xs, ys, strict=True))]
    return np.asarray(vals)


def assign_strata(records, raster_path) -> dict[str, int]:
    """Map each record's event id to its stratum class from a global raster.

    Returns ``{event_id: class_int}``. A fire off the raster (nodata) is omitted,
    so it falls back to the global threshold at evaluation time.
    """
    records = list(records)
    if not records:
        return {}
    lons = [r.lon for r in records]
    lats = [r.lat for r in records]
    classes = sample_class_raster(lons, lats, raster_path)
    out: dict[str, int] = {}
    for r, c in zip(records, classes, strict=True):
        c = int(c)
        if c != 0:  # 0 is the usual nodata / ocean class
            out[r.event_id] = c
    return out

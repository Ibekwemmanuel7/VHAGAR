"""Locate fire events on the analysis grid.

A :class:`~vhagar.labels.registry.FireEventRecord` carries a representative
lon/lat in EPSG:4326. Splits and per-tile training reads need it on the region's
equal-area analysis grid instead, as one or more tile ids. This projects the
point (and, when a bounding box is supplied, its extent) into the region CRS and
reads off the tiles. It reuses :mod:`vhagar.grid`, so tile ids stay identical to
everything else in the system.

Point assignment is the default. A large fire spans several tiles; pass its
bounding box to get every tile it touches, which is what per-tile training reads
need. An event whose point falls outside the region grid returns no tiles rather
than raising, because a mixed-region label set legitimately contains events that
are not in this grid.
"""

from __future__ import annotations

from functools import lru_cache

from vhagar.grid import REGION_CRS, AnalysisGrid

__all__ = ["assign_tiles"]


@lru_cache(maxsize=8)
def _transformer(region: str):
    from pyproj import Transformer

    return Transformer.from_crs("EPSG:4326", REGION_CRS[region], always_xy=True)


def assign_tiles(
    record,
    grid: AnalysisGrid | None = None,
    bbox_4326: tuple[float, float, float, float] | None = None,
) -> list[str]:
    """Return the analysis-grid tile ids an event falls in.

    ``grid`` defaults to the region grid for ``record.region``. Pass
    ``bbox_4326`` as ``(west, south, east, north)`` to cover a perimeter's full
    extent rather than just its representative point.
    """
    if record.region not in REGION_CRS:
        return []
    if grid is None:
        grid = AnalysisGrid(record.region)
    tf = _transformer(grid.region)

    if bbox_4326 is not None:
        west, south, east, north = bbox_4326
        xs, ys = tf.transform([west, east, west, east], [south, south, north, north])
        px0, px1 = min(xs), max(xs)
        py0, py1 = min(ys), max(ys)
        return [t.tile_id for t in grid.tiles_for_bounds((px0, py0, px1, py1))]

    x, y = tf.transform(record.lon, record.lat)
    try:
        return [grid.tile_for_point(x, y).tile_id]
    except IndexError:
        return []

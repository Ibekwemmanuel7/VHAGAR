"""The VHAGAR analysis grid.

One equal-area grid per region at VIIRS I-band native resolution (375 m).
Everything else is resampled onto it. Tiles are the atomic unit of both
storage chunking and *spatial splitting*, tile IDs are what gets versioned
into a split manifest, which is why they must be stable and deterministic.

Design notes
------------
* Equal-area projections are used so that area statistics (burned hectares,
  FRP density) are unbiased without per-pixel area weighting.
* Tiles are square in projected space, 256 cells on a side (96 km), with a
  32-cell halo so a convolutional model sees valid context at tile edges.
  The halo overlaps neighbouring tiles; it is *never* used for loss, only
  for context. See ``Tile.core_slice``.
* Tile IDs encode the region and the integer tile index, e.g.
  ``conus/x0031_y0072``. They are stable across code versions.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

Region = Literal["conus", "canada", "europe"]

#: Equal-area CRS per region.
REGION_CRS: dict[str, str] = {
    "conus": "EPSG:5070",  # NAD83 / Conus Albers
    "canada": "EPSG:3979",  # NAD83(CSRS) / Canada Atlas Lambert
    "europe": "EPSG:3035",  # ETRS89-extended / LAEA Europe
}

#: Projected bounds (xmin, ymin, xmax, ymax) in each region's CRS, snapped
#: outward to a whole number of tiles by :class:`AnalysisGrid`.
REGION_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "conus": (-2_400_000.0, 200_000.0, 2_300_000.0, 3_200_000.0),
    "canada": (-2_800_000.0, -1_000_000.0, 3_200_000.0, 3_900_000.0),
    "europe": (2_400_000.0, 1_300_000.0, 7_400_000.0, 5_500_000.0),
}

RESOLUTION_M = 375.0
TILE_CELLS = 256
HALO_CELLS = 32


@dataclass(frozen=True, slots=True)
class Tile:
    """A single analysis tile.

    Attributes
    ----------
    region, ix, iy
        Identity. ``ix``/``iy`` are integer tile indices from the grid origin.
    bounds
        Core (halo-excluded) bounds in the region CRS, as
        ``(xmin, ymin, xmax, ymax)``.
    """

    region: str
    ix: int
    iy: int
    bounds: tuple[float, float, float, float]

    @property
    def tile_id(self) -> str:
        return f"{self.region}/x{self.ix:04d}_y{self.iy:04d}"

    @property
    def crs(self) -> str:
        return REGION_CRS[self.region]

    @property
    def haloed_bounds(self) -> tuple[float, float, float, float]:
        pad = HALO_CELLS * RESOLUTION_M
        x0, y0, x1, y1 = self.bounds
        return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)

    @property
    def shape(self) -> tuple[int, int]:
        """Array shape *including* the halo, as ``(rows, cols)``."""
        n = TILE_CELLS + 2 * HALO_CELLS
        return (n, n)

    @property
    def core_slice(self) -> tuple[slice, slice]:
        """Slice that extracts the core (loss-eligible) region from a haloed array."""
        s = slice(HALO_CELLS, HALO_CELLS + TILE_CELLS)
        return (s, s)

    @property
    def transform(self) -> tuple[float, float, float, float, float, float]:
        """Affine transform (GDAL order) for the haloed array."""
        x0, _, _, y1 = self.haloed_bounds
        return (RESOLUTION_M, 0.0, x0, 0.0, -RESOLUTION_M, y1)

    def centroid(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bounds
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


class AnalysisGrid:
    """Deterministic tiling of a region.

    >>> g = AnalysisGrid("conus")
    >>> t = g.tile(10, 10)
    >>> t.tile_id
    'conus/x0010_y0010'
    >>> t.shape
    (320, 320)
    """

    def __init__(self, region: Region | str) -> None:
        if region not in REGION_CRS:
            raise ValueError(f"unknown region {region!r}; expected one of {list(REGION_CRS)}")
        self.region = str(region)
        self.crs = REGION_CRS[self.region]
        self.tile_size_m = TILE_CELLS * RESOLUTION_M

        x0, y0, x1, y1 = REGION_BOUNDS[self.region]
        # Snap the origin down and the extent up to whole tiles so that tile
        # indices never shift if the nominal bounds are later widened.
        self.origin_x = math.floor(x0 / self.tile_size_m) * self.tile_size_m
        self.origin_y = math.floor(y0 / self.tile_size_m) * self.tile_size_m
        self.n_x = math.ceil((x1 - self.origin_x) / self.tile_size_m)
        self.n_y = math.ceil((y1 - self.origin_y) / self.tile_size_m)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"AnalysisGrid(region={self.region!r}, crs={self.crs!r}, tiles={self.n_x}x{self.n_y})"

    @property
    def n_tiles(self) -> int:
        return self.n_x * self.n_y

    def tile(self, ix: int, iy: int) -> Tile:
        if not (0 <= ix < self.n_x and 0 <= iy < self.n_y):
            raise IndexError(f"tile ({ix}, {iy}) outside grid {self.n_x}x{self.n_y}")
        x0 = self.origin_x + ix * self.tile_size_m
        y0 = self.origin_y + iy * self.tile_size_m
        return Tile(self.region, ix, iy, (x0, y0, x0 + self.tile_size_m, y0 + self.tile_size_m))

    def tiles(self) -> Iterator[Tile]:
        for iy in range(self.n_y):
            for ix in range(self.n_x):
                yield self.tile(ix, iy)

    def tile_for_point(self, x: float, y: float) -> Tile:
        """Tile containing a projected coordinate (already in the region CRS)."""
        ix = int((x - self.origin_x) // self.tile_size_m)
        iy = int((y - self.origin_y) // self.tile_size_m)
        return self.tile(ix, iy)

    def tiles_for_bounds(self, bounds: tuple[float, float, float, float]) -> list[Tile]:
        """All tiles intersecting a projected bounding box."""
        x0, y0, x1, y1 = bounds
        ix0 = max(0, int((x0 - self.origin_x) // self.tile_size_m))
        iy0 = max(0, int((y0 - self.origin_y) // self.tile_size_m))
        ix1 = min(self.n_x - 1, int((x1 - self.origin_x) // self.tile_size_m))
        iy1 = min(self.n_y - 1, int((y1 - self.origin_y) // self.tile_size_m))
        return [
            self.tile(ix, iy)
            for iy in range(iy0, iy1 + 1)
            for ix in range(ix0, ix1 + 1)
        ]


def parse_tile_id(tile_id: str) -> Tile:
    """Inverse of :attr:`Tile.tile_id`."""
    region, _, xy = tile_id.partition("/")
    xs, _, ys = xy.partition("_")
    return AnalysisGrid(region).tile(int(xs[1:]), int(ys[1:]))

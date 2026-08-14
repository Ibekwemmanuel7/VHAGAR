"""Sentinel-2 surface reflectance to a burn-severity predictor, on a target grid.

This is the independent predictor for the T2 Stage-0 baseline. MTBS supplies the
reference; the predictor here is computed from Sentinel-2 L2A, which shares no
lineage with MTBS, so a threshold calibrated on it and evaluated against MTBS is
a real accuracy claim rather than a self-comparison.

The method follows the architecture (section 4.2): mean-composite pre-fire and
post-fire windows, compute NBR from B8A (nir08) and B12 (swir22) at their native
20 m, and difference to RBR. B8A, not B8, matches the SWIR bandpass and the 20 m
grid.

Design
------
The fallible logic, cloud masking, temporal compositing, and the band maths, is
pure and tested offline on synthetic stacks. The network edge, STAC search and
warping each COG onto the analysis grid, is separated and lazily imports
pystac-client and rasterio. Reading each scene **warped directly onto the target
grid** (the fire's MTBS 30 m Albers window) folds reprojection and windowing into
one step, so the predictor and the reference co-register with no separate
resampling pass.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from vhagar.features.indices import nbr, rbr

__all__ = [
    "SCL_KEEP",
    "TargetGrid",
    "composite_nbr",
    "mean_composite",
    "scl_valid_mask",
    "sentinel2_rbr",
]

#: Sen2Cor Scene Classification values to KEEP. Dropped are 0 nodata, 1 saturated,
#: 2 dark, 3 cloud shadow, 8/9 cloud, 10 thin cirrus. Kept are vegetation (4),
#: bare (5), water (6), unclassified (7) and snow (11). Burn scars usually fall in
#: 4/5; keeping 6/7/11 avoids over-masking valid ground.
SCL_KEEP = (4, 5, 6, 7, 11)


@dataclass(frozen=True, slots=True)
class TargetGrid:
    """The grid a predictor is warped onto: a fire's MTBS 30 m Albers window."""

    crs: str                       # e.g. "EPSG:5070"
    transform: tuple               # affine (a, b, c, d, e, f), GDAL/rasterio order
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


# ------------------------------------------------------------- pure -------


def scl_valid_mask(scl: np.ndarray, keep: tuple[int, ...] = SCL_KEEP) -> np.ndarray:
    """Boolean mask of usable pixels from a Sentinel-2 Scene Classification array."""
    return np.isin(np.asarray(scl), keep)


def mean_composite(stack: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-pixel mean over time of valid observations. NaN where none are valid.

    ``stack`` and ``valid`` are ``(T, H, W)``. A pixel cloudy in every scene has
    no valid observation and becomes NaN rather than a fabricated value.
    """
    stack = np.asarray(stack, dtype=np.float64)
    masked = np.where(np.asarray(valid, dtype=bool), stack, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices are expected
        return np.nanmean(masked, axis=0)


def composite_nbr(
    nir_stack: np.ndarray,
    swir_stack: np.ndarray,
    scl_stack: np.ndarray,
    keep: tuple[int, ...] = SCL_KEEP,
) -> np.ndarray:
    """Cloud-masked mean-composite NBR from stacks of one window's scenes."""
    valid = scl_valid_mask(scl_stack, keep)
    nir_c = mean_composite(nir_stack, valid)
    swir_c = mean_composite(swir_stack, valid)
    return nbr(nir_c, swir_c)


def rbr_from_windows(
    pre_nir, pre_swir, pre_scl, post_nir, post_swir, post_scl, keep: tuple[int, ...] = SCL_KEEP
) -> np.ndarray:
    """RBR from pre- and post-fire scene stacks. The independent predictor."""
    nbr_pre = composite_nbr(pre_nir, pre_swir, pre_scl, keep)
    nbr_post = composite_nbr(post_nir, post_swir, post_scl, keep)
    return rbr(nbr_pre, nbr_post)


# ---------------------------------------------------------- network -------


def _search_sentinel2(bbox_4326, start: str, end: str, max_cloud: float = 60.0):
    """STAC items for a bbox and date window. Needs pystac-client."""
    from pystac_client import Client

    client = Client.open("https://earth-search.aws.element84.com/v1")
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=list(bbox_4326),
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    return list(search.items())


def _warp_asset_to_grid(href: str, grid: TargetGrid, resampling: str = "bilinear") -> np.ndarray:
    """Read one COG asset warped onto the target grid. Needs rasterio."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.vrt import WarpedVRT

    dst = Affine(*grid.transform)
    with rasterio.open(href) as src, WarpedVRT(
        src,
        crs=grid.crs,
        transform=dst,
        width=grid.width,
        height=grid.height,
        resampling=getattr(Resampling, resampling),
    ) as vrt:
        return vrt.read(1).astype(np.float64)


def sentinel2_rbr(
    bbox_4326,
    ignition_date: str,
    grid: TargetGrid,
    pre_days: tuple[int, int] = (90, 15),
    post_days: tuple[int, int] = (15, 75),
    max_cloud: float = 60.0,
    keep: tuple[int, ...] = SCL_KEEP,
) -> np.ndarray:
    """Independent RBR for a fire on its MTBS grid. Needs network, pystac + rasterio.

    Searches Sentinel-2 in the pre and post windows around ``ignition_date``,
    warps B8A/B12/SCL onto ``grid``, cloud-masks, composites, and differences NBR
    to RBR. This is the one step that must run where the network is open.
    """
    from datetime import date, timedelta

    ig = date.fromisoformat(ignition_date)

    def window(days: tuple[int, int]) -> tuple[str, str]:
        return (
            (ig - timedelta(days=days[0])).isoformat(),
            (ig - timedelta(days=days[1])).isoformat(),
        )

    def post_window(days: tuple[int, int]) -> tuple[str, str]:
        return (
            (ig + timedelta(days=days[0])).isoformat(),
            (ig + timedelta(days=days[1])).isoformat(),
        )

    def stacks(start: str, end: str):
        items = _search_sentinel2(bbox_4326, start, end, max_cloud)
        if not items:
            raise RuntimeError(f"no Sentinel-2 scenes for {start}/{end}")
        nir, swir, scl = [], [], []
        for it in items:
            a = it.assets
            nir.append(_warp_asset_to_grid(a["nir08"].href, grid))
            swir.append(_warp_asset_to_grid(a["swir22"].href, grid))
            scl.append(_warp_asset_to_grid(a["scl"].href, grid, resampling="nearest"))
        return np.stack(nir), np.stack(swir), np.stack(scl)

    pre = stacks(*window(pre_days))
    post = stacks(*post_window(post_days))
    return rbr_from_windows(*pre, *post, keep=keep)

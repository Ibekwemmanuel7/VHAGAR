"""Per-parcel features from vector and raster sources, with vintage and leakage notes.

Every feature is computed from the parcel polygon (not just its centroid) in an
equal-area CRS (EPSG:5070) or, for the geographic DEM, with metric pixel spacing
computed at the parcel's latitude. ``FEATURE_SPECS`` records source, vintage, method, and
the leakage risk of each feature relative to the 2016 snapshot, and is written into every
study output.

Raw coordinates are deliberately NOT features: in VHAGAR's fire-detection work, raw
lat/lon supplied most of a model's gain under a random split and harmed transfer under a
spatial holdout (docs/02_VALIDATION.md). Location enters only through physical layers.
"""
from __future__ import annotations

import json
import math
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.merge
import rasterio.windows
import shapely

from vhagar.parcel.study.paths import EQUAL_AREA, GEOGRAPHIC, RAW

__all__ = ["FEATURE_SPECS", "FeatureSpec", "slope_stats", "zonal_classes", "zonal_mean",
           "nearest_distance_km", "overlap_fraction", "interp_grid", "NLCD_GROUPS",
           "vsizip_member"]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source: str
    vintage: str
    method: str
    leakage: str  # none | low | moderate | high, relative to the 2016 snapshot
    note: str = ""


FEATURE_SPECS: dict[str, FeatureSpec] = {s.name: s for s in [
    FeatureSpec("slope_mean_pct", "USGS 3DEP 1 arc-second DEM", "tile publication date",
                "mean of per-pixel slope (central differences, metric spacing) inside parcel",
                "low", "post-construction grading can lower measured slope on later lidar"),
    FeatureSpec("slope_p90_pct", "USGS 3DEP 1 arc-second DEM", "tile publication date",
                "90th percentile of slope inside parcel", "low"),
    FeatureSpec("ghi_kwh_m2_day", "NASA POWER climatology ALLSKY_SFC_SW_DWN", "multi-year mean",
                "bilinear interpolation of the 0.5 x 0.625 deg grid at the parcel centroid",
                "none", "coarse grid: captures the statewide gradient, not local shading"),
    FeatureSpec("dist_tx_km", "HIFLD Transmission Lines (archived 2025)", "2025",
                "distance from parcel boundary to nearest line, any voltage", "high",
                "includes generator tie lines built FOR later facilities; see dist_tx_robust_km"),
    FeatureSpec("dist_tx_robust_km", "HIFLD Transmission Lines (archived 2025)", "2025",
                "as dist_tx_km but excluding short lines (< 15 km) that end within 2 km of "
                "any USPVDB footprint (probable generator ties)", "moderate",
                "conservative: also removes some genuine pre-existing lines near facilities"),
    FeatureSpec("dist_tx_hv_km", "HIFLD Transmission Lines (archived 2025)", "2025",
                "distance to nearest line of 230 kV or more (robust set)", "moderate"),
    FeatureSpec("dist_road_km", "TIGER/Line 2016 primary and secondary roads", "2016",
                "distance from parcel boundary to nearest primary or secondary road", "none"),
    FeatureSpec("whp_mean", "USFS Wildfire Hazard Potential", "2018",
                "mean continuous WHP inside parcel (270 m, all touched)", "low"),
    FeatureSpec("floodplain_frac", "FEMA NFHL flood hazard zones (SFHA)", "effective at fetch",
                "fraction of parcel area in a Special Flood Hazard Area", "low",
                "missing (NaN) where NFHL has no effective mapping"),
    FeatureSpec("protected_overlap_frac", "PAD-US 3.0 Fee + Easement", "2022",
                "fraction of parcel area in protected fee or easement land (union)", "low"),
    FeatureSpec("log_area_ha", "TxGIO StratMap 2026 parcels", "2026",
                "log10 of parcel area in EPSG:5070", "low",
                "2026 parcel boundaries; lease parcels are rarely re-platted"),
    *[FeatureSpec(f"nlcd_{g}_frac", "Annual NLCD land cover", "2016",
                  f"fraction of parcel pixels in NLCD group '{g}'", "none")
      for g in ("developed", "cropland", "pasture", "shrub_grass", "forest", "wetland_water",
                "barren")],
]}

NLCD_GROUPS = {
    "developed": (21, 22, 23, 24), "cropland": (82,), "pasture": (81,),
    "shrub_grass": (52, 71), "forest": (41, 42, 43), "wetland_water": (11, 90, 95),
    "barren": (31,),
}

# ---------------------------------------------------------------------------- helpers


def vsizip_member(zpath: Path, pick: Callable[[str], bool]) -> str:
    """GDAL /vsizip path to the first member of ``zpath`` accepted by ``pick``."""
    with zipfile.ZipFile(zpath) as zf:
        names = [n for n in zf.namelist() if pick(n)]
    if not names:
        raise FileNotFoundError(f"no matching member in {zpath}")
    names.sort(key=len)
    return f"/vsizip/{zpath.as_posix()}/{names[0].rstrip('/')}"


def slope_stats(geom_ll: shapely.Geometry, tiles: Iterable[str],
                opener: Callable[[str], rasterio.io.DatasetReader]) -> dict[str, float]:
    """Slope statistics (percent rise) for one parcel from geographic DEM tiles."""
    srcs = [opener(t) for t in tiles]
    srcs = [s for s in srcs if s is not None]
    nan = {"slope_mean_pct": math.nan, "slope_p90_pct": math.nan}
    if not srcs:
        return nan
    x0, y0, x1, y1 = geom_ll.bounds
    res = abs(srcs[0].res[0])
    pad = 2 * res
    bounds = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    if len(srcs) == 1:
        s = srcs[0]
        win = _pixel_window(bounds, s.transform)
        arr = s.read(1, window=win, boundless=True, fill_value=np.nan,
                     out_dtype="float32")
        transform = s.window_transform(win)
        nodata = s.nodata
    else:
        arr, transform = rasterio.merge.merge(srcs, bounds=bounds, nodata=np.nan,
                                              dtype="float32")
        arr = arr[0]
        nodata = None
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        return nan
    lat = (y0 + y1) / 2
    dx = abs(transform.a) * 111_320.0 * math.cos(math.radians(lat))
    dy = abs(transform.e) * 110_574.0
    gy, gx = np.gradient(arr, dy, dx)
    slope = 100.0 * np.hypot(gx, gy)
    mask = rasterio.features.geometry_mask([geom_ll], arr.shape, transform, invert=True,
                                           all_touched=True)
    v = slope[mask & np.isfinite(slope)]
    if v.size == 0:
        return nan
    return {"slope_mean_pct": float(v.mean()), "slope_p90_pct": float(np.percentile(v, 90))}


def _pixel_window(bounds, transform) -> rasterio.windows.Window:
    """Integer window covering ``bounds`` (outward rounding, at least one pixel)."""
    w = rasterio.windows.from_bounds(*bounds, transform=transform)
    c0, r0 = math.floor(w.col_off), math.floor(w.row_off)
    c1, r1 = math.ceil(w.col_off + w.width), math.ceil(w.row_off + w.height)
    return rasterio.windows.Window(c0, r0, max(1, c1 - c0), max(1, r1 - r0))


def _window_read(src, geom, all_touched: bool):
    win = _pixel_window(geom.bounds, src.transform)
    arr = src.read(1, window=win, boundless=True, fill_value=src.nodata or 0)
    tr = src.window_transform(win)
    mask = rasterio.features.geometry_mask([geom], arr.shape, tr, invert=True,
                                           all_touched=all_touched)
    if not mask.any():  # sliver smaller than a pixel: fall back to touched pixels
        mask = rasterio.features.geometry_mask([geom], arr.shape, tr, invert=True,
                                               all_touched=True)
    return arr, mask


def zonal_classes(src, geom, groups: dict[str, tuple[int, ...]]) -> dict[str, float]:
    """Fraction of parcel pixels in each class group (geom in the raster's CRS)."""
    arr, mask = _window_read(src, geom, all_touched=False)
    vals = arr[mask]
    if src.nodata is not None:
        vals = vals[vals != src.nodata]
    n = vals.size
    out = {}
    for name, codes in groups.items():
        out[f"nlcd_{name}_frac"] = float(np.isin(vals, codes).sum() / n) if n else math.nan
    return out


def zonal_mean(src, geom, valid: Callable[[np.ndarray], np.ndarray] | None = None) -> float:
    """Mean raster value inside the parcel (all touched), ignoring nodata."""
    arr, mask = _window_read(src, geom, all_touched=True)
    vals = arr[mask].astype("float64")
    if src.nodata is not None:
        vals = vals[vals != src.nodata]
    if valid is not None:
        vals = vals[valid(vals)]
    return float(vals.mean()) if vals.size else math.nan


def nearest_distance_km(geoms: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Distance (km) from each geometry to the nearest target geometry (same metric CRS)."""
    if len(targets) == 0:
        return np.full(len(geoms), np.nan)
    tree = shapely.STRtree(targets)
    idx, dist = tree.query_nearest(geoms, return_distance=True, all_matches=False)
    out = np.full(len(geoms), np.nan)
    out[idx[0]] = dist / 1000.0
    return out


def overlap_fraction(geoms: np.ndarray, polys: np.ndarray) -> np.ndarray:
    """Fraction of each geometry's area covered by the union of ``polys``."""
    out = np.zeros(len(geoms))
    if len(polys) == 0:
        return out
    tree = shapely.STRtree(polys)
    gi, pi = tree.query(geoms, predicate="intersects")
    if len(gi) == 0:
        return out
    df = pd.DataFrame({"g": gi, "p": pi})
    for g, grp in df.groupby("g"):
        inter = shapely.union_all(shapely.intersection(geoms[g], polys[grp["p"].to_numpy()]))
        out[g] = shapely.area(inter) / shapely.area(geoms[g])
    return np.clip(out, 0.0, 1.0)


def interp_grid(points: list[dict], key: str, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a regular lon/lat point grid (NaN outside)."""
    from scipy.interpolate import RegularGridInterpolator

    df = pd.DataFrame([p for p in points if key in p])
    xs = np.sort(df["lon"].unique())
    ys = np.sort(df["lat"].unique())
    grid = df.pivot_table(index="lat", columns="lon", values=key).reindex(index=ys, columns=xs)
    f = RegularGridInterpolator((ys, xs), grid.to_numpy(), bounds_error=False,
                                fill_value=np.nan)
    return f(np.column_stack([lat, lon]))


# ---------------------------------------------------------------------------- loaders


def load_power(raw: Path = RAW) -> list[dict]:
    return json.loads((raw / "power" / "ghi_climatology_tx.json").read_text())["points"]


def to_metric(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.to_crs(EQUAL_AREA) if gdf.crs and gdf.crs.to_string() != EQUAL_AREA else gdf


def to_geographic(geoms_metric: gpd.GeoSeries) -> gpd.GeoSeries:
    return geoms_metric.to_crs(GEOGRAPHIC)

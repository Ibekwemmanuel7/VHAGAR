"""USPVDB facility footprints for Texas, in an equal-area CRS."""
from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd

from vhagar.parcel.study.paths import EQUAL_AREA, RAW

__all__ = ["load_facilities", "LABEL_SOURCE"]

LABEL_SOURCE = "USGS/LBNL U.S. Large-Scale Solar Photovoltaic Database (USPVDB) v4.0"

_KEEP = ["case_id", "p_name", "p_year", "p_cap_ac", "p_area", "p_sys_type", "p_county",
         "p_pwr_reg", "p_type", "eia_id", "ylat", "xlong"]


def _inner_path(zpath: Path, suffixes: tuple[str, ...]) -> str:
    with zipfile.ZipFile(zpath) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(suffixes)]
    if not names:
        raise FileNotFoundError(f"no {suffixes} inside {zpath}")
    return f"/vsizip/{zpath.as_posix()}/{names[0]}"


def load_facilities(raw: Path = RAW, state: str = "TX") -> gpd.GeoDataFrame:
    """Ground-mounted USPVDB facilities in ``state`` as footprints in EPSG:5070.

    Rooftop systems are excluded (they are not land-use decisions). Geometries are
    repaired with ``make_valid`` so overlay areas are well defined."""
    src = _inner_path(raw / "uspvdb" / "uspvdbGeoJSON.zip", (".geojson", ".json"))
    gdf = gpd.read_file(src)
    gdf = gdf[gdf["p_state"] == state].copy()
    if "p_sys_type" in gdf:
        gdf = gdf[gdf["p_sys_type"].fillna("ground").str.lower() != "rooftop"]
    keep = [c for c in _KEEP if c in gdf.columns]
    gdf = gdf[keep + ["geometry"]].to_crs(EQUAL_AREA)
    gdf["geometry"] = gdf.geometry.make_valid()
    gdf["p_year"] = gdf["p_year"].astype(int)
    return gdf.reset_index(drop=True)

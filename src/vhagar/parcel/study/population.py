"""Build the parcel population from the TxGIO county files and attach case labels.

For each county file the parcels are streamed in chunks (to stay within a few GB of
memory), projected to EPSG:5070, repaired, measured, filtered to the minimum size,
de-duplicated (appraisal rolls often stack identical polygons for multiple accounts), and
intersected with the USPVDB footprints. One GeoParquet per county is written to
``data/parcel/work/population/`` together with a per-county accounting record, so every
parcel that leaves the population leaves for a recorded reason.
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely

from vhagar.parcel.study.paths import (
    CASE_MIN_OVERLAP_FRAC,
    CASE_MIN_OVERLAP_HA,
    EQUAL_AREA,
    GEOGRAPHIC,
    LABEL_FIRST_YEAR,
    LABEL_LAST_YEAR,
    MIN_PARCEL_HA,
    PRE_SNAPSHOT_EXCLUDE_HA,
    RAW,
    SNAPSHOT_YEAR,
    WORK,
)

__all__ = ["CountyStats", "county_layer", "process_county", "label_overlaps", "build_all"]

_CHUNK = 100_000
_COLUMNS = ["prop_id", "fips", "county"]


@dataclass
class CountyStats:
    """Accounting for one county file: where every parcel went."""

    file: str
    county: str | None = None
    fips: str | None = None
    source_crs: str | None = None
    features: int = 0
    empty_or_invalid: int = 0
    below_min_area: int = 0
    duplicate_geometry: int = 0
    pre_snapshot_covered: int = 0
    population: int = 0
    cases: int = 0
    error: str | None = None


def county_layer(zpath: Path) -> str:
    """GDAL path to the parcel layer inside a TxGIO county zip (shapefile or geodatabase)."""
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
    gdb = sorted({n.split(".gdb/")[0] + ".gdb" for n in names if ".gdb/" in n})
    if gdb:
        return f"/vsizip/{zpath.as_posix()}/{gdb[0]}"
    shp = [n for n in names if n.lower().endswith(".shp")]
    if not shp:
        raise FileNotFoundError(f"no shapefile or geodatabase in {zpath.name}")
    shp.sort(key=lambda n: ("lp" not in n.lower(), len(n)))
    return f"/vsizip/{zpath.as_posix()}/{shp[0]}"


def label_overlaps(parcels: gpd.GeoSeries, facilities: gpd.GeoDataFrame) -> pd.DataFrame:
    """Overlap of each parcel with facility footprints, split into the label window, the
    pre-snapshot period, and after the window. Returns one row per parcel (aligned with
    ``parcels``); ``case_year``/``case_facility`` come from the facility with the largest
    in-window overlap (-1 when none)."""
    n = len(parcels)
    case_ha = np.zeros(n)
    pre_ha = np.zeros(n)
    after_ha = np.zeros(n)
    case_year = np.full(n, -1, dtype=np.int64)
    case_fac = np.full(n, -1, dtype=np.int64)
    if n and not facilities.empty:
        tree = shapely.STRtree(facilities.geometry.values)
        pi, fi = tree.query(parcels.values, predicate="intersects")
        if len(pi):
            ha = shapely.area(shapely.intersection(parcels.values[pi],
                                                   facilities.geometry.values[fi])) / 1e4
            years = facilities["p_year"].to_numpy(dtype=np.int64)[fi]
            fids = facilities["case_id"].to_numpy(dtype=np.int64)[fi]
            win = (years >= LABEL_FIRST_YEAR) & (years <= LABEL_LAST_YEAR)
            np.add.at(case_ha, pi[win], ha[win])
            np.add.at(pre_ha, pi[years <= SNAPSHOT_YEAR], ha[years <= SNAPSHOT_YEAR])
            np.add.at(after_ha, pi[years > LABEL_LAST_YEAR], ha[years > LABEL_LAST_YEAR])
            order = np.argsort(-ha[win], kind="stable")
            wp, wy, wf = pi[win][order], years[win][order], fids[win][order]
            _, first = np.unique(wp, return_index=True)
            case_year[wp[first]] = wy[first]
            case_fac[wp[first]] = wf[first]
    return pd.DataFrame({"overlap_case_ha": case_ha, "overlap_pre_ha": pre_ha,
                         "overlap_after_ha": after_ha, "case_year": case_year,
                         "case_facility": case_fac})


def process_county(zpath: Path, facilities: gpd.GeoDataFrame, out_dir: Path) -> CountyStats:
    """Process one county zip into ``out_dir/<stem>.parquet``. Never raises: errors are
    recorded in the returned stats so one bad county file cannot stop the build."""
    st = CountyStats(file=zpath.name)
    try:
        layer = county_layer(zpath)
        info = pyogrio.read_info(layer)
        st.features = int(info["features"])
        st.source_crs = info.get("crs")
        if not st.source_crs:
            raise ValueError("source layer has no CRS")
        by_lower = {f.lower(): f for f in info["fields"]}
        avail = [by_lower[c] for c in _COLUMNS if c in by_lower]
        parts: list[gpd.GeoDataFrame] = []
        seen: set[int] = set()
        for start in range(0, st.features, _CHUNK):
            g = pyogrio.read_dataframe(layer, columns=avail, skip_features=start,
                                       max_features=_CHUNK)
            g = g.rename(columns={c: c.lower() for c in avail})
            g = g[~g.geometry.isna()]
            g = g.to_crs(EQUAL_AREA)
            geom = shapely.make_valid(g.geometry.values)
            area_ha = shapely.area(geom) / 1e4
            ok = np.isfinite(area_ha) & (area_ha > 0)
            st.empty_or_invalid += int((~ok).sum())
            big = ok & (area_ha >= MIN_PARCEL_HA)
            st.below_min_area += int((ok & ~big).sum())
            g = g[big].copy()
            g["geometry"] = geom[big]
            g["area_ha"] = area_ha[big]
            # De-duplicate stacked polygons: identical geometry at 0.1 m precision.
            keys = pd.util.hash_array(shapely.to_wkb(
                shapely.set_precision(g.geometry.values, 0.1), hex=True).astype(object))
            first = ~pd.Series(keys).duplicated().to_numpy()
            fresh = np.array([k not in seen for k in keys]) & first
            seen.update(keys[fresh].tolist())
            st.duplicate_geometry += int((~fresh).sum())
            parts.append(g[fresh])
        pop = pd.concat(parts, ignore_index=True) if parts else gpd.GeoDataFrame(
            columns=[*[c.lower() for c in avail], "geometry", "area_ha"], geometry="geometry",
            crs=EQUAL_AREA)
        pop = gpd.GeoDataFrame(pop, geometry="geometry", crs=EQUAL_AREA)
        lab = label_overlaps(pop.geometry, facilities)
        pop = pd.concat([pop.reset_index(drop=True), lab], axis=1)
        pre = pop["overlap_pre_ha"] >= PRE_SNAPSHOT_EXCLUDE_HA
        st.pre_snapshot_covered = int(pre.sum())
        pop = pop[~pre].reset_index(drop=True)
        pop["is_case"] = ((pop["overlap_case_ha"] >= CASE_MIN_OVERLAP_HA)
                          | (pop["overlap_case_ha"] >= CASE_MIN_OVERLAP_FRAC * pop["area_ha"]))
        pop.loc[~pop["is_case"], ["case_year", "case_facility"]] = -1
        cent = shapely.centroid(pop.geometry.values)
        pop["cx"] = shapely.get_x(cent)
        pop["cy"] = shapely.get_y(cent)
        ll = gpd.GeoSeries(cent, crs=EQUAL_AREA).to_crs(GEOGRAPHIC)
        pop["lon"], pop["lat"] = ll.x.to_numpy(), ll.y.to_numpy()
        stem = zpath.stem
        pop["pid"] = [f"{stem}:{i}" for i in range(len(pop))]
        if "fips" in pop and len(pop):
            st.fips = str(pop["fips"].mode().iloc[0])
        if "county" in pop and len(pop):
            st.county = str(pop["county"].mode().iloc[0])
        st.population = len(pop)
        st.cases = int(pop["is_case"].sum())
        out_dir.mkdir(parents=True, exist_ok=True)
        pop.to_parquet(out_dir / f"{stem}.parquet", index=False)
    except Exception as exc:  # recorded, not raised
        st.error = f"{type(exc).__name__}: {exc}"
    return st


def build_all(facilities: gpd.GeoDataFrame, raw: Path = RAW, work: Path = WORK,
              only: list[str] | None = None, log=print) -> pd.DataFrame:
    """Process every county zip (resuming past finished ones) and return the accounting."""
    out_dir = work / "population"
    acct_path = work / "population_accounting.json"
    acct: dict[str, dict] = (json.loads(acct_path.read_text()) if acct_path.exists() else {})
    zips = sorted((raw / "txgio").glob("stratmap26-landparcels_48[0-9][0-9][0-9]_lp.zip"))
    if only:
        zips = [z for z in zips if any(o in z.name for o in only)]
    for i, z in enumerate(zips, 1):
        if z.name in acct and not acct[z.name].get("error") and (out_dir / f"{z.stem}.parquet").exists():
            continue
        st = process_county(z, facilities, out_dir)
        acct[z.name] = asdict(st)
        acct_path.write_text(json.dumps(acct, indent=1))
        log(f"[{i}/{len(zips)}] {z.name}: pop={st.population} cases={st.cases} "
            f"dupes={st.duplicate_geometry} err={st.error}")
    return pd.DataFrame(acct.values())

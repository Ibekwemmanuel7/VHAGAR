"""Build and evaluate the real-data Texas solar parcel study.

Run the stages in order (each is resumable and writes into data/parcel/work/)::

    python scripts/parcel_build.py population   # county parcels -> population + case labels
    python scripts/parcel_build.py sample       # case-control draw + stage-2 fetch plan
    #   (then: python scripts/parcel_fetch.py stage2   on a machine with internet)
    python scripts/parcel_build.py features     # vector + raster overlays per sampled parcel
    python scripts/parcel_build.py evaluate     # models, splits, metrics, figures, report JSON

Design and constants: ``src/vhagar/parcel/study/paths.py`` and
``docs/28_PARCEL_STUDY_TEXAS_SOLAR.md``. Requires the ``geo`` and ``gbdt`` extras plus
``duckdb`` (with ``duckdb-extension-spatial`` for the SQL summaries).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhagar.parcel.study import features as F  # noqa: E402
from vhagar.parcel.study import paths as P  # noqa: E402
from vhagar.parcel.study.labels import LABEL_SOURCE, load_facilities  # noqa: E402
from vhagar.parcel.study.population import build_all  # noqa: E402
from vhagar.parcel.study.sampling import (  # noqa: E402
    attach_geometry,
    dem_tile_name,
    draw_sample,
    fetch_plan,
    load_population,
    write_plan,
)

LOG = P.WORK / "build_log.txt"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    P.WORK.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------------------- stages


def stage_population(only: list[str] | None) -> None:
    fac = load_facilities()
    fac.to_parquet(P.WORK / "facilities.parquet")
    log(f"facilities: {len(fac)} ground-mounted TX footprints, years "
        f"{fac.p_year.min()}-{fac.p_year.max()}")
    acct = build_all(fac, only=only, log=log)
    log(f"population: {int(acct['population'].sum())} parcels, {int(acct['cases'].sum())} "
        f"cases, {int(acct['error'].notna().sum())} county errors")


def stage_sample() -> None:
    pop = load_population()
    sample, record = draw_sample(pop)
    fac = gpd.read_parquet(P.WORK / "facilities.parquet")
    cent = fac.geometry.centroid.to_crs(P.GEOGRAPHIC)
    fac_ll = pd.DataFrame({"case_facility": fac["case_id"], "fac_lon": cent.x,
                           "fac_lat": cent.y})
    sample = sample.merge(fac_ll, on="case_facility", how="left")
    sample["block_lon"] = sample["fac_lon"].fillna(sample["lon"])
    sample["block_lat"] = sample["fac_lat"].fillna(sample["lat"])
    geo = attach_geometry(sample)
    geo.to_parquet(P.WORK / "sample.parquet")
    record["case_facilities_in_population"] = int(
        geo.loc[geo.role == "case", "case_facility"].nunique())
    (P.WORK / "sampling_record.json").write_text(json.dumps(record, indent=1))
    plan = fetch_plan(geo)
    write_plan(plan)
    log(f"sample: {record}")
    log(f"fetch plan: {len(plan['nfhl_cells'])} NFHL cells, {len(plan['dem_tiles'])} DEM tiles")


# ---------------------------------------------------------------------------- features


def _unzip_member(zf: zipfile.ZipFile, name: str, out: Path) -> None:
    """Extract one member atomically and verify its size (a partial file is never kept)."""
    size = zf.getinfo(name).file_size
    if out.exists() and out.stat().st_size == size:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    with zf.open(name) as src, part.open("wb") as dst:
        while chunk := src.read(1 << 24):
            dst.write(chunk)
    if part.stat().st_size != size:
        raise OSError(f"extracted {part} has {part.stat().st_size} bytes, expected {size}")
    part.replace(out)


def _extract(zpath: Path, pick, dest_dir: Path) -> Path:
    """Extract the first matching member of a zip once (rasters read faster unzipped)."""
    with zipfile.ZipFile(zpath) as zf:
        names = sorted([n for n in zf.namelist() if pick(n)], key=len)
        if not names:
            raise FileNotFoundError(f"no matching raster in {zpath}")
        out = dest_dir / Path(names[0]).name
        _unzip_member(zf, names[0], out)
    return out


def _extract_tree(zpath: Path, prefix: str, dest_dir: Path) -> Path:
    """Extract every member under ``prefix`` (an ESRI grid is a directory of files)."""
    with zipfile.ZipFile(zpath) as zf:
        members = [n for n in zf.namelist() if n.startswith(prefix) and not n.endswith("/")]
        if not members:
            raise FileNotFoundError(f"{prefix} not in {zpath}")
        for n in members:
            _unzip_member(zf, n, dest_dir / n)
    return dest_dir / prefix


def _read_lines(raw: Path) -> gpd.GeoDataFrame:
    path = raw / "hifld" / "transmission_lines.parquet"
    try:
        g = gpd.read_parquet(path)
    except Exception:
        t = pq.read_table(path).to_pandas()
        gcol = next(c for c in t.columns if c.lower() in ("geometry", "geom", "shape"))
        geom = shapely.from_wkb(t[gcol].to_numpy())
        x = shapely.get_coordinates(geom[:1000])[:, 0]
        crs = "EPSG:3857" if np.nanmax(np.abs(x)) > 180 else "EPSG:4326"
        g = gpd.GeoDataFrame(t.drop(columns=[gcol]), geometry=geom, crs=crs)
    if g.crs is None:
        x = shapely.get_coordinates(g.geometry.values[:1000])[:, 0]
        g = g.set_crs("EPSG:3857" if np.nanmax(np.abs(x)) > 180 else "EPSG:4326")
    if "STATUS" in g:  # keep in-service (and unknown-status) lines only
        g = g[~g["STATUS"].isin(["INACTIVE", "UNDER CONSTRUCTION", "PROPOSED"])]
    vcol = next((c for c in g.columns if c.upper() == "VOLTAGE"), None)
    g["kv"] = pd.to_numeric(g[vcol], errors="coerce") if vcol else np.nan
    g.loc[g["kv"] <= 0, "kv"] = np.nan
    g = g.to_crs(P.EQUAL_AREA)
    tx = gpd.GeoSeries.from_xy([-106.7, -93.5], [25.8, 36.5], crs=P.GEOGRAPHIC).to_crs(
        P.EQUAL_AREA).total_bounds
    return g.cx[tx[0] - 2e5:tx[2] + 2e5, tx[1] - 2e5:tx[3] + 2e5].reset_index(drop=True)


def _robust_lines(lines: gpd.GeoDataFrame, fac: gpd.GeoDataFrame) -> np.ndarray:
    """Drop probable generator-tie lines: shorter than 15 km and ending within 2 km of any
    USPVDB footprint. Returns a boolean keep-mask."""
    length_km = lines.length.to_numpy() / 1000
    ends = shapely.union(shapely.get_point(shapely.line_merge(lines.geometry.values), 0),
                         shapely.get_point(shapely.line_merge(lines.geometry.values), -1))
    near = np.zeros(len(lines), bool)
    tree = shapely.STRtree(fac.geometry.buffer(2000).values)
    li, _ = tree.query(ends, predicate="intersects")
    near[np.unique(li)] = True
    return ~((length_km < 15) & near)


def _sample() -> gpd.GeoDataFrame:
    return gpd.read_parquet(P.WORK / "sample.parquet")


def feat_vector() -> None:
    """Transmission, roads, protected areas, irradiance, parcel size (fast, vectorised)."""
    raw = P.RAW
    s = _sample()
    geoms = s.geometry.values
    fac = gpd.read_parquet(P.WORK / "facilities.parquet")
    feat = pd.DataFrame({"pid": s["pid"].to_numpy()})
    feat["log_area_ha"] = np.log10(s["area_ha"].to_numpy())

    lines = _read_lines(raw)
    keep = _robust_lines(lines, fac)
    hv = keep & (lines["kv"].to_numpy() >= 230)
    feat["dist_tx_km"] = F.nearest_distance_km(geoms, lines.geometry.values)
    feat["dist_tx_robust_km"] = F.nearest_distance_km(geoms, lines.geometry.values[keep])
    feat["dist_tx_hv_km"] = F.nearest_distance_km(geoms, lines.geometry.values[hv])
    log(f"transmission: {len(lines)} in-service lines near TX, {int((~keep).sum())} probable "
        f"generator ties removed, {int(hv.sum())} at >= 230 kV")

    roads = gpd.read_file(F.vsizip_member(raw / "tiger" / "tl_2016_48_prisecroads.zip",
                                          lambda n: n.endswith(".shp"))).to_crs(P.EQUAL_AREA)
    feat["dist_road_km"] = F.nearest_distance_km(geoms, roads.geometry.values)

    pz = raw / "padus" / "PADUS3_0_State_TX_SHP.zip"
    with zipfile.ZipFile(pz) as zf:
        shps = [n for n in zf.namelist() if n.lower().endswith(".shp")
                and any(k in n.lower() for k in ("fee_", "easement_"))]
    prot = pd.concat([gpd.read_file(f"/vsizip/{pz.as_posix()}/{n}").to_crs(P.EQUAL_AREA)
                      for n in shps], ignore_index=True)
    prot = prot[~prot.geometry.isna()]
    feat["protected_overlap_frac"] = F.overlap_fraction(
        geoms, shapely.make_valid(prot.geometry.values))
    log(f"PAD-US: {len(prot)} fee and easement polygons from {shps}")

    feat["ghi_kwh_m2_day"] = F.interp_grid(F.load_power(raw), "ghi_kwh_m2_day",
                                           s["lon"].to_numpy(), s["lat"].to_numpy())
    feat.to_parquet(P.WORK / "feat_vector.parquet", index=False)
    log(f"vector features: {feat.shape}")


def feat_nfhl() -> None:
    """Floodplain share and NFHL mapping status from the stage-2 cells.

    The cells total several GB of GeoJSON, so they are read one at a time and only
    polygons that intersect a sampled parcel are kept (bounded memory)."""
    s = _sample()
    geoms = s.geometry.values
    tree = shapely.STRtree(s.to_crs(P.GEOGRAPHIC).geometry.values)

    def _keep(features: list) -> gpd.GeoDataFrame | None:
        if not features:
            return None
        g = gpd.GeoDataFrame.from_features(features, crs=P.GEOGRAPHIC)
        g = g[~g.geometry.isna()]
        if not len(g):
            return None
        hit = np.unique(tree.query(shapely.make_valid(g.geometry.values),
                                   predicate="intersects")[0])
        return g.iloc[hit] if len(hit) else None

    sfha_parts, avail_parts = [], []
    cells = sorted((P.RAW / "nfhl").glob("cell_*.geojson"))
    for i, cf in enumerate(cells, 1):
        j = json.loads(cf.read_text(encoding="utf-8"))
        for feats, parts in ((j["sfha"]["features"], sfha_parts),
                             (j["availability"]["features"], avail_parts)):
            k = _keep(feats)
            if k is not None:
                parts.append(k)
        del j
        if i % 100 == 0:
            log(f"NFHL: read {i}/{len(cells)} cells")

    def _cat(parts: list) -> gpd.GeoDataFrame:
        if not parts:
            return gpd.GeoDataFrame(geometry=[], crs=P.EQUAL_AREA)
        return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True),
                                crs=P.GEOGRAPHIC).to_crs(P.EQUAL_AREA)

    gs, ga = _cat(sfha_parts), _cat(avail_parts)
    if len(gs):  # adjacent cells return the same polygon; keep one copy
        gs = gs.loc[~pd.Series(shapely.to_wkb(gs.geometry.values)).duplicated().to_numpy()]
    if len(ga):
        ga = ga.loc[~pd.Series(shapely.to_wkb(ga.geometry.values)).duplicated().to_numpy()]
    mapped = F.overlap_fraction(geoms, shapely.make_valid(ga.geometry.values)) >= 0.5
    ff = F.overlap_fraction(geoms, shapely.make_valid(gs.geometry.values))
    out = pd.DataFrame({"pid": s["pid"].to_numpy(), "nfhl_mapped": mapped,
                        "floodplain_frac": np.where(mapped, ff, np.nan)})
    out.to_parquet(P.WORK / "feat_nfhl.parquet", index=False)
    log(f"NFHL: {len(cells)} cells, {len(gs)} SFHA polygons, {len(ga)} availability polygons; "
        f"{mapped.mean():.1%} of sampled parcels on effective NFHL mapping")


def _chunked(name: str, compute, budget_s: float, chunk: int = 1500) -> bool:
    """Run ``compute(sample_slice) -> DataFrame`` over the sample in checkpointed chunks
    until the time budget is spent. Returns True when every chunk is done."""
    s = _sample()
    d = P.WORK / f"feat_{name}"
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_chunks = (len(s) + chunk - 1) // chunk
    for k in range(n_chunks):
        out = d / f"part_{k:04d}.parquet"
        if out.exists():
            continue
        if time.time() - t0 > budget_s:
            log(f"{name}: paused at chunk {k}/{n_chunks} (time budget)")
            return False
        part = s.iloc[k * chunk:(k + 1) * chunk]
        df = compute(part)
        df.insert(0, "pid", part["pid"].to_numpy())
        df.to_parquet(out, index=False)
        log(f"{name}: chunk {k + 1}/{n_chunks} done ({time.time() - t0:.0f} s)")
    return True


def _whp_path() -> Path:
    for p in (P.RAW / "whp" / "whp2018_cnt",
              P.RAW / "whp" / "Data" / "whp_2018_continuous" / "whp2018_cnt"):
        if (p / "hdr.adf").exists():
            return p
    raise FileNotFoundError("WHP grid not extracted; run parcel_fetch.py stage2")


def _nlcd_path() -> Path:
    p = P.RAW / "nlcd" / "Annual_NLCD_LndCov_2016_CU_C1V1.tif"
    with zipfile.ZipFile(P.RAW / "nlcd" / "Annual_NLCD_LndCov_2016_CU.zip") as zf:
        size = zf.getinfo(p.name).file_size
    if not p.exists() or p.stat().st_size != size:
        raise FileNotFoundError("NLCD GeoTIFF not fully extracted; run parcel_fetch.py stage2")
    return p


def feat_rasters(budget_s: float) -> bool:
    nlcd = rasterio.open(_nlcd_path())
    whp = rasterio.open(_whp_path())

    def compute(part: gpd.GeoDataFrame) -> pd.DataFrame:
        gn = part.geometry.to_crs(nlcd.crs).values
        gw = part.geometry.to_crs(whp.crs).values
        rows = []
        for a, b in zip(gn, gw, strict=True):
            r = F.zonal_classes(nlcd, a, F.NLCD_GROUPS)
            r["whp_mean"] = F.zonal_mean(whp, b, valid=lambda v: v >= 0)
            rows.append(r)
        return pd.DataFrame(rows)

    return _chunked("rasters", compute, budget_s)


def feat_slope(budget_s: float) -> bool:
    cache: dict[str, object] = {}

    def opener(tag: str):
        if tag not in cache:
            p = P.RAW / "dem" / f"{tag}.tif"
            cache[tag] = rasterio.open(p) if p.exists() else None
        return cache[tag]

    def compute(part: gpd.GeoDataFrame) -> pd.DataFrame:
        rows = []
        for g in part.geometry.to_crs(P.GEOGRAPHIC).values:
            x0, y0, x1, y1 = g.bounds
            tiles = sorted({dem_tile_name(x, y) for x in (x0, x1) for y in (y0, y1)})
            rows.append(F.slope_stats(g, tiles, opener))
        return pd.DataFrame(rows)

    return _chunked("slope", compute, budget_s, chunk=1000)


def feat_merge() -> None:
    s = _sample()
    feat = pd.DataFrame({"pid": s["pid"].to_numpy()})
    for name in ("feat_vector.parquet", "feat_nfhl.parquet"):
        feat = feat.merge(pd.read_parquet(P.WORK / name), on="pid", how="left")
    for name in ("rasters", "slope"):
        parts = sorted((P.WORK / f"feat_{name}").glob("part_*.parquet"))
        feat = feat.merge(pd.concat([pd.read_parquet(p) for p in parts]), on="pid", how="left")
    feat["whp_pct"] = feat["whp_mean"].rank(pct=True) * 100.0
    feat.to_parquet(P.WORK / "features.parquet", index=False)
    miss = {c: round(float(feat[c].isna().mean()), 4) for c in feat.columns
            if feat[c].isna().any()}
    log(f"features merged: {feat.shape}; missing share: {json.dumps(miss)}")


# ---------------------------------------------------------------------------- evaluate


def stage_evaluate(n_boot: int) -> None:
    from vhagar.parcel.study.report import run_evaluation

    s = gpd.read_parquet(P.WORK / "sample.parquet")
    feat = pd.read_parquet(P.WORK / "features.parquet")
    record = json.loads((P.WORK / "sampling_record.json").read_text())
    df = s.drop(columns="geometry").merge(feat, on="pid", how="inner")
    df["y"] = (df["role"] == "case").astype(int)
    results = run_evaluation(df, record, n_boot=n_boot, log=log,
                             facilities=gpd.read_parquet(P.WORK / "facilities.parquet"),
                             label_source=LABEL_SOURCE)
    log(f"evaluation written: {results}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["population", "sample", "features", "evaluate"])
    ap.add_argument("--part", choices=["vector", "nfhl", "rasters", "slope", "merge"],
                    help="features: which part to build (rasters/slope resume in chunks)")
    ap.add_argument("--budget", type=float, default=95.0,
                    help="features rasters/slope: seconds to work before checkpointing")
    ap.add_argument("--only", nargs="*", help="population: restrict to county files")
    ap.add_argument("--n-boot", type=int, default=300)
    a = ap.parse_args()
    P.WORK.mkdir(parents=True, exist_ok=True)
    log(f"=== {a.stage} ===")
    if a.stage == "population":
        stage_population(a.only)
    elif a.stage == "sample":
        stage_sample()
    elif a.stage == "features":
        {"vector": feat_vector, "nfhl": feat_nfhl, "merge": feat_merge,
         "rasters": lambda: feat_rasters(a.budget),
         "slope": lambda: feat_slope(a.budget)}[a.part or "vector"]()
    else:
        stage_evaluate(a.n_boot)


if __name__ == "__main__":
    main()

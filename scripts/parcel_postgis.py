"""Mirror the Texas solar parcel study into PostGIS and check the SQL features.

Start the database first (see docs/29_POSTGIS_PARCEL_STUDY.md)::

    docker compose -f db/compose.postgis.yml up -d

Then run the stages in order, or all of them at once::

    python scripts/parcel_postgis.py load       # sample + lines + facilities + PAD-US -> PostGIS
    python scripts/parcel_postgis.py features   # tie rule + 3 features recomputed in SQL
    python scripts/parcel_postgis.py check      # SQL vs GeoPandas, parcel by parcel
    python scripts/parcel_postgis.py screen     # customer-style screen in SQL
    python scripts/parcel_postgis.py all

The connection string comes from --dsn, else VHAGAR_PG_DSN, else
postgresql://vhagar:vhagar@localhost:5433/vhagar (the compose file's defaults).
Results go to outputs/parcel_study/postgis_check.json.
Requires the study's work files (run scripts/parcel_build.py features first) and
psycopg 3 (pip install -e .[postgis]).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import parcel_build as B  # noqa: E402  (reuses the study's own loaders)

from vhagar.parcel.study import paths as P  # noqa: E402
from vhagar.parcel.study import postgis as PG  # noqa: E402

REPORT = P.OUT / "postgis_check.json"


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}", flush=True)


def _report() -> dict:
    if REPORT.exists():
        return json.loads(REPORT.read_text(encoding="utf-8"))
    return {"study_version": P.STUDY_VERSION}


def _save(rep: dict) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=2), encoding="utf-8")


def _lines() -> gpd.GeoDataFrame:
    return B._read_lines(P.RAW)


def _protected() -> gpd.GeoSeries:
    import zipfile
    pz = P.RAW / "padus" / "PADUS3_0_State_TX_SHP.zip"
    with zipfile.ZipFile(pz) as zf:
        shps = [n for n in zf.namelist() if n.lower().endswith(".shp")
                and any(k in n.lower() for k in ("fee_", "easement_"))]
    prot = pd.concat([gpd.read_file(f"/vsizip/{pz.as_posix()}/{n}").to_crs(P.EQUAL_AREA)
                      for n in shps], ignore_index=True)
    return prot.geometry[~prot.geometry.isna()]


def stage_load(conn) -> None:
    s = B._sample()
    feats = pd.read_parquet(P.WORK / "features.parquet")
    s = s.merge(feats[["pid", "floodplain_frac", "slope_mean_pct"]], on="pid", how="left")
    fac = gpd.read_parquet(P.WORK / "facilities.parquet").to_crs(P.EQUAL_AREA)
    lines = _lines()
    prot = _protected()
    t0 = time.time()
    counts = PG.load_tables(conn, s, s.geometry.values, lines["kv"].to_numpy(),
                            lines.geometry.values, fac.geometry.values, prot.values)
    log(f"loaded {counts} in {time.time() - t0:.0f} s")
    rep = _report()
    rep["load"] = {"counts": counts, "seconds": round(time.time() - t0, 1)}
    _save(rep)


def stage_features(conn) -> None:
    t0 = time.time()
    ties = PG.apply_tie_rule(conn)
    log(f"tie rule in SQL: {ties}")
    t1 = time.time()
    df = PG.compute_features(conn)
    log(f"features_sql: {len(df)} parcels in {time.time() - t1:.0f} s")
    rep = _report()
    rep["features"] = {"tie_rule": ties, "rows": len(df),
                       "seconds": round(time.time() - t0, 1)}
    _save(rep)


def stage_check(conn) -> None:
    sql = PG.read_features(conn)
    py = pd.read_parquet(P.WORK / "features.parquet",
                         columns=["pid", "dist_tx_robust_km", "dist_tx_hv_km",
                                  "protected_overlap_frac"])
    sql["pid"] = sql["pid"].astype(str)
    py["pid"] = py["pid"].astype(str)
    res = PG.compare(sql, py)

    # Line-by-line agreement of the generator-tie rule (SQL vs shapely).
    lines = _lines()
    fac = gpd.read_parquet(P.WORK / "facilities.parquet").to_crs(P.EQUAL_AREA)
    keep_py = B._robust_lines(lines, fac)
    keep_sql = PG.robust_flags(conn)
    res["tie_rule"] = {"lines": int(len(keep_py)),
                       "removed_py": int((~keep_py).sum()),
                       "removed_sql": int((~keep_sql).sum()),
                       "disagreements": int((keep_py != keep_sql).sum())}

    for f, v in res["features"].items():
        log(f"{f:24s} within tolerance {v['within_tolerance_share']:.4%}  "
            f"mismatches {v['mismatches']}  max |diff| {v['max_abs_diff']:.6f}")
    log(f"tie rule: {res['tie_rule']}")
    log("ALL FEATURES AGREE" if res["all_within_tolerance"] and
        res["tie_rule"]["disagreements"] == 0 else "DIFFERENCES FOUND: see " + str(REPORT))
    rep = _report()
    rep["check"] = res
    _save(rep)


def stage_screen(conn) -> None:
    df = PG.run_screen(conn)
    log("screen (sample rows, unweighted):\n" + df.to_string(index=False))
    rep = _report()
    rep["screen"] = {"params": PG.SCREEN_DEFAULTS,
                     "note": "counts are case-control sample rows, not population totals",
                     "rows": json.loads(df.to_json(orient="records"))}
    _save(rep)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["load", "features", "check", "screen", "all"])
    ap.add_argument("--dsn", help="PostgreSQL connection string (else VHAGAR_PG_DSN)")
    a = ap.parse_args()
    with PG.connect(a.dsn) as conn:
        ver = conn.execute("SELECT postgis_full_version()").fetchone()[0].split(" ")[0:2]
        log(f"connected: {' '.join(ver)}")
        stages = (["load", "features", "check", "screen"] if a.stage == "all" else [a.stage])
        for st in stages:
            log(f"=== {st} ===")
            {"load": stage_load, "features": stage_features, "check": stage_check,
             "screen": stage_screen}[st](conn)
    log(f"report: {REPORT}")


if __name__ == "__main__":
    main()

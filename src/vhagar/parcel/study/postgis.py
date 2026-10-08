"""PostGIS mirror of the parcel study's vector features.

Loads the study sample, transmission lines, solar facility footprints and PAD-US
protected areas into PostGIS (EPSG:5070, GiST-indexed) and recomputes three features in
SQL, so that they can be checked parcel by parcel against the GeoPandas/shapely pipeline:

* ``dist_tx_robust_km``: distance to the nearest in-service line after removing probable
  generator ties. The tie rule (shorter than 15 km and ending within 2 km of a USPVDB
  footprint) is itself evaluated in SQL.
* ``dist_tx_hv_km``: the same for lines of 230 kV or more.
* ``protected_overlap_frac``: share of parcel area in PAD-US fee or easement land (union,
  so overlapping fee and easement polygons are not counted twice).

It also runs a customer-style screen in SQL (see ``SCREEN_SQL``).

``psycopg`` (v3) is imported lazily, so this module imports without a database driver and
the unit test suite never needs one. Integration tests live in
``tests/test_parcel_postgis.py`` and run only when ``VHAGAR_PG_DSN`` is set.
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import shapely

SRID = 5070  # CONUS Albers equal-area, metres (same as paths.EQUAL_AREA)
SCHEMA = "parcel_study"
DEFAULT_DSN = "postgresql://vhagar:vhagar@localhost:5433/vhagar"

# Tie-rule constants, identical to scripts/parcel_build.py::_robust_lines.
TIE_MAX_LENGTH_M = 15_000.0
TIE_END_BUFFER_M = 2_000.0
HV_KV = 230.0

DDL = """
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE SCHEMA IF NOT EXISTS {s};
DROP TABLE IF EXISTS {s}.parcels, {s}.tx_lines, {s}.facilities, {s}.protected,
                     {s}.features_sql CASCADE;
CREATE TABLE {s}.parcels (
    pid             text PRIMARY KEY,
    county          text,
    area_ha         double precision,
    is_case         boolean,
    floodplain_frac double precision,   -- NULL where FEMA NFHL has no effective mapping
    slope_mean_pct  double precision,
    geom            geometry(Geometry, {srid}) NOT NULL
);
CREATE TABLE {s}.tx_lines (
    line_id integer PRIMARY KEY,
    kv      double precision,           -- NULL when HIFLD voltage is unknown
    robust  boolean,                    -- set by TIE_SQL
    geom    geometry(Geometry, {srid}) NOT NULL
);
CREATE TABLE {s}.facilities (
    fac_id integer PRIMARY KEY,
    geom   geometry(Geometry, {srid}) NOT NULL
);
CREATE TABLE {s}.protected (
    prot_id integer PRIMARY KEY,
    geom    geometry(Geometry, {srid}) NOT NULL
);
"""

INDEX_SQL = """
CREATE INDEX ON {s}.parcels    USING gist (geom);
CREATE INDEX ON {s}.tx_lines   USING gist (geom);
CREATE INDEX ON {s}.facilities USING gist (geom);
CREATE INDEX ON {s}.protected  USING gist (geom);
ANALYZE {s}.parcels; ANALYZE {s}.tx_lines; ANALYZE {s}.facilities; ANALYZE {s}.protected;
"""

# Probable generator ties: short lines with an end point near a solar footprint.
# Parity note: shapely.get_point returns None for a multi-part line that does not merge
# into one LineString, so the Python rule never flags such lines. PostGIS 3.4
# ST_StartPoint would instead return the first part's first point, so the SQL restricts
# the end-point test to merged single LineStrings to match the study exactly. (Known
# limitation of the rule in both implementations: unmergeable multi-part ties are kept.)
TIE_SQL = """
UPDATE {s}.tx_lines l
SET robust = NOT (
    ST_Length(l.geom) < {max_len}
    AND GeometryType(ST_LineMerge(l.geom)) = 'LINESTRING'
    AND EXISTS (
        SELECT 1 FROM {s}.facilities f
        WHERE ST_DWithin(f.geom, ST_StartPoint(ST_LineMerge(l.geom)), {buf})
           OR ST_DWithin(f.geom, ST_EndPoint(ST_LineMerge(l.geom)), {buf})
    )
);
CREATE INDEX IF NOT EXISTS tx_robust_gix ON {s}.tx_lines USING gist (geom) WHERE robust;
CREATE INDEX IF NOT EXISTS tx_hv_gix ON {s}.tx_lines USING gist (geom) WHERE robust AND kv >= {hv};
ANALYZE {s}.tx_lines;
"""

# KNN (<->) narrows to the 8 nearest candidates through the GiST index; ST_Distance then
# gives the exact distance, so the result does not depend on how <-> ranks long lines.
FEATURES_SQL = """
CREATE TABLE {s}.features_sql AS
SELECT
    p.pid,
    (SELECT min(ST_Distance(p.geom, c.geom)) FROM (
        SELECT l.geom FROM {s}.tx_lines l WHERE l.robust
        ORDER BY l.geom <-> p.geom LIMIT 8) c) / 1000.0            AS dist_tx_robust_km,
    (SELECT min(ST_Distance(p.geom, c.geom)) FROM (
        SELECT l.geom FROM {s}.tx_lines l WHERE l.robust AND l.kv >= {hv}
        ORDER BY l.geom <-> p.geom LIMIT 8) c) / 1000.0            AS dist_tx_hv_km,
    LEAST(1.0, GREATEST(0.0, COALESCE((
        SELECT ST_Area(ST_Union(ST_Intersection(p.geom, pr.geom)))
        FROM {s}.protected pr
        WHERE ST_Intersects(p.geom, pr.geom)), 0.0) / ST_Area(p.geom))) AS protected_overlap_frac
FROM {s}.parcels p;
ALTER TABLE {s}.features_sql ADD PRIMARY KEY (pid);
"""

# A customer-style screen: large parcels near high-voltage transmission, mostly outside
# protected land and the mapped floodplain. Unmapped floodplain is reported separately,
# never treated as "not in a floodplain".
SCREEN_SQL = """
WITH screened AS (
    SELECT p.pid, p.county, p.is_case, p.area_ha,
           CASE
             WHEN p.area_ha < %(min_ha)s                       THEN 'too_small'
             WHEN f.protected_overlap_frac >= %(max_protected)s THEN 'protected'
             WHEN f.dist_tx_hv_km > %(max_hv_km)s              THEN 'far_from_hv'
             WHEN p.floodplain_frac IS NULL                    THEN 'pass_flood_unknown'
             WHEN p.floodplain_frac >= %(max_flood)s           THEN 'floodplain'
             ELSE 'pass'
           END AS outcome
    FROM {s}.parcels p JOIN {s}.features_sql f USING (pid)
)
SELECT outcome,
       count(*)                          AS parcels,
       sum(is_case::int)                 AS case_parcels,
       round(sum(area_ha))::float8       AS sample_area_ha
FROM screened
GROUP BY outcome
ORDER BY outcome;
"""

SCREEN_DEFAULTS = {"min_ha": 40.0, "max_protected": 0.10, "max_hv_km": 5.0, "max_flood": 0.10}


def _psycopg():
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - only without the optional driver
        raise RuntimeError("PostGIS steps need psycopg 3: pip install 'psycopg[binary]' "
                           "(or pip install -e .[postgis])") from exc
    return psycopg


def dsn_from_env(dsn: str | None = None) -> str:
    return dsn or os.environ.get("VHAGAR_PG_DSN") or DEFAULT_DSN


def connect(dsn: str | None = None):
    return _psycopg().connect(dsn_from_env(dsn), autocommit=True)


def _ewkb(geoms: Sequence) -> np.ndarray:
    """Hex EWKB in EPSG:5070. Z is dropped: some TxGIO parcels carry Z = 0, the columns
    are 2D, and shapely's distances and areas (the Python side) ignore Z anyway."""
    g = shapely.force_2d(np.asarray(geoms, dtype=object))
    return shapely.to_wkb(shapely.set_srid(g, SRID), hex=True, include_srid=True)


def _clean(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v


def _copy(cur, table: str, columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> int:
    n = 0
    with cur.copy(f"COPY {table} ({', '.join(columns)}) FROM STDIN") as cp:
        for r in rows:
            cp.write_row([_clean(v) for v in r])
            n += 1
    return n


def create_schema(conn, schema: str = SCHEMA) -> None:
    conn.execute(DDL.format(s=schema, srid=SRID))


def load_tables(conn, parcels: pd.DataFrame, parcel_geoms: Sequence,
                lines_kv: Sequence, line_geoms: Sequence,
                facility_geoms: Sequence, protected_geoms: Sequence,
                schema: str = SCHEMA) -> dict[str, int]:
    """(Re)create the schema and bulk-load all four tables with COPY.

    ``parcels`` needs columns pid, county, area_ha, is_case, floodplain_frac and
    slope_mean_pct (the last two may be NaN). All geometries must already be in EPSG:5070.
    Protected polygons are made valid on the way in, as the Python pipeline does.
    """
    create_schema(conn, schema)
    counts: dict[str, int] = {}
    with conn.cursor() as cur:
        cols = ["pid", "county", "area_ha", "is_case", "floodplain_frac", "slope_mean_pct"]
        df = parcels.reindex(columns=cols)
        counts["parcels"] = _copy(
            cur, f"{schema}.parcels", cols + ["geom"],
            (list(r) + [g] for r, g in zip(df.itertuples(index=False, name=None),
                                           _ewkb(parcel_geoms), strict=True)))
        counts["tx_lines"] = _copy(
            cur, f"{schema}.tx_lines", ["line_id", "kv", "geom"],
            ((i, kv, g) for i, (kv, g) in enumerate(zip(lines_kv, _ewkb(line_geoms), strict=True))))
        counts["facilities"] = _copy(
            cur, f"{schema}.facilities", ["fac_id", "geom"],
            enumerate(_ewkb(facility_geoms)))
        prot = shapely.make_valid(np.asarray(protected_geoms, dtype=object))
        counts["protected"] = _copy(
            cur, f"{schema}.protected", ["prot_id", "geom"], enumerate(_ewkb(prot)))
    conn.execute(INDEX_SQL.format(s=schema))
    return counts


def apply_tie_rule(conn, schema: str = SCHEMA) -> dict[str, int]:
    conn.execute(TIE_SQL.format(s=schema, max_len=TIE_MAX_LENGTH_M, buf=TIE_END_BUFFER_M,
                                hv=HV_KV))
    row = conn.execute(
        f"SELECT count(*) FILTER (WHERE NOT robust), "
        f"count(*) FILTER (WHERE robust AND kv >= {HV_KV}), count(*) FROM {schema}.tx_lines"
    ).fetchone()
    return {"ties_removed": row[0], "hv_lines": row[1], "lines": row[2]}


def robust_flags(conn, schema: str = SCHEMA) -> np.ndarray:
    rows = conn.execute(f"SELECT line_id, robust FROM {schema}.tx_lines ORDER BY line_id"
                        ).fetchall()
    return np.array([r[1] for r in rows], dtype=bool)


def compute_features(conn, schema: str = SCHEMA) -> pd.DataFrame:
    conn.execute(f"DROP TABLE IF EXISTS {schema}.features_sql")
    conn.execute(FEATURES_SQL.format(s=schema, hv=HV_KV))
    return read_features(conn, schema)


def read_features(conn, schema: str = SCHEMA) -> pd.DataFrame:
    rows = conn.execute(
        f"SELECT pid, dist_tx_robust_km, dist_tx_hv_km, protected_overlap_frac "
        f"FROM {schema}.features_sql ORDER BY pid").fetchall()
    return pd.DataFrame(rows, columns=["pid", "dist_tx_robust_km", "dist_tx_hv_km",
                                       "protected_overlap_frac"])


def run_screen(conn, schema: str = SCHEMA, **params: float) -> pd.DataFrame:
    p = {**SCREEN_DEFAULTS, **params}
    with conn.cursor() as cur:
        cur.execute(SCREEN_SQL.format(s=schema), p)
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


FEATURE_TOLERANCE = {"dist_tx_robust_km": 0.001,   # 1 m
                     "dist_tx_hv_km": 0.001,
                     "protected_overlap_frac": 0.001}  # 0.1 percentage point


def compare(sql: pd.DataFrame, py: pd.DataFrame,
            tolerance: dict[str, float] | None = None, worst: int = 5) -> dict[str, Any]:
    """Parcel-by-parcel agreement between SQL and Python feature values."""
    tol = tolerance or FEATURE_TOLERANCE
    m = sql.merge(py, on="pid", suffixes=("_sql", "_py"), how="outer", indicator=True)
    out: dict[str, Any] = {"parcels_sql": int(len(sql)), "parcels_py": int(len(py)),
                           "unmatched_pids": int((m["_merge"] != "both").sum()),
                           "features": {}}
    m = m[m["_merge"] == "both"]
    for f, t in tol.items():
        a = pd.to_numeric(m[f"{f}_sql"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(m[f"{f}_py"], errors="coerce").to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        d = np.abs(a - b)
        d[both_nan] = 0.0
        d[np.isnan(d)] = np.inf  # missing on one side only
        bad = d > t
        idx = np.argsort(-d)[:worst]
        out["features"][f] = {
            "tolerance": t,
            "within_tolerance_share": float(1 - bad.mean()) if len(d) else 1.0,
            "mismatches": int(bad.sum()),
            "max_abs_diff": float(d[np.isfinite(d)].max()) if np.isfinite(d).any() else 0.0,
            "p99_abs_diff": float(np.quantile(d[np.isfinite(d)], 0.99))
            if np.isfinite(d).any() else 0.0,
            "worst": [{"pid": str(m["pid"].iloc[i]), "sql": _clean(a[i]), "py": _clean(b[i])}
                      for i in idx if d[i] > t],
        }
    out["all_within_tolerance"] = all(v["mismatches"] == 0 for v in out["features"].values())
    return out

"""Database integration tests for the PostGIS mirror of the parcel study.

They run only when ``VHAGAR_PG_DSN`` points at a PostGIS database, for example::

    docker compose -f db/compose.postgis.yml up -d
    $env:VHAGAR_PG_DSN = "postgresql://vhagar:vhagar@localhost:5433/vhagar"   # PowerShell
    pytest tests/test_parcel_postgis.py -v

Each test works in its own throwaway schema and drops it afterwards. The expected values
come from the same shapely functions the study uses, so the tests check SQL-to-Python
agreement on geometry with known answers.
"""
from __future__ import annotations

import os
import uuid

import numpy as np
import pandas as pd
import pytest

shapely = pytest.importorskip("shapely")
pytest.importorskip("psycopg")
DSN = os.environ.get("VHAGAR_PG_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="VHAGAR_PG_DSN not set (no PostGIS)")

from shapely.geometry import LineString, MultiLineString, box  # noqa: E402

from vhagar.parcel.study import features as F  # noqa: E402
from vhagar.parcel.study import postgis as PG  # noqa: E402

# Metric coordinates (EPSG:5070 metres). Parcels are 1 km squares.
PARCELS = [box(0, 0, 1000, 1000), box(5000, 0, 6000, 1000), box(20000, 0, 21000, 1000),
           box(40000, 0, 41000, 1000)]
FACILITY = box(19000, 2000, 19500, 2500)  # a solar footprint north-west of parcel 2
LINES = [
    LineString([(0, 3000), (10000, 3000)]),           # 0: 10 km, far from facility: keep
    LineString([(19400, 2600), (25000, 2600)]),       # 1: 5.6 km, starts at facility: tie
    LineString([(30000, -50000), (30000, 50000)]),    # 2: 100 km, near nothing: keep (HV)
    LineString([(5500, 500), (5500, 9000)]),          # 3: crosses parcel 1: distance 0
    MultiLineString([[(18000, 2600), (19000, 2600)],  # 4: two disjoint parts, short and
                     [(19100, 2700), (19200, 2700)]]),  # near the facility: no single end
]
KV = [69.0, 345.0, 500.0, np.nan, 138.0]
PROTECTED = [box(500, 0, 1500, 1000),                 # half of parcel 0 ...
             box(700, 0, 1200, 1000),                 # ... overlapping again (union)
             box(40000, 0, 40250, 1000)]              # a quarter of parcel 3


@pytest.fixture()
def conn_schema():
    schema = f"vhagar_test_{uuid.uuid4().hex[:8]}"
    conn = PG.connect(DSN)
    try:
        yield conn, schema
    finally:
        conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        conn.close()


def _load(conn, schema):
    df = pd.DataFrame({"pid": [f"p{i}" for i in range(len(PARCELS))], "county": "X",
                       "area_ha": 100.0, "is_case": [False, True, False, False],
                       "floodplain_frac": [0.0, np.nan, 0.5, 0.0],
                       "slope_mean_pct": 1.0})
    return PG.load_tables(conn, df, PARCELS, KV, LINES, [FACILITY], PROTECTED, schema=schema)


def _python_keep():
    import sys
    from pathlib import Path

    gpd = pytest.importorskip("geopandas")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    B = pytest.importorskip("parcel_build")  # the study's own shapely tie rule
    lines = gpd.GeoDataFrame({"kv": KV}, geometry=LINES, crs="EPSG:5070")
    fac = gpd.GeoDataFrame(geometry=[FACILITY], crs="EPSG:5070")
    return B._robust_lines(lines, fac)


def test_load_counts(conn_schema):
    conn, schema = conn_schema
    counts = _load(conn, schema)
    assert counts == {"parcels": 4, "tx_lines": 5, "facilities": 1, "protected": 3}


def test_tie_rule_matches_shapely(conn_schema):
    conn, schema = conn_schema
    _load(conn, schema)
    ties = PG.apply_tie_rule(conn, schema)
    keep_sql = PG.robust_flags(conn, schema)
    assert keep_sql.tolist() == [True, False, True, True, True]
    assert ties["ties_removed"] == 1
    assert keep_sql.tolist() == _python_keep().tolist()


def test_features_match_shapely(conn_schema):
    conn, schema = conn_schema
    _load(conn, schema)
    PG.apply_tie_rule(conn, schema)
    sql = PG.compute_features(conn, schema)

    geoms = np.array(PARCELS, dtype=object)
    lines = np.array(LINES, dtype=object)
    keep = PG.robust_flags(conn, schema)
    hv = keep & (np.nan_to_num(np.array(KV), nan=0.0) >= PG.HV_KV)
    py = pd.DataFrame({
        "pid": [f"p{i}" for i in range(len(PARCELS))],
        "dist_tx_robust_km": F.nearest_distance_km(geoms, lines[keep]),
        "dist_tx_hv_km": F.nearest_distance_km(geoms, lines[hv]),
        "protected_overlap_frac": F.overlap_fraction(
            geoms, shapely.make_valid(np.array(PROTECTED, dtype=object))),
    })
    res = PG.compare(sql, py)
    assert res["all_within_tolerance"], res
    s = sql.set_index("pid")
    assert s.loc["p1", "dist_tx_robust_km"] == pytest.approx(0.0)      # line 3 crosses it
    assert s.loc["p0", "protected_overlap_frac"] == pytest.approx(0.5)  # union, not 0.5+0.2
    assert s.loc["p3", "protected_overlap_frac"] == pytest.approx(0.25)
    assert s.loc["p2", "dist_tx_hv_km"] == pytest.approx(9.0)          # tie removed: line 2


def test_screen_keeps_unknown_flood_separate(conn_schema):
    conn, schema = conn_schema
    _load(conn, schema)
    PG.apply_tie_rule(conn, schema)
    PG.compute_features(conn, schema)
    out = PG.run_screen(conn, schema, min_ha=50.0, max_protected=0.10, max_hv_km=50.0,
                        max_flood=0.10).set_index("outcome")["parcels"].to_dict()
    # p0 protected (50%), p1 flood unknown, p2 in floodplain (50%), p3 protected (25%)
    assert out == {"protected": 2, "pass_flood_unknown": 1, "floodplain": 1}


def test_load_drops_z(conn_schema):
    conn, schema = conn_schema
    z_parcels = [shapely.force_3d(PARCELS[0])] + PARCELS[1:]  # TxGIO has some Z = 0 rings
    df = pd.DataFrame({"pid": [f"p{i}" for i in range(4)], "county": "X", "area_ha": 100.0,
                       "is_case": False, "floodplain_frac": 0.0, "slope_mean_pct": 1.0})
    PG.load_tables(conn, df, z_parcels, KV, LINES, [FACILITY], PROTECTED, schema=schema)
    dims = conn.execute(f"SELECT max(ST_NDims(geom)) FROM {schema}.parcels").fetchone()[0]
    assert dims == 2

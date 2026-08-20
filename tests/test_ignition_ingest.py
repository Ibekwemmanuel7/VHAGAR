"""T3 real ignition ingest tests (synthetic fixtures; no network, no real DB)."""
from __future__ import annotations

import numpy as np
import pytest

from vhagar.datasets import ignition_ingest as ing


def test_classify_cause_handles_both_schemas():
    assert ing.classify_cause("Natural") == "lightning"
    assert ing.classify_cause("Lightning") == "lightning"
    assert ing.classify_cause("Human") == "human"
    assert ing.classify_cause("Campfire") == "human"          # 4th-ed free text
    assert ing.classify_cause("Missing data/not specified/undetermined") == ""
    assert ing.classify_cause(None) == ""


def test_read_fpa_fod_sqlite_roundtrip(tmp_path):
    import sqlite3

    pytest.importorskip("pandas")
    db = tmp_path / "fpa.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE Fires (FOD_ID INTEGER, LONGITUDE REAL, LATITUDE REAL, "
                "FIRE_YEAR INTEGER, STAT_CAUSE_DESCR TEXT)")
    rows = [(1, -120.0, 40.0, 2020, "Lightning"),
            (2, -119.5, 40.5, 2020, "Arson"),
            (3, -118.0, 41.0, 2021, "Missing/Undefined"),
            (4, -100.0, 55.0, 2020, "Lightning")]     # outside the test bbox
    con.executemany("INSERT INTO Fires VALUES (?,?,?,?,?)", rows)
    con.commit()
    con.close()

    df = ing.read_fpa_fod_sqlite(db, bbox=(-124.0, 32.0, -114.0, 42.0), years=[2020, 2021])
    assert len(df) == 3                                   # the 55N fire is filtered out
    assert set(df["cause"]) == {"lightning", "human", ""}
    assert {"id", "lon", "lat", "year", "cause"} <= set(df.columns)


def test_ingest_end_to_end_feeds_assemble(tmp_path):
    pytest.importorskip("pandas")
    sk = pytest.importorskip("sklearn")  # noqa: F841  assemble/eval path uses it downstream
    import sqlite3

    from vhagar.datasets import danger as dg

    # a small real-shaped occurrence DB
    db = tmp_path / "fpa.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE Fires (FOD_ID INTEGER, LONGITUDE REAL, LATITUDE REAL, "
                "FIRE_YEAR INTEGER, NWCG_CAUSE_CLASSIFICATION TEXT)")
    rng = np.random.default_rng(0)
    recs = []
    for i in range(60):
        lon = rng.uniform(-124, -114)
        lat = rng.uniform(33, 42)
        yr = int(rng.choice([2020, 2021]))
        cause = rng.choice(["Human", "Natural"])
        recs.append((i, lon, lat, yr, cause))
    con.executemany("INSERT INTO Fires VALUES (?,?,?,?,?)", recs)
    con.commit()
    con.close()

    out = ing.ingest_fpa_fod(
        db, bbox=(-124.0, 33.0, -114.0, 42.0), years=[2020, 2021],
        out_dir=tmp_path / "t3", cell_deg=0.5, seed=1)
    assert out["synthetic_covariates"] is True
    assert out["features"] == list(ing.DEFAULT_FEATURES)
    assert out["n_presence"] == 60 and out["n_candidates"] > 60

    import pandas as pd
    pres = pd.read_parquet(out["paths"]["occurrence"])
    cand = pd.read_parquet(out["paths"]["candidates"])
    presence, candidates, feat = dg.frames_to_records(pres, cand, list(ing.DEFAULT_FEATURES))
    # lon/lat must never leak into the feature matrix
    assert "lon" not in feat and "lat" not in feat
    s = dg.assemble_ignition_samples(presence, candidates, feat,
                                     np.random.default_rng(2), tau=0.02, neg_per_pos=3.0)
    assert s.X.shape[1] == len(feat)
    assert s.y.sum() == 60                                # all presences are positives

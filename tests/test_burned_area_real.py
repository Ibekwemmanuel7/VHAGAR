"""Tests for the real FPA-FOD burned-area size model (leak-free features, blocked
CV, climatology baseline). Uses a small synthetic-but-realistic SQLite fixture so
the tests need no network and no multi-GB download; the demo SQLite (no FIRE_SIZE)
is used to prove the clear-error path."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from vhagar.eval.burned_area_real import (
    ACRES_TO_HA,
    build_size_features,
    evaluate_real_burned_area,
    fit_real_burned_area_model,
    read_fpa_fod_sizes,
)

DEMO = Path(__file__).resolve().parents[1] / "demo" / "FPA_FOD_demo.sqlite"


def _make_sqlite(path, n=600, seed=0):
    """A FPA-FOD-shaped SQLite with real column names, spread over several 5-deg
    blocks, positive heavy-tailed sizes, and both causes."""
    rng = np.random.default_rng(seed)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE Fires (FOD_ID INT, LONGITUDE REAL, LATITUDE REAL, "
                "FIRE_YEAR INT, DISCOVERY_DOY INT, FIRE_SIZE REAL, "
                "NWCG_CAUSE_CLASSIFICATION TEXT)")
    rows = []
    for i in range(n):
        lon = -124.0 + 24.0 * rng.random()      # CONUS-ish, many 5-deg blocks
        lat = 33.0 + 16.0 * rng.random()
        year = int(rng.choice([2018, 2019, 2020, 2021]))
        doy = int(rng.integers(60, 320))
        light = rng.random() < 0.4
        loga = 1.0 + 0.9 * light + 0.6 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 1.1)
        if rng.random() < 0.05:
            loga += rng.gamma(2.0, 1.0)
        acres = float(np.clip(np.expm1(loga), 0.1, 500_000))
        cause = "Natural" if light else "Human"
        rows.append((i, lon, lat, year, doy, acres, cause))
    con.executemany("INSERT INTO Fires VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


def test_reader_requires_fire_size_demo_raises():
    if not DEMO.exists():
        pytest.skip("demo sqlite absent")
    with pytest.raises(ValueError, match="full FPA-FOD"):
        read_fpa_fod_sizes(DEMO)


def test_reader_and_units(tmp_path):
    p = tmp_path / "full.sqlite"
    _make_sqlite(p)
    df = read_fpa_fod_sizes(p)
    assert set(df.columns) == {"id", "lon", "lat", "year", "doy", "cause", "area_ha"}
    assert (df["area_ha"] > 0).all()
    # spot-check acres -> ha on the raw row
    con = sqlite3.connect(p)
    fid, acres = con.execute("SELECT FOD_ID, FIRE_SIZE FROM Fires LIMIT 1").fetchone()
    con.close()
    got = float(df.loc[df.id == fid, "area_ha"].iloc[0])
    assert abs(got - acres * ACRES_TO_HA) < 1e-6


def test_features_are_leak_free(tmp_path):
    p = tmp_path / "full.sqlite"
    _make_sqlite(p)
    df = read_fpa_fod_sizes(p)
    X, area, year, lon, lat, fn = build_size_features(df)
    assert fn == ["lightning", "sin_doy", "cos_doy"]
    assert X.shape == (len(df), 3)
    # lon/lat must NOT appear as feature columns
    for col in (lon, lat):
        assert not any(np.allclose(X[:, j], col) for j in range(X.shape[1]))
    assert np.isfinite(X).all() and (area > 0).all()


def test_evaluate_real_burned_area_runs_and_scores(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    p = tmp_path / "full.sqlite"
    _make_sqlite(p, n=800)
    res = evaluate_real_burned_area(p, n_folds=3)
    assert res["n_fires"] == 800
    assert np.isfinite(res["crps"]) and np.isfinite(res["crps_climatology"])
    assert res["crps"] > 0
    # honest reporting: skill vs climatology present and RMSE instability exposed
    assert "crps_skill_vs_climatology" in res
    assert res["rmse_std"] >= 0
    assert set(res["pinball"]) and all(v >= 0 for v in res["pinball"].values())


def test_fit_model_predicts_positive_quantiles(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    p = tmp_path / "full.sqlite"
    _make_sqlite(p)
    model, fn, summary = fit_real_burned_area_model(p)
    # predict for a lightning fire mid-summer and a human fire in spring
    doy = 200
    ang = 2 * np.pi * (doy - 1) / 365
    X = np.array([[1.0, np.sin(ang), np.cos(ang)], [0.0, np.sin(ang), np.cos(ang)]])
    q = model.predict_quantiles(X)
    assert q.shape[0] == 2
    assert (q > 0).all()
    assert (np.diff(q, axis=1) >= -1e-9).all()   # monotone quantiles
    assert summary["n_fires"] > 0 and summary["median_ha"] > 0

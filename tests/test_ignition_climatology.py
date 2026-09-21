"""Tests for the real FPA-FOD ignition occurrence-climatology and its honest
two-way (temporal + spatial) evaluation. Uses an in-memory-style SQLite fixture
with recurring fire cells so the temporal-persistence signal is present."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from vhagar.eval.ignition_climatology import (
    climatology_frequency,
    evaluate_ignition_climatology,
    read_fpa_fod_occurrence,
)

DEMO = Path(__file__).resolve().parents[1] / "demo" / "FPA_FOD_demo.sqlite"


def _make_occ_sqlite(path, seed=1):
    rng = np.random.default_rng(seed)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE Fires (LONGITUDE REAL, LATITUDE REAL, FIRE_YEAR INT, "
                "DISCOVERY_DOY INT)")
    hot = [(-120 + 0.5 * rng.integers(0, 40), 34 + 0.5 * rng.integers(0, 30))
           for _ in range(40)]
    rows = []
    for y in (2018, 2019, 2020, 2021):
        for (lo, la) in hot:                       # recurring hotspots (persistence)
            for _ in range(int(rng.integers(1, 6))):
                rows.append((lo + rng.normal(0, 0.05), la + rng.normal(0, 0.05), y,
                             int(rng.integers(150, 280))))
        for _ in range(200):                       # diffuse background
            rows.append((-120 + 20 * rng.random(), 34 + 15 * rng.random(), y,
                         int(rng.integers(1, 366))))
    con.executemany("INSERT INTO Fires VALUES (?,?,?,?)", rows)
    con.commit()
    con.close()


def test_reader_requires_doy_demo_raises():
    if not DEMO.exists():
        pytest.skip("demo sqlite absent")
    with pytest.raises(ValueError, match="full FPA-FOD"):
        read_fpa_fod_occurrence(DEMO)


def test_reader_builds_month(tmp_path):
    p = tmp_path / "occ.sqlite"
    _make_occ_sqlite(p)
    df = read_fpa_fod_occurrence(p)
    assert set(df.columns) == {"lon", "lat", "year", "month"}
    assert df["month"].between(1, 12).all()
    assert df["year"].nunique() == 4


def test_climatology_frequency_smoothed(tmp_path):
    p = tmp_path / "occ.sqlite"
    _make_occ_sqlite(p)
    df = read_fpa_fod_occurrence(p)
    freq, base, seen = climatology_frequency(df, cell_deg=0.5)
    assert 0.0 < base < 1.0
    assert all(0.0 <= v <= 1.0 for v in freq.values())
    assert len(seen) > 0


def test_two_way_eval_temporal_beats_spatial(tmp_path):
    p = tmp_path / "occ.sqlite"
    _make_occ_sqlite(p)
    r = evaluate_ignition_climatology(p, cell_deg=0.5)
    t, s = r["temporal_holdout"], r["spatial_block"]
    # temporal: climatology AUPRC should exceed the base rate (persistence signal)
    assert t["auprc_climatology"] >= t["auprc_baserate"]
    assert t["n_positive"] > 0
    # spatial: on unseen blocks, climatology should NOT beat base rate (honest limit)
    if s["n_blocks_scored"] > 0:
        assert s["auprc_climatology_mean"] <= s["auprc_baserate_mean"] + 1e-6

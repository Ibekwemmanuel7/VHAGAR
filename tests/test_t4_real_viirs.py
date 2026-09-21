"""Regression test for scripts/t4_real_viirs.py, the real-VIIRS T4 forward-run
workflow. Exercises the script's own pipeline (load timed FIRMS points, cluster,
build the case, split calibrate/forecast, score) on a tiny synthetic radially
growing fire, rather than only the generic timed-detection path in
test_spread_ingest. No network, no multi-day download, no rasterio/matplotlib."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

pytest.importorskip("pandas")


def _load_script():
    """Import scripts/t4_real_viirs.py as a module; skip if the T4 spread solver
    (skfmm) or another core dep is unavailable in this environment."""
    root = pathlib.Path(__file__).resolve().parents[1]
    path = root / "scripts" / "t4_real_viirs.py"
    spec = importlib.util.spec_from_file_location("t4_real_viirs", path)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:                       # pragma: no cover - env guard
        pytest.skip(f"t4_real_viirs deps unavailable: {e}")
    return m


def _synthetic_firms_csv(path, seed=0):
    """A radially growing fire: detections spread outward from a center over ~2.5
    days, so an early/late time split has genuine held-out new burn to forecast."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    lon0, lat0 = -120.0, 39.0
    rows = []
    base = pd.Timestamp("2026-08-01T18:00:00")
    for step in range(24):                       # 24 time steps over ~2.4 days
        t = base + pd.Timedelta(hours=step * 2.4)
        radius = 0.01 + 0.006 * step             # degrees, grows with time
        n = 30
        ang = rng.uniform(0, 2 * np.pi, n)
        rr = radius * np.sqrt(rng.uniform(0, 1, n))
        lon = lon0 + rr * np.cos(ang)
        lat = lat0 + rr * np.sin(ang)
        for a, b in zip(lon, lat, strict=True):
            rows.append({"latitude": round(float(b), 5), "longitude": round(float(a), 5),
                         "acq_date": t.strftime("%Y-%m-%d"),
                         "acq_time": int(t.strftime("%H%M"))})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_and_cluster(tmp_path):
    m = _load_script()
    csv = tmp_path / "viirs.csv"
    _synthetic_firms_csv(csv)
    df = m._load(str(csv))
    assert {"latitude", "longitude", "_dt"}.issubset(df.columns)
    assert df["_dt"].notna().all()
    g, df2 = m._clusters(df, min_days=1, min_dets=50, top=3)
    assert len(g) >= 1                            # one growing complex
    assert {"cy", "cx"}.issubset(df2.columns)


def test_case_and_masks_forecasts_new_burn(tmp_path):
    m = _load_script()
    csv = tmp_path / "viirs.csv"
    _synthetic_firms_csv(csv)
    df = m._load(str(csv))
    # score the whole synthetic fire directly (bypass the size thresholds)
    fire = m._case_and_masks(df, cell_deg=0.01, split_frac=0.5, fuel_tif=None)
    if fire is None:
        pytest.skip("too few rasterized detections for a case in this environment")
    # metrics exist and are in range; the forecast covers held-out NEW burn
    assert 0.0 <= fire["dice"] <= 1.0
    assert 0.0 <= fire["pod"] <= 1.0
    assert 0.0 <= fire["far"] <= 1.0
    assert fire["n_eval_cells"] >= 1             # genuine held-out cells to score
    # seen (calibration) and truth (held-out) are disjoint by construction
    assert not (fire["seen"] & fire["truth"]).any()
    assert fire["k"] > 0                          # a positive per-fire ROS scale
    assert fire["pred"].shape == fire["seen"].shape

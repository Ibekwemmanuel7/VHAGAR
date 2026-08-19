"""T1 Stage-2 leakage experiment: labelling, and the with/without-lat/lon collapse."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyproj")  # geo CRS stack (geo extra); skip in a minimal env

from vhagar.eval.t1_classifier import LabeledSamples, build_samples  # noqa: E402


def test_build_samples_labels_viirs_coincidence():
    # two GOES detections at roughly (-120, 38); one has a VIIRS hit ~same place+time.
    t0 = pd.Timestamp("2026-08-01T21:00:00")
    fdc = pd.DataFrame({
        "lon": [-120.00, -110.00], "lat": [38.00, 45.00],
        "t": [t0, t0], "frp_mw": [50.0, 20.0], "temp_k": [400.0, 350.0],
        "confidence": [0.8, 0.4], "area_m2": [1e6, 5e5], "view_zenith_deg": [45.0, 50.0],
    })
    viirs_ll = np.array([[-120.001, 38.001]])           # ~100 m from the first GOES pixel
    viirs_t = np.array([t0.timestamp() + 300])          # 5 min later
    s = build_samples(fdc, viirs_ll, viirs_t, cell_m=4_000.0, window_min=30.0)
    assert s.y.tolist() == [1, 0]                       # first confirmed, second not
    assert s.X.shape == (2, len(s.feature_names))
    assert s.lonlat.shape == (2, 2)


def _leaky_dataset(n_cells=60, per_cell=40, seed=0):
    """Each cell gets a random label; physical features are pure noise, so the ONLY way
    to predict is to memorise the cell's location. That is textbook leakage: it works
    when cells are shared (random split) and vanishes when whole cells/blocks are held
    out. lon/lat are placed on a coarse grid so cells map to 5-degree blocks."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n_cells)
    X, lonlat, y, cell, block = [], [], [], [], []
    for c in range(n_cells):
        clon = -120 + (c % 10) * 1.5
        clat = 30 + (c // 10) * 1.5
        for _ in range(per_cell):
            X.append(rng.normal(0, 1, 3))               # noise features
            lonlat.append([clon + rng.normal(0, 0.05), clat + rng.normal(0, 0.05)])
            y.append(labels[c])
            cell.append(c)
            block.append(int(np.floor(clon / 5)) * 1000 + int(np.floor(clat / 5)))
    return LabeledSamples(
        X=np.array(X), lonlat=np.array(lonlat), y=np.array(y),
        cell_group=np.array(cell), block_group=np.array(block), feature_names=("a", "b", "c"),
    )


def test_latlon_leakage_helps_on_random_and_collapses_out_of_region():
    pytest.importorskip("sklearn")
    from vhagar.eval.t1_classifier import evaluate_leakage

    s = _leaky_dataset()
    r = evaluate_leakage(s, n_folds=4)
    # physical (noise) features cannot predict the random per-cell label anywhere
    assert r["random"]["physical"] < 0.7
    # lat/lon memorises the cell on a random split (big gain)...
    assert r["random"]["latlon_gain"] > 0.2
    # ...but that gain evaporates once whole cells/blocks are held out (the leak)
    assert r["spatial_block_5deg"]["latlon_gain"] < r["random"]["latlon_gain"]

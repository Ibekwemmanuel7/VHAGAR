"""T4 real spread ingest tests (synthetic timed detections; no network)."""
from __future__ import annotations

import numpy as np
import pytest

from vhagar.datasets import spread_ingest as si


def test_gridspec_maps_lonlat_and_flags_out_of_bounds():
    spec = si.GridSpec.from_bbox_res((-121.0, 39.0, -120.0, 40.0), cell_deg=0.1)
    assert spec.shape == (10, 10)
    row, col, inb = spec.lonlat_to_rc([-120.95, -119.0], [39.95, 39.5])
    assert inb[0] and not inb[1]                 # second point is east of the bbox
    assert row[0] == 0 and col[0] == 0           # NW corner: top row (north), left col


def test_rasterize_keeps_earliest_time_per_cell():
    spec = si.GridSpec.from_bbox_res((0.0, 0.0, 1.0, 1.0), cell_deg=0.1)
    # two detections in the same cell at different times, plus one elsewhere
    lon = [0.05, 0.05, 0.55]
    lat = [0.05, 0.05, 0.55]
    t = [5.0, 2.0, 9.0]
    det_rc, det_times = si.rasterize_detections(lon, lat, t, spec)
    assert det_rc.shape == (2, 2)                 # deduped to two cells
    assert det_times.min() == 2.0                 # kept the earlier time in the shared cell
    assert list(det_times) == sorted(det_times)   # returned in time order


def test_build_and_assimilate_real_scores_in_range():
    pytest.importorskip("scipy")
    spec = si.GridSpec.from_bbox_res((0.0, 0.0, 1.0, 1.0), cell_deg=0.05)  # 20x20
    # a fire spreading outward from the SW: time grows with distance from origin
    rng = np.random.default_rng(0)
    lons, lats, times = [], [], []
    for _ in range(400):
        lo = rng.uniform(0.0, 1.0)
        la = rng.uniform(0.0, 1.0)
        # arrival ~ distance from the SW corner, with noise
        d = np.hypot(lo, la)
        lons.append(lo)
        lats.append(la)
        times.append(d * 10 + rng.uniform(0, 0.5))
    case = si.build_spread_case(lons, lats, times, spec, seed=0)
    assert case["prior_ros"].shape == spec.shape
    assert case["ignition"].any()
    out = si.assimilate_real(case, split_frac=0.5)
    assert out["k"] > 0.0
    for key in ("dice", "pod", "far"):
        assert 0.0 <= out[key] <= 1.0
    assert out["n_early"] >= 1 and out["n_late"] >= 1


def test_assimilate_excludes_calibration_cells_from_truth():
    """Regression: Dice/POD must be scored on held-out post-cutoff NEW burn only,
    not against the calibration detections the analysis was fit on (which inflated
    the score). The evaluable region is exactly the later detections and is a
    strict subset of all detected cells."""
    pytest.importorskip("scipy")
    spec = si.GridSpec.from_bbox_res((0.0, 0.0, 1.0, 1.0), cell_deg=0.05)
    rng = np.random.default_rng(1)
    lons, lats, times = [], [], []
    for _ in range(400):
        lo, la = rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)
        d = np.hypot(lo, la)
        lons.append(lo)
        lats.append(la)
        times.append(d * 10 + rng.uniform(0, 0.5))
    case = si.build_spread_case(lons, lats, times, spec, seed=1)
    out = si.assimilate_real(case, split_frac=0.5)
    assert out["scoring"].startswith("held-out")
    # evaluable cells are the later detections, and exclude the calibration footprint
    assert out["n_eval_cells"] == out["n_late"]
    assert out["n_eval_cells"] < out["n_early"] + out["n_late"]
    assert 0.0 <= out["dice"] <= 1.0

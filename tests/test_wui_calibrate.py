"""WUI DINS calibration harness: synthetic fires with a known 'true' parameter set,
so grid search should recover good parameters and leave-one-fire-out should run and
score sensibly. Deterministic, no RNG, no external data."""
from __future__ import annotations

import numpy as np
import pytest

from vhagar.eval import wui_calibrate as wc
from vhagar.models import wui


def _synthetic_fire(name, seed_xy, struct_specs, wind_dir=0.0, horizon=8.0,
                    true_params=None):
    """Build a WuiFire whose truth_destroyed is produced by the model at true_params,
    so a good calibration can recover it. struct_specs: list of (row, col)."""
    H = W = 41
    ros = np.ones((H, W))
    burned = np.zeros((H, W), bool)
    burned[seed_xy] = True
    rows = np.array([r for r, _ in struct_specs])
    cols = np.array([c for _, c in struct_specs])
    tp = true_params or {"spotting_max_dist": 6.0, "spotting_intensity": 4.0,
                         "struct_radius_cells": 3.0, "struct_base_p": 0.9}
    out = wui.wui_spread(burned, ros, rows, cols, horizon=horizon, wind_speed=1.0,
                         wind_dir=wind_dir, spotting_max_dist=tp["spotting_max_dist"],
                         spotting_intensity=tp["spotting_intensity"],
                         struct_radius_cells=tp["struct_radius_cells"],
                         struct_base_p=tp["struct_base_p"])
    return wc.WuiFire(name=name, ros=ros, burned_seed=burned, struct_rows=rows,
                      struct_cols=cols, truth_destroyed=out["destroyed"],
                      wind_speed=1.0, wind_dir=wind_dir, horizon=horizon)


def _fires():
    # downwind (+x) clusters near ignition (should be destroyed) + a far survivor
    return [
        _synthetic_fire("A", (20, 20), [(20, 24), (20, 25), (20, 26), (0, 40)]),
        _synthetic_fire("B", (10, 10), [(10, 14), (10, 15), (10, 16), (39, 0)]),
        _synthetic_fire("C", (30, 15), [(30, 19), (30, 20), (30, 21), (0, 0)]),
    ]


def test_wuifire_validates_lengths():
    with pytest.raises(ValueError):
        wc.WuiFire(name="bad", ros=np.ones((4, 4)), burned_seed=np.zeros((4, 4), bool),
                   struct_rows=np.array([0, 1]), struct_cols=np.array([0]),
                   truth_destroyed=np.array([True, False]))


def test_predict_and_score_fire_recovers_truth_at_true_params():
    fire = _synthetic_fire("A", (20, 20), [(20, 24), (20, 25), (20, 26), (0, 40)])
    tp = {"spotting_max_dist": 6.0, "spotting_intensity": 4.0,
          "struct_radius_cells": 3.0, "struct_base_p": 0.9}
    s = wc.score_fire(fire, tp)
    # scored against its own generating params -> perfect
    assert s.fp == 0 and s.fn == 0
    assert s.f1 == 1.0 or np.isnan(s.f1)         # nan only if no positives at all
    assert s.tp >= 1                              # near cluster is destroyed


def test_grid_search_returns_valid_params():
    fires = _fires()
    grid = {"spotting_max_dist": [6.0], "spotting_intensity": [1.0, 4.0],
            "struct_radius_cells": [3.0], "struct_base_p": [0.5, 0.9]}
    params, f1 = wc.grid_search(fires, grid)
    assert set(params) == set(wc._PARAM_KEYS)
    assert np.isfinite(f1) and 0.0 <= f1 <= 1.0


def test_calibrate_lofo_runs_and_generalizes():
    fires = _fires()
    # grid includes the generating params, so held-out F1 should be high
    grid = {"spotting_max_dist": [6.0], "spotting_intensity": [1.0, 4.0],
            "struct_radius_cells": [3.0], "struct_base_p": [0.5, 0.9]}
    rep = wc.calibrate_lofo(fires, grid)
    assert rep["n_fires"] == 3
    assert len(rep["folds"]) == 3
    assert set(rep["selected"]) == set(wc._PARAM_KEYS)
    assert np.isfinite(rep["mean_heldout_f1"])
    assert rep["mean_heldout_f1"] > 0.8          # recovers generalizable params


def test_calibrate_lofo_requires_two_fires():
    with pytest.raises(ValueError):
        wc.calibrate_lofo(_fires()[:1])


def test_wind_from_deg_to_grid():
    # west wind (from 270) -> toward east -> +col -> grid heading 0
    assert abs(wc.wind_from_deg_to_grid(270.0) - 0.0) < 1e-9
    # south wind (from 180) -> toward north -> +row -> +pi/2
    assert abs(wc.wind_from_deg_to_grid(180.0) - np.pi / 2) < 1e-9


def test_fire_from_points_builds_faithful_wuifire():
    # tiny cluster of structures with an ignition point just west of them
    lat = np.array([34.190, 34.191, 34.192, 34.193])
    lon = np.array([-118.130, -118.129, -118.128, -118.127])
    destroyed = np.array([True, True, False, False])
    fire = wc.fire_from_points("Test", lat, lon, destroyed,
                               ignition_lat=34.1905, ignition_lon=-118.131,
                               wind_speed_ms=7.5, wind_from_deg=270.0, cell_m=30.0)
    assert fire.anisotropic is True
    assert fire.struct_rows.size == 4 and fire.truth_destroyed.tolist() == [True, True, False, False]
    assert abs(fire.wind_speed - 0.5) < 1e-9              # 7.5 / 15 ref
    assert fire.burned_seed.any()                          # ignition seeded
    H, W = fire.ros.shape
    assert (fire.struct_rows >= 0).all() and (fire.struct_rows < H).all()
    assert (fire.struct_cols >= 0).all() and (fire.struct_cols < W).all()
    # a WuiFire this shape runs through the model end to end
    pred = wc.predict_fire(fire, dict(zip(wc._PARAM_KEYS,
                                          (6.0, 3.0, 3.0, 0.7), strict=True)))
    assert pred.shape == fire.truth_destroyed.shape

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

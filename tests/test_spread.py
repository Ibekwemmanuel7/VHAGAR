"""T4 spread: fast-marching propagation + honest incremental validation tests."""
from __future__ import annotations

import numpy as np
import pytest

from vhagar.models.spread import (
    fast_marching_arrival,
    persistence_buffer,
    rate_of_spread,
    spread_forecast,
)


def test_fast_marching_recovers_distance():
    speed = np.ones((41, 41))
    seed = np.zeros((41, 41), dtype=bool)
    seed[20, 20] = True
    T = fast_marching_arrival(speed, seed)
    assert abs(T[20, 0] - 20.0) < 1.0            # axis distance exact-ish
    assert abs(T[0, 20] - 20.0) < 1.0
    assert abs(T[10, 10] - np.hypot(10, 10)) < 1.3   # diagonal within FMM tolerance
    assert T[20, 20] == 0.0


def test_faster_ros_arrives_sooner():
    fast = rate_of_spread(np.full((30, 30), 0.9), np.full((30, 30), 0.9), np.zeros((30, 30)))
    slow = rate_of_spread(np.full((30, 30), 0.2), np.zeros((30, 30)), np.zeros((30, 30)))
    seed = np.zeros((30, 30), dtype=bool)
    seed[15, 15] = True
    assert fast_marching_arrival(fast, seed)[15, 0] < fast_marching_arrival(slow, seed)[15, 0]


def test_rate_of_spread_monotone():
    lo = rate_of_spread(0.2, 0.2, 0.2)
    assert rate_of_spread(0.9, 0.2, 0.2) > lo        # more fuel
    assert rate_of_spread(0.2, 0.9, 0.2) > lo        # more wind
    assert rate_of_spread(0.2, 0.2, 0.9) > lo        # steeper upslope


def test_spread_forecast_grows_and_probabilistic():
    ros = rate_of_spread(np.full((30, 30), 0.7), np.full((30, 30), 0.6), np.zeros((30, 30)))
    burned = np.zeros((30, 30), dtype=bool)
    burned[13:17, 13:17] = True
    mask, prob, arrival = spread_forecast(burned, ros, horizon=8.0)
    assert mask.mean() > burned.mean()               # the fire grows
    assert prob.min() >= 0 and prob.max() <= 1
    assert np.all(prob[burned] == 1.0)               # already burned stays certain
    assert np.all(mask[burned])                       # persistence of burned cells


def test_persistence_buffer_dilates():
    burned = np.zeros((30, 30), dtype=bool)
    burned[15, 15] = True
    mask, prob = persistence_buffer(burned, radius_cells=4)
    assert 1 < mask.sum() <= (2 * 4 + 1) ** 2
    assert prob.max() <= 1.0 and prob[15, 15] == 1.0


@pytest.mark.slow
def test_physics_beats_baselines_incrementally():
    pytest.importorskip("scipy")
    from vhagar.eval.spread import evaluate_spread
    r = evaluate_spread(n_fires=10, seed=1)
    for regime in ("wind", "plume"):
        ag = r[regime]
        phys, buf, per = ag["physics"], ag["persistence_buffer"], ag["persistence"]
        # scored on a thin incremental band, not cumulative area
        assert 0.03 < ag["new_burn_rate"] < 0.25
        # physics beats persistence+buffer beats persistence on IoU
        assert phys["iou"] > buf["iou"] > per["iou"]
        assert phys["ap"] > per["ap"]
        # burned-area ratio exposes the honest over-prediction from hidden suppression
        assert phys["ba_ratio"] > 1.0
        assert np.isfinite(phys["arrival_mae"])

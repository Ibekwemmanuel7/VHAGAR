"""T4-M residual-correction contracts: physics stays the prior and folds are fires."""
from __future__ import annotations

import numpy as np
import pytest

from vhagar.models.spread_residual import (
    OODGate,
    RidgeResidualCorrector,
    leave_one_fire_out,
    leave_one_region_out,
    observed_log_ros_ratio,
)


def test_observed_log_ros_ratio_uses_inverse_arrival_scaling():
    prior = np.array([[0.0, 10.0], [20.0, np.inf]])
    rc = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    got = observed_log_ros_ratio(prior, rc, np.array([1.0, 5.0, 40.0, 10.0]))
    assert np.isnan(got[0]) and np.isnan(got[3])
    assert got[1] == pytest.approx(np.log(2.0))
    assert got[2] == pytest.approx(np.log(0.5))


def test_ridge_corrector_learns_bounded_multiplicative_adjustment():
    x = np.linspace(-1, 1, 30)[:, None]
    target = 0.5 * x[:, 0]
    model = RidgeResidualCorrector(alpha=0.001).fit(x, target)
    pred = model.predict_log_ratio(np.array([[-1.0], [1.0], [10.0]]))
    assert pred[0] < 0 < pred[1]
    assert abs(pred[2]) <= np.log(3.0) + 1e-12
    assert model.correct_ros(np.array([2.0]), np.array([[0.0]]))[0] == pytest.approx(2.0, rel=0.1)


def test_leave_one_fire_out_never_trains_on_held_incident():
    # Same stable relationship on three fires: the held-out fit should improve on
    # no correction, while the returned records make the split auditable.
    feature = np.tile(np.array([-1.0, 0.0, 1.0]), 3)[:, None]
    target = 0.35 * feature[:, 0]
    fire = np.repeat(["a", "b", "c"], 3)
    scores = leave_one_fire_out(feature, target, fire, alpha=0.001)
    assert {s["fire_id"] for s in scores} == {"a", "b", "c"}
    assert all(s["n_test"] == 3 for s in scores)
    assert all(s["residual_rmse"] < s["baseline_rmse"] for s in scores)


def test_corrector_rejects_prediction_before_fit():
    with pytest.raises(RuntimeError):
        RidgeResidualCorrector().predict_log_ratio([[0.0]])


def test_leave_one_region_out_reports_geographic_transfer():
    x = np.tile(np.array([-1.0, 0.0, 1.0]), 3)[:, None]
    y = 0.2 * x[:, 0]
    region = np.repeat(["coastal", "mountain", "interior"], 3)
    scores = leave_one_region_out(x, y, region, alpha=0.001)
    assert {s["region_id"] for s in scores} == {"coastal", "mountain", "interior"}
    assert all(s["residual_rmse"] < s["baseline_rmse"] for s in scores)


def test_ood_gate_falls_back_toward_physics_outside_training_support():
    gate = OODGate(threshold=2.0).fit(np.array([[0.0, 0.0], [0.2, -0.1], [-0.2, 0.1]]))
    trust = gate.trust(np.array([[0.0, 0.0], [20.0, 20.0]]))
    assert trust[0] > 0.9 and trust[1] == 0.0
    assert gate.blend_log_ratio(np.array([0.4, 0.4]), np.array([[0.0, 0.0], [20.0, 20.0]]))[1] == 0.0

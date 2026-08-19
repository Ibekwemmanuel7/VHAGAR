"""T4 state estimation + assimilation tests."""
from __future__ import annotations

import numpy as np
import pytest

from vhagar.models.spread import fast_marching_arrival, rate_of_spread
from vhagar.models.state_estimation import calibrate_ros_scale, estimate_arrival_field


def test_calibrate_recovers_known_scale():
    ros = rate_of_spread(np.full((40, 40), 0.6), np.full((40, 40), 0.5), np.zeros((40, 40)))
    ign = np.zeros((40, 40), dtype=bool)
    ign[20, 20] = True
    prior_arrival = fast_marching_arrival(ros, ign)
    k_true = 1.8
    reach = np.argwhere(np.isfinite(prior_arrival) & (prior_arrival > 0))
    # observed times if the true ROS were k_true x the prior: arrival scales 1/k
    det_times = prior_arrival[reach[:, 0], reach[:, 1]] / k_true
    k = calibrate_ros_scale(prior_arrival, reach, det_times)
    assert abs(k - k_true) < 0.05
    assert calibrate_ros_scale(prior_arrival, np.empty((0, 2), int), np.array([])) == 1.0


def test_estimate_arrival_field_scales_prior():
    ros = rate_of_spread(np.full((30, 30), 0.6), np.full((30, 30), 0.5), np.zeros((30, 30)))
    ign = np.zeros((30, 30), dtype=bool)
    ign[15, 15] = True
    prior_arrival = fast_marching_arrival(ros, ign)
    reach = np.argwhere(np.isfinite(prior_arrival) & (prior_arrival > 0))
    det_times = prior_arrival[reach[:, 0], reach[:, 1]] / 1.5
    st = estimate_arrival_field(ros, ign, reach, det_times)
    assert abs(st.k - 1.5) < 0.1
    # analysis arrival ~ prior / k
    fin = np.isfinite(prior_arrival) & np.isfinite(st.arrival)
    assert np.allclose(st.arrival[fin], prior_arrival[fin] / st.k, rtol=1e-6)
    assert st.burned_by(np.nanmax(st.arrival[fin])).all()


@pytest.mark.slow
def test_assimilation_beats_baselines_and_calibrates():
    pytest.importorskip("scipy")
    from vhagar.eval.assimilation import assimilation_experiment
    r = assimilation_experiment(n_fires=10, n_passes=6, prior_bias=0.6, seed=0)
    o = r["overall"]
    # calibration recovers the ROS bias (ideal k = 1/0.6 ~ 1.67)
    assert abs(o["k_final"] - o["k_ideal"]) < 0.4
    # the calibrated analysis forecasts new burn far better than persistence / prior
    assert o["soren_analysis"] > 0.25
    assert o["soren_analysis"] > o["soren_prior"]
    assert o["soren_naive"] == 0.0
    # full-perimeter reconstruction is in the cited ballpark
    assert 0.55 < o["soren_full"] < 0.95

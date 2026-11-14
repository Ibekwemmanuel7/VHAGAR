"""T4-M replay test: historical-only learning, pre-cutoff adaptation, future score."""
from __future__ import annotations

import numpy as np
import pandas as pd

from vhagar.eval.t4m_replay import run_t4m_time_forward_replay


def _fire(fire_id: str, start: str, *, bias: float = 0.0) -> pd.DataFrame:
    x = np.array([-1.0, -0.5, 0.2, 0.7, 1.0])
    # Stable cross-fire relationship plus a named event offset.  The physical
    # arrival is 4h, so first-seen time encodes the implied ROS correction.
    log_ratio = 0.25 * x + bias
    prior = np.full(len(x), 4.0)
    return pd.DataFrame({
        "fire_id": [fire_id] * len(x), "observation_id": [f"{fire_id}-{i}" for i in range(len(x))],
        "timestamp_utc": pd.date_range(start, periods=len(x), freq="h", tz="UTC"),
        "source": ["thermal"] * len(x), "source_product": ["test"] * len(x),
        "observation_kind": ["thermal_front"] * len(x), "analysis_grid_id": [fire_id] * len(x),
        "row": np.arange(len(x)), "col": np.arange(len(x)),
        "first_seen_hours": prior / np.exp(log_ratio),
        "time_sigma_min": np.full(len(x), 5.0), "location_sigma_m": np.full(len(x), 30.0),
        "quality": np.full(len(x), 0.9), "prior_arrival_hours": prior,
        "prior_ros_m_min": np.ones(len(x)), "fuel_moisture": 0.08 + x * 0.01,
        "wind_midflame_ms": 4.0 + x, "slope_tan": 0.15 + x * 0.01,
    })


def test_replay_scores_only_future_target_observations_and_improves_prior():
    hist = pd.concat([_fire("h1", "2026-07-01", bias=0.0), _fire("h2", "2026-07-02", bias=0.0)])
    target = _fire("target", "2026-08-01", bias=0.10)
    out = run_t4m_time_forward_replay(hist, target, "2026-08-01T01:30:00Z", alpha=0.01)
    assert out["n_historical_fires"] == 2
    assert out["n_calibration"] == 2 and out["n_evaluation"] == 3
    assert out["residual_log_ros_rmse"] < out["physics_log_ros_rmse"]
    assert out["residual_arrival_mae_hours"] < out["physics_arrival_mae_hours"]
    assert 0 <= out["interval_10_90_coverage"] <= 1
    assert "post-cutoff" in out["scoring"]

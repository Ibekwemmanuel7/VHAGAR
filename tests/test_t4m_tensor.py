"""T4-M tensor contract: future labels must not enter an issue-time input."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vhagar.datasets.t4m_tensor import tensorize_issue_time


def _obs():
    return pd.DataFrame({
        "fire_id": ["f"] * 3, "observation_id": ["a", "b", "c"],
        "timestamp_utc": ["2026-08-01T00:00:00Z", "2026-08-01T01:00:00Z", "2026-08-01T02:00:00Z"],
        "source": ["airborne"] * 3, "source_product": ["thermal"] * 3,
        "observation_kind": ["thermal_front"] * 3, "analysis_grid_id": ["g"] * 3,
        "row": [0, 1, 2], "col": [0, 1, 2], "first_seen_hours": [0.0, 1.0, 2.0],
        "time_sigma_min": [1.0] * 3, "location_sigma_m": [10.0] * 3, "quality": [0.9] * 3,
        "prior_arrival_hours": [0.0, 1.2, 2.2], "prior_ros_m_min": [1.0] * 3,
        "fuel_moisture": [0.08] * 3, "wind_midflame_ms": [3.0] * 3, "slope_tan": [0.1] * 3,
    })


def test_tensor_has_only_pre_cutoff_observations_in_conditioning():
    grids = {"fuel_moisture": np.full((4, 4), 0.08), "wind_midflame_ms": np.full((4, 4), 3.0), "slope_tan": np.full((4, 4), 0.1)}
    case = tensorize_issue_time(_obs(), grid_shape=(4, 4), covariate_grids=grids,
                                 prior_arrival_hours=np.ones((4, 4)), prior_ros_m_min=np.ones((4, 4)),
                                 cutoff_utc="2026-08-01T00:30:00Z", tmax_hours=6.0)
    assert case.conditioning.shape == (6, 4, 4)
    assert case.conditioning[0, 0, 0] == 1.0
    assert case.conditioning[0, 1, 1] == 0.0  # later observations are not input
    assert case.target_valid[1, 1] and case.target_valid[2, 2]
    assert not case.target_valid[0, 0]
    assert case.n_input_observations == 1 and case.n_target_observations == 2


def test_tensor_requires_complete_grid_covariates():
    with pytest.raises(ValueError, match="wind_midflame_ms"):
        tensorize_issue_time(_obs(), grid_shape=(2, 2), covariate_grids={"fuel_moisture": np.ones((2, 2))},
                             prior_arrival_hours=np.ones((2, 2)), prior_ros_m_min=np.ones((2, 2)),
                             cutoff_utc="2026-08-01T00:00:00Z", tmax_hours=6.0)

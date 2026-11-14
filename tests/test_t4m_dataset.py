"""T4-M event-table tests: preserve uncertainty and protect time-forward scoring."""
from __future__ import annotations

import pandas as pd
import pytest

from vhagar.datasets.t4m import (
    T4M_FEATURE_COLUMNS,
    time_forward_split,
    training_rows,
    validate_t4m_observations,
)


def _rows():
    return pd.DataFrame({
        "fire_id": ["fire-a", "fire-a", "fire-a"],
        "observation_id": ["o1", "o2", "o3"],
        "timestamp_utc": ["2026-08-01T00:00:00Z", "2026-08-01T01:00:00Z", "2026-08-01T02:00:00Z"],
        "source": ["airborne", "airborne", "airborne"],
        "source_product": ["orbit-like", "orbit-like", "orbit-like"],
        "observation_kind": ["thermal_front", "thermal_front", "thermal_front"],
        "analysis_grid_id": ["fire-a-30m"] * 3,
        "row": [2, 3, 4], "col": [2, 3, 4],
        "first_seen_hours": [0.0, 1.0, 4.0],
        "time_sigma_min": [2.0, 10.0, 60.0],
        "location_sigma_m": [15.0, 15.0, 20.0],
        "quality": [1.0, 0.8, 0.4],
        "prior_arrival_hours": [0.0, 2.0, 2.0],
        "prior_ros_m_min": [1.0, 1.0, 1.0],
        "fuel_moisture": [0.06, 0.07, 0.08],
        "wind_midflame_ms": [3.0, 4.0, 5.0],
        "slope_tan": [0.1, 0.2, 0.3],
    })


def test_training_rows_excludes_seed_and_low_quality_and_retains_features():
    rows = training_rows(_rows(), min_quality=0.5)
    assert len(rows) == 1
    assert tuple(T4M_FEATURE_COLUMNS) == ("fuel_moisture", "wind_midflame_ms", "slope_tan")
    assert rows.loc[0, "log_ros_ratio"] == pytest.approx(__import__("math").log(2.0))
    assert 0 < rows.loc[0, "sample_weight"] < 0.8


def test_time_forward_split_never_exposes_future_observations_to_calibration():
    early, future = time_forward_split(_rows(), "2026-08-01T01:00:00+00:00")
    assert list(early["observation_id"]) == ["o1", "o2"]
    assert list(future["observation_id"]) == ["o3"]
    assert early["timestamp_utc"].max() < future["timestamp_utc"].min()


def test_contract_refuses_missing_provenance_and_bad_quality():
    with pytest.raises(ValueError, match="source_product"):
        validate_t4m_observations(_rows().drop(columns="source_product"))
    bad = _rows()
    bad.loc[1, "quality"] = 1.5
    with pytest.raises(ValueError, match="quality"):
        validate_t4m_observations(bad)

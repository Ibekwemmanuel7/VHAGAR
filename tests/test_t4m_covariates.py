from __future__ import annotations

import numpy as np
import pytest

from vhagar.datasets.t4m_covariates import (
    CovariateCube,
    select_forecast_trajectory,
    select_issue_time_covariates,
)
from vhagar.datasets.t4m_weather_io import load_cached_npz_weather_cube


def test_issue_time_uses_latest_available_not_future_weather():
    cube = CovariateCube(
        valid_time_utc=np.array(["2026-08-01T00:00Z", "2026-08-01T01:00Z"]),
        available_time_utc=np.array(["2026-08-01T00:20Z", "2026-08-01T01:20Z"]),
        wind_10m_ms=np.array([np.ones((2, 2)) * 4, np.ones((2, 2)) * 8]),
        relative_humidity=np.ones((2, 2, 2)) * 0.3,
        slope_tan=np.ones((2, 2)) * 0.2, aspect_rad=np.zeros((2, 2)), source="test",
    )
    out = select_issue_time_covariates(cube, "2026-08-01T01:00Z", wind_adjustment=0.25)
    assert np.all(out["wind_midflame_ms"] == 1.0)
    assert "00:20" in out["provenance"]["available_time_utc"]
    with pytest.raises(ValueError, match="no weather"):
        select_issue_time_covariates(cube, "2026-07-31T23:00Z", wind_adjustment=0.25)


def test_trajectory_uses_only_future_valid_frames_available_at_issue_time() -> None:
    cube = CovariateCube(
        valid_time_utc=np.array(["2026-08-01T01:00Z", "2026-08-01T02:00Z", "2026-08-01T03:00Z"]),
        available_time_utc=np.array(["2026-08-01T00:30Z", "2026-08-01T00:40Z", "2026-08-01T02:20Z"]),
        wind_10m_ms=np.array([np.ones((2, 2)) * 4, np.ones((2, 2)) * 8, np.ones((2, 2)) * 16]),
        relative_humidity=np.ones((3, 2, 2)) * 0.3,
        slope_tan=np.ones((2, 2)) * 0.2,
        aspect_rad=np.zeros((2, 2)),
        source="test",
    )
    out = select_forecast_trajectory(cube, "2026-08-01T01:00Z", horizon_hours=3, wind_adjustment=0.25)
    assert out["wind_midflame_ms"].shape == (2, 2, 2)
    assert np.all(out["wind_midflame_ms"][:, 0, 0] == [1.0, 2.0])


def test_npz_weather_loader_keeps_valid_and_available_times(tmp_path) -> None:
    terrain = tmp_path / "terrain.npz"
    weather = tmp_path / "weather.npz"
    np.savez(terrain, slope_tan=np.ones((2, 2)), aspect_rad=np.zeros((2, 2)))
    np.savez(
        weather,
        u10_ms=np.ones((2, 2)) * 3,
        v10_ms=np.ones((2, 2)) * 4,
        relative_humidity=np.ones((2, 2)) * 0.5,
        valid_time_utc=np.array("2026-08-01T01:00:00+00:00"),
        available_time_utc=np.array("2026-08-01T00:30:00+00:00"),
    )
    cube = load_cached_npz_weather_cube([weather], terrain_path=terrain, source="HRRR")
    assert cube.wind_10m_ms.shape == (1, 2, 2)
    assert cube.wind_10m_ms[0, 0, 0] == 5

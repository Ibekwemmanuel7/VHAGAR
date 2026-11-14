"""Issue-time covariate selection for T4-M replay and training.

This small contract prevents future weather leakage.  It accepts already
downloaded/regridded weather and terrain arrays (HRRR, ERA5, WRF, etc.) and
selects the latest weather frame available at a declared forecast issue time.
Acquisition/availability time is distinct from valid time so delayed products
cannot be used as if crews had them in real time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CovariateCube", "select_forecast_trajectory", "select_issue_time_covariates"]


@dataclass(frozen=True)
class CovariateCube:
    valid_time_utc: np.ndarray
    available_time_utc: np.ndarray
    wind_10m_ms: np.ndarray       # [T,H,W]
    relative_humidity: np.ndarray # [T,H,W], fraction
    slope_tan: np.ndarray         # [H,W]
    aspect_rad: np.ndarray        # [H,W]
    source: str


def _issue_time(value):
    import pandas as pd

    issue = pd.Timestamp(value)
    if issue.tzinfo is None:
        raise ValueError("issue_time_utc must include a timezone")
    return issue.tz_convert("UTC")


def _time_axes(cube: CovariateCube):
    import pandas as pd

    valid = pd.to_datetime(cube.valid_time_utc, utc=True)
    available = pd.to_datetime(cube.available_time_utc, utc=True)
    if len(valid) != len(available) or cube.wind_10m_ms.shape[0] != len(valid):
        raise ValueError("weather time axes must match wind_10m_ms")
    return valid, available


def select_issue_time_covariates(cube: CovariateCube, issue_time_utc, *, wind_adjustment: float):
    """Return the latest *available* weather state at issue time plus provenance.

    ``wind_adjustment`` must be declared by the caller: it converts the weather
    product's 10-m wind to midflame wind, and is never silently assumed.  RH is
    retained for fuel-moisture estimation/validation rather than fed directly
    into Rothermel.
    """
    issue = _issue_time(issue_time_utc)
    valid, available = _time_axes(cube)
    eligible = (available <= issue) & (valid <= issue)
    if not eligible.any():
        raise ValueError("no weather frame was available by the forecast issue time")
    idx = int(np.flatnonzero(eligible)[-1])
    wind = np.asarray(cube.wind_10m_ms[idx], dtype=float)
    rh = np.asarray(cube.relative_humidity[idx], dtype=float)
    if wind.shape != cube.slope_tan.shape or rh.shape != wind.shape or cube.aspect_rad.shape != wind.shape:
        raise ValueError("weather and terrain grids must share [H,W] shape")
    if not np.isfinite(wind).all() or not np.isfinite(rh).all():
        raise ValueError("selected weather frame contains non-finite values")
    if wind_adjustment <= 0 or wind_adjustment > 1:
        raise ValueError("wind_adjustment must be in (0, 1]")
    return {"wind_midflame_ms": wind * wind_adjustment,
            "relative_humidity": np.clip(rh, 0, 1),
            "slope_tan": np.asarray(cube.slope_tan, dtype=float),
            "aspect_rad": np.asarray(cube.aspect_rad, dtype=float),
            "provenance": {"weather_source": cube.source, "valid_time_utc": str(valid[idx]),
                           "available_time_utc": str(available[idx]), "issue_time_utc": str(issue),
                           "wind_adjustment": wind_adjustment}}


def select_forecast_trajectory(
    cube: CovariateCube, issue_time_utc, *, horizon_hours: float, wind_adjustment: float
) -> dict:
    """Select future-valid forecasts that were available at the issue time.

    This is the input contract for a time-varying spread solve. A forecast may
    be valid after the issue time, but it must have been available by then.
    Later analyses/reanalyses are excluded even if they look more accurate.
    """
    import pandas as pd

    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")
    if wind_adjustment <= 0 or wind_adjustment > 1:
        raise ValueError("wind_adjustment must be in (0, 1]")
    issue = _issue_time(issue_time_utc)
    valid, available = _time_axes(cube)
    horizon = issue + pd.Timedelta(hours=horizon_hours)
    selected = np.flatnonzero((available <= issue) & (valid >= issue) & (valid <= horizon))
    if len(selected) == 0:
        raise ValueError("no future-valid weather frame was available by the forecast issue time")
    selected = selected[np.argsort(valid[selected].to_numpy())]
    wind = np.asarray(cube.wind_10m_ms[selected], dtype=float)
    humidity = np.asarray(cube.relative_humidity[selected], dtype=float)
    expected = (len(selected), *cube.slope_tan.shape)
    if wind.shape != expected or humidity.shape != expected or not np.isfinite(wind).all():
        raise ValueError("selected weather and terrain grids must share [T,H,W] shape")
    return {
        "valid_time_utc": valid[selected].astype(str).to_numpy(),
        "wind_midflame_ms": wind * wind_adjustment,
        "relative_humidity": np.clip(humidity, 0, 1),
        "slope_tan": np.asarray(cube.slope_tan, dtype=float),
        "aspect_rad": np.asarray(cube.aspect_rad, dtype=float),
        "provenance": {
            "weather_source": cube.source,
            "issue_time_utc": str(issue),
            "horizon_hours": horizon_hours,
            "wind_adjustment": wind_adjustment,
            "n_frames": int(len(selected)),
        },
    }

"""Issue-time tensorisation for T4-M physics-guided neural arrival models.

Thermal observations are sparse, irregular, and time ordered.  This module turns
one fire at one declared issue time into a model input without filling unknown
cells with invented labels.  Future observations appear only in a masked target;
full-grid fuel/weather/terrain and a physics-prior arrival field are required
inputs, because front points alone cannot forecast unobserved terrain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vhagar.datasets.t4m import T4M_FEATURE_COLUMNS, time_forward_split, validate_t4m_observations

__all__ = ["T4MTensorCase", "tensorize_issue_time"]


@dataclass(frozen=True)
class T4MTensorCase:
    """One leakage-safe T4-M input/target pair for a declared forecast cutoff."""

    conditioning: np.ndarray       # [2 + 1 + C, H, W]: fire, detection time, physics, covariates
    target_arrival: np.ndarray     # [H, W], NaN where no held-out label exists
    target_valid: np.ndarray       # [H, W] boolean
    prior_ros: np.ndarray          # [H, W], used by the Eikonal loss
    tmax_hours: float
    fire_id: str
    cutoff_utc: str
    n_input_observations: int
    n_target_observations: int


def _grid(name: str, value, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"{name} must have grid shape {shape}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def tensorize_issue_time(
    observations,
    *,
    grid_shape: tuple[int, int],
    covariate_grids: dict[str, np.ndarray],
    prior_arrival_hours,
    prior_ros_m_min,
    cutoff_utc,
    tmax_hours: float,
    min_quality: float = 0.5,
) -> T4MTensorCase:
    """Create a leakage-safe neural example from one fire at one issue time.

    Conditioning channels are: observed-burned mask, normalised early detection
    time (zero away from observations), normalised physical-prior arrival time,
    then full-grid ``fuel_moisture``, ``wind_midflame_ms``, and ``slope_tan``.
    Target cells are *only* quality-filtered observations after ``cutoff_utc``.
    Their arrival labels retain the absolute event-origin clock; NaN means unknown
    rather than unburned.  A training loop must use ``target_valid`` as its mask.
    """
    if len(grid_shape) != 2 or min(grid_shape) < 1:
        raise ValueError("grid_shape must be a positive (H, W) tuple")
    if tmax_hours <= 0:
        raise ValueError("tmax_hours must be positive")
    if not 0 <= min_quality <= 1:
        raise ValueError("min_quality must lie in [0, 1]")
    df = validate_t4m_observations(observations)
    fire_ids = set(df["fire_id"])
    if len(fire_ids) != 1:
        raise ValueError("tensorize_issue_time requires exactly one fire")
    H, W = grid_shape
    prior_t = _grid("prior_arrival_hours", prior_arrival_hours, grid_shape)
    prior_ros = _grid("prior_ros_m_min", prior_ros_m_min, grid_shape)
    if (prior_ros < 0).any():
        raise ValueError("prior_ros_m_min must be non-negative everywhere")
    missing = [name for name in T4M_FEATURE_COLUMNS if name not in covariate_grids]
    if missing:
        raise ValueError(f"missing full-grid covariates: {', '.join(missing)}")
    grids = [_grid(name, covariate_grids[name], grid_shape) for name in T4M_FEATURE_COLUMNS]

    early, future = time_forward_split(df, cutoff_utc)
    early = early.loc[early["quality"] >= min_quality]
    future = future.loc[future["quality"] >= min_quality]
    obs = np.zeros(grid_shape, dtype=np.float32)
    det_time = np.zeros(grid_shape, dtype=np.float32)
    if len(early):
        r, c = early["row"].to_numpy(int), early["col"].to_numpy(int)
        if ((r < 0) | (r >= H) | (c < 0) | (c >= W)).any():
            raise ValueError("early observation row/col falls outside grid_shape")
        obs[r, c] = 1.0
        det_time[r, c] = np.clip(early["first_seen_hours"].to_numpy(float) / tmax_hours, 0.0, 1.0)

    target = np.full(grid_shape, np.nan, dtype=np.float32)
    valid = np.zeros(grid_shape, dtype=bool)
    if len(future):
        r, c = future["row"].to_numpy(int), future["col"].to_numpy(int)
        if ((r < 0) | (r >= H) | (c < 0) | (c >= W)).any():
            raise ValueError("future observation row/col falls outside grid_shape")
        # A cell can be emitted more than once by multiple sensors.  First seen
        # is the label; sorting in the contract makes this deterministic.
        for rr, cc, tt in zip(r, c, future["first_seen_hours"].to_numpy(float), strict=True):
            value = np.float32(np.clip(tt / tmax_hours, 0.0, 1.0))
            target[rr, cc] = value if not valid[rr, cc] else min(target[rr, cc], value)
            valid[rr, cc] = True

    prior_norm = np.clip(prior_t / tmax_hours, 0.0, 1.0)
    conditioning = np.stack([obs, det_time, prior_norm, *grids], axis=0).astype(np.float32)
    return T4MTensorCase(
        conditioning=conditioning,
        target_arrival=target,
        target_valid=valid,
        prior_ros=prior_ros,
        tmax_hours=float(tmax_hours),
        fire_id=next(iter(fire_ids)),
        cutoff_utc=str(cutoff_utc),
        n_input_observations=int(len(early)),
        n_target_observations=int(valid.sum()),
    )

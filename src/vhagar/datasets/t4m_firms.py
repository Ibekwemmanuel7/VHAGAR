"""Convert NASA FIRMS-style thermal detections into the T4-M event contract.

This adapter intentionally does *not* fabricate fire-weather or fuel inputs.
The caller must supply the issue-time physics-prior arrival/ROS and full-grid
covariates.  FIRMS provides the observation layer: timing, geolocation, sensor
metadata, confidence, and (where available) FRP.
"""

from __future__ import annotations

import numpy as np

from vhagar.datasets.spread_ingest import GridSpec
from vhagar.datasets.t4m import T4M_FEATURE_COLUMNS, validate_t4m_observations

__all__ = ["footprint_fuel_support", "firms_to_t4m_observations"]


def _timestamp(frame):
    import pandas as pd

    if not {"acq_date", "acq_time"}.issubset(frame.columns):
        raise ValueError("FIRMS frame needs acq_date and acq_time")
    times = pd.to_numeric(frame["acq_time"], errors="coerce")
    if times.isna().any():
        raise ValueError("FIRMS acq_time contains non-numeric values")
    # FIRMS encodes zero-padded HHMM as an integer, e.g. 822 == 08:22 UTC.
    hours, minutes = (times.astype(int) // 100), (times.astype(int) % 100)
    if ((hours > 23) | (minutes > 59)).any():
        raise ValueError("FIRMS acq_time has invalid HHMM values")
    return pd.to_datetime(frame["acq_date"], utc=True, errors="coerce") + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")


def _confidence(value) -> float:
    """Map FIRMS confidence strings/numbers to a documented conservative weight."""
    if isinstance(value, str):
        return {"h": 0.90, "n": 0.75, "l": 0.50}.get(value.strip().lower(), 0.50)
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.50
    return float(np.clip(x / 100.0 if x > 1.0 else x, 0.0, 1.0))


def _cell_values(name: str, value, shape: tuple[int, int], row, col) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(len(row), float(arr))
    if arr.shape != shape:
        raise ValueError(f"{name} must be scalar or have grid shape {shape}")
    return arr[row, col]


def footprint_fuel_support(ros, arrival, row: int, col: int, *, cell_size_m: tuple[float, float],
                           radius_m: float) -> tuple[float, float, float]:
    """Approximate thermal-pixel fuel support on a fine physical grid.

    Uses cell centres inside a circular footprint as an explicit, reproducible
    approximation to a sensor footprint.  Returns burnable-area fraction,
    mean positive ROS, and earliest finite burnable prior arrival.  It is more
    defensible than a centroid lookup, but a future sensor-specific polygon and
    off-nadir ellipse is still preferable.
    """
    r = np.asarray(ros, dtype=float)
    a = np.asarray(arrival, dtype=float)
    dy, dx = cell_size_m
    if radius_m <= 0 or dy <= 0 or dx <= 0:
        raise ValueError("footprint radius and grid cell sizes must be positive")
    ry, rx = int(np.ceil(radius_m / dy)), int(np.ceil(radius_m / dx))
    y0, y1 = max(0, row - ry), min(r.shape[0], row + ry + 1)
    x0, x1 = max(0, col - rx), min(r.shape[1], col + rx + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = ((yy - row) * dy) ** 2 + ((xx - col) * dx) ** 2 <= radius_m ** 2
    if not inside.any():
        return 0.0, 0.0, float("inf")
    values, times = r[y0:y1, x0:x1][inside], a[y0:y1, x0:x1][inside]
    burn = values > 0
    if not burn.any():
        return 0.0, 0.0, float("inf")
    return float(burn.mean()), float(values[burn].mean()), float(np.nanmin(times[burn]))


def firms_to_t4m_observations(
    firms,
    *,
    fire_id: str,
    spec: GridSpec,
    analysis_grid_id: str,
    prior_arrival_hours,
    prior_ros_m_min,
    covariate_grids: dict[str, np.ndarray],
    source: str = "NASA FIRMS",
    source_product: str = "VIIRS active fire",
    time_sigma_min: float = 360.0,
    location_sigma_m: float = 375.0,
    footprint_radius_m: float | None = None,
    min_burnable_support: float = 0.10,
):
    """Rasterise real FIRMS first-seen detections into one row per grid cell.

    The stated default timing uncertainty reflects sparse polar-orbiter revisit,
    not the timestamp precision.  It should be replaced by product-specific
    cadence/latency evidence when using another sensor.  ``location_sigma_m`` is
    likewise an explicit nominal footprint scale, not an accuracy certification.
    """
    import pandas as pd

    if time_sigma_min < 0 or location_sigma_m < 0:
        raise ValueError("FIRMS uncertainty values must be non-negative")
    if not {"latitude", "longitude"}.issubset(firms.columns):
        raise ValueError("FIRMS frame needs latitude and longitude")
    missing = [name for name in T4M_FEATURE_COLUMNS if name not in covariate_grids]
    if missing:
        raise ValueError(f"missing full-grid covariates: {', '.join(missing)}")
    shape = spec.shape
    # Validate grids up front, including the physics prior that produces targets.
    for name, value in [("prior_arrival_hours", prior_arrival_hours), ("prior_ros_m_min", prior_ros_m_min),
                        *((name, covariate_grids[name]) for name in T4M_FEATURE_COLUMNS)]:
        arr = np.asarray(value)
        if arr.shape != shape or not np.isfinite(arr).all():
            raise ValueError(f"{name} must be a finite grid with shape {shape}")
    if (np.asarray(prior_ros_m_min) < 0).any():
        raise ValueError("prior_ros_m_min must be non-negative")

    df = firms.copy()
    df["timestamp_utc"] = _timestamp(df)
    lon = pd.to_numeric(df["longitude"], errors="coerce").to_numpy()
    lat = pd.to_numeric(df["latitude"], errors="coerce").to_numpy()
    row, col, inside = spec.lonlat_to_rc(lon, lat)
    keep = inside & df["timestamp_utc"].notna().to_numpy() & np.isfinite(lon) & np.isfinite(lat)
    df, row, col = df.loc[keep].copy(), row[keep], col[keep]
    if df.empty:
        raise ValueError("no finite FIRMS detections lie inside the analysis grid")
    df["row"], df["col"] = row, col
    df = df.sort_values("timestamp_utc", kind="stable")
    # One earliest first-seen observation per analysis cell.  Repeated later heat
    # belongs in a future intensity/front product, not an arrival-time target.
    df = df.drop_duplicates(["row", "col"], keep="first").reset_index(drop=True)
    row, col = df["row"].to_numpy(int), df["col"].to_numpy(int)
    cell_ros = _cell_values("prior_ros_m_min", prior_ros_m_min, shape, row, col)
    cell_arrival = _cell_values("prior_arrival_hours", prior_arrival_hours, shape, row, col)
    support = np.where(cell_ros > 0, 1.0, 0.0)
    nonburnable = cell_ros <= 0.0
    if nonburnable.any():
        if footprint_radius_m is not None and footprint_radius_m > 0:
            dy, dx = spec.cell_size_m()
            for i in np.flatnonzero(nonburnable):
                frac, effective_ros, effective_arrival = footprint_fuel_support(
                    prior_ros_m_min, prior_arrival_hours, int(row[i]), int(col[i]),
                    cell_size_m=(dy, dx), radius_m=footprint_radius_m,
                )
                support[i] = frac
                if frac >= min_burnable_support and effective_ros > 0 and np.isfinite(effective_arrival):
                    cell_ros[i], cell_arrival[i] = effective_ros, effective_arrival
            nonburnable = cell_ros <= 0.0
        if nonburnable.any():
            raise ValueError(
                f"{int(nonburnable.sum())} of {len(df)} earliest FIRMS cells map to zero-Ros "
                "LANDFIRE fuel. Refusing to drop or invent a surface-fire rate for them. "
                "Use a finer/footprint-aware fuel intersection or treat this incident as outside "
                "the surface-fire model's supported fuel domain."
            )
    origin = df["timestamp_utc"].min()
    first_seen = (df["timestamp_utc"] - origin).dt.total_seconds().to_numpy() / 3600.0
    confidence = df["confidence"] if "confidence" in df.columns else [0.50] * len(df)
    product = (df["satellite"].astype(str) + " " + df["instrument"].astype(str)
               if {"satellite", "instrument"}.issubset(df.columns) else source_product)
    out = pd.DataFrame({
        "fire_id": fire_id,
        "observation_id": [f"{fire_id}:firms:{r}:{c}:{t.isoformat()}" for r, c, t in zip(row, col, df["timestamp_utc"], strict=True)],
        "timestamp_utc": df["timestamp_utc"], "source": source,
        "source_product": product, "observation_kind": "first_seen_point",
        "analysis_grid_id": analysis_grid_id, "row": row, "col": col,
        "first_seen_hours": first_seen, "time_sigma_min": float(time_sigma_min),
        "location_sigma_m": float(location_sigma_m),
        "quality": [_confidence(x) for x in confidence],
        "prior_arrival_hours": cell_arrival,
        "prior_ros_m_min": cell_ros,
        **{name: _cell_values(name, covariate_grids[name], shape, row, col) for name in T4M_FEATURE_COLUMNS},
        # Extra provenance fields are preserved by the contract and useful for
        # later sensor-specific quality models.
        "latitude": lat[keep][df.index], "longitude": lon[keep][df.index],
        "frp_mw": pd.to_numeric(df["frp"], errors="coerce") if "frp" in df.columns else np.nan,
        "fuel_support_fraction": support,
    })
    return validate_t4m_observations(out)

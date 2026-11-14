"""T4-M event table: auditable training data for thermal spread correction.

The unit of learning is one observed front cell at one time, not an arbitrary
pixel.  This schema preserves the observation process alongside the physical
covariates and the physics prior, so a model can be reproduced, filtered, and
evaluated incident by incident.  It accommodates satellite point detections,
airborne thermal front extractions, or mapped perimeter cells without pretending
they have identical location or timing error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "T4M_REQUIRED_COLUMNS",
    "T4M_FEATURE_COLUMNS",
    "validate_t4m_observations",
    "training_rows",
    "time_forward_split",
    "write_t4m_parquet",
]


# Identity, provenance, and observation uncertainty are required.  A residual
# without these fields cannot later be traced to a sensor/product or correctly
# down-weighted for timing and geolocation uncertainty.
T4M_REQUIRED_COLUMNS = (
    "fire_id",
    "observation_id",
    "timestamp_utc",
    "source",
    "source_product",
    "observation_kind",
    "analysis_grid_id",
    "row",
    "col",
    "first_seen_hours",
    "time_sigma_min",
    "location_sigma_m",
    "quality",
    "prior_arrival_hours",
    "prior_ros_m_min",
    "fuel_moisture",
    "wind_midflame_ms",
    "slope_tan",
)

# The initial residual baseline uses only variables available at forecast issue
# time.  No future perimeter geometry, final area, or post-cutoff observation
# may be introduced here.
T4M_FEATURE_COLUMNS = ("fuel_moisture", "wind_midflame_ms", "slope_tan")
_KINDS = {"first_seen_point", "thermal_front", "perimeter_cell"}


def _frame(frame):
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("T4-M observations must be a pandas DataFrame")
    return frame.copy()


def validate_t4m_observations(frame):
    """Validate and normalise a T4-M observation frame.

    Raises a descriptive ``ValueError`` rather than filling missing scientific
    inputs.  Timestamps are converted to UTC, identity is unique within a fire,
    and quality / uncertainty values are constrained to physical ranges.
    """
    df = _frame(frame)
    missing = sorted(set(T4M_REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"T4-M observations missing required columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("T4-M observations are empty")

    import pandas as pd

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    finite = ("first_seen_hours", "time_sigma_min", "location_sigma_m", "quality",
              "prior_arrival_hours", "prior_ros_m_min", *T4M_FEATURE_COLUMNS)
    for col in finite:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[list(finite)].isna().any().any() or df["timestamp_utc"].isna().any():
        raise ValueError("T4-M observations contain missing or non-numeric scientific values")
    if (df["first_seen_hours"] < 0).any() or (df["prior_arrival_hours"] < 0).any():
        raise ValueError("arrival times must be non-negative hours since event origin")
    if (df["time_sigma_min"] < 0).any() or (df["location_sigma_m"] < 0).any():
        raise ValueError("observation uncertainty must be non-negative")
    if ((df["quality"] < 0) | (df["quality"] > 1)).any():
        raise ValueError("quality must lie in [0, 1]")
    if (df["prior_ros_m_min"] <= 0).any():
        raise ValueError("prior_ros_m_min must be positive")
    if not set(df["observation_kind"].astype(str)).issubset(_KINDS):
        raise ValueError(f"observation_kind must be one of {sorted(_KINDS)}")
    if df.duplicated(["fire_id", "observation_id"]).any():
        raise ValueError("observation_id must be unique within each fire")
    return df.sort_values(["fire_id", "timestamp_utc", "observation_id"], kind="stable").reset_index(drop=True)


def training_rows(frame, *, min_quality: float = 0.5):
    """Return forecast-available features, residual target, and uncertainty weight.

    Seed cells have a zero arrival time and cannot define a ROS ratio, so they
    are excluded.  The target is ``log(prior_arrival / first_seen)``.  The sample
    weight combines the supplied quality with inverse timing uncertainty; it does
    not turn uncertainty into an invented label.
    """
    if not 0 <= min_quality <= 1:
        raise ValueError("min_quality must lie in [0, 1]")
    df = validate_t4m_observations(frame)
    keep = ((df["quality"] >= min_quality) & (df["first_seen_hours"] > 0)
            & (df["prior_arrival_hours"] > 0))
    out = df.loc[keep, ["fire_id", "observation_id", "timestamp_utc", *T4M_FEATURE_COLUMNS]].copy()
    selected = df.loc[keep]
    out["log_ros_ratio"] = np.log(
        selected["prior_arrival_hours"].to_numpy() / selected["first_seen_hours"].to_numpy()
    )
    # A one-hour timing sigma gets half the weight of a zero-sigma observation;
    # location error remains explicit in the source table for a future footprint
    # likelihood rather than being quietly folded into this first baseline.
    out["sample_weight"] = selected["quality"].to_numpy() / (
        1.0 + selected["time_sigma_min"].to_numpy() / 60.0
    )
    return out.reset_index(drop=True)


def time_forward_split(frame, cutoff_utc):
    """Split one incident into calibration and future observations by real time.

    The cutoff is supplied by the caller and recorded in its run manifest.  Rows
    at or before it are calibration data; later rows are held-out forecast truth.
    This guards against a subtle but common spread-model leak: fitting a forecast
    with a perimeter observed after its stated issue time.
    """
    df = validate_t4m_observations(frame)
    import pandas as pd

    cutoff = pd.Timestamp(cutoff_utc)
    if cutoff.tzinfo is None:
        raise ValueError("cutoff_utc must include a timezone")
    cutoff = cutoff.tz_convert("UTC")
    early = df["timestamp_utc"] <= cutoff
    return df.loc[early].reset_index(drop=True), df.loc[~early].reset_index(drop=True)


def write_t4m_parquet(frame, path: str | Path) -> Path:
    """Validate then write a portable, versionable T4-M event table."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    validate_t4m_observations(frame).to_parquet(out, index=False)
    return out

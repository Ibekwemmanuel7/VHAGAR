"""Read cached ERA5/HRRR-style weather cubes into the T4-M availability contract."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vhagar.datasets.t4m_covariates import CovariateCube

__all__ = ["load_cached_npz_weather_cube", "load_cached_weather_cube"]


def load_cached_npz_weather_cube(
    weather_paths: list[str | Path], *, terrain_path: str | Path, source: str
) -> CovariateCube:
    """Load regridded HRRR-style NPZ frames plus a same-grid terrain package.

    Each weather frame must retain ``valid_time_utc`` and
    ``available_time_utc``. Frames are sorted by valid time; the issue-time
    selector, not this loader, decides which frames are operationally eligible.
    """
    if not weather_paths:
        raise ValueError("at least one weather package is required")
    with np.load(terrain_path) as terrain:
        for name in ("slope_tan", "aspect_rad"):
            if name not in terrain:
                raise ValueError(f"terrain package missing {name}")
        slope = np.asarray(terrain["slope_tan"], dtype=float)
        aspect = np.asarray(terrain["aspect_rad"], dtype=float)

    frames = []
    for path in weather_paths:
        with np.load(path) as weather:
            required = {"u10_ms", "v10_ms", "relative_humidity", "valid_time_utc", "available_time_utc"}
            missing = required - set(weather.files)
            if missing:
                raise ValueError(f"weather package {path} missing {', '.join(sorted(missing))}")
            u10 = np.asarray(weather["u10_ms"], dtype=float)
            v10 = np.asarray(weather["v10_ms"], dtype=float)
            humidity = np.asarray(weather["relative_humidity"], dtype=float)
            if u10.shape != slope.shape or v10.shape != slope.shape or humidity.shape != slope.shape:
                raise ValueError(f"weather package {path} does not match terrain shape")
            frames.append(
                (
                    str(np.asarray(weather["valid_time_utc"]).item()),
                    str(np.asarray(weather["available_time_utc"]).item()),
                    np.hypot(u10, v10),
                    humidity,
                )
            )
    frames.sort(key=lambda frame: frame[0])
    return CovariateCube(
        valid_time_utc=np.asarray([frame[0] for frame in frames]),
        available_time_utc=np.asarray([frame[1] for frame in frames]),
        wind_10m_ms=np.asarray([frame[2] for frame in frames]),
        relative_humidity=np.asarray([frame[3] for frame in frames]),
        slope_tan=slope,
        aspect_rad=aspect,
        source=source,
    )


def load_cached_weather_cube(
    path: str | Path,
    *,
    source: str,
    available_time_utc,
    wind_u: str = "u10",
    wind_v: str = "v10",
    humidity: str = "r",
    slope: str = "slope_tan",
    aspect: str = "aspect_rad",
    time: str = "time",
) -> CovariateCube:
    """Load a pre-fetched NetCDF/Zarr cube; downloading is intentionally separate."""
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover
        raise ImportError("cached weather ingest needs vhagar[geo]") from exc
    path = Path(path)
    ds = xr.open_zarr(path) if path.suffix == ".zarr" else xr.open_dataset(path)
    for name in (wind_u, wind_v, humidity, slope, aspect, time):
        if name not in ds:
            raise ValueError(f"weather cube missing {name}")
    u, v = np.asarray(ds[wind_u]), np.asarray(ds[wind_v])
    if u.shape != v.shape or u.ndim != 3:
        raise ValueError("wind components must be matching [time,y,x] arrays")
    available = np.asarray(available_time_utc)
    if available.ndim == 0:
        available = np.repeat(available, u.shape[0])
    return CovariateCube(
        np.asarray(ds[time].values),
        available,
        np.hypot(u, v),
        np.asarray(ds[humidity]),
        np.asarray(ds[slope]),
        np.asarray(ds[aspect]),
        source,
    )

"""Create declared T4-M physics grids from LANDFIRE fuel, weather assumptions, and an issue time.

This is a reproducible *scenario builder*, not a weather-data downloader.  Wind,
fuel moisture, and terrain slope must be declared on the command line and are
written to a JSON sidecar.  For a defensible replay, supply values extracted from
the relevant weather and DEM products; a single wind and zero slope remain named
limitations, not hidden defaults.
"""

from __future__ import annotations

import argparse
import json
import os
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd

from vhagar.datasets.spread_ingest import GridSpec, ignition_from_detections, rasterize_detections
from vhagar.models.rothermel import fbfm13_to_fbfm40_proxy, rothermel_head_ros_field
from vhagar.models.spread import fast_marching_arrival


def _bbox(value: str) -> tuple[float, float, float, float]:
    try:
        x = tuple(float(v) for v in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north") from exc
    if len(x) != 4 or x[0] >= x[2] or x[1] >= x[3]:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north with positive area")
    return x


def _timestamps(frame):
    t = pd.to_numeric(frame["acq_time"], errors="coerce")
    return (pd.to_datetime(frame["acq_date"], utc=True, errors="coerce")
            + pd.to_timedelta(t.astype(int) // 100, unit="h")
            + pd.to_timedelta(t.astype(int) % 100, unit="m"))


def _fuel_codes(spec: GridSpec, raster_path: str) -> np.ndarray:
    """Nearest-neighbour sample FBFM40 codes to the regular lon/lat analysis grid."""
    # Some Windows GIS installs export PROJ_LIB globally (for example PostGIS),
    # which can be ABI-incompatible with Rasterio's bundled GDAL.  Select the
    # Rasterio wheel's matching projection database for this process only.
    rasterio_spec = find_spec("rasterio")
    if rasterio_spec is not None and rasterio_spec.origin:
        bundled_proj = Path(rasterio_spec.origin).parent / "proj_data"
        if bundled_proj.is_dir():
            os.environ["PROJ_LIB"] = str(bundled_proj)
            os.environ["PROJ_DATA"] = str(bundled_proj)
    import rasterio
    from rasterio.warp import transform as project
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window

    with rasterio.open(raster_path) as ds:
        h, w = spec.shape
        west, south, east, north = spec.bbox
        rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        lon = west + (cc.ravel() + 0.5) / w * (east - west)
        lat = north - (rr.ravel() + 0.5) / h * (north - south)
        x0, y0, x1, y1 = transform_bounds("EPSG:4326", ds.crs, west, south, east, north)
        win = ds.window(x0, y0, x1, y1).round_offsets().round_lengths()
        win = Window(win.col_off - 2, win.row_off - 2, win.width + 4, win.height + 4)
        raw = ds.read(1, window=win, boundless=True, fill_value=91)
        xx, yy = project("EPSG:4326", ds.crs, list(lon), list(lat))
        col, row = (~ds.window_transform(win)) * (np.asarray(xx), np.asarray(yy))
        col = np.clip(np.round(col).astype(int), 0, raw.shape[1] - 1)
        row = np.clip(np.round(row).astype(int), 0, raw.shape[0] - 1)
    return raw[row, col].reshape(spec.shape).astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("firms_csv")
    ap.add_argument("--bbox", type=_bbox, required=True)
    ap.add_argument("--cell-deg", type=float, required=True)
    ap.add_argument("--cutoff-utc", required=True, help="forecast issue time, timezone required")
    ap.add_argument("--lf-fbfm40", required=True, help="LANDFIRE FBFM40 GeoTIFF or /vsizip/ path")
    ap.add_argument(
        "--fuel-model-scheme", choices=("FBFM40", "FBFM13"), default="FBFM40",
        help="raster coding scheme; FBFM13 is mapped to declared representative FBFM40 groups",
    )
    wind_group = ap.add_mutually_exclusive_group(required=True)
    wind_group.add_argument("--wind-midflame-ms", type=float, help="declared uniform midflame wind")
    wind_group.add_argument(
        "--weather-npz",
        help="same-grid weather package from t4m_hrrr_to_weather.py; uses 10 m U/V wind",
    )
    ap.add_argument("--fuel-moisture", type=float, required=True, help="dead-fuel moisture fraction")
    ap.add_argument(
        "--ellipse-wind-reference-ms",
        type=float,
        default=5.0,
        help="midflame wind mapped to ellipse index 1; must be calibrated/declared",
    )
    terrain_group = ap.add_mutually_exclusive_group(required=True)
    terrain_group.add_argument(
        "--slope-tan",
        type=float,
        help="declared uniform rise/run; use only for an explicitly labelled scenario",
    )
    terrain_group.add_argument(
        "--terrain-npz",
        help="same-grid terrain package from t4m_terrain_grid.py; uses its slope_tan array",
    )
    ap.add_argument("--wind-adjustment", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cell_deg <= 0 or args.fuel_moisture < 0 or args.ellipse_wind_reference_ms <= 0:
        ap.error("cell size and moisture must be non-negative (cell size positive)")
    if args.wind_midflame_ms is not None and args.wind_midflame_ms < 0:
        ap.error("--wind-midflame-ms must be non-negative")
    if args.slope_tan is not None and args.slope_tan < 0:
        ap.error("--slope-tan must be non-negative")
    cutoff = pd.Timestamp(args.cutoff_utc)
    if cutoff.tzinfo is None:
        ap.error("--cutoff-utc must include a timezone, e.g. 2026-08-03T00:00:00Z")
    cutoff = cutoff.tz_convert("UTC")
    spec = GridSpec.from_bbox_res(args.bbox, args.cell_deg)
    df = pd.read_csv(args.firms_csv)
    df["_time"] = _timestamps(df)
    df = df.loc[df["_time"].notna() & (df["_time"] <= cutoff)].copy()
    if df.empty:
        ap.error("no FIRMS detections at or before cutoff")
    origin = df["_time"].min()
    hours = (df["_time"] - origin).dt.total_seconds().to_numpy() / 3600.0
    rc, t = rasterize_detections(df.longitude, df.latitude, hours, spec)
    if len(rc) < 2:
        ap.error("fewer than two in-grid, pre-cutoff first-seen detections")
    ignition = ignition_from_detections(rc, t, spec, seed_quantile=0.05)
    source_codes = _fuel_codes(spec, args.lf_fbfm40)
    codes = fbfm13_to_fbfm40_proxy(source_codes) if args.fuel_model_scheme == "FBFM13" else source_codes
    moisture = np.full(spec.shape, args.fuel_moisture, dtype=np.float32)
    if args.weather_npz:
        with np.load(args.weather_npz) as weather:
            required_weather = {"u10_ms", "v10_ms", "relative_humidity"}
            missing_weather = required_weather - set(weather.files)
            if missing_weather:
                ap.error(f"--weather-npz missing {', '.join(sorted(missing_weather))}")
            wind = np.hypot(weather["u10_ms"], weather["v10_ms"]).astype(np.float32)
            wind_dir_grid_rad = np.arctan2(-weather["v10_ms"], weather["u10_ms"]).astype(np.float32)
            relative_humidity = np.asarray(weather["relative_humidity"], dtype=np.float32)
        if wind.shape != spec.shape or not np.isfinite(wind).all():
            ap.error("weather wind must be finite and match the analysis grid")
        if relative_humidity.shape != spec.shape or not np.isfinite(relative_humidity).all():
            ap.error("weather RH must be finite and match the analysis grid")
        weather_source = args.weather_npz
        wind_mode = "gridded_10m_weather"
    else:
        wind = np.full(spec.shape, args.wind_midflame_ms, dtype=np.float32)
        wind_dir_grid_rad = np.zeros(spec.shape, dtype=np.float32)
        relative_humidity = np.full(spec.shape, np.nan, dtype=np.float32)
        weather_source = None
        wind_mode = "uniform_midflame_scenario"
    if args.terrain_npz:
        with np.load(args.terrain_npz) as terrain:
            if "slope_tan" not in terrain:
                ap.error("--terrain-npz must contain slope_tan")
            slope = np.asarray(terrain["slope_tan"], dtype=np.float32)
        if slope.shape != spec.shape or not np.isfinite(slope).all() or (slope < 0).any():
            ap.error("terrain slope_tan must be finite, non-negative, and match the analysis grid")
        terrain_source = args.terrain_npz
        terrain_mode = "gridded_dem"
    else:
        slope = np.full(spec.shape, args.slope_tan, dtype=np.float32)
        terrain_source = None
        terrain_mode = "uniform_scenario"
    ros_wind_adjustment = args.wind_adjustment if args.weather_npz else 1.0
    midflame_wind = wind * ros_wind_adjustment
    wind_ellipse_index = np.clip(midflame_wind / args.ellipse_wind_reference_ms, 0, 1)
    ros = rothermel_head_ros_field(codes, m_f=moisture, wind_ms=wind, slope_tan=slope,
                                   wind_adj=ros_wind_adjustment).astype(np.float32)
    # FMM represents a zero-ROS nonburnable cell as a practically unreachable
    # arrival.  Keep the zero ROS in the package; neural losses mask/clamp it.
    arrival = fast_marching_arrival(ros, ignition, dx=np.mean(spec.cell_size_m())).astype(np.float32) / 60.0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, fuel_model_code=codes, source_fuel_model_code=source_codes, ignition_seed=ignition,
                        prior_arrival_hours=arrival, prior_ros_m_min=ros,
                        fuel_moisture=moisture, wind_input_ms=wind,
                        wind_midflame_ms=midflame_wind, relative_humidity=relative_humidity,
                        wind_dir_grid_rad=wind_dir_grid_rad, wind_ellipse_index=wind_ellipse_index,
                        slope_tan=slope)
    # The arrival solver uses one effective grid spacing. Persist it with the
    # case rather than letting a downstream dynamic replay inherit the spacing
    # of whichever example happened to be built first. A lon/lat grid is not
    # square away from the equator, so this is the explicit mean-spacing
    # approximation used by the present 8-neighbour solver.
    cell_y_m, cell_x_m = spec.cell_size_m()
    metadata = {"bbox": args.bbox, "shape": spec.shape, "cell_deg": args.cell_deg,
                "cell_size_m": {"north_south": cell_y_m, "east_west": cell_x_m,
                                "effective_isotropic": float(np.mean((cell_y_m, cell_x_m)))},
                "grid_spacing_convention": (
                    "mean north-south/east-west cell size; replace with a projected "
                    "or anisotropic metric solver for large domains"),
                "cutoff_utc": cutoff.isoformat(), "n_pre_cutoff_cells": int(len(rc)),
                "fuel": args.lf_fbfm40,
                "fuel_model_scheme": args.fuel_model_scheme,
                "fuel_model_interpretation": (
                    "native Scott and Burgan FBFM40 codes"
                    if args.fuel_model_scheme == "FBFM40" else
                    "Anderson FBFM13 mapped to representative FBFM40 fuel groups; retain as a model-form limitation"
                ),
                "wind_mode": wind_mode,
                "weather_source": weather_source,
                "wind_midflame_ms": args.wind_midflame_ms,
                "fuel_moisture": args.fuel_moisture,
                "terrain_mode": terrain_mode,
                "terrain_source": terrain_source,
                "slope_tan": args.slope_tan,
                "wind_adjustment": ros_wind_adjustment,
                "ellipse_wind_reference_ms": args.ellipse_wind_reference_ms,
                "limitations": (
                    "fuel moisture remains a declared scalar; wind is "
                    f"{wind_mode}; terrain is "
                    f"{terrain_mode}. Replace with issue-time gridded weather and validate."
                )}
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {out}: {spec.shape[0]}x{spec.shape[1]} physics grid; {len(rc)} pre-cutoff detected cells")
    print(f"wrote {out.with_suffix('.json')}: provenance and declared limitations")


if __name__ == "__main__":
    main()

"""Build a time-varying Rothermel/ellipse forecast tensor from cached HRRR frames.

Each weather package must have been extracted to the same T4-M grid and retain
its valid and availability timestamps. This script preserves those timestamps;
it does not decide whether they were operationally available at an issue time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vhagar.models.rothermel import rothermel_head_ros_field


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-physics", required=True)
    parser.add_argument("--weather-npz", nargs="+", required=True)
    parser.add_argument("--wind-adjustment", type=float, required=True)
    parser.add_argument("--ros-scale", type=float, default=1.0, help="pre-cutoff calibrated multiplicative ROS scale")
    parser.add_argument("--ellipse-wind-reference-ms", type=float, default=5.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not 0 < args.wind_adjustment <= 1 or args.ellipse_wind_reference_ms <= 0 or args.ros_scale <= 0:
        parser.error("wind adjustment must be in (0, 1] and ellipse reference must be positive")

    with np.load(args.base_physics) as base:
        required = {"fuel_model_code", "fuel_moisture", "slope_tan"}
        missing = required - set(base.files)
        if missing:
            parser.error(f"base physics missing {', '.join(sorted(missing))}; rebuild it with current script")
        fuel_code = np.asarray(base["fuel_model_code"])
        fuel_moisture = np.asarray(base["fuel_moisture"])
        slope_tan = np.asarray(base["slope_tan"])

    frames = []
    for weather_path in args.weather_npz:
        with np.load(weather_path) as weather:
            required = {"u10_ms", "v10_ms", "relative_humidity", "valid_time_utc", "available_time_utc"}
            missing = required - set(weather.files)
            if missing:
                parser.error(f"weather {weather_path} missing {', '.join(sorted(missing))}")
            u10 = np.asarray(weather["u10_ms"], dtype=float)
            v10 = np.asarray(weather["v10_ms"], dtype=float)
            if u10.shape != fuel_code.shape or v10.shape != fuel_code.shape:
                parser.error(f"weather {weather_path} does not match base physics grid")
            midflame_wind = np.hypot(u10, v10) * args.wind_adjustment
            frames.append(
                {
                    "valid_time_utc": str(np.asarray(weather["valid_time_utc"]).item()),
                    "available_time_utc": str(np.asarray(weather["available_time_utc"]).item()),
                    "head_ros_m_min": args.ros_scale * rothermel_head_ros_field(
                        fuel_code, m_f=fuel_moisture, wind_ms=np.hypot(u10, v10),
                        slope_tan=slope_tan, wind_adj=args.wind_adjustment,
                    ),
                    "wind_midflame_ms": midflame_wind,
                    "wind_dir_grid_rad": np.arctan2(-v10, u10),
                    "wind_ellipse_index": np.clip(
                        midflame_wind / args.ellipse_wind_reference_ms, 0, 1
                    ),
                }
            )
    frames.sort(key=lambda frame: frame["valid_time_utc"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        valid_time_utc=np.asarray([frame["valid_time_utc"] for frame in frames]),
        available_time_utc=np.asarray([frame["available_time_utc"] for frame in frames]),
        head_ros_m_min=np.asarray([frame["head_ros_m_min"] for frame in frames]),
        wind_midflame_ms=np.asarray([frame["wind_midflame_ms"] for frame in frames]),
        wind_dir_grid_rad=np.asarray([frame["wind_dir_grid_rad"] for frame in frames]),
        wind_ellipse_index=np.asarray([frame["wind_ellipse_index"] for frame in frames]),
    )
    metadata = {
        "base_physics": args.base_physics,
        "weather_packages": args.weather_npz,
        "wind_adjustment": args.wind_adjustment,
        "ros_scale": args.ros_scale,
        "ellipse_wind_reference_ms": args.ellipse_wind_reference_ms,
        "n_frames": len(frames),
        "availability_note": "Eligibility for an operational replay must be checked against declared issue time.",
    }
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {out}: {len(frames)} time-varying forecast frame(s)")


if __name__ == "__main__":
    main()

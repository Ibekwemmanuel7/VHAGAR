"""Run VHAGAR's segmented anisotropic forecast from a dynamic physics package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vhagar.models.spread import dynamic_anisotropic_forecast


def _effective_grid_spacing(base_physics: str) -> float:
    """Read the spacing declared by ``t4m_build_physics_grid.py``.

    Dynamic propagation must not assume the resolution of an unrelated case.
    The current discrete solver takes one scalar ``dx``; the physics builder
    therefore records its stated mean metric spacing. Older packages without
    that provenance are intentionally rejected instead of receiving a hidden
    333 m default.
    """
    metadata_path = Path(base_physics).with_suffix(".json")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        spacing = float(metadata["cell_size_m"]["effective_isotropic"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{metadata_path} lacks declared cell_size_m.effective_isotropic; "
            "rebuild the base physics package with t4m_build_physics_grid.py"
        ) from exc
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"{metadata_path} has an invalid effective grid spacing")
    return spacing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic-npz", required=True)
    parser.add_argument("--base-physics", required=True)
    parser.add_argument("--frame-hours", type=float, default=None, help="constant duration; default derives hourly cadence")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        cell_size_m = _effective_grid_spacing(args.base_physics)
    except ValueError as exc:
        parser.error(str(exc))
    with np.load(args.dynamic_npz) as dynamic, np.load(args.base_physics) as base:
        required_dynamic = {"valid_time_utc", "head_ros_m_min", "wind_dir_grid_rad", "wind_ellipse_index"}
        missing = required_dynamic - set(dynamic.files)
        if missing or "ignition_seed" not in base:
            parser.error("dynamic package or base physics lacks required forecast/ignition fields")
        valid = pd.to_datetime(dynamic["valid_time_utc"], utc=True)
        if args.frame_hours is None:
            if len(valid) < 2:
                parser.error("one-frame forecast needs explicit --frame-hours")
            durations = np.diff(valid.asi8) / 3.6e12
            durations = np.append(durations, durations[-1])
        else:
            if args.frame_hours <= 0:
                parser.error("--frame-hours must be positive")
            durations = np.full(len(valid), args.frame_hours)
        burned, arrival = dynamic_anisotropic_forecast(
            dynamic["head_ros_m_min"], dynamic["wind_ellipse_index"], dynamic["wind_dir_grid_rad"],
            base["ignition_seed"], durations, dx=cell_size_m,
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, burned_mask=burned, arrival_hours=arrival, valid_time_utc=np.asarray(valid.astype(str)))
    out.with_suffix(".json").write_text(
        json.dumps({"dynamic_physics": args.dynamic_npz, "base_physics": args.base_physics,
                    "frame_hours": durations.tolist(), "cell_size_m": cell_size_m,
                    "n_burned_cells": int(burned.sum()),
                    "scope": "research forecast; no suppression, spotting, or coupled-atmosphere solve"}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out}: {int(burned.sum())} burned cells after {durations.sum():.2f} h")


if __name__ == "__main__":
    main()

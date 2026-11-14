"""Build a T4-M event table from a real NASA FIRMS CSV and declared physics grids.

This script has no nominal-fuel or nominal-wind fallback.  The supplied ``.npz``
must contain the issue-time grids ``prior_arrival_hours``, ``prior_ros_m_min``,
``fuel_moisture``, ``wind_midflame_ms``, and ``slope_tan`` on the exact analysis
grid.  That refusal is intentional: a real thermal observation table is useful
only when its physical inputs and provenance are equally real and versioned.

Example:
  python scripts/t4m_from_firms.py viirs_truth.csv --fire-id example-2026 \
    --bbox -121,38,-120,39 --cell-deg 0.01 --physics-npz example_grids.npz \
    --out data/t4m/example-2026.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vhagar.datasets.spread_ingest import GridSpec
from vhagar.datasets.t4m import write_t4m_parquet
from vhagar.datasets.t4m_firms import firms_to_t4m_observations

_REQUIRED = ("prior_arrival_hours", "prior_ros_m_min", "fuel_moisture", "wind_midflame_ms", "slope_tan")


def _bbox(value: str) -> tuple[float, float, float, float]:
    try:
        parts = tuple(float(x) for x in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north") from exc
    if len(parts) != 4 or parts[0] >= parts[2] or parts[1] >= parts[3]:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north with positive area")
    return parts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("firms_csv")
    ap.add_argument("--fire-id", required=True)
    ap.add_argument("--bbox", type=_bbox, required=True)
    ap.add_argument("--cell-deg", type=float, required=True)
    ap.add_argument("--physics-npz", required=True,
                    help="issue-time, same-grid physics arrays; see script docstring")
    ap.add_argument("--analysis-grid-id", default=None)
    ap.add_argument("--time-sigma-min", type=float, default=360.0)
    ap.add_argument("--location-sigma-m", type=float, default=375.0)
    ap.add_argument("--fuel-footprint-radius-m", type=float, default=None,
                    help="optional thermal-footprint radius for zero-ROS centroid cells")
    ap.add_argument("--min-burnable-support", type=float, default=0.10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cell_deg <= 0:
        ap.error("--cell-deg must be positive")

    spec = GridSpec.from_bbox_res(args.bbox, args.cell_deg)
    grids = np.load(args.physics_npz)
    missing = [key for key in _REQUIRED if key not in grids]
    if missing:
        ap.error(f"physics NPZ missing required arrays: {', '.join(missing)}")
    table = firms_to_t4m_observations(
        pd.read_csv(args.firms_csv), fire_id=args.fire_id, spec=spec,
        analysis_grid_id=args.analysis_grid_id or f"{args.fire_id}-{spec.shape[0]}x{spec.shape[1]}",
        prior_arrival_hours=grids["prior_arrival_hours"], prior_ros_m_min=grids["prior_ros_m_min"],
        covariate_grids={key: grids[key] for key in ("fuel_moisture", "wind_midflame_ms", "slope_tan")},
        time_sigma_min=args.time_sigma_min, location_sigma_m=args.location_sigma_m,
        footprint_radius_m=args.fuel_footprint_radius_m,
        min_burnable_support=args.min_burnable_support,
    )
    out = write_t4m_parquet(table, Path(args.out))
    print(f"wrote {out}: {len(table):,} earliest-cell observations, grid {spec.shape[0]}x{spec.shape[1]}")
    print("scope: FIRMS first-seen points with explicit timing/footprint uncertainty; not a perimeter label")


if __name__ == "__main__":
    main()

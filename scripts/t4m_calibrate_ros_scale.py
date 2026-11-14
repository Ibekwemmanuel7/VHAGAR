"""Estimate a pre-cutoff multiplicative ROS scale from timed FIRMS detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vhagar.datasets.spread_ingest import GridSpec, rasterize_detections
from vhagar.models.state_estimation import calibrate_ros_scale


def _bbox(value: str) -> tuple[float, float, float, float]:
    values = tuple(float(part) for part in value.split(","))
    if len(values) != 4 or values[0] >= values[2] or values[1] >= values[3]:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("firms_csv")
    parser.add_argument("--base-physics", required=True)
    parser.add_argument("--bbox", type=_bbox, required=True)
    parser.add_argument("--cell-deg", type=float, required=True)
    parser.add_argument("--cutoff-utc", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cutoff = pd.Timestamp(args.cutoff_utc)
    if cutoff.tzinfo is None or args.cell_deg <= 0:
        parser.error("cutoff must include timezone and cell size must be positive")
    cutoff = cutoff.tz_convert("UTC")
    detections = pd.read_csv(args.firms_csv)
    hhmm = pd.to_numeric(detections["acq_time"], errors="coerce")
    detections["_time"] = (
        pd.to_datetime(detections["acq_date"], utc=True, errors="coerce")
        + pd.to_timedelta(hhmm // 100, unit="h")
        + pd.to_timedelta(hhmm % 100, unit="m")
    )
    west, south, east, north = args.bbox
    detections = detections.loc[
        detections["_time"].notna()
        & detections["_time"].le(cutoff)
        & detections["longitude"].between(west, east)
        & detections["latitude"].between(south, north)
    ].copy()
    if len(detections) < 2:
        parser.error("need at least two pre-cutoff detections")
    origin = detections["_time"].min()
    hours = (detections["_time"] - origin).dt.total_seconds().to_numpy() / 3600.0
    spec = GridSpec.from_bbox_res(args.bbox, args.cell_deg)
    rc, times = rasterize_detections(detections.longitude, detections.latitude, hours, spec)
    with np.load(args.base_physics) as base:
        scale = calibrate_ros_scale(base["prior_arrival_hours"], rc, times)
    result = {
        "ros_scale": scale,
        "n_pre_cutoff_first_seen_cells": int(len(rc)),
        "cutoff_utc": cutoff.isoformat(),
        "calibration": "median prior-arrival / observed-first-detection time; pre-cutoff only",
        "limitation": "FIRMS first detections are sparse/censored thermal observations, not exact arrival times.",
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

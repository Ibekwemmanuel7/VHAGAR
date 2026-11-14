"""Evaluate a dynamic T4-M forecast against held-out FIRMS first-seen cells.

FIRMS detections are sparse thermal observations, not a perimeter. Metrics here
measure agreement with post-cutoff detected *new cells* and are reported with a
persistence baseline; they are not perimeter-validation claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vhagar.datasets.spread_ingest import GridSpec
from vhagar.eval.metrics import burned_area_ratio, dice, pod_far


def _bbox(value: str) -> tuple[float, float, float, float]:
    try:
        bbox = tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north") from exc
    if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise argparse.ArgumentTypeError("bbox must have positive area")
    return bbox


def _timestamp(frame: pd.DataFrame) -> pd.Series:
    hhmm = pd.to_numeric(frame["acq_time"], errors="coerce")
    return (
        pd.to_datetime(frame["acq_date"], utc=True, errors="coerce")
        + pd.to_timedelta(hhmm // 100, unit="h")
        + pd.to_timedelta(hhmm % 100, unit="m")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("firms_csv")
    parser.add_argument("--base-physics", required=True)
    parser.add_argument("--forecast-npz", required=True)
    parser.add_argument("--bbox", type=_bbox, required=True)
    parser.add_argument("--cell-deg", type=float, required=True)
    parser.add_argument("--cutoff-utc", required=True)
    parser.add_argument("--horizon-hours", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cutoff = pd.Timestamp(args.cutoff_utc)
    if cutoff.tzinfo is None or args.cell_deg <= 0 or args.horizon_hours <= 0:
        parser.error("cutoff needs timezone; cell size and horizon must be positive")
    cutoff = cutoff.tz_convert("UTC")
    end = cutoff + pd.Timedelta(hours=args.horizon_hours)
    with np.load(args.base_physics) as base, np.load(args.forecast_npz) as forecast:
        initial = np.asarray(base["ignition_seed"], dtype=bool)
        predicted = np.asarray(forecast["burned_mask"], dtype=bool)
    if initial.shape != predicted.shape:
        parser.error("forecast and base physics grids do not match")

    observations = pd.read_csv(args.firms_csv)
    observations["_time"] = _timestamp(observations)
    west, south, east, north = args.bbox
    selected = observations.loc[
        observations["_time"].gt(cutoff)
        & observations["_time"].le(end)
        & observations["longitude"].between(west, east)
        & observations["latitude"].between(south, north)
    ]
    spec = GridSpec.from_bbox_res(args.bbox, args.cell_deg)
    rows, columns, inside = spec.lonlat_to_rc(selected.longitude, selected.latitude)
    truth = np.zeros(spec.shape, dtype=bool)
    truth[rows[inside], columns[inside]] = True
    truth &= ~initial
    prediction = predicted & ~initial
    persistence = np.zeros(spec.shape, dtype=bool)
    if not truth.any():
        parser.error("no post-cutoff FIRMS first-seen cells in the forecast horizon")

    def score(mask: np.ndarray) -> dict:
        pod, far = pod_far(truth, mask)
        return {
            "sorensen": float(dice(truth, mask)),
            "pod": float(pod),
            "far": float(far),
            "burned_area_ratio": float(burned_area_ratio(truth, mask)),
            "predicted_new_cells": int(mask.sum()),
        }

    result = {
        "scoring": "post-cutoff FIRMS first-seen cells; sparse-detection agreement, not perimeter validation",
        "cutoff_utc": cutoff.isoformat(),
        "end_utc": end.isoformat(),
        "n_raw_post_cutoff_detections": int(len(selected)),
        "n_truth_new_cells": int(truth.sum()),
        "dynamic_forecast": score(prediction),
        "persistence_baseline": score(persistence),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

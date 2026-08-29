"""T4 real spread ingest: turn timed active-fire observations into the arrival-time
assimilation problem that :mod:`vhagar.models.state_estimation` solves.

The physics/assimilation core works on a regular grid and consumes, per fire, a
prior rate-of-spread field, an ignition seed mask, and a time-ordered set of
detections ``(det_rc, det_times)`` (burning cell row/cols and their first-seen
times). This module builds those from real inputs:

* a :class:`GridSpec` fixes a lon/lat analysis grid for one fire;
* :func:`rasterize_detections` bins timed active-fire points (VIIRS/MODIS from
  FIRMS, or rasterised NIROPS/NIFC perimeters) to cells, keeping the earliest
  time per cell;
* :func:`ignition_from_detections` seeds from the earliest detections;
* :func:`prior_ros_from_covariates` maps fuel/wind/slope to a prior ROS with the
  existing :func:`vhagar.models.spread.rate_of_spread` (synthetic fields as a
  labelled stand-in until real fuel/wind rasters are wired);
* :func:`assimilate_real` runs the one-parameter arrival-time analysis on an
  early time slice and scores the forecast against the held-out later detections
  with Sorensen/Dice and the false-alarm ratio, exactly as the synthetic
  ``assimilation_experiment`` does, so real and synthetic numbers are comparable.

No real perimeter/NIROPS parser ships yet; the ingest is written so that a thin
reader (points parquet, or a rasteriser over perimeter polygons) feeds these pure
functions at the edge.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "GridSpec",
    "rasterize_detections",
    "ignition_from_detections",
    "prior_ros_from_covariates",
    "build_spread_case",
    "assimilate_real",
    "read_firms_points_parquet",
]

_EARTH_M_PER_DEG = 111_320.0


@dataclass(frozen=True)
class GridSpec:
    """A regular lon/lat analysis grid for one fire. Row 0 is the north edge."""

    bbox: tuple[float, float, float, float]  # (west, south, east, north)
    shape: tuple[int, int]                   # (H rows = lat, W cols = lon)

    @classmethod
    def from_bbox_res(cls, bbox: tuple[float, float, float, float], cell_deg: float) -> GridSpec:
        w, s, e, n = bbox
        H = max(1, int(round((n - s) / cell_deg)))
        W = max(1, int(round((e - w) / cell_deg)))
        return cls(bbox=bbox, shape=(H, W))

    def lonlat_to_rc(self, lon, lat):
        """Map lon/lat to (row, col, in_bounds) integer cell indices."""
        w, s, e, n = self.bbox
        H, W = self.shape
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        col = np.floor((lon - w) / (e - w) * W).astype(np.int64)
        row = np.floor((n - lat) / (n - s) * H).astype(np.int64)
        inb = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        return row, col, inb

    def cell_size_m(self) -> tuple[float, float]:
        """Approximate (dy, dx) cell size in metres at the grid's mid-latitude."""
        w, s, e, n = self.bbox
        H, W = self.shape
        mid = np.radians((s + n) / 2.0)
        dy = (n - s) / H * _EARTH_M_PER_DEG
        dx = (e - w) / W * _EARTH_M_PER_DEG * np.cos(mid)
        return float(dy), float(dx)


def rasterize_detections(lon, lat, time_hours, spec: GridSpec):
    """Bin timed detections to grid cells, keeping the EARLIEST time per cell.

    ``time_hours`` is hours since the fire's first detection (any consistent
    monotonic clock works; calibration is scale-relative). Returns
    ``(det_rc: (m,2) int, det_times: (m,) float)`` sorted by time, one row per
    burning cell."""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    t = np.asarray(time_hours, dtype=np.float64)
    row, col, inb = spec.lonlat_to_rc(lon, lat)
    row, col, t = row[inb], col[inb], t[inb]
    if row.size == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float64)
    H, W = spec.shape
    flat = row * W + col
    order = np.argsort(t, kind="stable")           # earliest first
    flat, t = flat[order], t[order]
    _, first = np.unique(flat, return_index=True)   # first occurrence == earliest time
    cells, times = flat[first], t[first]
    keep = np.argsort(times, kind="stable")
    cells, times = cells[keep], times[keep]
    det_rc = np.column_stack([cells // W, cells % W]).astype(np.int64)
    return det_rc, times


def ignition_from_detections(det_rc, det_times, spec: GridSpec, *, seed_quantile: float = 0.05):
    """Seed mask from the earliest detections (arrival time ~ 0). Falls back to the
    single first cell if the quantile selects nothing."""
    H, W = spec.shape
    ign = np.zeros((H, W), dtype=bool)
    if len(det_times) == 0:
        return ign
    thresh = np.quantile(det_times, seed_quantile)
    sel = det_times <= thresh
    if not sel.any():
        sel = det_times == det_times.min()
    ign[det_rc[sel, 0], det_rc[sel, 1]] = True
    return ign


def prior_ros_from_covariates(
    spec: GridSpec, *, fuel=None, wind=None, slope=None, seed: int = 0,
) -> np.ndarray:
    """Prior rate-of-spread field from fuel/wind/slope via the existing physics.

    When a covariate is None a LABELLED synthetic field is used (smooth, positive)
    so the pipeline runs end to end; replace with real rasters before quoting
    numbers."""
    from vhagar.models.spread import rate_of_spread

    H, W = spec.shape
    rng = np.random.default_rng(seed)

    def _synth(scale: float, base: float) -> np.ndarray:
        yy, xx = np.mgrid[0:H, 0:W] / max(H, W)
        ph = rng.uniform(0, 2 * np.pi, size=2)
        return base + scale * (0.5 + 0.5 * np.sin(6 * yy + ph[0]) * np.cos(6 * xx + ph[1]))

    fuel = _synth(0.6, 0.4) if fuel is None else np.asarray(fuel, dtype=np.float64)
    wind = _synth(4.0, 1.0) if wind is None else np.asarray(wind, dtype=np.float64)
    slope = np.zeros((H, W)) if slope is None else np.asarray(slope, dtype=np.float64)
    ros = rate_of_spread(fuel, wind, slope)
    return np.clip(ros, 1e-3, None)


def build_spread_case(
    lon, lat, time_hours, spec: GridSpec, *,
    fuel=None, wind=None, slope=None, ros=None, seed: int = 0, seed_quantile: float = 0.05,
) -> dict:
    """Assemble one fire's arrival-time problem from timed detections + covariates.

    Returns ``{spec, prior_ros, ignition, det_rc, det_times}`` ready for
    :func:`assimilate_real` / :func:`vhagar.models.state_estimation.estimate_arrival_field`."""
    det_rc, det_times = rasterize_detections(lon, lat, time_hours, spec)
    if len(det_rc) < 2:
        raise ValueError("need at least two detected cells to assimilate a spread")
    ignition = ignition_from_detections(det_rc, det_times, spec, seed_quantile=seed_quantile)
    prior_ros = (np.asarray(ros, dtype=np.float64) if ros is not None
                 else prior_ros_from_covariates(spec, fuel=fuel, wind=wind, slope=slope, seed=seed))
    return {"spec": spec, "prior_ros": prior_ros, "ignition": ignition,
            "det_rc": det_rc, "det_times": det_times,
            "synthetic_ros": ros is None and fuel is None}


def assimilate_real(case: dict, *, split_frac: float = 0.5) -> dict:
    """Calibrate the arrival-time analysis on the early detections, then score the
    forecast against the held-out later detections.

    Detections are split in time at the ``split_frac`` quantile: the earlier
    fraction calibrates the per-fire ROS scale, the later fraction is truth for
    the forecast. Scoring (Sorensen/Dice, POD, FAR) is restricted to NEW burn: the
    cells first detected after the cutoff, over the region not already observed as
    burned at the cutoff. This avoids the inflation of scoring against the
    calibration detections the analysis was fit on. The returned dict records the
    scoring convention and the number of evaluable cells."""
    from vhagar.eval.metrics import dice, pod_far
    from vhagar.models.state_estimation import estimate_arrival_field

    det_rc, det_times = case["det_rc"], case["det_times"]
    prior_ros, ignition, spec = case["prior_ros"], case["ignition"], case["spec"]
    if len(det_times) < 2:
        raise ValueError("not enough detections to split")

    t_split = float(np.quantile(det_times, split_frac))
    early = det_times <= t_split
    if not early.any():
        early = det_times == det_times.min()

    state = estimate_arrival_field(prior_ros, ignition, det_rc[early], det_times[early])

    # Score only NEW burn: cells first detected AFTER the calibration cutoff, and
    # only over the region not already observed as burned at the cutoff. Scoring
    # against every detected cell (including the calibration detections the analysis
    # was fit on) inflates Dice/POD, since the forecast trivially recalls its own
    # calibration footprint.
    t_eval = float(det_times.max())
    H, W = spec.shape
    late = ~early
    seen = np.zeros((H, W), dtype=bool)
    seen[det_rc[early, 0], det_rc[early, 1]] = True        # burned as of the cutoff
    truth = np.zeros((H, W), dtype=bool)
    truth[det_rc[late, 0], det_rc[late, 1]] = True         # held-out later detections
    eval_mask = ~seen                                      # never credit already-seen cells
    truth &= eval_mask
    pred = state.burned_by(t_eval) & eval_mask

    if truth.any():
        d = float(dice(truth, pred))
        pod, far = pod_far(truth, pred)
        d_pod, d_far = float(pod), float(far)
        scoring = "held-out post-cutoff new burn; calibration cells excluded"
    else:
        d = d_pod = d_far = float("nan")
        scoring = "no held-out post-cutoff detections outside the calibration footprint"
    return {"k": float(state.k), "dice": d, "pod": d_pod, "far": d_far,
            "t_split": t_split, "t_eval": t_eval,
            "n_early": int(early.sum()), "n_late": int(late.sum()),
            "n_eval_cells": int(truth.sum()), "scoring": scoring}


def read_firms_points_parquet(
    path: str | Path, *, bbox: tuple[float, float, float, float] | None = None,
    lon_col: str = "lon", lat_col: str = "lat", time_col: str = "t",
):
    """Thin reader: load active-fire points and return ``(lon, lat, time_hours)``
    with time expressed as hours since the earliest point. Accepts the parquet the
    FIRMS ingest already writes (lon, lat, t)."""
    import pandas as pd

    df = pd.read_parquet(path)
    if bbox is not None:
        w, s, e, n = bbox
        m = ((df[lon_col] >= w) & (df[lon_col] <= e) & (df[lat_col] >= s) & (df[lat_col] <= n))
        df = df.loc[m]
    if df.empty:
        raise ValueError("no active-fire points in the requested bbox")
    t = pd.to_datetime(df[time_col], utc=True)
    hours = (t - t.min()).dt.total_seconds().to_numpy() / 3600.0
    return df[lon_col].to_numpy(), df[lat_col].to_numpy(), hours

"""T4 mapped-perimeter shape validation: does the elliptical-wavelet front
reproduce a REAL burned perimeter better than an equal-area circle?

This is the honest counterpart to the VIIRS forward-run proxy. Given a real burned
mask (e.g. an MTBS severity polygon rasterised), it measures the perimeter's final
area, dominant spread axis, and length-to-breadth, seeds the anisotropic
arrival-time front at the tail (backing) end, drives it with that axis and
elongation, grows it to the *observed* final area, and scores the predicted burned
mask against the real one with IoU and Sorensen (Dice).

Scope, stated plainly so it is not oversold: this is a SHAPE-FAMILY adequacy test,
not a blind forecast. The final size, dominant axis orientation and length-to-
breadth are taken from the truth mask; only the front geometry (Richards' elliptical
wavelet on the 8-connected arrival-time solver) is under test. It answers "is our
front the right shape for real fires?", and it is scored against a mandatory
equal-area circle baseline, the naive shape. A true next-day forecast validation
still needs the ignition point, timed perimeters, real wind and fuel, which are
future work. The point of this test is to retire the specific claim that VHAGAR has
never been compared to a real mapped perimeter.
"""
from __future__ import annotations

import numpy as np

from vhagar.eval.metrics import dice
from vhagar.models.spread import anisotropic_arrival

__all__ = [
    "mask_geometry",
    "front_prediction",
    "equal_area_disk",
    "iou",
    "evaluate_perimeter_shape",
]


def iou(a, b) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def mask_geometry(mask) -> dict:
    """Area, centroid, dominant-axis orientation and length-to-breadth of a burned
    mask, plus the tail (backing) seed cell at the far end of the major axis.

    Orientation ``theta`` is in the arrival-time solver's convention,
    ``atan2(drow, dcol)``, pointing from the tail toward the head."""
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.where(mask)
    if ys.size < 3:
        raise ValueError("mask too small for a geometry fit")
    cy, cx = float(ys.mean()), float(xs.mean())
    yr, xr = ys - cy, xs - cx
    cov = np.array([[np.mean(yr * yr), np.mean(yr * xr)],
                    [np.mean(yr * xr), np.mean(xr * xr)]])       # (row, col) order
    evals, evecs = np.linalg.eigh(cov)
    v = evecs[:, int(np.argmax(evals))]                          # major eigenvector (row, col)
    theta = float(np.arctan2(v[0], v[1]))
    lam_max, lam_min = float(evals.max()), float(max(evals.min(), 1e-9))
    lb = float(np.sqrt(lam_max / lam_min))
    proj = yr * v[0] + xr * v[1]                                 # project cells on major axis
    tail = int(np.argmin(proj))                                  # far backing end
    return {"area": int(mask.sum()), "centroid": (cy, cx), "theta": theta, "lb": lb,
            "tail_rc": (int(ys[tail]), int(xs[tail]))}


def equal_area_disk(shape, center, area) -> np.ndarray:
    """Boolean disk of the given area (in cells) centred at ``center=(row,col)``."""
    H, W = shape
    r = np.sqrt(area / np.pi)
    yy, xx = np.mgrid[0:H, 0:W]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= r * r


def front_prediction(mask, *, lb_cap: float = 12.0, seed_radius: int = 2) -> tuple[np.ndarray, dict]:
    """Run the elliptical front from the tail, elongated to the mask's measured
    length-to-breadth and axis, and threshold it to the observed area. Returns
    ``(pred_mask, geometry)``."""
    mask = np.asarray(mask, dtype=bool)
    g = mask_geometry(mask)
    H, W = mask.shape
    lb = float(np.clip(g["lb"], 1.0, lb_cap))
    ty, tx = g["tail_rc"]
    yy, xx = np.mgrid[0:H, 0:W]
    seeds = (yy - ty) ** 2 + (xx - tx) ** 2 <= seed_radius * seed_radius
    # wind=1 with lb_max=lb yields exactly length_to_breadth == lb; axis = theta.
    T = anisotropic_arrival(np.ones((H, W)), 1.0, g["theta"], seeds, dx=1.0, lb_max=lb)
    finite = np.isfinite(T)
    n = min(int(g["area"]), int(finite.sum()))
    thresh = np.partition(T[finite], n - 1)[n - 1]              # n-th smallest arrival time
    pred = finite & np.less_equal(T, thresh)
    return pred, g


def evaluate_perimeter_shape(mask, *, lb_cap: float = 12.0) -> dict:
    """Score the elliptical front against the real mask and the equal-area circle
    baselines (at the fire centroid and at the tail seed)."""
    mask = np.asarray(mask, dtype=bool)
    pred, g = front_prediction(mask, lb_cap=lb_cap)
    area = g["area"]
    circle_c = equal_area_disk(mask.shape, g["centroid"], area)
    circle_t = equal_area_disk(mask.shape, g["tail_rc"], area)
    return {
        "area_cells": int(area),
        "lb": round(g["lb"], 2),
        "iou_front": round(iou(pred, mask), 4),
        "dice_front": round(float(dice(mask, pred)), 4),
        "iou_circle_centroid": round(iou(circle_c, mask), 4),
        "iou_circle_tail": round(iou(circle_t, mask), 4),
        "front_beats_circle": bool(iou(pred, mask) > iou(circle_c, mask)),
    }

"""Tests for the T4 mapped-perimeter shape validation core (pure numpy: no
rasterio/scipy/skfmm, so it runs in the core CI env). Uses synthetic rotated
ellipses as stand-ins for a rasterised burned perimeter."""
from __future__ import annotations

import numpy as np

from vhagar.eval.perimeter_shape import (
    arrival_time_mae,
    evaluate_perimeter_shape,
    front_prediction,
    iou,
    mask_geometry,
)


def _ellipse(H, W, cy, cx, a, b, deg):
    yy, xx = np.mgrid[0:H, 0:W]
    th = np.deg2rad(deg)
    xr = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
    yr = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
    return (xr / a) ** 2 + (yr / b) ** 2 <= 1.0


def test_geometry_recovers_axis_and_elongation():
    m = _ellipse(200, 200, 100, 100, 70, 22, 35)
    g = mask_geometry(m)
    assert abs(g["lb"] - 70 / 22) < 0.5                      # true LB ~3.18
    ang = np.degrees(g["theta"]) % 180
    assert min(abs(ang - 35), abs(ang - (35 + 180))) < 8     # axis within a few degrees
    assert g["area"] == int(m.sum())


def test_front_beats_circle_on_elongated_fire():
    m = _ellipse(220, 220, 110, 110, 80, 22, 25)             # LB ~3.6, strongly elongated
    res = evaluate_perimeter_shape(m)
    assert res["front_beats_circle"] is True
    assert res["iou_front"] > res["iou_circle_centroid"]
    assert 0.0 < res["iou_front"] <= 1.0
    assert res["dice_front"] > res["iou_front"]              # Dice >= IoU always


def test_front_area_matches_observed():
    m = _ellipse(180, 180, 90, 90, 60, 25, 60)
    pred, g = front_prediction(m)
    # thresholded to the observed area: predicted count within a small tolerance
    assert abs(pred.sum() - g["area"]) <= max(3, int(0.02 * g["area"]))


def test_arrival_time_mae_scores_only_hits():
    # 3x3: forecast and truth overlap on two cells; arrival vs observed differ by 2 h and 4 h.
    pred = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]], bool)
    truth = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], bool)   # third truth cell is missed (not predicted)
    arrival = np.array([[10.0, 12.0, np.inf], [np.inf, np.inf, np.inf],
                        [np.inf, np.inf, np.inf]])
    truth_time = np.array([[12.0, 16.0, np.inf], [20.0, np.inf, np.inf],
                           [np.inf, np.inf, np.inf]])
    # hits are the two predicted-and-true cells: |10-12|=2, |12-16|=4 -> mean 3.0
    assert abs(arrival_time_mae(pred, truth, arrival, truth_time) - 3.0) < 1e-9


def test_arrival_time_mae_nan_without_overlap():
    pred = np.zeros((4, 4), bool)
    truth = np.ones((4, 4), bool)
    arrival = np.zeros((4, 4))
    truth_time = np.zeros((4, 4))
    assert np.isnan(arrival_time_mae(pred, truth, arrival, truth_time))


def test_iou_helper_bounds():
    a = np.zeros((10, 10), bool)
    a[2:6, 2:6] = True
    assert iou(a, a) == 1.0
    b = np.zeros((10, 10), bool)
    assert iou(a, b) == 0.0

"""Tests for the T4 mapped-perimeter shape validation core (pure numpy: no
rasterio/scipy/skfmm, so it runs in the core CI env). Uses synthetic rotated
ellipses as stand-ins for a rasterised burned perimeter."""
from __future__ import annotations

import numpy as np

from vhagar.eval.perimeter_shape import (
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


def test_iou_helper_bounds():
    a = np.zeros((10, 10), bool)
    a[2:6, 2:6] = True
    assert iou(a, a) == 1.0
    b = np.zeros((10, 10), bool)
    assert iou(a, b) == 0.0

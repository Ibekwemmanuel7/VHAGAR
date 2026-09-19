"""Rothermel (1972) surface fire spread: physical monotonicity and field mapping."""
from __future__ import annotations

import numpy as np

from vhagar.models.rothermel import (
    STANDARD_FUELS,
    params_for_codes,
    rothermel_head_ros_field,
    rothermel_ros,
)

GR = STANDARD_FUELS["GR"]


def test_wind_increases_ros():
    r = [float(rothermel_ros(GR, 0.06, w)) for w in (0.0, 2.0, 5.0, 10.0)]
    assert all(b > a for a, b in zip(r, r[1:], strict=False))
    assert r[0] > 0.0


def test_moisture_decreases_ros():
    r = [float(rothermel_ros(GR, m, 5.0)) for m in (0.03, 0.06, 0.10, 0.14)]
    assert all(b < a for a, b in zip(r, r[1:], strict=False))


def test_slope_increases_ros():
    flat = float(rothermel_ros(GR, 0.06, 5.0, 0.0))
    steep = float(rothermel_ros(GR, 0.06, 5.0, 0.4))
    assert steep > flat


def test_zero_at_and_above_extinction_moisture():
    assert float(rothermel_ros(GR, GR.m_x, 5.0)) == 0.0
    assert float(rothermel_ros(GR, GR.m_x + 0.05, 5.0)) == 0.0


def test_ros_is_physically_sane_magnitude():
    # grass at 6% moisture and a moderate midflame wind should spread, but not absurdly
    r = float(rothermel_ros(GR, 0.06, 5.0))
    assert 5.0 < r < 300.0


def test_fuel_ordering_grass_faster_than_timber_litter():
    g = float(rothermel_ros(STANDARD_FUELS["GR"], 0.06, 5.0))
    tl = float(rothermel_ros(STANDARD_FUELS["TL"], 0.06, 5.0))
    assert g > tl


def test_field_maps_codes_and_zeros_nonburnable():
    codes = np.array([[102, 145, 183], [91, 161, 201]])
    f = rothermel_head_ros_field(codes, m_f=0.06, wind_ms=5.0)
    assert f.shape == codes.shape
    assert f[1, 0] == 0.0                       # 91 is non-burnable
    assert (f[[0, 0, 0, 1, 1], [0, 1, 2, 1, 2]] > 0).all()


def test_wind_adj_reduces_ros():
    codes = np.array([102])
    full = float(rothermel_head_ros_field(codes, m_f=0.06, wind_ms=8.0, wind_adj=1.0)[0])
    mid = float(rothermel_head_ros_field(codes, m_f=0.06, wind_ms=8.0, wind_adj=0.3)[0])
    assert 0.0 < mid < full


def test_params_for_codes_flags_burnable():
    _, _, _, _, burn = params_for_codes(np.array([102, 91, 999]))
    assert burn.tolist() == [True, False, True]     # 999 unrecognised -> burnable default

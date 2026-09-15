"""FBFM40 fuel-model -> rate-of-spread factor, and the fuel-aware ROS wiring."""
from __future__ import annotations

import numpy as np

from vhagar.eval.wui_calibrate import fire_from_points
from vhagar.models import fuels


def test_fbfm40_spread_factor_bands_and_nonburnable():
    codes = np.array([101, 109, 121, 145, 163, 185, 203, 91, 93, 99, 999])
    f = fuels.fbfm40_spread_factor(codes)
    assert f[0] == 1.0 and f[1] == 1.0          # GR grass
    assert f[2] == 0.80                          # GS
    assert f[3] == 0.70                          # SH
    assert f[4] == 0.55                          # TU
    assert f[5] == 0.35                          # TL
    assert f[6] == 0.85                          # SB
    assert f[7] == 0.0 and f[8] == 0.0 and f[9] == 0.0   # NB1/NB3/NB9 non-burnable
    assert f[10] == 0.4                          # unrecognised -> default


def test_ros_from_fuel_codes_zero_on_nonburnable():
    ros = fuels.ros_from_fuel_codes(np.array([101, 91, 145]), base=1.0)
    assert ros[0] == 1.0
    assert ros[1] == 0.0                         # front cannot cross non-burnable
    assert ros[2] > 0.0


def test_fire_from_points_with_fuel_sampler_creates_breaks():
    lat = np.array([34.100, 34.101, 34.102, 34.103])
    lon = np.array([-118.100, -118.099, -118.098, -118.097])
    dest = np.array([True, True, False, False])

    def sampler(lon_g, lat_g):
        # urban (non-burnable 91) west of a line, grass (101) to the east
        return np.where(lon_g < -118.0985, 91, 101)

    fire = fire_from_points("T", lat, lon, dest, 34.1005, -118.0975,
                            wind_speed_ms=7.5, wind_from_deg=270.0, fuel_sampler=sampler)
    assert (fire.ros == 0).any()                 # non-burnable break present
    assert (fire.ros > 0).any()                  # burnable fuel present
    # uniform-ROS default is unchanged
    fire2 = fire_from_points("T", lat, lon, dest, 34.1005, -118.0975, 7.5, 270.0)
    assert np.allclose(fire2.ros, 1.0)

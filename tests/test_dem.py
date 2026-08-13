"""DEM sampling and its effect on the GEO/LEO matching tolerance."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from vhagar.harmonize.dem import DEM, attach_elevation
from vhagar.harmonize.fusion import Detection, geo_leo_tolerance_m

T = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)


def _ramp_dem(nx=11, ny=11, dx=1000.0, dy=-1000.0, x0=0.0, y0=10000.0):
    """A DEM whose elevation is a known linear function of position, so bilinear
    sampling is exact and testable to the metre."""
    cols = np.arange(nx)
    rows = np.arange(ny)
    xx = x0 + dx * cols[None, :]
    yy = y0 + dy * rows[:, None]
    elev = 0.5 * xx + 0.1 * yy  # linear surface
    return DEM(elevation=np.broadcast_to(elev, (ny, nx)).copy(), x0=x0, y0=y0, dx=dx, dy=dy)


def _expected(x, y):
    return 0.5 * x + 0.1 * y


def test_bilinear_sampling_is_exact_on_a_linear_surface():
    dem = _ramp_dem()
    for x, y in [(0.0, 10000.0), (2500.0, 7300.0), (9000.0, 1000.0)]:
        assert float(dem.sample(x, y)) == pytest.approx(_expected(x, y), abs=1e-6)


def test_sampling_off_the_grid_is_nan():
    dem = _ramp_dem()
    assert np.isnan(float(dem.sample(-5000.0, 5000.0)))   # west of the grid
    assert np.isnan(float(dem.sample(5000.0, 99999.0)))   # north of the grid


def test_sample_is_vectorised():
    dem = _ramp_dem()
    xs = np.array([0.0, 2000.0, 4000.0])
    ys = np.array([10000.0, 8000.0, 6000.0])
    got = dem.sample(xs, ys)
    assert np.allclose(got, _expected(xs, ys), atol=1e-6)


def test_nodata_neighbour_makes_the_sample_nan():
    dem = _ramp_dem()
    dem.elevation[5, 5] = np.nan
    # a point in the cell whose corner is the nodata pixel must be NaN
    assert np.isnan(float(dem.sample(5000.0 + 100.0, 5000.0 - 100.0)))


def test_attach_elevation_fills_detections_and_edges_stay_none():
    dem = _ramp_dem()
    inside = Detection("goes", x=3000.0, y=6000.0, when=T, view_zenith_deg=40.0)
    outside = Detection("goes", x=-9999.0, y=6000.0, when=T, view_zenith_deg=40.0)
    a, b = attach_elevation([inside, outside], dem)
    assert a.elevation_m == pytest.approx(_expected(3000.0, 6000.0), abs=1e-6)
    assert b.elevation_m is None


# --------------------------------------------- tolerance wiring ----------


def test_tolerance_accepts_per_pixel_elevation_arrays():
    """The float() cast used to block arrays; per-pixel elevation must work."""
    vza = np.array([48.1, 48.1])
    elev = np.array([0.0, 1500.0])
    tol = geo_leo_tolerance_m(vza, elevation_m=elev)
    assert tol.shape == (2,)
    assert tol[1] > tol[0]  # higher ground, more parallax, wider tolerance


def test_nan_elevation_falls_back_to_the_placeholder():
    at_nan = float(geo_leo_tolerance_m(48.1, elevation_m=np.nan))
    at_placeholder = float(geo_leo_tolerance_m(48.1, elevation_m=1000.0))
    assert at_nan == pytest.approx(at_placeholder)


def test_detection_uses_its_elevation_in_the_tolerance():
    low = Detection("goes", 0.0, 0.0, T, view_zenith_deg=48.0, elevation_m=0.0)
    high = Detection("goes", 0.0, 0.0, T, view_zenith_deg=48.0, elevation_m=2500.0)
    assert high.tolerance_m > low.tolerance_m


def test_detection_without_elevation_matches_the_placeholder():
    none = Detection("goes", 0.0, 0.0, T, view_zenith_deg=48.0)
    placeholder = Detection("goes", 0.0, 0.0, T, view_zenith_deg=48.0, elevation_m=1000.0)
    assert none.tolerance_m == pytest.approx(placeholder.tolerance_m)

"""Geospatial correctness tests.

These target the *silent* failures, the ones that do not raise, do not look
wrong on a map, and quietly corrupt the numbers you publish.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vhagar.grid import HALO_CELLS, TILE_CELLS, AnalysisGrid, parse_tile_id
from vhagar.harmonize.regrid import check_mass_conservation, conservative_regrid_2d

# ------------------------------------------------------------------ grid --


@pytest.mark.parametrize("region", ["conus", "canada", "europe"])
def test_grid_tiles_cover_and_do_not_overlap(region):
    g = AnalysisGrid(region)
    a = g.tile(3, 3)
    right = g.tile(4, 3)
    up = g.tile(3, 4)
    assert a.bounds[2] == pytest.approx(right.bounds[0]), "horizontal gap or overlap"
    assert a.bounds[3] == pytest.approx(up.bounds[1]), "vertical gap or overlap"


def test_tile_id_roundtrip():
    g = AnalysisGrid("conus")
    t = g.tile(12, 20)
    assert parse_tile_id(t.tile_id).bounds == t.bounds


def test_halo_shape_and_core_slice():
    t = AnalysisGrid("europe").tile(0, 0)
    assert t.shape == (TILE_CELLS + 2 * HALO_CELLS,) * 2
    arr = np.zeros(t.shape)
    assert arr[t.core_slice].shape == (TILE_CELLS, TILE_CELLS)


def test_tile_for_point_is_self_consistent():
    g = AnalysisGrid("conus")
    t = g.tile(7, 11)
    cx, cy = t.centroid()
    assert g.tile_for_point(cx, cy).tile_id == t.tile_id


def test_tiles_for_bounds_covers_a_span():
    g = AnalysisGrid("conus")
    t = g.tile(5, 5)
    x0, y0, x1, y1 = t.bounds
    tiles = g.tiles_for_bounds((x0 - 1, y0 - 1, x1 + 1, y1 + 1))
    assert t.tile_id in {tt.tile_id for tt in tiles}
    assert len(tiles) >= 4


def test_unknown_region_rejected():
    with pytest.raises(ValueError, match="unknown region"):
        AnalysisGrid("antarctica")


# -------------------------------------------------------------- regrid ----


def test_conservative_regrid_conserves_mass_on_aggregation():
    v = np.array([[1.0, 3.0], [5.0, 7.0]])
    e = np.array([0.0, 1.0, 2.0])
    out = conservative_regrid_2d(v, e, e, np.array([0.0, 2.0]), np.array([0.0, 2.0]))
    check_mass_conservation(v, out)
    assert out.shape == (1, 1)
    assert out[0, 0] == pytest.approx(16.0)


def test_conservative_regrid_conserves_mass_on_refinement():
    rng = np.random.default_rng(0)
    v = rng.random((4, 4)) * 100
    src = np.linspace(0.0, 4.0, 5)
    dst = np.linspace(0.0, 4.0, 9)  # 2x finer
    out = conservative_regrid_2d(v, src, src, dst, dst)
    check_mass_conservation(v, out)


@settings(max_examples=30, deadline=None)
@given(
    ny=st.integers(min_value=1, max_value=6),
    nx=st.integers(min_value=1, max_value=6),
    scale=st.floats(min_value=0.5, max_value=4.0),
)
def test_conservative_regrid_property_mass_is_invariant(ny, nx, scale):
    rng = np.random.default_rng(ny * 100 + nx)
    v = rng.random((ny, nx))
    sy = np.arange(ny + 1, dtype=float)
    sx = np.arange(nx + 1, dtype=float)
    dy = np.linspace(0.0, float(ny), max(2, int(ny * scale) + 1))
    dx = np.linspace(0.0, float(nx), max(2, int(nx * scale) + 1))
    out = conservative_regrid_2d(v, sx, sy, dx, dy)
    assert float(out.sum()) == pytest.approx(float(v.sum()), rel=1e-9)


def test_mass_conservation_check_catches_a_bilinear_style_error():
    before = np.array([[10.0, 0.0], [0.0, 0.0]])
    after = before * 0.9  # a resampler that dropped 10% of the flux
    with pytest.raises(AssertionError, match="did not conserve mass"):
        check_mass_conservation(before, after)


def test_regrid_rejects_non_monotonic_edges():
    with pytest.raises(ValueError, match="strictly increasing"):
        conservative_regrid_2d(
            np.ones((1, 1)),
            np.array([0.0, 1.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        )


# ------------------------------------------------------- nodata / masks ----


def test_nodata_does_not_silently_become_cold_ground():
    """The most common silent EO bug: nodata -> 0 -> a plausible physical value."""
    from vhagar.features.indices import nbr

    nir = np.array([np.nan, 0.3, 0.4])
    swir = np.array([0.1, np.nan, 0.05])
    out = nbr(nir, swir)
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert np.isfinite(out[2])


def test_zero_denominator_becomes_nan_not_inf():
    from vhagar.features.indices import normalized_difference

    out = normalized_difference(np.array([0.0]), np.array([0.0]))
    assert np.isnan(out[0]), "guarded denominator must yield NaN, not inf"

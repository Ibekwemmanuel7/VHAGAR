"""Independent optical predictor: cloud masking, compositing, RBR, and the
per-fire sample assembly. The network and rasterio edges are stubbed; the pure
logic and the window geometry are tested for real.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from vhagar.io.optical import (
    SCL_KEEP,
    composite_nbr,
    mean_composite,
    rbr_from_windows,
    scl_valid_mask,
)
from vhagar.labels.registry import FireEventRecord, LabelSource

# ---------------------------------------------------- pure compositing ----


def test_scl_valid_mask_keeps_only_ground_classes():
    scl = np.array([[4, 8], [9, 5], [3, 6]])   # veg, cloud, cloud, bare, shadow, water
    m = scl_valid_mask(scl)
    assert m.tolist() == [[True, False], [False, True], [False, True]]
    assert set(SCL_KEEP) == {4, 5, 6, 7, 11}


def test_mean_composite_ignores_masked_and_is_nan_when_all_masked():
    stack = np.array([[[10.0, 20.0]], [[30.0, 40.0]]])       # T=2, 1x2
    valid = np.array([[[True, False]], [[True, False]]])     # right pixel never valid
    out = mean_composite(stack, valid)
    assert out[0, 0] == pytest.approx(20.0)   # mean of 10 and 30
    assert np.isnan(out[0, 1])                # no valid observation


def test_composite_nbr_uses_masked_means():
    # one clear scene and one fully clouded scene; the clouded one must not count
    nir = np.array([[[0.4]], [[0.9]]])
    swir = np.array([[[0.1]], [[0.9]]])
    scl = np.array([[[4]], [[9]]])            # scene 0 vegetation, scene 1 cloud
    got = composite_nbr(nir, swir, scl)
    expected = (0.4 - 0.1) / (0.4 + 0.1)      # NBR of the clear scene only
    assert got[0, 0] == pytest.approx(expected)


def test_rbr_separates_burned_from_unburned():
    # unburned pixel: NBR stays high pre and post; burned: NBR drops post
    # column 0 unburned, column 1 burned
    pre_nir = np.array([[[0.5, 0.5]]])
    pre_swir = np.array([[[0.1, 0.1]]])
    post_nir = np.array([[[0.5, 0.2]]])
    post_swir = np.array([[[0.1, 0.4]]])
    scl = np.array([[[4, 4]]])
    r = rbr_from_windows(pre_nir, pre_swir, scl, post_nir, post_swir, scl)
    assert r[0, 1] > r[0, 0]        # burned column has the larger RBR
    assert r[0, 0] == pytest.approx(0.0, abs=1e-9)   # unburned: no change


# ------------------------------------------------------ window geometry ---


def _fire(area_ha=10_000.0, lon=-121.5, lat=39.8):
    return FireEventRecord(
        event_id="mtbs:TEST", source=LabelSource.MTBS, region="conus",
        ignition_date=date(2021, 8, 1), containment_date=None,
        area_ha=area_ha, lon=lon, lat=lat, tile_ids=["conus/x0001_y0001"],
    )


def test_target_grid_covers_the_fire_and_is_on_the_region_crs():
    from vhagar.datasets.t2_optical import target_grid_for_fire

    grid, bbox = target_grid_for_fire(_fire(area_ha=10_000.0))
    assert grid.crs == "EPSG:5070"
    # ~9 km half-window at 30 m is ~600 cells a side
    assert 400 < grid.width < 900 and 400 < grid.height < 900
    # lon/lat bbox brackets the fire point
    w, s, e, n = bbox
    assert w < -121.5 < e and s < 39.8 < n


def test_small_fire_still_gets_the_minimum_window():
    from vhagar.datasets.t2_optical import target_grid_for_fire

    grid, _ = target_grid_for_fire(_fire(area_ha=1.0), min_half_m=5_000.0)
    # 5 km half-window at 30 m is ~333 cells
    assert 300 < grid.width < 380


# ---------------------------------------------- sample assembly (stubbed) -


def test_build_optical_sample_wires_predictor_and_reference(monkeypatch):
    import vhagar.datasets.t2_optical as t2o
    import vhagar.io.optical as optical
    from vhagar.datasets.burned_area import T2Sample

    shape = (20, 20)
    rng = np.random.default_rng(0)
    fake_rbr = rng.normal(0.0, 0.1, shape)
    burned = np.zeros(shape, dtype=bool)
    burned[5:15, 5:15] = True
    valid = np.ones(shape, dtype=bool)

    def fake_sentinel2_rbr(bbox, ig, grid, **kw):
        # predictor must be on the grid the window builder produced
        assert (grid.height, grid.width) == grid.shape
        return np.resize(fake_rbr, grid.shape)

    def fake_reference(mosaic_path, grid):
        return np.resize(burned, grid.shape), np.resize(valid, grid.shape)

    monkeypatch.setattr(optical, "sentinel2_rbr", fake_sentinel2_rbr)
    monkeypatch.setattr(t2o, "read_mtbs_reference_on_grid", fake_reference)

    s = t2o.build_optical_sample(_fire(area_ha=2_000.0), "dummy_mosaic.tif")
    assert isinstance(s, T2Sample)
    assert s.predictor.shape == s.reference.shape == s.valid.shape
    assert s.reference.any()          # some burned pixels
    assert s.n_valid > 0


def test_batch_builder_skips_fires_that_error(monkeypatch):
    import vhagar.datasets.t2_optical as t2o

    def boom(record, mosaic_path, **kw):
        raise RuntimeError("no scenes")

    monkeypatch.setattr(t2o, "build_optical_sample", boom)
    skipped = []
    out = t2o.build_optical_samples(
        [_fire(), _fire()], "m.tif", on_error=lambda r, e: skipped.append(r.event_id)
    )
    assert out == {}
    assert len(skipped) == 2

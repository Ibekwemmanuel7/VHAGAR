"""Independent optical predictor: cloud masking, compositing, RBR, and the
per-fire sample assembly. The network and rasterio edges are stubbed; the pure
logic and the window geometry are tested for real.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from vhagar.io.optical import (
    PRITHVI_BAND_ASSETS,
    SCL_KEEP,
    composite_nbr,
    mean_composite,
    rbr_from_windows,
    scl_valid_mask,
    stream_band_composite,
    stream_nbr,
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


def test_stream_nbr_matches_stack_composite_but_bounds_memory():
    """Streaming one scene at a time must give the same composite NBR as stacking
    them all, which is the whole point of the memory-safe path."""
    rng = np.random.default_rng(0)
    nir = rng.uniform(0.2, 0.6, (4, 8, 8))
    swir = rng.uniform(0.05, 0.4, (4, 8, 8))
    scl = rng.integers(0, 12, (4, 8, 8))
    streamed = stream_nbr(zip(nir, swir, scl, strict=True), shape=(8, 8))
    stacked = composite_nbr(nir, swir, scl)
    assert np.allclose(streamed, stacked, equal_nan=True)


def test_stream_nbr_is_nan_where_every_scene_is_cloudy():
    nir = np.array([[[0.4]], [[0.5]]])
    swir = np.array([[[0.1]], [[0.2]]])
    scl = np.array([[[9]], [[8]]])   # cloud in both scenes
    out = stream_nbr(zip(nir, swir, scl, strict=True), shape=(1, 1))
    assert np.isnan(out[0, 0])


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


def test_stream_band_composite_masks_clouds_and_scales_reflectance():
    assert len(PRITHVI_BAND_ASSETS) == 6
    # two scenes, 1x2 grid, 6 bands. Right pixel is cloudy in scene 2 only.
    def scene(dn, scl):
        return ([np.full((1, 2), float(dn + b)) for b in range(6)], np.array(scl))

    scenes = [scene(1000, [[4, 4]]), scene(3000, [[4, 9]])]   # scene2 right pixel cloud
    out = stream_band_composite(scenes, (1, 2), n_bands=6)
    assert out.shape == (6, 1, 2)
    # band 0 left pixel: mean(1000, 3000)/10000 = 0.2; right pixel: only scene1 -> 1000/10000
    assert out[0, 0, 0] == pytest.approx(0.2)
    assert out[0, 0, 1] == pytest.approx(0.1)
    # band 5 carries its +5 offset; reflectance scaled
    assert out[5, 0, 0] == pytest.approx((1005 + 3005) / 2 / 10000)


def test_build_prithvi_sample_stacks_six_bands_against_mtbs(tmp_path):
    from vhagar.datasets.t2_optical import build_prithvi_sample

    rec = _fire(area_ha=10_000.0)

    def fake_bands6(bbox, ign_iso, grid, max_cloud, max_scenes):
        H, W = grid.shape
        b = np.tile(np.linspace(0.05, 0.4, 6)[:, None, None], (1, H, W)).astype("float32")
        b[:, 0, 0] = np.nan                       # one all-cloud pixel -> invalid
        return b

    def fake_reference(grid):
        H, W = grid.shape
        burned = np.zeros((H, W), bool)
        burned[: H // 2] = True
        return burned, np.ones((H, W), bool)

    s = build_prithvi_sample(rec, fake_reference, bands6_fn=fake_bands6, cache_dir=tmp_path)
    assert s.features.shape[0] == 6                # six bands ride in the stack
    assert s.features.shape[1:] == s.reference.shape
    assert not s.valid[0, 0]                       # the all-cloud pixel is invalid
    # cache round-trips (second call reads the .npz, no puller needed)
    s2 = build_prithvi_sample(rec, fake_reference, bands6_fn=None, cache_dir=tmp_path)
    assert s2.features.shape[0] == 6


# ------------------------------------------------------ window geometry ---


def _fire(area_ha=10_000.0, lon=-121.5, lat=39.8):
    return FireEventRecord(
        event_id="mtbs:TEST", source=LabelSource.MTBS, region="conus",
        ignition_date=date(2021, 8, 1), containment_date=None,
        area_ha=area_ha, lon=lon, lat=lat, tile_ids=["conus/x0001_y0001"],
    )


def test_target_grid_covers_the_fire_and_is_on_the_region_crs():
    from vhagar.datasets.t2_optical import target_grid_for_fire

    # A 10,000 ha fire has a ~5.6 km radius; at 2.5x buffer that is ~14 km, below
    # the 15 km floor, so the half-window is the 15 km floor. At 30 m that is
    # ~1000 cells a side.
    grid, bbox = target_grid_for_fire(_fire(area_ha=10_000.0))
    assert grid.crs == "EPSG:5070"
    assert 900 < grid.width < 1100 and 900 < grid.height < 1100
    # lon/lat bbox brackets the fire point
    w, s, e, n = bbox
    assert w < -121.5 < e and s < 39.8 < n


def test_small_fire_gets_the_widened_default_floor():
    from vhagar.datasets.t2_optical import target_grid_for_fire

    # A tiny fire is floored to the default 15 km half-window (docs/11: small
    # fires need a wide unburned ring, or the window is ~all burned).
    grid, _ = target_grid_for_fire(_fire(area_ha=1.0))
    assert 900 < grid.width < 1100
    # the floor is overridable for callers that want a tighter window
    tight, _ = target_grid_for_fire(_fire(area_ha=1.0), min_half_m=5_000.0)
    assert 300 < tight.width < 380


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

    def fake_reference(mosaic_path, grid, include_background=True):
        return np.resize(burned, grid.shape), np.resize(valid, grid.shape)

    monkeypatch.setattr(optical, "sentinel2_rbr", fake_sentinel2_rbr)
    monkeypatch.setattr(t2o, "read_mtbs_reference_on_grid", fake_reference)

    s = t2o.build_optical_sample(_fire(area_ha=2_000.0), "dummy_mosaic.tif")
    assert isinstance(s, T2Sample)
    assert s.predictor.shape == s.reference.shape == s.valid.shape
    assert s.reference.any()          # some burned pixels
    assert s.n_valid > 0


def test_select_fires_largest_vs_size_stratified():
    from vhagar.datasets.t2_optical import select_fires

    fires = [_fire(area_ha=a) for a in (100, 500, 1000, 5000, 20000, 100000)]
    largest = select_fires(fires, 3, "largest")
    assert sorted(r.area_ha for r in largest) == [5000, 20000, 100000]
    spread = select_fires(fires, 3, "size")
    areas = sorted(r.area_ha for r in spread)
    # spans small to large, not just the top
    assert areas[0] <= 500 and areas[-1] == 100000
    assert len(select_fires(fires, 99, "size")) == 6   # n >= len returns all


def test_select_fires_rejects_unknown_strategy():
    from vhagar.datasets.t2_optical import select_fires

    with pytest.raises(ValueError, match="strategy must be"):
        select_fires([_fire(area_ha=1), _fire(area_ha=2)], 1, "random")


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

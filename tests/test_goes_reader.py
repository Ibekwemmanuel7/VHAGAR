"""ABI navigation and FDC decoding.

These run entirely offline against a synthetic granule, so CI never depends on
S3 being reachable. The navigation tests check the GOES-R PUG algorithm against
values you can verify by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from vhagar.io.abi_grid import ABIProjection

xr = pytest.importorskip("xarray")

from vhagar.io.goes_reader import (  # noqa: E402
    decode_fdc,
    mask_summary,
    read_fdc_detections,
)

PROJ = ABIProjection(lon_origin_deg=-75.0)


# ----------------------------------------------------- navigation --------


def test_nadir_maps_to_the_subsatellite_point():
    lat, lon = PROJ.to_latlon(0.0, 0.0)
    assert float(lat) == pytest.approx(0.0, abs=1e-9)
    assert float(lon) == pytest.approx(-75.0, abs=1e-9)


def test_scan_angle_roundtrip():
    """to_latlon and to_scan_angles must be inverses over the visible disk."""
    lats = np.array([0.0, 25.0, 45.0, -20.0, 55.0])
    lons = np.array([-75.0, -100.0, -120.0, -60.0, -90.0])
    x, y = PROJ.to_scan_angles(lats, lons)
    assert np.all(np.isfinite(x)), "all these points are visible from 75W"
    lat2, lon2 = PROJ.to_latlon(x, y)
    assert np.allclose(lat2, lats, atol=1e-6)
    assert np.allclose(lon2, lons, atol=1e-6)


def test_offdisk_lines_of_sight_return_nan_not_a_clamp():
    """~15% of a full-disk grid is space. It must not become a coastline."""
    lat, lon = PROJ.to_latlon(0.16, 0.16)   # well past the limb
    assert np.isnan(float(lat)) and np.isnan(float(lon))


def test_far_side_of_the_earth_is_not_visible():
    x, y = PROJ.to_scan_angles(0.0, 105.0)   # antipodal to 75W
    assert np.isnan(float(x)) and np.isnan(float(y))


def test_view_zenith_is_zero_at_nadir_and_grows_outward():
    assert float(PROJ.view_zenith_deg(0.0, -75.0)) == pytest.approx(0.0, abs=1e-6)
    z_near = float(PROJ.view_zenith_deg(20.0, -80.0))
    z_far = float(PROJ.view_zenith_deg(55.0, -125.0))
    assert 0.0 < z_near < z_far < 90.0


def test_pixel_area_grows_off_nadir():
    """FRP is proportional to pixel area, using the nominal 2 km everywhere
    under-reports FRP off nadir by exactly this factor."""
    nadir = float(PROJ.pixel_area_m2(0.0, -75.0))
    oblique = float(PROJ.pixel_area_m2(50.0, -125.0))
    assert nadir == pytest.approx(2000.0**2, rel=0.01)
    assert oblique > 2.0 * nadir


def test_projection_reads_parameters_from_a_dataset():
    ds = _synthetic_fdc()
    proj = ABIProjection.from_dataset(ds)
    assert proj.lon_origin_deg == pytest.approx(-75.0)
    assert proj.semi_major_axis == pytest.approx(6378137.0)


# ------------------------------------------------------- decoding --------


from _fixtures import _synthetic_fdc  # noqa: E402


def test_decode_produces_finite_geolocation_and_geometry():
    g = decode_fdc(_synthetic_fdc(), satellite=19)
    assert g.lat.shape == g.mask.shape == g.view_zenith_deg.shape
    assert np.isfinite(g.lat).all() and np.isfinite(g.lon).all()
    assert np.all(g.view_zenith_deg >= 0.0)
    assert np.all(g.true_pixel_area_m2 > 0.0)
    assert g.scan_start.year >= 2000


def test_decode_counts_fire_pixels_in_both_streams():
    g = decode_fdc(_synthetic_fdc(), satellite=19)
    assert g.n_fire_pixels(filtered=False) == 4   # codes 10, 13, 11, 15
    assert g.n_fire_pixels(filtered=True) == 1    # code 30
    summary = mask_summary(g)
    assert summary["good_quality_fire"] == 1
    assert summary["good_quality_fire_temporally_filtered"] == 1


def test_bbox_crop_reduces_the_grid():
    full = decode_fdc(_synthetic_fdc(n=60), satellite=19)
    lat_mid = float(np.nanmedian(full.lat))
    lon_mid = float(np.nanmedian(full.lon))
    small = decode_fdc(
        _synthetic_fdc(n=60),
        satellite=19,
        bbox=(lon_mid - 0.3, lat_mid - 0.3, lon_mid + 0.3, lat_mid + 0.3),
    )
    assert small.mask.size < full.mask.size
    assert np.isfinite(small.lat).all()


def test_bbox_outside_the_disk_raises_rather_than_returning_empty():
    with pytest.raises(ValueError, match="not visible|outside"):
        decode_fdc(_synthetic_fdc(), satellite=19, bbox=(100.0, 0.0, 110.0, 10.0))


# ----------------------------------------------------- detections --------


def test_detections_carry_geometry_and_drop_unreliable_frp():
    """Saturated (11) and cloud-contaminated (12) pixels keep the detection but
    lose the FRP number, the fire is real, the megawatts are not."""
    g = decode_fdc(_synthetic_fdc(), satellite=19)
    dets = read_fdc_detections(g)
    assert len(dets) == 5
    assert all(d.sensor == "goes" for d in dets)
    assert all(d.view_zenith_deg is not None for d in dets)

    by_conf = {round(d.confidence, 2): d for d in dets}
    saturated = by_conf[0.90]          # mask code 11
    good = by_conf[0.95]               # mask code 10
    assert saturated.frp_mw is None, "saturated FRP must not be passed through"
    assert good.frp_mw is not None


def test_min_confidence_filters_low_probability_detections():
    g = decode_fdc(_synthetic_fdc(), satellite=19)
    assert len(read_fdc_detections(g, min_confidence=0.5)) == 4   # drops code 15
    assert len(read_fdc_detections(g, min_confidence=0.94)) == 2  # codes 10, 30


def test_excluding_filtered_stream_drops_the_30_series():
    g = decode_fdc(_synthetic_fdc(), satellite=19)
    assert len(read_fdc_detections(g, include_filtered=False)) == 4


def test_detections_feed_straight_into_the_fusion_clusterer():
    """The whole point of one detection schema: no adapter needed."""
    from vhagar.harmonize.fusion import cluster_detections

    g = decode_fdc(_synthetic_fdc(), satellite=19)
    dets = read_fdc_detections(g, crs="EPSG:5070")
    events = cluster_detections(dets)
    assert len(events) >= 1
    assert sum(len(e.detections) for e in events) == len(dets)


def test_empty_granule_yields_no_detections():
    ds = _synthetic_fdc(fire_codes=())
    g = decode_fdc(ds, satellite=19)
    assert read_fdc_detections(g) == []


# --------------------------------- CF time decoding (regression) ---------


def _cf_decoded_fdc(n: int = 30) -> xr.Dataset:
    """A granule as xarray actually returns it for a real ABI file.

    Real files carry ``units = "seconds since 2000-01-01 12:00:00"`` on ``t``,
    so xarray decodes it to datetime64 before our code sees it. The original
    fixture stored a plain float and never exercised this path, which is
    exactly how the OverflowError reached real data.
    """
    return _synthetic_fdc(n).assign(t=np.datetime64("2026-08-12T22:11:18", "ns"))


def test_cf_decoded_time_does_not_overflow():
    """Regression: datetime64 t added to the ABI epoch overflows timedelta."""
    g = decode_fdc(_cf_decoded_fdc(), satellite=18)
    assert (g.scan_start.year, g.scan_start.month, g.scan_start.day) == (2026, 8, 12)
    assert (g.scan_start.hour, g.scan_start.minute) == (22, 11)


def test_raw_seconds_time_still_works():
    """The undecoded path must keep working; decode_cf can be switched off."""
    assert decode_fdc(_synthetic_fdc(), satellite=18).scan_start.year >= 2000


@pytest.mark.parametrize(
    "value",
    [
        np.datetime64("2026-08-12T22:11:18", "ns"),
        np.datetime64("2026-08-12T22:11:18", "s"),
        np.float64(838_000_000.0),
        np.float32(838_000_000.0),
    ],
)
def test_scan_start_handles_every_dtype_xarray_might_hand_us(value):
    g = decode_fdc(_synthetic_fdc(30).assign(t=value), satellite=18)
    assert 2000 <= g.scan_start.year <= 2100, f"implausible year from {value.dtype}"
    assert g.scan_start.tzinfo is not None, "scan_start must be timezone-aware"


def test_cf_decoded_granule_still_yields_usable_detections():
    """End to end on the realistic fixture, not just the time field."""
    dets = read_fdc_detections(decode_fdc(_cf_decoded_fdc(), satellite=18), crs="EPSG:5070")
    assert len(dets) == 5
    assert all(d.when.year == 2026 for d in dets)


# ------------------------------ fixed-grid navigation cache --------------
# The ABI fixed grid does not move, so lat/lon/view-zenith/pixel-area are
# identical in every granule on the same grid. Recomputing them per granule was
# the dominant cost of a backfill; these pin the cache that removes it.


def test_navigation_is_computed_once_per_grid_and_then_reused():
    from vhagar.io.goes_reader import _NAV_CACHE_STATS, _clear_nav_cache

    _clear_nav_cache()
    g1 = decode_fdc(_synthetic_fdc(n=40), satellite=19)
    g2 = decode_fdc(_synthetic_fdc(n=40), satellite=19)
    assert _NAV_CACHE_STATS["misses"] == 1, "a second granule on the same grid must not recompute"
    assert _NAV_CACHE_STATS["hits"] >= 1
    # The reuse is literal: the same arrays are shared, not merely equal.
    assert g1.lat is g2.lat
    assert g1.lon is g2.lon
    assert g1.view_zenith_deg is g2.view_zenith_deg
    assert g1.true_pixel_area_m2 is g2.true_pixel_area_m2


def test_cached_navigation_arrays_are_read_only():
    """They are shared across every granule on the grid, so an in-place write
    would corrupt all of them. The arrays must refuse the write."""
    from vhagar.io.goes_reader import _clear_nav_cache

    _clear_nav_cache()
    g = decode_fdc(_synthetic_fdc(n=40), satellite=19)
    with pytest.raises(ValueError):
        g.lat[0, 0] = 0.0
    with pytest.raises(ValueError):
        g.true_pixel_area_m2[0, 0] = 0.0


def test_a_different_grid_is_a_separate_cache_entry():
    from vhagar.io.goes_reader import _NAV_CACHE_STATS, _clear_nav_cache

    _clear_nav_cache()
    decode_fdc(_synthetic_fdc(n=40), satellite=19)
    decode_fdc(_synthetic_fdc(n=50), satellite=19)  # different x/y bytes
    assert _NAV_CACHE_STATS["misses"] == 2


def test_navigation_is_computed_once_even_under_concurrency():
    """The double-checked lock exists so 16 workers hitting an empty cache at
    startup do not all compute the same grid. One miss, many hits."""
    from concurrent.futures import ThreadPoolExecutor

    from vhagar.io.goes_reader import _NAV_CACHE_STATS, _clear_nav_cache

    _clear_nav_cache()

    def work(_):
        return decode_fdc(_synthetic_fdc(n=40), satellite=19).lat.shape

    with ThreadPoolExecutor(max_workers=8) as pool:
        shapes = list(pool.map(work, range(32)))

    assert all(s == shapes[0] for s in shapes)
    assert _NAV_CACHE_STATS["misses"] == 1
    assert _NAV_CACHE_STATS["hits"] == 31


def test_cache_cap_of_zero_disables_caching():
    """The benchmark toggle: cap 0 forces a recompute every call on the same
    code path, which is how the before number is measured honestly."""
    import vhagar.io.goes_reader as reader

    reader._clear_nav_cache()
    original = reader._NAV_CACHE_MAX
    reader._NAV_CACHE_MAX = 0
    try:
        decode_fdc(_synthetic_fdc(n=40), satellite=19)
        decode_fdc(_synthetic_fdc(n=40), satellite=19)
        assert reader._NAV_CACHE_STATS["misses"] == 2
        assert reader._NAV_CACHE_STATS["hits"] == 0
        assert len(reader._NAV_CACHE) == 0
    finally:
        reader._NAV_CACHE_MAX = original
        reader._clear_nav_cache()


# ---------------------------- corrupt scan-start recovery ----------------
# A real GOES-18 granule in the 7-day backfill decoded its scan start to
# 2000-01-01 (the ABI epoch), which split the coverage record and stamped 60
# detection rows with the year 2000. The key carries the true time, so we
# recover from it rather than trust an impossible value.

_BAD_KEY = (
    "ABI-L2-FDCC/2026/215/15/"
    "OR_ABI-L2-FDCC-M4_G18_s20262151510224_e20262151510224_c20262151515522.nc"
)


def test_implausible_scan_start_is_recovered_from_the_key():
    from datetime import UTC as _UTC
    from datetime import datetime as _dt

    from vhagar.io.goes_reader import _validated_scan_start

    bad = _dt(2000, 1, 1, 11, 43, 21, tzinfo=_UTC)  # what the granule decoded to
    fixed = _validated_scan_start(bad, _BAD_KEY, satellite=18)
    # Day 215 of 2026 is 3 August; the key says 15:10:22.
    assert (fixed.year, fixed.month, fixed.day) == (2026, 8, 3)
    assert (fixed.hour, fixed.minute) == (15, 10)


def test_a_plausible_scan_start_is_returned_untouched():
    from datetime import UTC as _UTC
    from datetime import datetime as _dt

    from vhagar.io.goes_reader import _validated_scan_start

    good = _dt(2026, 8, 3, 15, 10, 22, tzinfo=_UTC)
    assert _validated_scan_start(good, _BAD_KEY, satellite=18) == good


def test_recovery_keeps_the_decoded_time_when_the_key_is_unparseable():
    from datetime import UTC as _UTC
    from datetime import datetime as _dt

    from vhagar.io.goes_reader import _validated_scan_start

    bad = _dt(2000, 1, 1, tzinfo=_UTC)
    # No 's' scan-start token in this key, so there is nothing to recover.
    kept = _validated_scan_start(bad, "ABI-L2-FDCC/junk/no_stamp_here.nc", satellite=18)
    assert kept == bad

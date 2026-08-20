from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from vhagar.harmonize.fusion import (
    Detection,
    cluster_detections,
    event_features,
    parallax_offset_m,
)

T0 = datetime(2026, 7, 15, 12, 0)


def test_geo_and_leo_detections_of_one_fire_merge():
    """A GOES pixel offset ~1.5 km from a VIIRS pixel is the same fire.

    Naive nearest-pixel matching would call this a false alarm; the 3x3
    tolerance is what drops the apparent FAR from ~30% to ~10%.
    """
    dets = [
        Detection("viirs", 0.0, 0.0, T0, frp_mw=12.0),
        Detection("goes", 1_500.0, 0.0, T0 + timedelta(minutes=20), frp_mw=30.0),
    ]
    assert len(cluster_detections(dets)) == 1


def test_distant_fires_stay_separate():
    dets = [
        Detection("viirs", 0.0, 0.0, T0),
        Detection("viirs", 50_000.0, 0.0, T0),
    ]
    assert len(cluster_detections(dets)) == 2


def test_temporal_gap_splits_events():
    dets = [
        Detection("viirs", 0.0, 0.0, T0),
        Detection("viirs", 100.0, 0.0, T0 + timedelta(hours=48)),
    ]
    assert len(cluster_detections(dets, max_gap_hours=12.0)) == 2


def test_forest_buffer_is_wider_than_crop_buffer():
    forest = Detection("viirs", 0.0, 0.0, T0, landcover="forest")
    crop = Detection("viirs", 0.0, 0.0, T0, landcover="crop")
    assert forest.tolerance_m > crop.tolerance_m


def test_single_link_chains_a_spreading_fire():
    """Detections strung along a run should be one event, not many."""
    dets = [
        Detection("viirs", i * 900.0, 0.0, T0 + timedelta(hours=i), landcover="grass")
        for i in range(8)
    ]
    events = cluster_detections(dets)
    assert len(events) == 1
    assert events[0].duration_h == pytest.approx(7.0)


def test_event_features_exclude_raw_coordinates():
    """The single most important guard against spatial memorisation."""
    event = cluster_detections(
        [
            Detection("viirs", 0.0, 0.0, T0, frp_mw=10.0, bt_mir_k=340.0, bt_tir_k=300.0),
            Detection("goes", 800.0, 0.0, T0 + timedelta(minutes=10), frp_mw=25.0),
        ]
    )[0]
    feats = event_features(event)
    forbidden = {"x", "y", "lat", "lon", "latitude", "longitude", "easting", "northing"}
    assert not (forbidden & set(feats)), f"coordinate leaked into features: {set(feats) & forbidden}"


def test_event_features_capture_multisensor_agreement_and_growth():
    event = cluster_detections(
        [
            Detection("viirs", 0.0, 0.0, T0, frp_mw=10.0),
            Detection("goes", 400.0, 0.0, T0 + timedelta(hours=1), frp_mw=30.0),
            Detection("goes", 500.0, 0.0, T0 + timedelta(hours=2), frp_mw=50.0),
        ]
    )[0]
    f = event_features(event)
    assert f["n_sensors"] == 2
    assert f["multi_sensor_agreement"] == 1.0
    assert f["frp_growth_mw_per_h"] == pytest.approx(20.0)
    assert f["peak_frp_mw"] == pytest.approx(50.0)


def test_growth_rate_is_nan_when_the_event_is_too_short_to_support_one():
    """Real GOES data produced -606 MW/h from two points 12 minutes apart.

    A rate needs enough points and enough elapsed time. Below that, NaN is the
    honest answer; a large number with a unit attached is not.
    """
    event = cluster_detections(
        [
            Detection("goes", 0.0, 0.0, T0, frp_mw=99.4),
            Detection("goes", 300.0, 0.0, T0 + timedelta(minutes=12), frp_mw=20.0),
        ]
    )[0]
    assert np.isnan(event_features(event)["frp_growth_mw_per_h"])


def test_growth_rate_uses_a_fit_not_the_endpoints():
    """One noisy final sample must not dominate the reported rate."""
    dets = [
        Detection("goes", i * 100.0, 0.0, T0 + timedelta(minutes=15 * i), frp_mw=frp)
        for i, frp in enumerate([10.0, 20.0, 30.0, 40.0, 5.0])
    ]
    growth = event_features(cluster_detections(dets)[0])["frp_growth_mw_per_h"]
    endpoint = (5.0 - 10.0) / 1.0
    assert growth > endpoint, "an endpoint difference would report a sharp decline"


def test_static_anomaly_fraction_flags_industrial_sources():
    event = cluster_detections(
        [
            Detection("viirs", 0.0, 0.0, T0, static_anomaly=True),
            Detection("viirs", 200.0, 0.0, T0 + timedelta(hours=1), static_anomaly=True),
        ]
    )[0]
    assert event_features(event)["static_anomaly_fraction"] == pytest.approx(1.0)


def test_parallax_grows_with_distance_from_subsatellite_point():
    near = parallax_offset_m(3_000.0, satellite_lon_deg=-75.2, pixel_lon_deg=-75.0, pixel_lat_deg=0.0)
    far = parallax_offset_m(3_000.0, satellite_lon_deg=-75.2, pixel_lon_deg=-125.0, pixel_lat_deg=50.0)
    assert far > near >= 0.0


def test_empty_input():
    assert cluster_detections([]) == []


# ------------------------------- geometry-derived matching tolerance -----


def test_tolerance_grows_with_view_zenith():
    """The ABI footprint grows with view zenith (slant-range geometry), so the
    tolerance must too. By ~70 deg the footprint alone exceeds a flat 2 km."""
    from vhagar.harmonize.fusion import geo_leo_tolerance_m

    nadir = float(geo_leo_tolerance_m(0.0, elevation_m=0.0))
    mid = float(geo_leo_tolerance_m(48.1, elevation_m=0.0))
    edge = float(geo_leo_tolerance_m(70.0, elevation_m=0.0))
    assert nadir < mid < edge
    assert edge > 2000.0, "at 70 deg the footprint alone must exceed a flat 2 km"


def test_tolerance_matches_the_measured_california_geometry():
    """GOES-18 over northern California: 48.1 deg vza, ~1500 m terrain.

    Reproduces the number derived from the 2026-08-12 run, where the observed
    median separation was 1.62 km and p75 was 2.80 km.
    """
    from vhagar.harmonize.fusion import geo_leo_tolerance_m

    tol = float(geo_leo_tolerance_m(48.1, elevation_m=1500.0))
    assert tol == pytest.approx(3500.0, rel=0.02)
    assert tol > 2800.0, "must cover the observed p75, not just the median"


def test_terrain_parallax_contributes_as_elevation_times_tan_vza():
    from vhagar.harmonize.fusion import geo_leo_tolerance_m

    flat = float(geo_leo_tolerance_m(48.1, elevation_m=0.0))
    high = float(geo_leo_tolerance_m(48.1, elevation_m=2000.0))
    assert high - flat == pytest.approx(2000.0 * np.tan(np.radians(48.1)), rel=1e-6)


def test_geo_detections_with_view_angle_get_a_computed_tolerance():
    """Landcover is held constant so the geometry term is what varies."""
    oblique = Detection("goes", 0.0, 0.0, T0, view_zenith_deg=48.1, landcover="other")
    nadir = Detection("goes", 0.0, 0.0, T0, view_zenith_deg=0.0, landcover="other")
    flat_guess = Detection("goes", 0.0, 0.0, T0, landcover="other")
    assert oblique.tolerance_m > nadir.tolerance_m
    assert nadir.tolerance_m < flat_guess.tolerance_m, (
        "at nadir the flat 6 km guess is far too loose and merges distinct fires"
    )


def test_polar_detections_are_unaffected_by_view_angle():
    """VIIRS aggregation caps footprint growth, so its tolerance is a constant."""
    a = Detection("viirs", 0.0, 0.0, T0, view_zenith_deg=0.0)
    b = Detection("viirs", 0.0, 0.0, T0, view_zenith_deg=60.0)
    assert a.tolerance_m == b.tolerance_m


def test_wider_tolerance_merges_a_pair_the_flat_2km_would_have_split():
    """The concrete failure the measurement exposed: one fire, two events."""
    goes = Detection("goes", 0.0, 0.0, T0, view_zenith_deg=48.1, landcover="forest")
    viirs = Detection("viirs", 2_700.0, 0.0, T0 + timedelta(minutes=20), landcover="forest")
    events = cluster_detections([goes, viirs], extra_tolerance_m=0.0)
    assert len(events) == 1, "a 2.7 km separation at 48 deg is one fire, not two"

"""Tests for the portfolio-to-event intersection core (pure numpy/stdlib, CI-safe).
Uses a synthetic square footprint as a stand-in for a detection hull."""
from __future__ import annotations

from datetime import UTC, datetime

from vhagar.intersect import (
    distance_to_ring_m,
    intersect_portfolio,
    parse_sensors,
    point_in_ring,
)

# a ~2.2 km square footprint centred at (-120.00, 39.00)
SQUARE = [[-120.02, 38.98], [-119.98, 38.98], [-119.98, 39.02], [-120.02, 39.02]]

EVENT = {
    "type": "Feature",
    "geometry": {"type": "Polygon", "coordinates": [SQUARE]},
    "properties": {"event_id": "E1", "label": "Test Fire", "sensors": "G18, VIIRS-NOAA20",
                   "n_detections": 12, "max_frp_mw": 240,
                   "first_seen": "2026-08-16 06:00:00", "last_seen": "2026-08-16 12:00:00"},
}
FC = {"type": "FeatureCollection", "features": [EVENT]}
NOW = datetime(2026, 8, 16, 15, 0, 0, tzinfo=UTC)


def test_point_in_ring_and_distance():
    assert point_in_ring(-120.00, 39.00, SQUARE) is True
    assert point_in_ring(-120.00, 39.10, SQUARE) is False
    assert distance_to_ring_m(-120.00, 39.00, SQUARE) == 0.0
    # a point ~0.005 deg north of the top edge is roughly 500 m out
    d = distance_to_ring_m(-120.00, 39.025, SQUARE)
    assert 400 < d < 700


def test_parse_sensors_dedup():
    assert parse_sensors("G18, VIIRS-NOAA20; g18") == ["G18", "VIIRS-NOAA20"]
    assert parse_sensors("") == []


def test_intersect_classifies_inside_buffer_and_clear():
    portfolio = [
        {"id": "inside", "lat": 39.00, "lon": -120.00},
        {"id": "near", "lat": 39.025, "lon": -120.00},   # ~500 m outside, inside 1 km buffer
        {"id": "far", "lat": 39.10, "lon": -120.00},     # ~9 km away
    ]
    res = intersect_portfolio(portfolio, FC, buffer_m=1000.0, now=NOW)
    by_id = {r["id"]: r for r in res["affected"]}
    assert by_id["inside"]["status"] == "inside_footprint"
    assert by_id["inside"]["distance_m"] == 0
    assert by_id["inside"]["confidence"] == "confirmed_multi_sensor"
    assert by_id["inside"]["data_age_hours"] == 3.0
    assert by_id["near"]["status"] == "within_buffer"
    assert res["summary"]["affected_count"] == 2
    assert res["summary"]["inside_footprint_count"] == 1
    clear_ids = {r["id"] for r in res["clear"]}
    assert "far" in clear_ids
    assert "disclosure" in res


def test_confidence_single_sensor():
    ev = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [SQUARE]},
          "properties": {"event_id": "E2", "sensors": "G18", "n_detections": 2,
                         "last_seen": "2026-08-16 12:00:00"}}
    res = intersect_portfolio([{"id": "p", "lat": 39.0, "lon": -120.0}],
                              {"type": "FeatureCollection", "features": [ev]}, now=NOW)
    assert res["affected"][0]["confidence"] == "single_sensor_sparse"

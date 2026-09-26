"""Tests for the Evidence Pack HTML report renderer (pure stdlib, CI-safe)."""
from __future__ import annotations

from vhagar.report import render_evidence_pack

SQUARE = [[-120.02, 38.98], [-119.98, 38.98], [-119.98, 39.02], [-120.02, 39.02]]
EVENTS = {"type": "FeatureCollection",
          "features": [{"type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": [SQUARE]},
                        "properties": {"event_id": "E1", "label": "Test Fire"}}]}
RESULT = {
    "affected": [{"id": "P-inside", "lat": 39.0, "lon": -120.0, "status": "inside_footprint",
                  "distance_m": 0, "event_id": "E1", "event_label": "Test Fire",
                  "last_seen_utc": "2026-08-16 12:00:00", "data_age_hours": 3.0,
                  "sensors": ["G18", "VIIRS-NOAA20"], "confidence": "confirmed_multi_sensor"}],
    "clear": [{"id": "P-far", "lat": 39.1, "lon": -120.0, "status": "clear"}],
    "summary": {"portfolio_size": 2, "affected_count": 1, "inside_footprint_count": 1,
                "within_buffer_count": 0, "buffer_m": 1000.0},
    "disclosure": "A footprint is a detection hull, not an agency perimeter.",
    "metadata": {"region": "california", "days": 3, "mode": "live", "event_count": 1},
}


def test_render_contains_core_sections():
    html = render_evidence_pack(RESULT, EVENTS, portfolio_name="ACME Book",
                                generated="2026-09-26 12:00 UTC")
    assert "<!doctype html>" in html.lower()
    assert "Wildfire Event Evidence Pack" in html
    assert "ACME Book" in html and "2026-09-26 12:00 UTC" in html
    assert "P-inside" in html and "confirmed_multi_sensor" in html
    assert "Inside footprint" in html
    assert "detection hull, not an agency perimeter" in html
    assert "<svg" in html and "<polygon" in html        # event footprint drawn
    # summary cards reflect the counts
    assert ">2<" in html and ">1<" in html


def test_render_escapes_html():
    r = dict(RESULT)
    r["affected"] = [dict(RESULT["affected"][0], id="<script>alert(1)</script>")]
    html = render_evidence_pack(r, EVENTS)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_render_empty_portfolio():
    r = {"affected": [], "clear": [], "summary": {"portfolio_size": 0, "affected_count": 0},
         "disclosure": "x", "metadata": {}}
    html = render_evidence_pack(r, {})
    assert "No portfolio locations were affected" in html

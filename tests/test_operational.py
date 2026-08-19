"""Operational parity modules: weather parse, spread-risk, KMZ export."""
from __future__ import annotations

import io
import zipfile

from vhagar.export.kmz import events_to_kmz
from vhagar.features.spread_risk import risk_class, risk_color, spread_risk_score
from vhagar.weather.open_meteo import parse_current


def test_parse_current():
    w = parse_current({"current": {"temperature_2m": 30.0, "relative_humidity_2m": 15,
                                   "wind_speed_10m": 9.0, "wind_direction_10m": 200,
                                   "wind_gusts_10m": 13}})
    assert w["wind_speed_ms"] == 9.0 and w["rh_pct"] == 15.0 and w["temp_c"] == 30.0
    assert parse_current({}) is None and parse_current("x") is None


def test_spread_risk_monotone_and_bands():
    hi = spread_risk_score(35, 8, 13)
    lo = spread_risk_score(16, 75, 1)
    assert hi > lo
    assert risk_class(hi) in ("High", "Extreme") and risk_class(lo) == "Low"
    assert spread_risk_score(None, None, None) is None
    assert risk_class(None) == "Unknown"
    assert risk_color(hi).startswith("#")
    # wind dominates: raising wind alone raises the score
    assert spread_risk_score(20, 50, 12) > spread_risk_score(20, 50, 1)


def test_events_to_kmz_roundtrip():
    fc = {"type": "FeatureCollection", "features": [{
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[-120.0, 38.0], [-119.9, 38.0],
                                                         [-119.9, 38.1], [-120.0, 38.1],
                                                         [-120.0, 38.0]]]},
        "properties": {"label": "Cluster 7", "n_detections": 40, "footprint_ha": 900,
                       "perimeter_km": 12, "total_frp_mw": 2500, "max_frp_mw": 700,
                       "risk_class": "Extreme", "risk_score": 82, "sensors": "GOES-18"}}]}
    kmz = events_to_kmz(fc)
    doc = zipfile.ZipFile(io.BytesIO(kmz)).read("doc.kml").decode()
    assert "<Placemark>" in doc and "Cluster 7" in doc
    assert "-120.00000,38.00000,0" in doc          # coordinates present
    assert "7f1c1cb7" in doc                        # Extreme fill colour
    # a non-polygon feature is skipped, not crashed
    fc["features"].append({"type": "Feature", "geometry": {"type": "Point",
                          "coordinates": [-120, 38]}, "properties": {}})
    assert events_to_kmz(fc)

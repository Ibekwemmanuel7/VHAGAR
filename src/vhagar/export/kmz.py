"""KMZ export for fire events: agency-ready perimeters for Google Earth / GIS.

Takes a GeoJSON FeatureCollection of event polygons (the shape the serving layer
emits) and writes a zipped KML with risk-styled polygons and a readable
description per event. Hand-rolled KML, no ``simplekml`` dependency.
"""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime
from xml.sax.saxutils import escape

__all__ = ["events_to_kmz"]

# KML polygon fill colours (aabbggrr, ~50% alpha) by spread-risk class.
_FILL = {"Low": "7f5b9e2e", "Moderate": "7f3ab4e0", "High": "7f2b55e8",
         "Extreme": "7f1c1cb7", "Unknown": "7f8f8f8f"}
_LINE = {"Low": "ff5b9e2e", "Moderate": "ff3ab4e0", "High": "ff2b55e8",
         "Extreme": "ff1c1cb7", "Unknown": "ff8f8f8f"}


def _num(props, key) -> str:
    v = props.get(key)
    try:
        return f"{float(v):.1f}"
    except (TypeError, ValueError):
        return "-"


def _description(props) -> str:
    def g(k):
        v = props.get(k)
        return "-" if v is None or (isinstance(v, float) and v != v) else v

    lines = [
        f"Detections: {g('n_detections')}",
        f"Footprint: {_num(props, 'footprint_ha' if 'footprint_ha' in props else 'area_ha')} ha",
        f"Perimeter: {_num(props, 'perimeter_km')} km",
        f"Total FRP: {_num(props, 'total_frp_mw')} MW   Peak: {_num(props, 'max_frp_mw')} MW",
        f"Spread risk: {g('risk_class')} ({_num(props, 'risk_score')})",
        f"Wind: {_num(props, 'wind_speed_ms')} m/s @ {_num(props, 'wind_dir_deg')} deg",
        f"RH: {_num(props, 'rh_pct')} %   Temp: {_num(props, 'temp_c')} C",
        f"Sensors: {g('sensors')}",
        f"First seen: {g('first_seen')}   Last: {g('last_seen')}",
    ]
    return escape("\n".join(str(x) for x in lines))


def _placemark(feat) -> str:
    geom = feat.get("geometry") or {}
    if geom.get("type") != "Polygon":
        return ""
    props = feat.get("properties") or {}
    rings = geom.get("coordinates") or []
    if not rings:
        return ""
    risk = props.get("risk_class") or "Unknown"

    def ring_xml(ring, tag):
        coords = " ".join(f"{float(lon):.5f},{float(lat):.5f},0" for lon, lat in ring)
        return (f"<{tag}><LinearRing><coordinates>{coords}</coordinates>"
                f"</LinearRing></{tag}>")

    boundaries = ring_xml(rings[0], "outerBoundaryIs")
    for hole in rings[1:]:
        boundaries += ring_xml(hole, "innerBoundaryIs")
    name = escape(str(props.get("label", "Fire event")))
    return (f"<Placemark><name>{name}</name>"
            f"<description>{_description(props)}</description>"
            f"<Style><LineStyle><color>{_LINE.get(risk, _LINE['Unknown'])}</color>"
            f"<width>2</width></LineStyle>"
            f"<PolyStyle><color>{_FILL.get(risk, _FILL['Unknown'])}</color><fill>1</fill>"
            f"</PolyStyle></Style>"
            f"<Polygon>{boundaries}</Polygon></Placemark>")


def _build_kml(fc) -> str:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    placemarks = "".join(_placemark(f) for f in (fc.get("features") or []))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        "<name>VHAGAR fire events</name>"
        f"<description>Satellite-derived fire events. Generated {stamp}.</description>"
        f"{placemarks}</Document></kml>"
    )


def events_to_kmz(fc) -> bytes:
    """Serialise a GeoJSON FeatureCollection of event polygons to KMZ bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", _build_kml(fc))
    return buf.getvalue()

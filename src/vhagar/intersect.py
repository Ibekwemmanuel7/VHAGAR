"""Portfolio-to-event intersection: the core of the Wildfire Event Evidence Pack.

Given a customer portfolio of locations and a set of fire events (VHAGAR's fused
detection footprints), this answers the one question a claims or exposure team asks
after a fire is reported: which of my locations are affected, by which event, how
close, how fresh is the evidence, and how well corroborated is it.

Honest scope, enforced in the output: an event footprint here is the CONVEX HULL of
observed thermal detections, an evidence extent, NOT a legal or agency fire
perimeter, and NOT a burned-area measurement. A location flagged "inside footprint"
means it falls inside that detection hull, which is a screening signal for
inspection, not a proof of loss. Confidence reflects sensor corroboration and
detection count, not a probability of damage.

Pure numpy and stdlib only, so it runs in the core CI env with no geospatial stack.
"""
from __future__ import annotations

import math
from datetime import UTC, datetime

__all__ = [
    "haversine_m",
    "point_in_ring",
    "distance_to_ring_m",
    "parse_sensors",
    "intersect_portfolio",
]

_EARTH_R_M = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in metres between two lon/lat points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _EARTH_R_M * math.asin(min(1.0, math.sqrt(a)))


def point_in_ring(lon: float, lat: float, ring) -> bool:
    """Ray-casting point-in-polygon on a lon/lat ring (list of [lon, lat]).

    The ring may be open or closed; the test wraps to the first vertex."""
    n = len(ring)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > lat) != (yj > lat)) and \
           (lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _local_m(lon, lat, lon0, lat0):
    """Equirectangular metres offset of (lon, lat) from an anchor, good for the
    small distances (a few km) an intersection needs."""
    x = math.radians(lon - lon0) * _EARTH_R_M * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * _EARTH_R_M
    return x, y


def _seg_dist_m(px, py, ax, ay, bx, by) -> float:
    """Point-to-segment distance in the local metric plane."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def distance_to_ring_m(lon: float, lat: float, ring) -> float:
    """Minimum distance in metres from a point to a lon/lat ring's edges. Returns
    0.0 when the point is inside the ring."""
    if point_in_ring(lon, lat, ring):
        return 0.0
    px, py = _local_m(lon, lat, lon, lat)  # anchor at the point -> (0, 0)
    best = math.inf
    n = len(ring)
    for i in range(n):
        ax, ay = _local_m(ring[i][0], ring[i][1], lon, lat)
        j = (i + 1) % n
        bx, by = _local_m(ring[j][0], ring[j][1], lon, lat)
        best = min(best, _seg_dist_m(px, py, ax, ay, bx, by))
    return best


def parse_sensors(s) -> list[str]:
    """Split an event 'sensors' string ('G18, MODIS; VIIRS-NOAA20') into a clean
    de-duplicated list, preserving order."""
    if not s:
        return []
    out, seen = [], set()
    for tok in str(s).replace(";", ",").split(","):
        t = tok.strip()
        if t and t.upper() not in seen:
            seen.add(t.upper())
            out.append(t)
    return out


def _parse_time(s):
    """Parse an event timestamp to an aware UTC datetime, or None."""
    if not s:
        return None
    txt = str(s).strip().replace("T", " ").replace("Z", "")
    txt = txt.split(".")[0].split("+")[0].strip()   # drop microseconds / offset if present
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(txt, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _confidence(sensors: list[str], n_detections) -> str:
    """Corroboration tier from sensor diversity and detection count. Mirrors the
    console's candidate-versus-confirmed logic; this is evidence strength, NOT a
    damage probability."""
    distinct = len({s.upper().split("-")[0] for s in sensors})
    nd = int(n_detections or 0)
    if distinct >= 2:
        return "confirmed_multi_sensor"
    if nd >= 5:
        return "single_sensor_repeated"
    return "single_sensor_sparse"


def _ring_of(feature) -> list:
    geom = (feature or {}).get("geometry") or {}
    coords = geom.get("coordinates") or []
    if geom.get("type") == "Polygon" and coords:
        return coords[0]
    return []


def intersect_portfolio(portfolio, events, *, buffer_m: float = 1000.0, now=None) -> dict:
    """Intersect a portfolio of point locations with fire-event footprints.

    portfolio: iterable of dicts with 'lat' and 'lon' (and optional 'id', 'name').
    events:    a GeoJSON FeatureCollection dict, or an iterable of Feature dicts,
               each a Polygon footprint with VHAGAR event properties.
    buffer_m:  a location is flagged 'affected' if it is inside a footprint or
               within this many metres of one (screening buffer, not a perimeter).
    now:       aware datetime used to compute data age; defaults to current UTC.

    Returns a dict with 'affected' (the flagged locations, each carrying the nearest
    event, distance, inside flag, provenance, confidence, and data age), 'clear'
    (locations with no event within the buffer), and a 'disclosure' block.
    """
    feats = events.get("features", []) if isinstance(events, dict) else list(events)
    now = now or datetime.now(UTC)

    affected, clear = [], []
    for i, loc in enumerate(portfolio):
        lat, lon = float(loc["lat"]), float(loc["lon"])
        pid = loc.get("id", loc.get("name", f"loc-{i}"))
        best = None
        for f in feats:
            ring = _ring_of(f)
            if len(ring) < 3:
                continue
            d = distance_to_ring_m(lon, lat, ring)
            if best is None or d < best[0]:
                best = (d, f)
        if best is None:
            clear.append({"id": pid, "lat": lat, "lon": lon, "status": "no_events_in_view"})
            continue
        dist_m, f = best
        p = f.get("properties", {})
        inside = dist_m == 0.0
        if not (inside or dist_m <= buffer_m):
            clear.append({"id": pid, "lat": lat, "lon": lon, "status": "clear",
                          "nearest_event_id": p.get("event_id"),
                          "nearest_distance_m": round(dist_m)})
            continue
        sensors = parse_sensors(p.get("sensors"))
        last = _parse_time(p.get("last_seen"))
        age_h = round((now - last).total_seconds() / 3600.0, 1) if last else None
        affected.append({
            "id": pid, "lat": lat, "lon": lon,
            "status": "inside_footprint" if inside else "within_buffer",
            "distance_m": round(dist_m),
            "event_id": p.get("event_id"), "event_label": p.get("label"),
            "first_seen_utc": p.get("first_seen"), "last_seen_utc": p.get("last_seen"),
            "data_age_hours": age_h,
            "sensors": sensors, "n_detections": p.get("n_detections"),
            "max_frp_mw": p.get("max_frp_mw"),
            "confidence": _confidence(sensors, p.get("n_detections")),
        })

    affected.sort(key=lambda r: (r["status"] != "inside_footprint", r["distance_m"]))
    return {
        "affected": affected,
        "clear": clear,
        "summary": {"portfolio_size": len(affected) + len(clear),
                    "affected_count": len(affected),
                    "inside_footprint_count": sum(1 for r in affected if r["status"] == "inside_footprint"),
                    "within_buffer_count": sum(1 for r in affected if r["status"] == "within_buffer"),
                    "buffer_m": buffer_m, "generated_utc": now.isoformat()},
        "disclosure": (
            "An event footprint is the convex hull of observed thermal detections, an "
            "evidence extent, not a legal or agency fire perimeter and not a burned-area "
            "measurement. 'Inside footprint' is a screening signal for inspection, not a "
            "proof of loss. Confidence reflects sensor corroboration and detection count, "
            "not a probability of damage."),
    }

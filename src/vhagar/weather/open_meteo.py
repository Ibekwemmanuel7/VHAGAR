"""Open-Meteo fire-weather client: current wind, RH, temperature per point.

Open-Meteo is free and key-less. Points are fetched in a single batched request
(comma-separated coordinates), which returns one object per point. Network or
parse failures degrade gracefully to ``None`` so callers still render perimeters,
just without weather or risk. Uses the standard library only (``urllib``); no
``requests`` dependency.

Weather is *current* conditions at the location. For a live NRT feed that is
coincident with the detections; for an archived detection window it is
present-day weather at that place, not the fire's time. Label it accordingly.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

__all__ = ["fetch_weather", "parse_current", "WEATHER_KEYS"]

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_CURRENT_VARS = ("temperature_2m,relative_humidity_2m,"
                 "wind_speed_10m,wind_direction_10m,wind_gusts_10m")
WEATHER_KEYS = ["temp_c", "rh_pct", "wind_speed_ms", "wind_dir_deg", "wind_gust_ms"]

_CACHE: dict = {}          # coarse (lat, lon) -> (timestamp, weather dict)
_TTL = 900                 # 15 min: weather changes slowly, quota is shared
_MAX_POINTS = 100


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_current(item) -> dict | None:
    """Map one Open-Meteo forecast object to the standard weather dict."""
    if not isinstance(item, dict):
        return None
    cur = item.get("current") or {}
    if not cur:
        return None
    return {
        "temp_c": _num(cur.get("temperature_2m")),
        "rh_pct": _num(cur.get("relative_humidity_2m")),
        "wind_speed_ms": _num(cur.get("wind_speed_10m")),
        "wind_dir_deg": _num(cur.get("wind_direction_10m")),
        "wind_gust_ms": _num(cur.get("wind_gusts_10m")),
    }


def fetch_weather(points, timeout: float = 10.0):
    """Current weather per ``(lat, lon)``; list aligned to input, ``None`` on failure.

    Batched into one request (capped at 100 points). Cached results (coarse
    coordinate, 15 min) are reused and only the uncached points are fetched.
    """
    if not points:
        return []
    pts = [(float(la), float(lo)) for la, lo in points]
    out: list = [None] * len(pts)
    need: list = []
    now = time.time()
    for i, (la, lo) in enumerate(pts):
        hit = _CACHE.get((round(la, 2), round(lo, 2)))
        if hit and now - hit[0] < _TTL:
            out[i] = hit[1]
        elif len(need) < _MAX_POINTS:
            need.append(i)
    if not need:
        return out
    lats = ",".join(f"{pts[i][0]:.4f}" for i in need)
    lons = ",".join(f"{pts[i][1]:.4f}" for i in need)
    q = urllib.parse.urlencode({"latitude": lats, "longitude": lons,
                                "current": _CURRENT_VARS, "wind_speed_unit": "ms",
                                "timezone": "UTC"})
    try:
        with urllib.request.urlopen(f"{OPEN_METEO_URL}?{q}", timeout=timeout) as r:  # noqa: S310
            data = json.loads(r.read().decode("utf-8"))
    except Exception:            # noqa: BLE001 - degrade gracefully
        return out
    arr = data if isinstance(data, list) else [data]
    for k, i in enumerate(need):
        w = parse_current(arr[k]) if k < len(arr) else None
        out[i] = w
        if w is not None:
            _CACHE[(round(pts[i][0], 2), round(pts[i][1], 2))] = (now, w)
    return out

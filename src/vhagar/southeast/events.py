"""Turn FIRMS detections into :class:`ObservedEvent` objects.

Two clusterers, same rule (single link: two detections join when they are
within the spatial tolerance and the time gap):

* :func:`cluster_obs` reuses :func:`vhagar.harmonize.fusion.cluster_detections`
  with its per-sensor tolerances. Exact and O(n^2): use it for live windows.
* :func:`cluster_obs_grid` uses a fixed tolerance and a spatial hash, so it
  scales to a full Southeast burn season (10^5 detections). It matches the
  fusion result when the tolerances agree (tested).

Both work on a local equirectangular plane. Over Georgia plus Florida the
scale error of that plane is a few percent, small next to 1 km tolerances.
"""

from __future__ import annotations

import math

from vhagar.harmonize.fusion import Detection, cluster_detections
from vhagar.southeast.assess import Obs, ObservedEvent

__all__ = ["cluster_obs", "cluster_obs_grid", "obs_from_firms"]

_R = 6_371_000.0
_GEOLOCATION_M = 100.0   # VIIRS I-band geolocation, order of magnitude


def obs_from_firms(records, static_mask=None) -> list[Obs]:
    """Convert :class:`vhagar.io.firms.FirmsRecord` objects.

    Location error is half the pixel diagonal (FIRMS ``scan`` and ``track`` are
    pixel sizes in km) plus a geolocation term. ``static_mask(lon, lat) ->
    bool`` marks persistent industrial sources when supplied.
    """
    out = []
    for r in records:
        sensor = "viirs" if "VIIRS" in (r.instrument or "").upper() else (
            "modis" if "MODIS" in (r.instrument or "").upper() else "viirs")
        err = area = None
        if math.isfinite(r.scan) and math.isfinite(r.track) and r.scan > 0 and r.track > 0:
            err = 0.5 * math.hypot(r.scan, r.track) * 1000.0 + _GEOLOCATION_M
            area = r.scan * r.track * 1e6
        out.append(Obs(
            lon=r.longitude, lat=r.latitude, when=r.acq_datetime, sensor=sensor,
            frp_mw=r.frp if math.isfinite(r.frp) else None,
            location_error_m=err, pixel_area_m2=area,
            static_anomaly=bool(static_mask(r.longitude, r.latitude)) if static_mask else False,
        ))
    return out


def cluster_obs(
    obs: list[Obs],
    *,
    max_gap_hours: float = 24.0,
    id_prefix: str = "se",
    lat0: float | None = None,
) -> list[ObservedEvent]:
    """Cluster detections into events (single link in space and time).

    ``max_gap_hours=24`` bridges the day and night VIIRS passes of one burn.
    """
    if not obs:
        return []
    lat0 = sum(o.lat for o in obs) / len(obs) if lat0 is None else lat0
    k = math.cos(math.radians(lat0))
    dets = []
    back: dict[int, Obs] = {}
    for o in obs:
        d = Detection(
            sensor=o.sensor,
            x=math.radians(o.lon) * _R * k,
            y=math.radians(o.lat) * _R,
            when=o.when,
            frp_mw=o.frp_mw,
            static_anomaly=o.static_anomaly,
        )
        dets.append(d)
        back[id(d)] = o
    events = cluster_detections(dets, max_gap_hours=max_gap_hours, id_prefix=id_prefix)
    return [
        ObservedEvent(e.event_id, sorted((back[id(d)] for d in e.detections), key=lambda o: o.when))
        for e in events
    ]


def cluster_obs_grid(
    obs: list[Obs],
    *,
    tolerance_m: float = 1_125.0,
    max_gap_hours: float = 24.0,
    id_prefix: str = "se",
    lat0: float | None = None,
) -> list[ObservedEvent]:
    """Single-link clustering with a spatial hash.

    Cells are ``tolerance_m`` wide, so every neighbour within tolerance is in
    the same or an adjacent cell. Detections are visited in time order and
    compared only with detections in the 3 x 3 cell block that are inside the
    time gap. ``tolerance_m=1125`` is the VIIRS value in
    :data:`vhagar.harmonize.fusion.SENSOR_TOLERANCE_M`.
    """
    from collections import defaultdict
    from datetime import timedelta

    n = len(obs)
    if n == 0:
        return []
    lat0 = sum(o.lat for o in obs) / n if lat0 is None else lat0
    k = math.cos(math.radians(lat0))
    xs = [math.radians(o.lon) * _R * k for o in obs]
    ys = [math.radians(o.lat) * _R for o in obs]
    order = sorted(range(n), key=lambda i: obs[i].when)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    gap = timedelta(hours=max_gap_hours)
    tol2 = tolerance_m * tolerance_m
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i in order:
        cx, cy = int(xs[i] // tolerance_m), int(ys[i] // tolerance_m)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                bucket = cells.get((cx + dx, cy + dy))
                if not bucket:
                    continue
                # Drop members older than the gap: visits are in time order.
                keep = [j for j in bucket if obs[i].when - obs[j].when <= gap]
                if len(keep) != len(bucket):
                    cells[(cx + dx, cy + dy)] = keep
                for j in keep:
                    if (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 <= tol2:
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            parent[rj] = ri
        cells[(cx, cy)].append(i)

    groups: dict[int, list[Obs]] = defaultdict(list)
    for i in order:
        groups[find(i)].append(obs[i])
    ordered = sorted(groups.values(), key=lambda g: g[0].when)
    return [ObservedEvent(f"{id_prefix}_{m:06d}", g) for m, g in enumerate(ordered)]

"""Multi-sensor detection fusion into fire *events*.

Why events and not pixels
-------------------------
A GOES pixel and a VIIRS pixel over the same fire will not coincide. Parallax
(the geostationary view is oblique, and the emitting fire has height above the
ellipsoid), differing point spread functions, and remapping all displace them.
Measured across the western US, Amazonas and Patagonia, **~12% of fire events
show spatial misalignment between VIIRS and GOES detections, and naive
nearest-pixel matching yields 26-36% apparent false alarm rate -- falling to
7-15% with a 3x3 pixel buffer.** That difference is geometry, not model
quality.

So VHAGAR clusters detections from all sensors into spatiotemporal events
with an explicit, parallax-aware tolerance, and classifies *events*.

The clustering is a DBSCAN-like single-link pass in (x, y, t) with a
land-cover-dependent spatial buffer (wider in forest, where fires are large
and detections sparse) following the established VIIRS fire-tracking practice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

__all__ = [
    "Detection",
    "geo_leo_tolerance_m",
    "FireEvent",
    "SENSOR_TOLERANCE_M",
    "cluster_detections",
    "event_features",
    "parallax_offset_m",
]

#: Base spatial matching tolerance by sensor, in metres. Roughly 3x the
#: nominal pixel size, i.e. the "3x3 buffer" that collapses the naive-matching
#: false alarm rate.
SENSOR_TOLERANCE_M: dict[str, float] = {
    "viirs": 1_125.0,   # 3 x 375 m
    "modis": 3_000.0,   # 3 x 1 km
    "goes": 6_000.0,    # 3 x 2 km
    "fci": 3_000.0,     # 3 x 1 km
    "slstr": 3_000.0,   # 3 x 1 km
    "landsat": 90.0,
    "sentinel2": 60.0,
}

#: Sensors on a geostationary platform, where footprint growth and terrain
#: parallax both scale steeply with view zenith angle.
GEO_SENSORS = frozenset({"goes", "abi", "fci", "seviri", "ahi"})

#: An FRP growth rate needs enough points and enough elapsed time to mean
#: anything. Below these, report NaN rather than a large fabricated number.
MIN_POINTS_FOR_GROWTH = 3
MIN_HOURS_FOR_GROWTH = 0.5

#: Land-cover dependent association buffer (metres). Forest fires are large
#: and detections sparse, so a wider buffer avoids fragmenting one incident
#: into many. Wider buffers also fuse genuinely separate fires -- this is a
#: known and accepted failure mode; record it in the event provenance.
LANDCOVER_BUFFER_M: dict[str, float] = {
    "forest": 5_000.0,
    "shrub": 2_000.0,
    "grass": 1_500.0,
    "crop": 1_000.0,
    "other": 1_000.0,
}


@dataclass(frozen=True, slots=True)
class Detection:
    """A single sensor-native fire detection, already projected to the region CRS."""

    sensor: str
    x: float
    y: float
    when: datetime
    frp_mw: float | None = None
    bt_mir_k: float | None = None
    bt_tir_k: float | None = None
    confidence: float | None = None
    view_zenith_deg: float | None = None
    landcover: str = "other"
    #: Ground elevation in metres, from a DEM if attached. Drives the terrain
    #: parallax term of the geostationary matching tolerance. ``None`` falls back
    #: to the placeholder in :func:`geo_leo_tolerance_m`.
    elevation_m: float | None = None
    #: True if the detection falls inside the FIRMS static thermal anomaly mask
    #: (persistent industrial/volcanic heat source).
    static_anomaly: bool = False

    @property
    def tolerance_m(self) -> float:
        """Matching radius for this detection.

        Geostationary detections that carry a view zenith angle get a
        geometry-derived tolerance rather than a flat constant, because their
        footprint grows more than threefold between nadir and 48 degrees. See
        :func:`geo_leo_tolerance_m` for the measurement that motivated it.
        """
        if self.view_zenith_deg is not None and self.sensor.lower() in GEO_SENSORS:
            # A computed value supersedes the flat guess rather than being
            # max()'d with it. The flat 6 km happens to be about right over
            # California, but it is four times too loose at nadir and too
            # tight above 60 degrees. Being loose is not free: it merges
            # neighbouring fires into one event.
            base = float(
                geo_leo_tolerance_m(
                    self.view_zenith_deg,
                    elevation_m=1000.0 if self.elevation_m is None else self.elevation_m,
                )
            )
        else:
            base = SENSOR_TOLERANCE_M.get(self.sensor.lower(), 2_000.0)
        return max(base, LANDCOVER_BUFFER_M.get(self.landcover, 1_000.0))


@dataclass(slots=True)
class FireEvent:
    """A spatiotemporal cluster of detections believed to be one fire."""

    event_id: str
    detections: list[Detection] = field(default_factory=list)

    @property
    def sensors(self) -> set[str]:
        return {d.sensor.lower() for d in self.detections}

    @property
    def start(self) -> datetime:
        return min(d.when for d in self.detections)

    @property
    def end(self) -> datetime:
        return max(d.when for d in self.detections)

    @property
    def duration_h(self) -> float:
        return (self.end - self.start).total_seconds() / 3600.0

    def centroid(self) -> tuple[float, float]:
        xs = np.array([d.x for d in self.detections])
        ys = np.array([d.y for d in self.detections])
        return float(xs.mean()), float(ys.mean())

    def frp_series(self) -> tuple[np.ndarray, np.ndarray]:
        pts = sorted(
            ((d.when, d.frp_mw) for d in self.detections if d.frp_mw is not None),
            key=lambda p: p[0],
        )
        if not pts:
            return np.array([]), np.array([])
        t0 = pts[0][0]
        t = np.array([(w - t0).total_seconds() / 3600.0 for w, _ in pts])
        f = np.array([v for _, v in pts], dtype=np.float64)
        return t, f


def geo_leo_tolerance_m(
    view_zenith_deg,
    elevation_m: float = 1000.0,
    nominal_pixel_m: float = 2000.0,
    floor_m: float = 1000.0,
    orbit_altitude_km: float = 35_786.0,
):
    """Spatial tolerance for matching a geostationary detection to a polar one.

    A flat constant is wrong, and real data says so. Measured over northern
    California on 2026-08-12, GOES-18 against VIIRS on NOAA-20 and NOAA-21:
    median separation 1.62 km, p75 2.80 km, only 58% inside a 2 km tolerance.

    Two effects produce that separation and both scale with view zenith angle.

    **Footprint quantisation.** At 48 degrees view zenith a nominally 2 km ABI
    pixel covers 13.3 km2, an effective side of 3.6 km. A VIIRS detection
    anywhere inside that footprint is legitimately the same fire, yet it can
    sit 1.8 km from the pixel centre and 2.6 km at the corner. A 2 km tolerance
    is therefore smaller than a single GOES pixel at this geometry.

    **Terrain parallax.** ABI navigation solves for the ellipsoid, so ground at
    elevation h appears displaced by ``h * tan(vza)``. Over the Sierra at
    1500 m that is 1.67 km, which on its own accounts for the observed median.

    The two are not separable without a DEM, so this returns their sum, which
    is the conservative choice: too tight fragments one fire into several
    events, too loose merges neighbouring fires. Fragmentation is the worse
    failure because it also destroys the FRP time series.

    Pass a real per-pixel elevation from a DEM when you have one, as a scalar or
    an array broadcastable against ``view_zenith_deg``. See
    :class:`vhagar.harmonize.dem.DEM` and
    :func:`vhagar.harmonize.dem.attach_elevation`. The default of 1000 m is a
    placeholder for mountainous western North America and will be wrong for the
    Central Valley or the Gulf coast, where it overstates the tolerance.

    >>> float(round(geo_leo_tolerance_m(48.1, elevation_m=1500) / 1000, 2))
    4.25
    >>> float(round(geo_leo_tolerance_m(0.0, elevation_m=0) / 1000, 2))
    1.41
    """
    from vhagar.physics.geometry import pixel_area_growth

    # The geostationary altitude matters: pixel_area_growth defaults to a polar
    # orbit at 833 km, which understates ABI footprint growth by nearly a
    # factor of two at 48 degrees (1.90x versus the correct 3.32x).
    vza = np.clip(np.asarray(view_zenith_deg, dtype=np.float64), 0.0, 80.0)
    side = nominal_pixel_m * np.sqrt(pixel_area_growth(vza, orbit_altitude_km=orbit_altitude_km))
    quantisation = side * np.sqrt(2.0) / 2.0
    # Elevation may be an array (per-pixel from a DEM) or a scalar. A NaN
    # elevation, off the edge of the DEM, falls back to the placeholder rather
    # than poisoning the tolerance with NaN.
    elev = np.asarray(elevation_m, dtype=np.float64)
    elev = np.where(np.isnan(elev), 1000.0, elev)
    parallax = elev * np.tan(np.radians(vza))
    return np.maximum(quantisation + parallax, floor_m)


def parallax_offset_m(
    fire_height_m: float,
    satellite_lon_deg: float,
    pixel_lon_deg: float,
    pixel_lat_deg: float,
) -> float:
    """Approximate geostationary parallax displacement for an elevated source.

    A fire (or its plume top) at height ``h`` above the ellipsoid appears
    displaced away from the sub-satellite point by roughly ``h * tan(theta)``,
    where ``theta`` is the satellite viewing zenith angle. This is a
    first-order estimate adequate for setting a matching tolerance -- it is not
    a replacement for a proper navigation correction.

    Returns metres.
    """
    r_e = 6_378_137.0
    r_geo = 42_164_000.0
    dlon = math.radians(pixel_lon_deg - satellite_lon_deg)
    lat = math.radians(pixel_lat_deg)
    cos_psi = math.cos(lat) * math.cos(dlon)
    cos_psi = max(min(cos_psi, 1.0), -1.0)
    # Viewing zenith at the surface point.
    denom = math.sqrt(1.0 + (r_e / r_geo) ** 2 - 2.0 * (r_e / r_geo) * cos_psi)
    sin_theta = min(1.0, (r_geo / r_e) * math.sqrt(max(1.0 - cos_psi**2, 0.0)) / (denom * r_geo / r_e))
    theta = math.asin(max(min(sin_theta, 1.0), 0.0))
    return float(fire_height_m * math.tan(min(theta, math.radians(85.0))))


def cluster_detections(
    detections: list[Detection],
    max_gap_hours: float = 12.0,
    extra_tolerance_m: float = 0.0,
    id_prefix: str = "evt",
) -> list[FireEvent]:
    """Single-link spatiotemporal clustering into fire events.

    Two detections join the same event when their planar separation is within
    the **larger** of their two tolerances (plus ``extra_tolerance_m``, which
    is where you inject a parallax allowance) and their time separation is
    within ``max_gap_hours``.

    Complexity is O(n^2) and this is deliberate for the reference
    implementation: it is exact, easy to audit, and fine up to ~10^4
    detections per tile-window. Swap in a ball-tree for production scale.

    >>> from datetime import datetime
    >>> d = [Detection("viirs", 0, 0, datetime(2026, 7, 1, 12)),
    ...      Detection("goes", 800, 0, datetime(2026, 7, 1, 13)),
    ...      Detection("viirs", 500_000, 0, datetime(2026, 7, 1, 12))]
    >>> len(cluster_detections(d))
    2
    """
    n = len(detections)
    if n == 0:
        return []

    order = sorted(range(n), key=lambda i: detections[i].when)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    max_gap = timedelta(hours=max_gap_hours)
    for oi, i in enumerate(order):
        di = detections[i]
        for j in order[oi + 1 :]:
            dj = detections[j]
            if dj.when - di.when > max_gap:
                break  # sorted by time, no later detection can match
            tol = max(di.tolerance_m, dj.tolerance_m) + extra_tolerance_m
            if (di.x - dj.x) ** 2 + (di.y - dj.y) ** 2 <= tol * tol:
                union(i, j)

    groups: dict[int, list[Detection]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(detections[i])

    events = []
    for k, (_, members) in enumerate(sorted(groups.items(), key=lambda kv: min(d.when for d in kv[1]))):
        events.append(FireEvent(event_id=f"{id_prefix}_{k:06d}", detections=members))
    return events


def event_features(event: FireEvent) -> dict[str, float]:
    """Event-level features for the T1 classifier.

    **Raw latitude and longitude are deliberately absent.** In a published
    FIRMS wildfire/non-wildfire classification, raw coordinates supplied ~89%
    of model gain while *harming* out-of-region transfer, and F1 collapsed from
    0.985 (random split) to 0.627 (5-degree spatial block). Anything that lets
    the model memorise "fires happen here" is excluded by construction.

    Density/persistence features are computed causally from past detections
    only -- never from the full event -- when used in an NRT setting; the
    offline version below is for training on completed events.
    """
    t, frp = event.frp_series()
    n = len(event.detections)

    growth = float("nan")
    peak_frp = float("nan")
    total_frp = float("nan")
    if frp.size:
        peak_frp = float(np.nanmax(frp))
        total_frp = float(np.nansum(frp))
        # Least-squares slope, not an endpoint difference, and only when the
        # event is long enough for a rate to mean anything. Real data produced
        # -606 MW/h from two points 12 minutes apart, which is noise wearing a
        # unit label.
        span = float(t[-1] - t[0])
        if frp.size >= MIN_POINTS_FOR_GROWTH and span >= MIN_HOURS_FOR_GROWTH:
            finite = np.isfinite(frp)
            if finite.sum() >= MIN_POINTS_FOR_GROWTH:
                growth = float(np.polyfit(t[finite], frp[finite], 1)[0])

    d_mir_tir = [
        d.bt_mir_k - d.bt_tir_k
        for d in event.detections
        if d.bt_mir_k is not None and d.bt_tir_k is not None
    ]
    vza = [d.view_zenith_deg for d in event.detections if d.view_zenith_deg is not None]
    conf = [d.confidence for d in event.detections if d.confidence is not None]

    xs = np.array([d.x for d in event.detections])
    ys = np.array([d.y for d in event.detections])
    spread_m = float(np.sqrt(xs.var() + ys.var())) if n > 1 else 0.0

    lc_counts: dict[str, int] = {}
    for d in event.detections:
        lc_counts[d.landcover] = lc_counts.get(d.landcover, 0) + 1
    dominant_lc = max(lc_counts, key=lambda k: lc_counts[k]) if lc_counts else "other"

    return {
        "n_detections": float(n),
        "n_sensors": float(len(event.sensors)),
        "multi_sensor_agreement": float(len(event.sensors) > 1),
        "duration_h": event.duration_h,
        "detections_per_hour": n / max(event.duration_h, 1e-3),
        "peak_frp_mw": peak_frp,
        "total_frp_mw": total_frp,
        "frp_growth_mw_per_h": growth,
        "mean_dt_mir_tir_k": float(np.mean(d_mir_tir)) if d_mir_tir else float("nan"),
        "max_dt_mir_tir_k": float(np.max(d_mir_tir)) if d_mir_tir else float("nan"),
        "mean_view_zenith_deg": float(np.mean(vza)) if vza else float("nan"),
        "mean_confidence": float(np.mean(conf)) if conf else float("nan"),
        "spatial_spread_m": spread_m,
        "static_anomaly_fraction": float(
            np.mean([d.static_anomaly for d in event.detections])
        ),
        "landcover_forest": float(dominant_lc == "forest"),
        "landcover_crop": float(dominant_lc == "crop"),
        "landcover_grass": float(dominant_lc == "grass"),
        "landcover_shrub": float(dominant_lc == "shrub"),
    }

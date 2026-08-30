"""Read GOES-R ABI L2 Fire Detection (FDC) granules into VHAGAR detections.

This is the first module that touches real bytes.

Design decisions worth knowing
------------------------------
**Crop before you decode.** A full-disk FDC granule is tens of megabytes; a
CONUS one is a few. If you want one fire, converting the whole grid to lat/lon
first is the difference between a fast loop and an unusable one. So the reader
takes a bounding box, converts it *once* to scan-angle limits via the inverse
projection, and slices the arrays before doing anything else.

**Keep both mask series, and understand that they are mutually exclusive.**
Codes 10-15 and 30-35 are not two parallel streams. A pixel carries one or the
other: 30-35 once the Part-II temporal filter has confirmed it over successive
scans, 10-15 while it has not.

Confirmed on real GOES-18 data over northern California, 2026-08-12: of 43
detections in two hours, 37 were 30-series and zero were code 10. The only
10-series values present were 15 (low probability) and 12 (cloud contaminated),
which are exactly the categories that fail temporal confirmation.

The operational consequence is the important part. A genuinely new fire appears
first as 13/14/15 and is promoted to 33/34/35 only after the filter agrees. So
**the 10-15 series is where early detection lives, and it is also where the
false alarms live.** Discarding it to raise precision discards your latency
advantage at the same time. That tension is the whole T1 problem.

**Attach geometry at read time.** View zenith angle and true pixel area come out
with every detection, because FRP is proportional to pixel area and to
1/transmittance, and both depend on that angle. Deriving them later, from a
lat/lon you already computed, is how the nominal 2 km ends up silently used at
the disk edge.

Anonymous S3 access, the NOAA buckets are public and not requester-pays, so
this costs nothing and needs no credentials.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta

try:                                    # datetime.UTC is 3.11+; fall back on 3.10
    from datetime import UTC
except ImportError:                     # pragma: no cover
    from datetime import timezone as _timezone
    UTC = _timezone.utc

import numpy as np

from vhagar.harmonize.fusion import Detection
from vhagar.io.abi_grid import ABIProjection
from vhagar.io.goes import (
    FDC_MASK_MEANINGS,
    GOES_BUCKETS,
    _as_utc,
    fdc_key_prefix,
    parse_goes_key,
)

log = logging.getLogger(__name__)

__all__ = ["FDCGranule", "list_fdc_granules", "open_fdc", "read_fdc_detections"]

#: ABI epoch for the ``t`` coordinate: 2000-01-01 12:00:00 UTC.
_ABI_EPOCH = datetime(2000, 1, 1, 12, 0, 0, tzinfo=UTC)
_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
#: GOES-R first light. No ABI granule can predate this, so a decoded scan start
#: earlier than it is corrupt, not real. GOES-16 returned first imagery in
#: January 2017; using the start of that year is a safe floor for the whole
#: constellation. A real granule whose ``t`` decodes to, say, 2000-01-01 (the
#: ABI epoch) has a bad time field, and trusting it silently poisons the
#: coverage record and any diurnal or persistence statistics built on it.
_GOES_R_FIRST_LIGHT = datetime(2017, 1, 1, tzinfo=UTC)


def _scan_start(ds) -> datetime:
    """Read the granule scan-start time, however xarray decided to decode it.

    Real ABI files carry CF metadata (``units = "seconds since 2000-01-01
    12:00:00"``), so xarray auto-decodes ``t`` into datetime64. Adding that to
    the ABI epoch as if it were raw seconds overflows ``timedelta``, because
    1.7e18 seconds is roughly 5e10 years.

    That was a real bug on real data, and it survived the test suite because
    the synthetic fixture stored ``t`` as a plain float and never exercised the
    CF path. Both paths are handled here and both are now tested.
    """
    raw = np.asarray(np.asarray(ds["t"].values).ravel()[0])
    if np.issubdtype(raw.dtype, np.datetime64):
        micros = int(raw.astype("datetime64[us]").astype("int64"))
        return _UNIX_EPOCH + timedelta(microseconds=micros)
    if np.issubdtype(raw.dtype, np.timedelta64):
        micros = int(raw.astype("timedelta64[us]").astype("int64"))
        return _ABI_EPOCH + timedelta(microseconds=micros)
    return _ABI_EPOCH + timedelta(seconds=float(raw))


def _validated_scan_start(decoded: datetime, key: str, satellite: int) -> datetime:
    """Return a trustworthy scan start, recovering from the key if ``t`` is bad.

    The granule ``t`` field is normally the truth, but it can arrive corrupt: a
    real GOES-18 CONUS granule in the 7-day backfill decoded to 2000-01-01, the
    ABI epoch, which split the coverage record into two intervals 26 years apart
    and left 60 detection rows stamped with the year 2000. The filename carries
    the authoritative scan start independently (the ``s`` token), and
    :func:`~vhagar.io.goes.parse_goes_key` already extracts it, so when the
    decoded time predates GOES-R first light we recover from the key rather than
    trusting the impossible value. The recovery is logged, loudly and by name,
    because a silently corrected time is only marginally better than a silently
    wrong one.
    """
    if decoded >= _GOES_R_FIRST_LIGHT:
        return decoded
    try:
        recovered = parse_goes_key(key, satellite).start
    except ValueError:
        log.warning(
            "granule %s decoded an implausible scan start %s and its key carries no "
            "parseable time; keeping the decoded value",
            key,
            decoded.isoformat(),
        )
        return decoded
    log.warning(
        "granule %s decoded an implausible scan start %s (before GOES-R first light); "
        "recovering %s from the filename",
        key,
        decoded.isoformat(),
        recovered.isoformat(),
    )
    return recovered


#: Mask codes that denote fire. Split so callers can choose their tradeoff.
UNFILTERED_FIRE_CODES = (10, 11, 12, 13, 14, 15)
FILTERED_FIRE_CODES = (30, 31, 32, 33, 34, 35)
#: Codes whose FRP is unreliable: saturated (11/31) and cloud-contaminated
#: (12/32). Detections are kept, FRP is not trusted.
UNRELIABLE_FRP_CODES = (11, 12, 31, 32)

#: Rough confidence for each mask category, for downstream ranking.
MASK_CONFIDENCE = {
    10: 0.95, 30: 0.98,   # good quality
    11: 0.90, 31: 0.93,   # saturated, real fire, unreliable FRP
    12: 0.55, 32: 0.60,   # cloud contaminated
    13: 0.80, 33: 0.85,   # high probability
    14: 0.55, 34: 0.60,   # medium probability
    15: 0.30, 35: 0.35,   # low probability
}


@dataclass(slots=True)
class FDCGranule:
    """A decoded FDC granule, cropped to an area of interest."""

    satellite: int
    scan_start: datetime
    mask: np.ndarray
    power_mw: np.ndarray
    area_m2: np.ndarray
    temp_k: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    view_zenith_deg: np.ndarray
    true_pixel_area_m2: np.ndarray
    projection: ABIProjection

    @property
    def shape(self) -> tuple[int, ...]:
        return self.mask.shape

    def fire_mask(self, filtered: bool = False) -> np.ndarray:
        codes = FILTERED_FIRE_CODES if filtered else UNFILTERED_FIRE_CODES
        return np.isin(self.mask, codes)

    def n_fire_pixels(self, filtered: bool = False) -> int:
        return int(self.fire_mask(filtered).sum())


def list_fdc_granules(
    satellite: int,
    start: datetime,
    end: datetime,
    domain: str = "C",
    anon: bool = True,
) -> list[str]:
    """List FDC S3 keys in a time window. Requires ``s3fs``; no credentials."""
    try:
        import s3fs
    except ImportError as exc:  # pragma: no cover
        raise ImportError("list_fdc_granules requires s3fs: pip install s3fs") from exc

    fs = s3fs.S3FileSystem(anon=anon)
    bucket = GOES_BUCKETS[satellite]
    out: list[str] = []
    start, end = _as_utc(start), _as_utc(end)          # naive is treated as UTC
    cursor = start.replace(minute=0, second=0, microsecond=0)
    while cursor <= end:
        prefix = fdc_key_prefix(satellite, cursor, domain)
        try:
            listing = fs.ls(f"{bucket}/{prefix}", detail=False)
        except FileNotFoundError:
            listing = []
        for full in listing:
            key = full.split("/", 1)[1]
            try:
                g = parse_goes_key(key, satellite)
            except ValueError:
                continue
            if start <= g.start <= end:
                out.append(key)
        cursor += timedelta(hours=1)
    return sorted(out)


def open_fdc(
    key: str,
    satellite: int,
    bbox: tuple[float, float, float, float] | None = None,
    anon: bool = True,
):
    """Open one FDC granule and decode it, optionally cropped to a bbox.

    ``bbox`` is ``(west, south, east, north)`` in degrees. Cropping happens in
    **scan-angle space before decoding**, so an area-of-interest read touches a
    small slice rather than the whole grid.
    """
    import s3fs
    import xarray as xr

    fs = s3fs.S3FileSystem(anon=anon)
    uri = f"{GOES_BUCKETS[satellite]}/{key}"
    with fs.open(uri, "rb") as fh:
        ds = xr.open_dataset(fh, engine="h5netcdf").load()
    granule = decode_fdc(ds, satellite=satellite, bbox=bbox)
    # The key carries the authoritative scan start, so use it to catch a granule
    # whose ``t`` field is corrupt before the bad time reaches the archive.
    granule.scan_start = _validated_scan_start(granule.scan_start, key, satellite)
    return granule


@dataclass(frozen=True, slots=True)
class _Navigation:
    """Everything about a granule that depends only on the fixed grid.

    The ABI fixed grid does not move. For a given satellite and domain the
    scan-angle arrays ``x`` and ``y`` are byte-for-byte identical in every
    granule, so latitude, longitude, view zenith and pixel area are identical
    too. Deriving them is the dominant cost of a decode (``to_latlon`` alone is
    tens of seconds over a CONUS-scale grid), and doing it once per granule is
    pure waste. This holds the derived arrays so they can be computed once and
    reused. The arrays are marked read-only, because they are shared across
    every granule that lands on the same grid and an in-place write would
    corrupt the lot.
    """

    lat: np.ndarray
    lon: np.ndarray
    view_zenith_deg: np.ndarray
    pixel_area_m2: np.ndarray
    nominal_m: float


#: Cache of fixed-grid navigation, keyed by projection parameters plus the exact
#: ``x`` and ``y`` coordinate bytes. Two entries is enough to hold, say, a CONUS
#: grid and a cropped bbox at the same time; each CONUS entry is roughly 120 MB
#: (four float64 arrays over 3.75M points), so the cap is deliberately small.
_NAV_CACHE: dict[tuple, _Navigation] = {}
_NAV_CACHE_LOCK = threading.Lock()
_NAV_CACHE_MAX = 2
#: Hit and miss counters, for tests and for the wall-clock story. Not load
#: bearing, so the fast-path increment is left unlocked.
_NAV_CACHE_STATS = {"hits": 0, "misses": 0}


def _clear_nav_cache() -> None:
    """Drop the navigation cache. For tests and benchmarks, not the hot path."""
    with _NAV_CACHE_LOCK:
        _NAV_CACHE.clear()
        _NAV_CACHE_STATS["hits"] = 0
        _NAV_CACHE_STATS["misses"] = 0


def _fixed_grid_navigation(proj: ABIProjection, x: np.ndarray, y: np.ndarray) -> _Navigation:
    """Return the navigation for one fixed grid, computing it at most once.

    Keyed by the projection parameters and the raw bytes of ``x`` and ``y``, so
    a granule that lands on an already-seen grid pays a dict lookup rather than
    a full meshgrid and inverse projection.

    The lookup is double-checked: an unlocked read handles the common case where
    the entry already exists, and only a miss takes the lock. The compute itself
    runs while the lock is held, on purpose. Without that, sixteen workers
    hitting an empty cache at startup would all miss, all compute the same grid,
    and only then store it. Holding the lock means the first worker computes and
    the other fifteen wait for its result.
    """
    key = (
        proj.lon_origin_deg,
        proj.perspective_point_height,
        proj.semi_major_axis,
        proj.semi_minor_axis,
        x.tobytes(),
        y.tobytes(),
    )
    hit = _NAV_CACHE.get(key)
    if hit is not None:
        _NAV_CACHE_STATS["hits"] += 1
        return hit

    with _NAV_CACHE_LOCK:
        hit = _NAV_CACHE.get(key)
        if hit is not None:
            _NAV_CACHE_STATS["hits"] += 1
            return hit

        xx, yy = np.meshgrid(x, y)
        lat, lon = proj.to_latlon(xx, yy)
        vza = proj.view_zenith_deg(lat, lon)
        nominal = (
            float(abs(x[1] - x[0])) * proj.perspective_point_height if x.size > 1 else 2000.0
        )
        area = proj.pixel_area_m2(lat, lon, nominal_m=nominal)
        for arr in (lat, lon, vza, area):
            arr.flags.writeable = False

        nav = _Navigation(
            lat=lat, lon=lon, view_zenith_deg=vza, pixel_area_m2=area, nominal_m=nominal
        )
        _NAV_CACHE[key] = nav
        _NAV_CACHE_STATS["misses"] += 1
        # Evict in insertion order once over the cap. A cap of 0 turns the cache
        # off, which is how the before/after benchmark reproduces the old
        # recompute-every-granule behaviour on the same code path.
        while len(_NAV_CACHE) > max(_NAV_CACHE_MAX, 0):
            oldest = next(iter(_NAV_CACHE))
            del _NAV_CACHE[oldest]
        return nav


def decode_fdc(ds, satellite: int, bbox=None) -> FDCGranule:
    """Decode an already-open ABI FDC Dataset. Separated so it is testable offline."""
    proj = ABIProjection.from_dataset(ds)
    x = ds["x"].values.astype(np.float64)
    y = ds["y"].values.astype(np.float64)

    if bbox is not None:
        west, south, east, north = bbox
        # Corners plus edge midpoints: the fixed grid is curvilinear, so the
        # extreme scan angles of a lat/lon box are not always at its corners.
        lats = np.array([south, south, north, north, (south + north) / 2, (south + north) / 2,
                         south, north])
        lons = np.array([west, east, west, east, west, east,
                         (west + east) / 2, (west + east) / 2])
        gx, gy = proj.to_scan_angles(lats, lons)
        if np.all(np.isnan(gx)):
            raise ValueError(
                f"bbox {bbox} is not visible from GOES-{satellite} "
                f"(sub-satellite longitude {proj.lon_origin_deg})"
            )
        pad = 3.0 * float(np.abs(np.diff(x)).mean())  # a few pixels of slack
        xi = np.where((x >= np.nanmin(gx) - pad) & (x <= np.nanmax(gx) + pad))[0]
        yi = np.where((y >= np.nanmin(gy) - pad) & (y <= np.nanmax(gy) + pad))[0]
        if xi.size == 0 or yi.size == 0:
            raise ValueError(f"bbox {bbox} falls outside this granule's grid")
        sl = {"x": slice(int(xi[0]), int(xi[-1]) + 1), "y": slice(int(yi[0]), int(yi[-1]) + 1)}
        ds = ds.isel(**sl)
        x, y = x[sl["x"]], y[sl["y"]]

    # The fixed grid does not move, so lat, lon, view zenith and pixel area are
    # computed once per (projection, x, y) and shared across granules.
    nav = _fixed_grid_navigation(proj, x, y)
    lat, lon = nav.lat, nav.lon
    vza = nav.view_zenith_deg
    area = nav.pixel_area_m2

    def band(name: str) -> np.ndarray:
        if name not in ds:
            return np.full(lat.shape, np.nan)
        return np.asarray(ds[name].values, dtype=np.float64)

    mask = np.asarray(ds["Mask"].values) if "Mask" in ds else np.zeros(lat.shape)
    mask = np.nan_to_num(mask, nan=0).astype(np.int16)

    return FDCGranule(
        satellite=satellite,
        scan_start=_scan_start(ds),
        mask=mask,
        power_mw=band("Power"),
        area_m2=band("Area"),
        temp_k=band("Temp"),
        lat=lat,
        lon=lon,
        view_zenith_deg=vza,
        true_pixel_area_m2=area,
        projection=proj,
    )


def read_fdc_detections(
    granule: FDCGranule,
    crs: str | None = None,
    include_filtered: bool = True,
    min_confidence: float = 0.0,
) -> list[Detection]:
    """Convert a decoded granule into :class:`~vhagar.harmonize.fusion.Detection` objects.

    Coordinates are projected into the VHAGAR analysis CRS when ``crs`` is
    given (requires ``pyproj``); otherwise ``x``/``y`` carry lon/lat degrees,
    which is fine for inspection but **not** for the metric-distance clustering
    in :func:`vhagar.harmonize.fusion.cluster_detections`.

    FRP from saturated or cloud-contaminated pixels is set to NaN rather than
    passed through, the detection is real, the number is not.
    """
    codes = list(UNFILTERED_FIRE_CODES)
    if include_filtered:
        codes += list(FILTERED_FIRE_CODES)
    sel = np.isin(granule.mask, codes)
    if not sel.any():
        return []

    idx = np.argwhere(sel)
    lats = granule.lat[sel]
    lons = granule.lon[sel]
    good = np.isfinite(lats) & np.isfinite(lons)
    idx, lats, lons = idx[good], lats[good], lons[good]

    if crs:
        from pyproj import Transformer

        tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        xs, ys = tf.transform(lons, lats)
    else:
        xs, ys = lons, lats

    codes_sel = granule.mask[sel][good]
    frp = granule.power_mw[sel][good]
    temp = granule.temp_k[sel][good]
    vza = granule.view_zenith_deg[sel][good]
    unreliable = np.isin(codes_sel, UNRELIABLE_FRP_CODES)
    frp = np.where(unreliable, np.nan, frp)

    out: list[Detection] = []
    for k in range(len(idx)):
        conf = MASK_CONFIDENCE.get(int(codes_sel[k]), 0.5)
        if conf < min_confidence:
            continue
        out.append(
            Detection(
                sensor="goes",
                x=float(xs[k]),
                y=float(ys[k]),
                when=granule.scan_start,
                frp_mw=None if not np.isfinite(frp[k]) else float(frp[k]),
                bt_mir_k=None if not np.isfinite(temp[k]) else float(temp[k]),
                confidence=conf,
                view_zenith_deg=float(vza[k]) if np.isfinite(vza[k]) else None,
            )
        )
    return out


def mask_summary(granule: FDCGranule) -> dict[str, int]:
    """Count pixels by mask meaning, the first thing to look at on real data."""
    vals, counts = np.unique(granule.mask, return_counts=True)
    return {
        FDC_MASK_MEANINGS.get(int(v), f"code_{int(v)}"): int(c)
        for v, c in zip(vals, counts, strict=True)
        if int(v) in FDC_MASK_MEANINGS
    }

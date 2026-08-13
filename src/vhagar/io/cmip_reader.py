"""Read GOES-R ABI L2 CMIP thermal channels into brightness-temperature grids.

This is the radiance-tier counterpart to :mod:`vhagar.io.goes_reader`, and it is
deliberately built to look like it. If you know the FDC reader you know this one.

The two facts that shape it
---------------------------
**CMIP hands us brightness temperature, not radiance.** For the emissive bands
VHAGAR uses (C07, C11, C13, C14, C15) the ``CMI`` variable is already calibrated,
gap-filled brightness temperature in kelvin. So there is no radiance calibration
step here. Where a physics stage wants radiance, for instance Wooster FRP on the
MIR channel, invert with :func:`vhagar.physics.planck.planck_radiance` at the
band centre. Note the planck caveat: a monochromatic centre-wavelength conversion
differs from a band-integrated radiance by a few kelvin, which is fine for
features and climatology and should be revisited only for quantitative FRP.

**CMIP rides the same ABI fixed grid as FDC.** The 2 km thermal channels use the
identical scan-angle grid, so :func:`vhagar.io.goes_reader._fixed_grid_navigation`
applies unchanged and lat/lon/view-zenith/pixel-area are computed once and shared
across every channel and every timestep. That shared cache is the whole reason
the radiance tier is affordable in wall clock, so this module reuses it rather
than recomputing anything.

Design rules carried over from the FDC reader, unchanged
--------------------------------------------------------
* Crop before decode. Convert the bbox to scan-angle limits once and slice.
* Attach geometry at read time, because FRP is proportional to pixel area and to
  1/transmittance and both depend on the view zenith angle.
* Saturation is censoring, not noise. ABI Ch7 saturates near 400 K; a saturated
  MIR pixel is a real hot source with an unusable value, so its BT is set to NaN
  and a separate mask records that it was saturated.
* Fill and bad-DQF pixels become NaN, never a number, so nothing downstream
  averages nodata into "cold ground".
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from vhagar.io.abi_grid import ABIProjection
from vhagar.io.goes import GOES_BUCKETS, parse_goes_key
from vhagar.io.goes_reader import (
    _fixed_grid_navigation,
    _scan_start,
    _validated_scan_start,
)

log = logging.getLogger(__name__)

__all__ = [
    "CMIP_CHANNELS",
    "CMIPChannel",
    "CMIPStack",
    "cmip_key_prefix",
    "decode_cmip",
    "group_cmip_keys_by_timestamp",
    "list_cmip_granules",
    "open_cmip",
    "open_cmip_stack",
    "stack_channels",
]

#: How far apart the per-channel files of one nominal timestep may scan. The
#: five channels of a CONUS frame are written close together but not to the same
#: second, so grouping needs a small tolerance. Well under the 5-minute cadence,
#: so it can never merge two distinct timesteps.
CMIP_STACK_TOLERANCE = timedelta(minutes=2)

#: Emissive ABI channels VHAGAR reads, with their central wavelengths in microns.
#: C07 is the mid-infrared fire channel; C13/C14/C15 are the split window and
#: C11 adds cloud-phase context. All five are 2 km, which is what makes the
#: radiance tier affordable (see :mod:`vhagar.archive.plan`).
CMIP_CHANNELS = {
    "C07": 3.9,
    "C11": 8.4,
    "C13": 10.3,
    "C14": 11.2,
    "C15": 12.3,
}

#: Approximate saturation brightness temperature per channel, kelvin. Only the
#: MIR channel saturates within the range of real fire scenes; the window
#: channels saturate far above any surface temperature, so they are not censored.
#: A BT at or above the threshold is a censored hot source, not a measurement.
CMIP_SATURATION_K = {
    "C07": 400.0,
}

#: CMIP ``DQF`` flag meanings. 0 is good and 1 is conditionally usable, both of
#: which we keep; 2 and above (out of range, no value) are set to NaN.
_DQF_KEEP_MAX = 1


@dataclass(slots=True)
class CMIPChannel:
    """One decoded CMIP channel on the ABI fixed grid, cropped to an area."""

    satellite: int
    band: str
    wavelength_um: float
    scan_start: datetime
    bt_k: np.ndarray            # brightness temperature, NaN where fill/bad/saturated
    dqf: np.ndarray             # raw data quality flag
    saturated: np.ndarray       # bool: real hot source, value censored
    lat: np.ndarray
    lon: np.ndarray
    view_zenith_deg: np.ndarray
    true_pixel_area_m2: np.ndarray
    projection: ABIProjection

    @property
    def shape(self) -> tuple[int, ...]:
        return self.bt_k.shape

    @property
    def n_saturated(self) -> int:
        return int(np.count_nonzero(self.saturated))

    def valid_mask(self) -> np.ndarray:
        """Pixels carrying a usable brightness temperature."""
        return np.isfinite(self.bt_k)


def cmip_key_prefix(satellite: int, when: datetime, domain: str = "C") -> str:
    """S3 key prefix for CMIP files in a given hour.

    Layout mirrors FDC: ``ABI-L2-CMIP{C,F}/YYYY/DDD/HH/``.
    """
    if satellite not in GOES_BUCKETS:
        raise ValueError(f"unknown GOES satellite {satellite}")
    if domain not in {"C", "F", "M1", "M2"}:
        raise ValueError(f"unknown ABI domain {domain!r}")
    w = when.astimezone(when.tzinfo or None)
    return f"ABI-L2-CMIP{domain}/{w.year:04d}/{w.timetuple().tm_yday:03d}/{w.hour:02d}/"


def list_cmip_granules(
    satellite: int,
    start: datetime,
    end: datetime,
    channel: str,
    domain: str = "C",
    anon: bool = True,
) -> list[str]:
    """List CMIP S3 keys for one channel in a time window. Requires ``s3fs``.

    CMIP ships one file per channel, so a caller building a multi-band stack
    lists each channel and groups the results by timestamp.
    """
    if channel not in CMIP_CHANNELS:
        raise ValueError(f"unknown channel {channel!r}, expected one of {sorted(CMIP_CHANNELS)}")
    try:
        import s3fs
    except ImportError as exc:  # pragma: no cover
        raise ImportError("list_cmip_granules requires s3fs: pip install s3fs") from exc

    fs = s3fs.S3FileSystem(anon=anon)
    bucket = GOES_BUCKETS[satellite]
    out: list[str] = []
    cursor = start.astimezone(start.tzinfo or None).replace(minute=0, second=0, microsecond=0)
    while cursor <= end:
        prefix = cmip_key_prefix(satellite, cursor, domain)
        try:
            listing = fs.ls(f"{bucket}/{prefix}", detail=False)
        except FileNotFoundError:
            listing = []
        for full in listing:
            key = full.split("/", 1)[1]
            # The channel token in the filename looks like "...-M6C07_G18_...".
            # Matching "C07_G" is mode-agnostic and unambiguous for two-digit
            # channel ids.
            if f"{channel}_G" not in key.rsplit("/", 1)[-1]:
                continue
            try:
                g = parse_goes_key(key, satellite)
            except ValueError:
                continue
            if start <= g.start <= end:
                out.append(key)
        cursor += timedelta(hours=1)
    return sorted(out)


def open_cmip(
    key: str,
    satellite: int,
    channel: str,
    bbox: tuple[float, float, float, float] | None = None,
    anon: bool = True,
) -> CMIPChannel:
    """Open one CMIP channel granule and decode it, optionally cropped to a bbox.

    ``bbox`` is ``(west, south, east, north)`` in degrees, cropped in scan-angle
    space before decoding, exactly like :func:`vhagar.io.goes_reader.open_fdc`.
    """
    import s3fs
    import xarray as xr

    fs = s3fs.S3FileSystem(anon=anon)
    uri = f"{GOES_BUCKETS[satellite]}/{key}"
    with fs.open(uri, "rb") as fh:
        ds = xr.open_dataset(fh, engine="h5netcdf").load()
    granule = decode_cmip(ds, satellite=satellite, channel=channel, bbox=bbox)
    # The key carries the authoritative scan start, so use it to catch a granule
    # whose ``t`` field is corrupt before the bad time reaches the archive.
    granule.scan_start = _validated_scan_start(granule.scan_start, key, satellite)
    return granule


def decode_cmip(ds, satellite: int, channel: str, bbox=None) -> CMIPChannel:
    """Decode an already-open ABI CMIP Dataset. Separated so it is testable offline."""
    if channel not in CMIP_CHANNELS:
        raise ValueError(f"unknown channel {channel!r}, expected one of {sorted(CMIP_CHANNELS)}")

    proj = ABIProjection.from_dataset(ds)
    x = ds["x"].values.astype(np.float64)
    y = ds["y"].values.astype(np.float64)

    if bbox is not None:
        west, south, east, north = bbox
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
        pad = 3.0 * float(np.abs(np.diff(x)).mean())
        xi = np.where((x >= np.nanmin(gx) - pad) & (x <= np.nanmax(gx) + pad))[0]
        yi = np.where((y >= np.nanmin(gy) - pad) & (y <= np.nanmax(gy) + pad))[0]
        if xi.size == 0 or yi.size == 0:
            raise ValueError(f"bbox {bbox} falls outside this granule's grid")
        sl = {"x": slice(int(xi[0]), int(xi[-1]) + 1), "y": slice(int(yi[0]), int(yi[-1]) + 1)}
        ds = ds.isel(**sl)
        x, y = x[sl["x"]], y[sl["y"]]

    # The fixed grid does not move, so geometry is computed once per (projection,
    # x, y) and shared with FDC and every other channel on the same grid.
    nav = _fixed_grid_navigation(proj, x, y)

    if "CMI" not in ds:
        raise ValueError("CMIP granule has no 'CMI' variable")
    bt = np.asarray(ds["CMI"].values, dtype=np.float64)
    dqf = (
        np.asarray(ds["DQF"].values)
        if "DQF" in ds
        else np.zeros(bt.shape, dtype=np.int16)
    )
    dqf = np.nan_to_num(dqf, nan=_DQF_KEEP_MAX + 1).astype(np.int16)

    # Fill and out-of-range or no-value DQF become NaN, never a number.
    bad = ~np.isfinite(bt) | (dqf > _DQF_KEEP_MAX)
    bt = np.where(bad, np.nan, bt)

    # Saturation is censoring, not noise: a saturated MIR pixel is a real hot
    # source with an unusable value. Record it, then drop the value.
    sat_threshold = CMIP_SATURATION_K.get(channel)
    if sat_threshold is not None:
        saturated = np.isfinite(bt) & (bt >= sat_threshold)
        bt = np.where(saturated, np.nan, bt)
    else:
        saturated = np.zeros(bt.shape, dtype=bool)

    return CMIPChannel(
        satellite=satellite,
        band=channel,
        wavelength_um=CMIP_CHANNELS[channel],
        scan_start=_scan_start(ds),
        bt_k=bt,
        dqf=dqf,
        saturated=saturated,
        lat=nav.lat,
        lon=nav.lon,
        view_zenith_deg=nav.view_zenith_deg,
        true_pixel_area_m2=nav.pixel_area_m2,
        projection=proj,
    )


# ---------------------------------------------------------------------------
# Multi-channel stacks
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CMIPStack:
    """Several CMIP channels at one timestep, sharing one grid and its geometry.

    Because every channel rides the same ABI fixed grid, the geometry is held
    once, not per channel, and the brightness temperatures are a dict keyed by
    band. The channels co-register exactly, so band arithmetic like the
    C07 minus C14 contextual fire signal is a plain array subtraction.
    """

    satellite: int
    scan_start: datetime
    bands: tuple[str, ...]
    bt_k: dict[str, np.ndarray]
    dqf: dict[str, np.ndarray]
    saturated: dict[str, np.ndarray]
    lat: np.ndarray
    lon: np.ndarray
    view_zenith_deg: np.ndarray
    true_pixel_area_m2: np.ndarray
    projection: ABIProjection

    @property
    def shape(self) -> tuple[int, ...]:
        return self.lat.shape

    def bt(self, band: str) -> np.ndarray:
        """Brightness temperature for one band."""
        return self.bt_k[band]

    def bt_difference(self, band_a: str, band_b: str) -> np.ndarray:
        """``BT(band_a) - BT(band_b)``, NaN-propagating.

        The MIR-minus-window difference (C07 minus C14) is the classic
        contextual fire signal; the split-window differences carry the
        atmospheric and cloud context.
        """
        return self.bt_k[band_a] - self.bt_k[band_b]


def _same_grid(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two navigation arrays describe the same grid.

    The navigation cache returns the identical array object for a repeated grid,
    so an identity check is the fast and normal path. The value fallback exists
    only for the pathological case where the cache evicted the grid between two
    decodes; it checks shape and the three corners rather than the whole array.
    """
    if a is b:
        return True
    if a.shape != b.shape:
        return False
    return bool(a[0, 0] == b[0, 0] and a[-1, -1] == b[-1, -1] and a[0, -1] == b[0, -1])


def stack_channels(channels: Sequence[CMIPChannel]) -> CMIPStack:
    """Assemble decoded channels that share a grid into one :class:`CMIPStack`.

    Pure and offline-testable. Raises if the channels are not on the same grid,
    which is the failure mode that would otherwise silently miscombine bands.
    """
    if not channels:
        raise ValueError("no channels to stack")
    ref = channels[0]
    bt_k: dict[str, np.ndarray] = {}
    dqf: dict[str, np.ndarray] = {}
    saturated: dict[str, np.ndarray] = {}
    for c in channels:
        if c.bt_k.shape != ref.bt_k.shape:
            raise ValueError(
                f"channel {c.band} shape {c.bt_k.shape} does not match {ref.band} "
                f"shape {ref.bt_k.shape}"
            )
        if not _same_grid(c.lat, ref.lat):
            raise ValueError(f"channel {c.band} is on a different grid than {ref.band}")
        bt_k[c.band] = c.bt_k
        dqf[c.band] = c.dqf
        saturated[c.band] = c.saturated
    return CMIPStack(
        satellite=ref.satellite,
        # The nominal timestep: the earliest channel scan start. The spread
        # across channels is under CMIP_STACK_TOLERANCE by construction.
        scan_start=min(c.scan_start for c in channels),
        bands=tuple(c.band for c in channels),
        bt_k=bt_k,
        dqf=dqf,
        saturated=saturated,
        lat=ref.lat,
        lon=ref.lon,
        view_zenith_deg=ref.view_zenith_deg,
        true_pixel_area_m2=ref.true_pixel_area_m2,
        projection=ref.projection,
    )


def group_cmip_keys_by_timestamp(
    keys_by_channel: dict[str, Sequence[str]],
    satellite: int,
    tolerance: timedelta = CMIP_STACK_TOLERANCE,
) -> list[dict[str, str]]:
    """Group per-channel keys into complete same-timestep sets.

    ``keys_by_channel`` maps each band to its list of keys, as returned by
    :func:`list_cmip_granules`. The per-channel files of one timestep do not
    share an exact scan start, so this pairs them by nearest time within
    ``tolerance``. Only timesteps that have every requested channel are returned,
    an incomplete timestep is dropped, because a stack with a missing band would
    quietly bias any band difference computed from it. Pure and offline-testable.
    """
    bands = list(keys_by_channel)
    if not bands:
        return []
    parsed: dict[str, list[tuple[datetime, str]]] = {
        b: sorted((parse_goes_key(k, satellite).start, k) for k in keys)
        for b, keys in keys_by_channel.items()
    }
    ref_band = bands[0]
    tol = tolerance.total_seconds()
    groups: list[dict[str, str]] = []
    for start, key in parsed[ref_band]:
        group = {ref_band: key}
        complete = True
        for b in bands[1:]:
            best_key, best_dt = None, None
            for s, k in parsed[b]:
                dt = abs((s - start).total_seconds())
                if dt <= tol and (best_dt is None or dt < best_dt):
                    best_key, best_dt = k, dt
            if best_key is None:
                complete = False
                break
            group[b] = best_key
        if complete:
            groups.append(group)
    return groups


def open_cmip_stack(
    keys_by_channel: dict[str, str],
    satellite: int,
    bbox: tuple[float, float, float, float] | None = None,
    anon: bool = True,
) -> CMIPStack:
    """Open one timestep's channel files and stack them on the shared grid.

    ``keys_by_channel`` maps each band to a single key, one entry from
    :func:`group_cmip_keys_by_timestamp`. Each channel is opened, then stacked;
    the grid check in :func:`stack_channels` guards against combining files that
    do not co-register.
    """
    channels = [
        open_cmip(key, satellite, channel=band, bbox=bbox, anon=anon)
        for band, key in keys_by_channel.items()
    ]
    return stack_channels(channels)

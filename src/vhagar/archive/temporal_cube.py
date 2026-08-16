"""Pull a time-ordered 3.9 um (C07) brightness-temperature cube for the temporal detector.

The T1 differentiator (``eval.t1_temporal``) forecasts the expected per-pixel BT and
flags residual excursions. Demonstrated on synthetic series and grounded on the on-disk
diurnal climatology, the one heavy piece left is the real input: a ``[T, H, W]`` cube of
the mid-infrared fire channel at native 5-minute cadence over a region, on which the
forecaster is trained (clear-sky) and the residual lead over GOES FDC first-detection is
measured. This module is that pull.

Why a dense cube here, and only a climatology in Tier B
-------------------------------------------------------
:mod:`vhagar.archive.climatology_backfill` folds every CMIP stack into a per-pixel,
per-hour mean/variance and keeps **no frames**, because a diurnal baseline needs only the
statistics. The temporal detector is the opposite: it needs the frames *in order*, so it
can see a fire develop between one 5-minute scan and the next. So this module keeps the
actual time series, for one channel and a bounded region, which is why the region must be
small (a fire-prone box, not CONUS) or the cube will not fit in memory.

What makes the cube trustworthy
-------------------------------
* **One grid, checked.** The ABI fixed grid is stationary, so cropping the same bbox in
  scan-angle space yields the identical pixel window every timestep. That is asserted, not
  assumed: a frame whose shape or corner navigation does not match the reference is
  dropped, never silently misaligned into the stack (the same rule as ``stack_channels``).
* **NaN is nodata, never a number.** Cloud, fill, bad-DQF and saturated pixels arrive as
  NaN from :mod:`vhagar.io.cmip_reader` and stay NaN, so the forecaster never averages a
  cloud into a "cold ground" baseline.
* **Every array states what it measured.** The saved cube carries its own UTC timestamps,
  latitude, longitude and view-zenith, so a later comparison (residual vs FDC) walks the
  same pixels and the same clock; two measurements are never compared across a regrid.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

import vhagar.io.cmip_reader as cmip
from vhagar.io.goes import parse_goes_key

log = logging.getLogger(__name__)

__all__ = [
    "TemporalCube",
    "TemporalCubeConfig",
    "assemble_cube",
    "fdc_first_detection_grid",
    "load_bt_cube",
    "pull_bt_cube",
    "solar_zenith_cube",
]


@dataclass(frozen=True, slots=True)
class TemporalCubeConfig:
    """Everything that determines the pulled cube."""

    out_path: Path
    start: datetime
    end: datetime
    #: ``(west, south, east, north)`` in degrees. Required and should be small: the cube
    #: is dense, so a CONUS box would not fit in memory.
    bbox: tuple[float, float, float, float]
    satellite: int = 18
    domain: str = "C"
    channel: str = "C07"
    #: Native CONUS cadence is 5 minutes; keep it for the temporal detector.
    cadence_min: int = 5
    workers: int = 8


@dataclass(slots=True)
class TemporalCube:
    """A time-ordered single-channel BT cube on one ABI grid, with its geometry."""

    bt: np.ndarray              # [T, H, W] brightness temperature, NaN where nodata
    times: list[datetime]       # length T, UTC scan starts, ascending
    lat: np.ndarray             # [H, W]
    lon: np.ndarray             # [H, W]
    view_zenith_deg: np.ndarray  # [H, W]
    satellite: int
    channel: str

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.bt.shape

    def hours_of_day(self) -> np.ndarray:
        """UTC hour-of-day (decimal) per frame, the diurnal forecaster's regressor."""
        return np.array([t.hour + t.minute / 60.0 + t.second / 3600.0 for t in self.times])


def _cadence_subsample(
    parsed: list[tuple[datetime, str]], cadence_min: int
) -> list[tuple[datetime, str]]:
    """Keep the earliest granule in each cadence bucket, so 5-min data stays 5-min but a
    coarser request thins cleanly."""
    step = timedelta(minutes=cadence_min)
    kept: list[tuple[datetime, str]] = []
    seen: set[int] = set()
    for start, key in sorted(parsed):
        bucket = int((start - datetime(start.year, 1, 1, tzinfo=UTC)) / step)
        if bucket in seen:
            continue
        seen.add(bucket)
        kept.append((start, key))
    return kept


def _same_grid(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two navigation arrays describe the same fixed-grid window."""
    if a.shape != b.shape:
        return False
    return bool(a[0, 0] == b[0, 0] and a[-1, -1] == b[-1, -1] and a[0, -1] == b[0, -1])


def assemble_cube(frames: list[cmip.CMIPChannel], channel: str) -> TemporalCube:
    """Stack decoded single-channel frames into a ``[T, H, W]`` cube on one grid.

    Pure and offline-testable. Frames are ordered by scan start; the first defines the
    reference grid, and any frame whose shape or corner navigation disagrees is dropped
    with a warning rather than misaligned into the stack.
    """
    if not frames:
        raise ValueError("no frames to assemble")
    ordered = sorted(frames, key=lambda f: f.scan_start)
    ref = ordered[0]
    kept: list[cmip.CMIPChannel] = []
    for f in ordered:
        if f.bt_k.shape != ref.bt_k.shape or not _same_grid(f.lat, ref.lat):
            log.warning(
                "dropping frame %s: grid %s does not match reference %s",
                f.scan_start, f.bt_k.shape, ref.bt_k.shape,
            )
            continue
        kept.append(f)
    bt = np.stack([f.bt_k.astype(np.float32) for f in kept], axis=0)
    return TemporalCube(
        bt=bt,
        times=[f.scan_start for f in kept],
        lat=ref.lat.astype(np.float32),
        lon=ref.lon.astype(np.float32),
        view_zenith_deg=ref.view_zenith_deg.astype(np.float32),
        satellite=ref.satellite,
        channel=channel,
    )


def pull_bt_cube(config: TemporalCubeConfig, progress=None) -> TemporalCube:
    """List, decode and stack one channel over the window into a cube; save to ``.npz``.

    Reads GOES ABI L2 CMIP for ``config.channel`` from the public S3 archive (needs
    ``s3fs`` + ``xarray``), crops each granule to ``config.bbox`` before decoding, thins to
    ``config.cadence_min``, and assembles a time-ordered cube. The frames open
    concurrently (reads are the slow part); assembly is single-threaded and pure.
    """
    keys = cmip.list_cmip_granules(
        config.satellite, config.start, config.end, config.channel, domain=config.domain
    )
    parsed = [(parse_goes_key(k, config.satellite).start, k) for k in keys]
    schedule = _cadence_subsample(parsed, config.cadence_min)
    if not schedule:
        raise ValueError(
            f"no {config.channel} granules for {config.start}..{config.end} "
            f"(satellite {config.satellite}, domain {config.domain})"
        )

    def _open(key: str):
        try:
            return cmip.open_cmip(key, config.satellite, config.channel, bbox=config.bbox), None
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"

    frames: list[cmip.CMIPChannel] = []
    n_fail = 0
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures = {pool.submit(_open, k): k for _, k in schedule}
        for i, fut in enumerate(as_completed(futures)):
            frame, err = fut.result()
            if frame is None:
                n_fail += 1
                log.warning("frame open failed: %s", err)
            else:
                frames.append(frame)
            if progress:
                progress(i + 1, len(schedule))
    if not frames:
        raise RuntimeError(f"all {len(schedule)} frame opens failed; last error logged above")

    cube = assemble_cube(frames, config.channel)
    save_bt_cube(cube, config.out_path)
    log.info(
        "pulled %d frames (%d failed) into %s, cube %s",
        len(cube.times), n_fail, config.out_path, cube.shape,
    )
    return cube


def save_bt_cube(cube: TemporalCube, path: Path) -> None:
    """Save a cube to ``.npz`` with its own timestamps and geometry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        bt=cube.bt,
        times_posix=np.array([t.timestamp() for t in cube.times], dtype=np.float64),
        lat=cube.lat,
        lon=cube.lon,
        view_zenith_deg=cube.view_zenith_deg,
        satellite=np.int32(cube.satellite),
        channel=np.array(cube.channel),
    )


def load_bt_cube(path: Path) -> TemporalCube:
    """Load a cube saved by :func:`save_bt_cube`."""
    z = np.load(path, allow_pickle=False)
    times = [datetime.fromtimestamp(float(p), tz=UTC) for p in z["times_posix"]]
    return TemporalCube(
        bt=z["bt"],
        times=times,
        lat=z["lat"],
        lon=z["lon"],
        view_zenith_deg=z["view_zenith_deg"],
        satellite=int(z["satellite"]),
        channel=str(z["channel"]),
    )


def solar_zenith_cube(lat: np.ndarray, lon: np.ndarray, times: list[datetime]) -> np.ndarray:
    """Per-frame solar zenith angle ``[T, H, W]`` as a covariate for the forecaster.

    The 3.9 um channel carries a solar-reflectance component in daylight, so the expected
    BT depends on where the sun is, not just the clock. Uses the NOAA low-precision solar
    position (``physics.geometry.solar_position``), accurate to ~0.2 degrees, ample for a
    covariate.
    """
    from vhagar.physics.geometry import solar_position

    out = np.empty((len(times), *lat.shape), dtype=np.float32)
    for i, t in enumerate(times):
        doy = t.timetuple().tm_yday
        utc_hour = t.hour + t.minute / 60.0 + t.second / 3600.0
        zen, _ = solar_position(lat, lon, doy, utc_hour)
        out[i] = zen.astype(np.float32)
    return out


def fdc_first_detection_grid(
    root, bbox: tuple[float, float, float, float], times: list[datetime],
    lat: np.ndarray, lon: np.ndarray, max_pixel_km: float = 3.0,
) -> np.ndarray:
    """First FDC-detection **frame index** per cube pixel, or -1 where never detected.

    Reads the GOES FDC parquet, restricts to the cube's bbox and time span, snaps each
    detection to the nearest cube pixel (rejecting matches farther than ``max_pixel_km``,
    so a detection outside the grid is not force-fit), and records the earliest frame index
    whose scan time is at or after the detection. This is the reference the residual
    detector is timed against: for a fire pixel, does the residual cross its threshold
    before this frame? Needs pandas.
    """
    import glob as _glob

    import pandas as pd

    files = sorted(_glob.glob(f"{root}/**/*.parquet", recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {root}")
    df = pd.concat((pd.read_parquet(f, columns=["lon", "lat", "t"]) for f in files),
                   ignore_index=True)
    west, south, east, north = bbox
    t0, t1 = times[0], times[-1]
    df["t"] = pd.to_datetime(df["t"], utc=True)
    m = (
        (df["lon"] >= west) & (df["lon"] <= east)
        & (df["lat"] >= south) & (df["lat"] <= north)
        & (df["t"] >= t0) & (df["t"] <= t1)
    )
    df = df.loc[m]
    H, W = lat.shape
    first = np.full((H, W), -1, dtype=np.int64)
    if df.empty:
        return first

    # Nearest cube pixel by lat/lon; the grid is smooth over a small bbox, so a per-axis
    # nearest index is adequate and avoids a KD-tree dependency here.
    lat_col = lat[:, W // 2]      # latitude varies mainly down rows
    lon_row = lon[H // 2, :]      # longitude varies mainly across columns
    frame_posix = np.array([t.timestamp() for t in times])
    deg_km = 111.0
    for lon_d, lat_d, t in zip(df["lon"].to_numpy(), df["lat"].to_numpy(),
                               df["t"].to_numpy(), strict=False):
        r = int(np.argmin(np.abs(lat_col - lat_d)))
        c = int(np.argmin(np.abs(lon_row - lon_d)))
        dr_km = abs(lat[r, c] - lat_d) * deg_km
        dc_km = abs(lon[r, c] - lon_d) * deg_km * np.cos(np.radians(lat_d))
        if (dr_km * dr_km + dc_km * dc_km) ** 0.5 > max_pixel_km:
            continue
        det_posix = pd.Timestamp(t).timestamp()
        idx = int(np.searchsorted(frame_posix, det_posix, side="left"))
        if idx >= len(times):
            continue
        if first[r, c] == -1 or idx < first[r, c]:
            first[r, c] = idx
    return first

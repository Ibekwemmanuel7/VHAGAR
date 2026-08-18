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
    "FireSpec",
    "TemporalCube",
    "TemporalCubeConfig",
    "assemble_cube",
    "cohort_pull",
    "fdc_first_detection_grid",
    "load_bt_cube",
    "pull_bt_cube",
    "select_fire_cohort",
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


@dataclass(frozen=True, slots=True)
class FireSpec:
    """One fire selected for the lead-time cohort, with a ready-to-pull cube window.

    ``stratum`` is the scientific grouping: ``night_coldstart`` fires (ignition in local
    night, slow early ramp) are where a diurnal-residual detector *should* beat an absolute
    contextual threshold; ``day`` fires are the control where it should not. ``clear_frac``
    is set so the baseline span ends before ignition (no contamination).
    """

    name: str
    lon: float
    lat: float
    ignition_utc: datetime
    local_solar_hour: float
    stratum: str
    ramp_slope_mw_per_h: float
    n_detections: int
    bbox: tuple[float, float, float, float]
    pull_start: datetime
    pull_end: datetime
    clear_frac: float

    def to_json(self) -> dict:
        return {
            "name": self.name, "lon": self.lon, "lat": self.lat,
            "ignition_utc": self.ignition_utc.isoformat(),
            "local_solar_hour": self.local_solar_hour, "stratum": self.stratum,
            "ramp_slope_mw_per_h": self.ramp_slope_mw_per_h,
            "n_detections": self.n_detections, "bbox": list(self.bbox),
            "pull_start": self.pull_start.isoformat(), "pull_end": self.pull_end.isoformat(),
            "clear_frac": self.clear_frac,
        }

    def pull_command(self, cube_dir: str = "cohort") -> str:
        w, s, e, n = self.bbox
        return (
            f"vhagar t1-pull-cube {cube_dir}/{self.name}.npz "
            f"--start {self.pull_start:%Y-%m-%dT%H:%M} --end {self.pull_end:%Y-%m-%dT%H:%M} "
            f"--bbox {w},{s},{e},{n} --satellite 18"
        )


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope, NaN if fewer than two distinct x (guards degenerate fits)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.unique(x).size < 2:
        return float("nan")
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    return float((xc * (y - y.mean())).sum() / denom) if denom > 0 else float("nan")


def select_fire_cohort(
    root, box_deg: float = 0.35, cell_deg: float = 0.25, baseline_hours: float = 30.0,
    post_hours: float = 8.0, min_detections: int = 50, per_stratum: int = 3,
    data_start: datetime | None = None, night_lst: tuple[float, float] = (20.0, 6.0),
    slow_ramp_mw_per_h: float = 200.0,
) -> list[FireSpec]:
    """Select a stratified cohort of fires from the FDC parquet for the lead-time eval.

    Clusters detections on a ``cell_deg`` grid into fires, and for each computes ignition
    time (first detection), local solar hour at ignition (``utc + lon/15``), and the early
    FRP ramp slope over the first two hours. Fires are stratified into ``night_coldstart``
    (ignition in local night and a slow ramp, the residual detector's theoretical edge) and
    ``day`` (control), keeping up to ``per_stratum`` of each with the most pre-fire baseline.
    Each spec carries a cube window (``baseline_hours`` before ignition to ``post_hours``
    after) and a ``clear_frac`` that ends the baseline before ignition. Needs pandas.
    """
    import glob as _glob

    import pandas as pd

    files = sorted(_glob.glob(f"{root}/**/*.parquet", recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {root}")
    df = pd.concat((pd.read_parquet(f, columns=["lon", "lat", "t", "frp_mw"]) for f in files),
                   ignore_index=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    d0 = pd.Timestamp(data_start) if data_start is not None else df["t"].min()
    df["gx"] = (df["lon"] / cell_deg).round().astype(int)
    df["gy"] = (df["lat"] / cell_deg).round().astype(int)

    night_lo, night_hi = night_lst
    specs: list[FireSpec] = []
    for _cell, g in df.groupby(["gx", "gy"]):
        if len(g) < min_detections:
            continue
        g = g.sort_values("t")
        ign = g["t"].iloc[0]
        pre_h = (ign - d0).total_seconds() / 3600.0
        if pre_h < baseline_hours:
            continue                        # not enough pre-fire baseline for this window
        lon, lat = float(g["lon"].mean()), float(g["lat"].mean())
        lst = (ign.hour + ign.minute / 60.0 + lon / 15.0) % 24
        is_night = lst >= night_lo or lst < night_hi
        early = g[g["t"] <= ign + pd.Timedelta(hours=2)]
        slope = _ols_slope((early["t"] - ign).dt.total_seconds().to_numpy() / 3600.0,
                           early["frp_mw"].fillna(0.0).to_numpy())
        if is_night and (slope == slope) and abs(slope) <= slow_ramp_mw_per_h:
            stratum = "night_coldstart"
        elif not is_night:
            stratum = "day"
        else:
            continue                        # night but fast/ambiguous ramp: skip
        half = box_deg / 2.0
        west, south = round(lon - half, 2), round(lat - half, 2)
        east, north = round(lon + half, 2), round(lat + half, 2)
        # Anchor the window on the earliest FDC detection anywhere in the *box* (not just the
        # cluster cell); the box is wider than the cell, so a neighbouring earlier fire would
        # otherwise land inside the baseline span and contaminate it. Baselining before this
        # time guarantees a fire-free training window by construction.
        in_box = df[(df["lon"] >= west) & (df["lon"] <= east)
                    & (df["lat"] >= south) & (df["lat"] <= north)]
        box_ign = in_box["t"].min()
        ign_eff = min(ign, box_ign)
        if (ign_eff - d0).total_seconds() / 3600.0 < baseline_hours:
            continue                        # not enough clean pre-fire baseline for the box
        pull_start = ign_eff - timedelta(hours=baseline_hours)
        pull_end = ign_eff + timedelta(hours=post_hours)
        span = (pull_end - pull_start).total_seconds()
        # end the baseline a clear margin before the earliest in-box fire
        clear_frac = max(0.1, min(0.95, (baseline_hours - 3.0) * 3600.0 / span))
        specs.append(FireSpec(
            name=f"{stratum}_{abs(lat):.0f}n_{abs(lon):.0f}w_{ign:%m%d%H%M}",
            lon=round(lon, 2), lat=round(lat, 2), ignition_utc=ign.to_pydatetime(),
            local_solar_hour=round(float(lst), 1), stratum=stratum,
            ramp_slope_mw_per_h=round(float(slope), 1) if slope == slope else float("nan"),
            n_detections=int(len(g)),
            bbox=(round(lon - half, 2), round(lat - half, 2),
                  round(lon + half, 2), round(lat + half, 2)),
            pull_start=pull_start.to_pydatetime(), pull_end=pull_end.to_pydatetime(),
            clear_frac=round(clear_frac, 3),
        ))

    chosen: list[FireSpec] = []
    for stratum in ("night_coldstart", "day"):
        pool = [s for s in specs if s.stratum == stratum]
        pool.sort(key=lambda s: (abs(s.ramp_slope_mw_per_h), -s.n_detections))
        chosen.extend(pool[:per_stratum])
    return chosen


def cohort_pull(
    spec_path, cube_dir=None, workers: int = 8, only_missing: bool = True, progress=None,
) -> dict[str, list[str]]:
    """Pull every cube named in a cohort spec, resumably. Returns ``{pulled, skipped, failed}``.

    Reads the JSON written by ``select_fire_cohort`` / ``t1-cohort-select`` and pulls each
    fire's 3.9 um cube into ``cube_dir`` (default: the spec's own directory). A cube already
    on disk is skipped when ``only_missing`` (so an interrupted cohort resumes without
    re-pulling), and a failed pull is recorded rather than aborting the rest. Needs s3fs +
    xarray + network for the pulls themselves; the spec parsing and skip logic are pure.
    """
    import json as _json

    spec_path = Path(spec_path)
    cube_dir = Path(cube_dir) if cube_dir is not None else spec_path.parent
    cube_dir.mkdir(parents=True, exist_ok=True)
    specs = _json.loads(spec_path.read_text(encoding="utf-8"))

    out: dict[str, list[str]] = {"pulled": [], "skipped": [], "failed": []}
    for i, s in enumerate(specs):
        target = cube_dir / f"{s['name']}.npz"
        if only_missing and target.exists():
            out["skipped"].append(s["name"])
            if progress:
                progress(i + 1, len(specs), s["name"], "skip")
            continue
        cfg = TemporalCubeConfig(
            out_path=target,
            start=datetime.fromisoformat(s["pull_start"]),
            end=datetime.fromisoformat(s["pull_end"]),
            bbox=tuple(s["bbox"]), satellite=int(s.get("satellite", 18)), workers=workers,
        )
        try:
            pull_bt_cube(cfg)
            out["pulled"].append(s["name"])
            if progress:
                progress(i + 1, len(specs), s["name"], "ok")
        except Exception as exc:  # noqa: BLE001
            out["failed"].append(f"{s['name']}: {type(exc).__name__}: {exc}")
            if progress:
                progress(i + 1, len(specs), s["name"], "fail")
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

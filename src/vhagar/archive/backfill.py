"""Tier A backfill: build the FDC detection history, resumably.

What this is for
----------------
An FDC granule is 0.32 MB on the wire, so the detection tier costs roughly
0.1 TB for 500 tiles over three years at five-minute cadence. It needs no
decoder that does not already exist, which is why it gets built first. The wall
clock is a separate budget and was badly underestimated at first: the 0.8 s
per-granule placeholder implied about six hours, but the real 7-day run came in
near 14.7 single-worker-equivalent seconds per granule, dominated by fixed-grid
navigation. Caching that navigation (see
:func:`vhagar.io.goes_reader._fixed_grid_navigation`) is what brings the wall
clock back down; re-measure it on the target machine before quoting hours.

The four rules this module exists to enforce
--------------------------------------------

**1. Granules in the outer loop, tiles in the inner loop.** Each granule is
fetched exactly once and every detection in it is assigned to a tile in memory.
The number of S3 reads therefore depends on cadence and duration alone. Looping
per tile and re-reading the same granule turns six hours into weeks.

**2. Concurrency is the lever, but measure it on the real operation.** A bare
S3 read of an FDC granule takes about 0.12 s. The full path the backfill
actually walks, fetch plus HDF5 parse plus navigation plus tabulation, takes
about 0.75 s. So roughly six sevenths of the per-granule cost is not the
network, and a worker count chosen by timing bare reads is tuned for the wrong
bottleneck. :func:`probe_workers` defaults to ``mode="full"`` for that reason.
Do not guess the number, and do not measure it on a proxy.

**3. Absence must be recorded, not inferred.** This is the price of storing a
sparse product sparsely. If a tile has no rows for 14:35 on some Tuesday, that
means either "observed, nothing burning" or "never read that granule", and
those are opposite facts. Every attempted granule gets a manifest line whether
it succeeded or failed, and :func:`coverage_intervals` turns those lines back
into the observed periods a training loader needs in order to mine honest
negatives. A backfill without a coverage record produces a dataset whose
negatives are quietly wrong, and nothing downstream will tell you.

**4. Interruption is the normal case, not the exception.** A run measured in
hours will be interrupted. The manifest is append-only JSONL flushed after
every granule, so a kill loses at most the current day's unwritten rows, and a
restart re-reads only what is genuinely missing.

What it deliberately does not do
--------------------------------
No radiance. No regridding onto the 375 m analysis grid. Detections are stored
at their native ABI pixel centres with the tile assignment attached, because
resampling a point detection onto a finer grid invents precision that the
instrument never had. Regridding, if it is ever wanted, belongs downstream
where the choice is explicit.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "BackfillConfig",
    "BackfillResult",
    "GranuleRecord",
    "backfill",
    "coverage_gaps",
    "coverage_intervals",
    "detection_table",
    "failed_records",
    "load_manifest",
    "probe_workers",
    "recommend_workers",
]

#: Columns written to the detection Parquet dataset, in order. Anything not on
#: this list is not stored, and anything stored is on this list.
DETECTION_COLUMNS = (
    "t",              # scan start, UTC
    "tile_id",        # VHAGAR analysis tile, e.g. conus/x0031_y0072
    "x", "y",         # region CRS metres, what clustering uses
    "lon", "lat",     # degrees, for inspection and joins to other sensors
    "mask_code",      # raw FDC code. Never collapse this to a boolean.
    "confidence",     # mapped from mask_code, see MASK_CONFIDENCE
    "frp_mw",         # NaN where the mask says the retrieval is unreliable
    "temp_k",
    "area_m2",        # FDC's own fire area estimate
    "view_zenith_deg",
    "true_pixel_area_m2",
    "granule_key",    # provenance. Cheap, and you will want it.
)

#: Manifest filename inside the output directory. Append-only JSONL.
MANIFEST_NAME = "_manifest.jsonl"
#: Config fingerprint, written once. Guards against mixing incompatible runs.
CONFIG_NAME = "_config.json"
#: Where the Parquet dataset lives, relative to the output directory.
DETECTIONS_DIR = "detections"


@dataclass(frozen=True, slots=True)
class BackfillConfig:
    """Everything that changes what ends up on disk.

    Written to ``_config.json`` on first run and compared on every later run.
    Two backfills with different bboxes must not share an output directory:
    the manifest would claim coverage the rows do not support, which is
    exactly the failure the coverage record exists to prevent.
    """

    out_dir: Path
    start: datetime
    end: datetime
    satellite: int = 18
    domain: str = "C"
    region: str = "conus"
    #: ``(west, south, east, north)`` in degrees. ``None`` reads the whole
    #: domain, which is correct for a real backfill and slow for a trial.
    bbox: tuple[float, float, float, float] | None = None
    workers: int = 12
    #: Keep the 10-15 series. It carries both the early detections and the
    #: false alarms, and Stage 2 exists to separate them. Dropping it here
    #: would discard the latency advantage before any model sees it.
    include_filtered: bool = True
    min_confidence: float = 0.0
    #: Re-attempt granules that previously failed. Most failures are transient
    #: S3 hiccups, so the default is yes.
    retry_failed: bool = True

    def fingerprint(self) -> dict[str, object]:
        """The subset that must match for two runs to share a directory."""
        return {
            "satellite": self.satellite,
            "domain": self.domain,
            "region": self.region,
            "bbox": list(self.bbox) if self.bbox else None,
            "include_filtered": self.include_filtered,
            "min_confidence": self.min_confidence,
        }


@dataclass(frozen=True, slots=True)
class GranuleRecord:
    """One manifest line. Written for successes and failures alike."""

    key: str
    scan_start: str | None
    status: str            # "ok" | "error"
    n_detections: int = 0
    elapsed_s: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def scan_start_dt(self) -> datetime | None:
        if not self.scan_start:
            return None
        return datetime.fromisoformat(self.scan_start)


@dataclass(slots=True)
class BackfillResult:
    """Summary of one run. Counts attempts, not granules in the archive."""

    granules_attempted: int = 0
    granules_ok: int = 0
    granules_failed: int = 0
    granules_skipped: int = 0
    detections: int = 0
    elapsed_s: float = 0.0
    errors: dict[str, int] = field(default_factory=dict)

    @property
    def granules_per_second(self) -> float:
        return self.granules_attempted / self.elapsed_s if self.elapsed_s > 0 else 0.0

    def __str__(self) -> str:
        parts = [
            f"{self.granules_ok} ok",
            f"{self.granules_failed} failed",
            f"{self.granules_skipped} skipped",
            f"{self.detections:,} detections",
            f"{self.elapsed_s / 60:.1f} min",
            f"{self.granules_per_second:.2f} granules/s",
        ]
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Manifest, the coverage record
# ---------------------------------------------------------------------------


def load_manifest(out_dir: Path | str) -> dict[str, GranuleRecord]:
    """Read the manifest into ``{granule_key: record}``, last line wins.

    Tolerates a truncated final line, which is what a kill during a flush
    leaves behind. That granule is simply re-attempted.
    """
    path = Path(out_dir) / MANIFEST_NAME
    if not path.exists():
        return {}
    out: dict[str, GranuleRecord] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                log.warning("ignoring truncated manifest line, granule will be re-attempted")
                continue
            out[d["key"]] = GranuleRecord(**d)
    return out


def _append_manifest(out_dir: Path, records: Iterable[GranuleRecord]) -> None:
    path = out_dir / MANIFEST_NAME
    with path.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r)) + "\n")
        fh.flush()


def _check_config(cfg: BackfillConfig) -> None:
    """Refuse to append to a directory built with different settings."""
    path = cfg.out_dir / CONFIG_NAME
    fp = cfg.fingerprint()
    if not path.exists():
        path.write_text(json.dumps(fp, indent=2), encoding="utf-8")
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing != fp:
        diff = {k: (existing.get(k), v) for k, v in fp.items() if existing.get(k) != v}
        raise ValueError(
            f"{cfg.out_dir} was built with different settings, so its manifest would "
            f"claim coverage these rows do not have. Differences (on disk, requested): "
            f"{diff}. Use a separate directory."
        )


def coverage_intervals(
    records: Iterable[GranuleRecord],
    max_gap: timedelta = timedelta(minutes=20),
) -> list[tuple[datetime, datetime]]:
    """Merge successful scan times into continuous observed intervals.

    This is the half of the archive that makes negatives usable. A training
    loader asking "was this tile observed at 14:35 with nothing burning?" needs
    an interval list; a table of detections cannot answer it.

    ``max_gap`` should exceed the nominal cadence with room for the odd dropped
    granule. At CONUS five-minute cadence, 20 minutes tolerates three misses in
    a row before it declares a hole, which is the point: a hole should mean a
    real outage, not one flaky read.
    """
    times = sorted(
        t for r in records if r.ok and (t := r.scan_start_dt()) is not None
    )
    if not times:
        return []
    intervals: list[tuple[datetime, datetime]] = []
    lo = hi = times[0]
    for t in times[1:]:
        if t - hi <= max_gap:
            hi = t
        else:
            intervals.append((lo, hi))
            lo = hi = t
    intervals.append((lo, hi))
    return intervals


def coverage_gaps(
    records: Iterable[GranuleRecord],
    max_gap: timedelta = timedelta(minutes=20),
) -> list[tuple[datetime, datetime, timedelta]]:
    """List the holes between observed intervals, as ``(start, end, duration)``.

    A gap is the span between the end of one observed interval and the start of
    the next: a period the archive did not observe. This is the tool that
    explains a multi-interval coverage report. A single dropped granule leaves a
    gap under ``max_gap`` and never splits an interval, so if the report shows a
    second interval, running this says exactly where the hole is and how long it
    lasted, rather than leaving you to guess between a real outage and a bad
    timestamp.
    """
    intervals = coverage_intervals(records, max_gap=max_gap)
    gaps: list[tuple[datetime, datetime, timedelta]] = []
    for (_, prev_end), (next_start, _) in zip(intervals, intervals[1:], strict=False):
        gaps.append((prev_end, next_start, next_start - prev_end))
    return gaps


def failed_records(records: Iterable[GranuleRecord]) -> list[GranuleRecord]:
    """The granules that were attempted and did not succeed, newest listing order."""
    return [r for r in records if not r.ok]


# ---------------------------------------------------------------------------
# Granule to rows
# ---------------------------------------------------------------------------


def detection_table(granule, region: str, key: str, config: BackfillConfig) -> dict[str, np.ndarray]:
    """Vectorised granule to columns, including tile assignment.

    Works off the granule arrays rather than building
    :class:`~vhagar.harmonize.fusion.Detection` objects, because a busy granule
    can carry thousands of fire pixels and this runs a few hundred thousand
    times. Returns empty arrays rather than ``None`` when nothing is burning,
    so callers do not need a special case.
    """
    from vhagar.grid import AnalysisGrid
    from vhagar.io.goes_reader import (
        FILTERED_FIRE_CODES,
        MASK_CONFIDENCE,
        UNFILTERED_FIRE_CODES,
        UNRELIABLE_FRP_CODES,
    )

    codes = list(UNFILTERED_FIRE_CODES)
    if config.include_filtered:
        codes += list(FILTERED_FIRE_CODES)
    sel = np.isin(granule.mask, codes)

    lat = granule.lat[sel]
    lon = granule.lon[sel]
    finite = np.isfinite(lat) & np.isfinite(lon)
    lat, lon = lat[finite], lon[finite]

    mask_code = granule.mask[sel][finite].astype(np.int16)
    conf = np.array(
        [MASK_CONFIDENCE.get(int(c), 0.5) for c in mask_code], dtype=np.float32
    )
    keep = conf >= config.min_confidence
    lat, lon, mask_code, conf = lat[keep], lon[keep], mask_code[keep], conf[keep]

    def col(arr, dtype=np.float32):
        return arr[sel][finite][keep].astype(dtype)

    frp = col(granule.power_mw)
    # A saturated or cloud-contaminated pixel is a real detection with a
    # meaningless number attached. Storing the number would be worse than
    # storing nothing, because something downstream would average it.
    frp = np.where(np.isin(mask_code, UNRELIABLE_FRP_CODES), np.nan, frp)

    if len(lat) == 0:
        empty_f = np.empty(0, dtype=np.float64)
        return {
            "t": np.empty(0, dtype="datetime64[us]"),
            "tile_id": np.empty(0, dtype=object),
            "x": empty_f, "y": empty_f, "lon": empty_f, "lat": empty_f,
            "mask_code": np.empty(0, dtype=np.int16),
            "confidence": np.empty(0, dtype=np.float32),
            "frp_mw": np.empty(0, dtype=np.float32),
            "temp_k": np.empty(0, dtype=np.float32),
            "area_m2": np.empty(0, dtype=np.float32),
            "view_zenith_deg": np.empty(0, dtype=np.float32),
            "true_pixel_area_m2": np.empty(0, dtype=np.float32),
            "granule_key": np.empty(0, dtype=object),
        }

    from pyproj import Transformer

    grid = AnalysisGrid(region)
    tf = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    x, y = tf.transform(lon, lat)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Tile index arithmetic inline rather than per-point tile_for_point calls,
    # which would be a Python loop over every detection. Out-of-grid points get
    # an empty tile_id rather than being dropped: a detection outside CONUS is
    # a real observation and losing it silently would bias the archive edge.
    ix = np.floor((x - grid.origin_x) / grid.tile_size_m).astype(np.int64)
    iy = np.floor((y - grid.origin_y) / grid.tile_size_m).astype(np.int64)
    inside = (ix >= 0) & (ix < grid.n_x) & (iy >= 0) & (iy < grid.n_y)
    tile_id = np.array(
        [
            f"{region}/x{a:04d}_y{b:04d}" if ok else ""
            for a, b, ok in zip(ix, iy, inside, strict=True)
        ],
        dtype=object,
    )

    n = len(lat)
    return {
        "t": np.full(n, np.datetime64(granule.scan_start.replace(tzinfo=None), "us")),
        "tile_id": tile_id,
        "x": x, "y": y,
        "lon": np.asarray(lon, dtype=np.float64),
        "lat": np.asarray(lat, dtype=np.float64),
        "mask_code": mask_code,
        "confidence": conf,
        "frp_mw": frp,
        "temp_k": col(granule.temp_k),
        "area_m2": col(granule.area_m2),
        "view_zenith_deg": col(granule.view_zenith_deg),
        "true_pixel_area_m2": col(granule.true_pixel_area_m2),
        "granule_key": np.full(n, key, dtype=object),
    }


def _read_one(key: str, config: BackfillConfig) -> tuple[GranuleRecord, dict | None]:
    """Fetch, decode and tabulate one granule. Never raises."""
    from vhagar.io.goes_reader import open_fdc

    t0 = time.perf_counter()
    try:
        granule = open_fdc(key, config.satellite, bbox=config.bbox)
        table = detection_table(granule, config.region, key, config)
        elapsed = time.perf_counter() - t0
        return (
            GranuleRecord(
                key=key,
                scan_start=granule.scan_start.isoformat(),
                status="ok",
                n_detections=len(table["lat"]),
                elapsed_s=elapsed,
            ),
            table,
        )
    except Exception as exc:  # noqa: BLE001
        return (
            GranuleRecord(
                key=key,
                scan_start=None,
                status="error",
                elapsed_s=time.perf_counter() - t0,
                error=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _write_day(out_dir: Path, day: datetime, tables: Sequence[dict]) -> int:
    """Write one day of detections, partitioned by year and tile.

    One file per (year, tile, day) rather than one per granule. A five-minute
    cadence would otherwise produce 288 tiny files per tile per day, and
    Parquet metadata would outweigh the data.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    nonempty = [t for t in tables if len(t["lat"]) > 0]
    if not nonempty:
        return 0

    merged = {
        col: np.concatenate([t[col] for t in nonempty]) for col in DETECTION_COLUMNS
    }
    table = pa.table({col: pa.array(merged[col]) for col in DETECTION_COLUMNS})

    root = out_dir / DETECTIONS_DIR
    written = 0
    tile_ids = merged["tile_id"]
    for tile in np.unique(tile_ids):
        rows = table.filter(pa.array(tile_ids == tile))
        label = str(tile).replace("/", "_") or "outside_grid"
        target = root / f"year={day.year}" / f"tile={label}"
        target.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            rows,
            target / f"part-{day:%Y%m%d}.parquet",
            compression="zstd",
        )
        written += rows.num_rows
    return written


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _days(start: datetime, end: datetime) -> Iterator[datetime]:
    cursor = start.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    last = end.astimezone(UTC)
    while cursor <= last:
        yield cursor
        cursor += timedelta(days=1)


def backfill(config: BackfillConfig, progress=None) -> BackfillResult:
    """Run the Tier A backfill, one day at a time, resumably.

    Days bound memory and give a natural checkpoint. Within a day, granules run
    concurrently because each granule spends most of its time waiting on S3 or
    inside numpy and pyproj, both of which release the GIL; across days they do
    not, because the writer would then have to hold several days of rows at
    once for no throughput gain.

    ``progress`` is called with ``(day, day_result)`` after each day, so a CLI
    can render a bar without this module importing rich.
    """
    from vhagar.io.goes_reader import list_fdc_granules

    config.out_dir.mkdir(parents=True, exist_ok=True)
    _check_config(config)
    done = load_manifest(config.out_dir)

    result = BackfillResult()
    t_start = time.perf_counter()

    for day in _days(config.start, config.end):
        day_t0 = time.perf_counter()
        day_end = min(day + timedelta(days=1) - timedelta(seconds=1), config.end)
        try:
            keys = list_fdc_granules(
                config.satellite, day, day_end, domain=config.domain
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("listing failed for %s: %s", day.date(), exc)
            continue

        pending = []
        for k in keys:
            rec = done.get(k)
            if rec is None or (not rec.ok and config.retry_failed):
                pending.append(k)
            else:
                result.granules_skipped += 1
        if not pending:
            if progress:
                progress(day, None)
            continue

        tables: list[dict] = []
        records: list[GranuleRecord] = []
        day_result = BackfillResult()
        with ThreadPoolExecutor(max_workers=config.workers) as pool:
            futures = {pool.submit(_read_one, k, config): k for k in pending}
            for fut in as_completed(futures):
                record, table = fut.result()
                records.append(record)
                result.granules_attempted += 1
                day_result.granules_attempted += 1
                if record.ok and table is not None:
                    tables.append(table)
                    result.granules_ok += 1
                    day_result.granules_ok += 1
                    result.detections += record.n_detections
                    day_result.detections += record.n_detections
                else:
                    result.granules_failed += 1
                    day_result.granules_failed += 1
                    kind = (record.error or "unknown").split(":", 1)[0]
                    result.errors[kind] = result.errors.get(kind, 0) + 1

        # Rows first, then the manifest. If the process dies between the two,
        # the granule is re-read and its rows rewritten to the same path, which
        # is idempotent. The other order would record coverage for rows that
        # are not on disk, and that lie is unrecoverable.
        _write_day(config.out_dir, day, tables)
        _append_manifest(config.out_dir, records)
        for r in records:
            done[r.key] = r
        # Set the day's own elapsed time before reporting it. Without this the
        # per-day granules/s line reads 0.0 for every day, because elapsed_s is
        # only ever set on the overall result at the end of the run.
        day_result.elapsed_s = time.perf_counter() - day_t0
        if progress:
            progress(day, day_result)

    result.elapsed_s = time.perf_counter() - t_start
    return result


# ---------------------------------------------------------------------------
# Worker probe
# ---------------------------------------------------------------------------


#: A probe run shorter than this is noise, not a measurement. At half a second
#: of wall clock, connection setup and scheduler jitter are the same order as
#: the thing being measured, and the ranking between worker counts flips run to
#: run. :func:`probe_workers` reports which settings fell below it.
MIN_PROBE_SECONDS = 3.0


def probe_workers(
    candidates: Sequence[int] = (1, 4, 8, 16, 32),
    satellite: int = 18,
    domain: str = "C",
    n_granules: int = 48,
    repeats: int = 3,
    mode: str = "full",
    bbox: tuple[float, float, float, float] | None = None,
) -> list[dict[str, float]]:
    """Measure throughput against worker count. Needs network.

    ``mode="full"`` (the default) times the operation the backfill actually
    performs: fetch, HDF5 parse, navigate, tabulate. ``mode="fetch"`` times a
    bare byte read instead.

    **Use "full" to choose a worker count.** The first version of this function
    only offered the "fetch" path, and the two are not interchangeable: on a
    real granule the bare read is a small fraction of the total, so a fetch
    probe finds where the *network* saturates and then that number gets applied
    to a workload whose bottleneck is somewhere else entirely. The gap between
    the two modes is itself worth looking at, which is why "fetch" is kept.

    ``bbox`` defaults to ``None``, the whole domain, because that is what a real
    CONUS backfill reads and the probe should walk the same path. The fixed-grid
    navigation is cached, so a warmup read is taken before timing to build the
    grid once; every timed read then pays the steady-state cost the backfill
    pays, not the one-off grid build. Before that cache existed this default
    meant every probe read recomputed a full-CONUS navigation, which made the
    probe itself take longer than it was worth. Pass a small ``bbox`` only if
    you intend to run the backfill with that same bbox.

    Each setting is run ``repeats`` times and the median is reported, because a
    single pass over a few granules is dominated by scheduler noise. Settings
    that finish faster than :data:`MIN_PROBE_SECONDS` are flagged in the output
    as ``too_fast``: their ranking cannot be trusted, and the fix is to raise
    ``n_granules`` rather than to believe the number.
    """
    import s3fs

    from vhagar.io.goes import GOES_BUCKETS
    from vhagar.io.goes_reader import list_fdc_granules, open_fdc

    if mode not in {"full", "fetch"}:
        raise ValueError(f"mode must be 'full' or 'fetch', got {mode!r}")

    end = datetime.now(UTC) - timedelta(hours=2)
    keys = list_fdc_granules(satellite, end - timedelta(hours=8), end, domain=domain)
    if not keys:
        raise RuntimeError("no granules found in the sampling window")
    keys = keys[-n_granules:]
    bucket = GOES_BUCKETS[satellite]

    def fetch(key: str) -> int:
        fs = s3fs.S3FileSystem(anon=True)
        with fs.open(f"{bucket}/{key}", "rb") as fh:
            return len(fh.read())

    def full(key: str) -> int:
        granule = open_fdc(key, satellite, bbox=bbox)
        return int(granule.mask.size)

    work = full if mode == "full" else fetch

    # Warm the navigation cache so the timed passes measure steady-state
    # per-granule cost rather than the one-off fixed-grid build. Without this
    # the first pass carries the whole grid computation and skews the median.
    if mode == "full":
        try:
            full(keys[0])
        except Exception as exc:  # noqa: BLE001
            log.warning("probe warmup read failed, timings may include grid build: %s", exc)

    out = []
    for n in candidates:
        timings = []
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=n) as pool:
                list(pool.map(work, keys))
            timings.append(time.perf_counter() - t0)
        timings.sort()
        elapsed = timings[len(timings) // 2]
        out.append(
            {
                "workers": float(n),
                "granules_per_second": len(keys) / elapsed,
                "seconds_per_granule": elapsed / len(keys),
                "seconds": elapsed,
                "spread": (timings[-1] - timings[0]) / elapsed if elapsed > 0 else 0.0,
                "too_fast": float(elapsed < MIN_PROBE_SECONDS),
            }
        )
    return out


def recommend_workers(rows: Sequence[dict[str, float]]) -> dict[str, object]:
    """Pick a worker count from probe rows, and say when the data cannot support one.

    Refuses to name a knee when the measurement is too noisy to distinguish
    settings. The alternative, picking the argmax of six noisy numbers, is how
    you end up confidently recommending a value that was third best on the
    previous run.
    """
    if not rows:
        return {"workers": None, "reason": "no measurements"}

    noisy = [r for r in rows if r.get("too_fast") or r.get("spread", 0.0) > 0.25]
    best = max(rows, key=lambda r: r["granules_per_second"])
    if len(noisy) > len(rows) / 2:
        return {
            "workers": None,
            "best": int(best["workers"]),
            "reason": (
                "runs too short or too variable to rank; re-probe with more granules"
            ),
        }
    knee = next(
        r for r in rows if r["granules_per_second"] >= 0.9 * best["granules_per_second"]
    )
    return {
        "workers": int(knee["workers"]),
        "best": int(best["workers"]),
        "reason": "smallest setting within 10% of the fastest",
    }

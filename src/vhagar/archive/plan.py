"""Archive sizing, work out what you can afford *before* downloading for weeks.

The arithmetic that matters
---------------------------
A VHAGAR tile is 96 km across (256 cells x 375 m). At GOES 2 km that is only
**48 x 48 pixels**, or 4.6 KB per band per timestep at int16. So tile *size* is
never the problem. The cost drivers are, in order:

    tiles x bands x timesteps_per_day x days

and the one with the steepest gradient is **cadence**. Going from 15-minute to
5-minute sampling triples everything, which is how a sensible 25 GB plan turns
into 1.4 TB for no modelling gain.

Hence the three-tier design (see :func:`three_tier_plan`):

* **Detection tier**, wide, long and fast. ABI L2 FDC only, measured at 0.32 MB
  per granule against 4.4 MB for a 2 km radiance band, and sparse enough to
  store as detection rows rather than a raster. 500 tiles, 3 years, 5-minute
  cadence comes to well under a gigabyte of disk and under 0.1 TB on the wire.
  Build this one first: it is nearly free and it already carries the
  persistence, diurnal and event-history features.
* **Climatology tier**, broad and shallow, radiance. Persistence and
  diurnal-baseline features are statistics, ``mu(pixel, hour)`` and
  ``sigma(pixel, hour)``; 15-minute sampling is ample. This is the expensive
  tier because it is wide.
* **High-cadence tier**, narrow and deep, radiance. Only the early-detection
  anomaly model needs true 5-minute radiance, and only over a handful of tiles
  for one fire season. That costs a few GB.

Sparse against dense is not cosmetic. The same FDC coverage stored as an int16
raster is 181 GB; stored as detection rows it is 0.6 GB. What you give up is
the explicit negative field, so the loader has to reconstruct negatives from
the coverage record. That is a bookkeeping cost, not an information loss.

Three separate budgets
----------------------
People size archives on disk and then get surprised. There are three:

1. **Disk**, what you keep. Small, because tiles are small.
2. **Download traffic**, what crosses the wire. Still 10-50x disk, because you
   pull a whole 4.4 MB CONUS granule to keep 4.6 KB of it. Chunked range-reads
   would not help here: at 4.4 MB the file is smaller than one s3fs block, so
   you fetch it all whatever you ask for. That changes for Full Disk.
3. **Wall clock**, dominated by the *number of granule reads*, not bytes, but
   the per-read cost is mostly not the network. A bare S3 read of an FDC
   granule is about 0.12 s; fetching, parsing and navigating it is about
   0.75 s. Size the wall clock on the full operation, and pick a worker count
   by measuring that operation rather than a bare read. See
   :func:`vhagar.archive.backfill.probe_workers`.

**Read granules in the outer loop, tiles in the inner loop.** The number of
granule reads then depends only on cadence and duration, it is independent of
how many tiles you extract. Iterating per-tile instead multiplies your S3
requests by the tile count and turns hours into weeks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC

__all__ = [
    "ABI_BAND_RESOLUTION_M",
    "ABI_PRODUCTS",
    "ArchivePlan",
    "PlanCost",
    "recommend_plan",
    "three_tier_plan",
    "two_tier_plan",
]

#: VHAGAR tile edge, metres (256 cells x 375 m).
TILE_EDGE_M = 96_000.0
#: Bytes per sample. int16 brightness temperature with a scale/offset. Never
#: float32, it doubles storage for precision the instrument does not have.
BYTES_PER_SAMPLE = 2
#: Typical zstd ratio on smooth int16 brightness-temperature fields. Measured
#: values run 3-6x; 4 is a defensible planning default. Verify on real data.
DEFAULT_COMPRESSION = 4.0
#: ABI CONUS single-band CMIP granule size on the wire, **measured** at 4.4 MB
#: for a 2 km channel (2500 x 1500 pixels). This is resolution-dependent and
#: the spread is large, so check :data:`ABI_BAND_RESOLUTION_M` before reusing
#: it: a 0.5 km channel has 16x the pixels and a Full Disk granule is roughly
#: 6x a CONUS one. All five bands VHAGAR needs are 2 km, which is why the
#: radiance tier is affordable at all.
DEFAULT_GRANULE_MB = 4.5
#: Single-worker-equivalent seconds to fetch, parse, navigate and tabulate one
#: full-CONUS FDC granule. This is the wall clock the planner divides by the
#: worker count, so it must be a per-granule figure, not a per-worker one.
#:
#: **Measured** from the real 7-day GOES-18 CONUS backfill: 2015 granules in
#: 30.7 minutes at 16 workers is 0.914 s of wall clock per granule, which is
#: about 14.7 single-worker-equivalent seconds per granule assuming near-linear
#: scaling. That run predates the fixed-grid navigation cache, and navigation
#: was its dominant cost, so this is a deliberately conservative upper bound:
#: with the cache the steady-state decode no longer recomputes navigation per
#: granule, so the real figure is lower. Re-measure it on the target machine
#: with ``vhagar archive-plan --measure`` and ``vhagar probe-workers`` and set
#: this from that, rather than trusting either the old 0.8 placeholder (which
#: predicted 6 hours for a 3-year run that in fact took roughly 80) or a number
#: measured on a different machine. The CMIP radiance figure is still unknown:
#: no decoder exists, so any radiance wall clock is unmeasured.
DEFAULT_SECONDS_PER_GRANULE = 14.7
#: Native resolution of the ABI channels, metres at nadir. VHAGAR's thermal
#: set (C07, C11, C13, C14, C15) is entirely 2 km. Reaching for C02 to get
#: visible smoke context would cost 16x the pixels per granule.
ABI_BAND_RESOLUTION_M = {
    "C01": 1000, "C02": 500, "C03": 1000, "C04": 2000, "C05": 1000,
    "C06": 2000, "C07": 2000, "C08": 2000, "C09": 2000, "C10": 2000,
    "C11": 2000, "C12": 2000, "C13": 2000, "C14": 2000, "C15": 2000,
    "C16": 2000,
}
#: Bytes per stored detection row: time, tile, row, col, mask code, FRP, area,
#: temperature, DQF, plus Parquet dictionary and index overhead. Measured rows
#: come out near 40 B; 56 B leaves headroom for the provenance columns.
BYTES_PER_DETECTION = 56
#: Fraction of pixel-timesteps that carry a fire mask code in FDC. **Measured**
#: on the 7-day GOES-18 CONUS backfill, 2026-08-01 to 08-07: 188,639 detections
#: over 2015 granules of 2500 x 1500 = 3.75M pixels each, both mask series and
#: all confidences, is 2.5e-5. This supersedes the old 3e-5, which came from a
#: single 6-hour northern California window (160 detections over 71 granules).
#: Both are peak-summer figures, so this is an upper bound on the annual mean
#: that a winter-inclusive window would give, which is the safe direction for
#: sizing disk. Verify per region and per season before trusting it far from
#: CONUS August.
DEFAULT_DETECTION_RATE = 2.5e-5


@dataclass(frozen=True, slots=True)
class ArchivePlan:
    """A candidate archive configuration."""

    name: str
    n_tiles: int
    years: float
    cadence_min: float
    n_bands: int
    sensor_resolution_m: float = 2000.0
    compression: float = DEFAULT_COMPRESSION
    granule_mb: float = DEFAULT_GRANULE_MB
    seconds_per_granule: float = DEFAULT_SECONDS_PER_GRANULE
    workers: int = 12
    #: ABI L2 CMIP / L1b Rad products ship **one file per channel**, so reading
    #: 5 bands costs 5 S3 reads per timestep. FDC is multiband in one file.
    #: This is the single most commonly missed factor in archive sizing.
    one_file_per_band: bool = True
    #: ``"dense"`` keeps every pixel-timestep as an int16 raster. That is the
    #: right shape for radiance, where the whole field is signal. ``"sparse"``
    #: keeps only rows that carry a detection, which is the right shape for
    #: FDC, where 99.997% of the grid is "no fire" and storing it dense costs
    #: three orders of magnitude more disk to record the same information.
    #: The trade is real: sparse discards the explicit negative field, so the
    #: loader has to reconstruct negatives from the coverage record.
    storage: str = "dense"
    detection_rate: float = DEFAULT_DETECTION_RATE

    @property
    def pixels_per_tile_side(self) -> int:
        """How many sensor pixels cover one 96 km tile edge."""
        return max(1, int(round(TILE_EDGE_M / self.sensor_resolution_m)))

    @property
    def timesteps_per_day(self) -> float:
        return 1440.0 / self.cadence_min

    @property
    def days(self) -> float:
        return self.years * 365.25

    @property
    def pixel_timesteps(self) -> float:
        """Total pixel-timesteps the plan covers, across all tiles and bands."""
        px = self.pixels_per_tile_side
        return (
            self.n_tiles
            * self.n_bands
            * px
            * px
            * self.timesteps_per_day
            * self.days
        )

    def cost(self) -> PlanCost:
        if self.storage == "sparse":
            # BYTES_PER_DETECTION is already a post-encoding figure, so the
            # raster compression ratio does not apply on top of it.
            raw = self.pixel_timesteps * self.detection_rate * BYTES_PER_DETECTION
            disk = raw
        elif self.storage == "dense":
            raw = self.pixel_timesteps * BYTES_PER_SAMPLE
            disk = raw / self.compression
        else:
            raise ValueError(f"storage must be 'dense' or 'sparse', got {self.storage!r}")
        # Granule reads depend on cadence, duration and: for per-band products
        #: band count. Crucially NOT on tile count: that is the whole argument
        # for the granule-outer-loop architecture.
        files_per_step = self.n_bands if self.one_file_per_band else 1
        reads = self.timesteps_per_day * self.days * files_per_step
        return PlanCost(
            plan=self,
            disk_gb=disk / 1e9,
            raw_gb=raw / 1e9,
            granule_reads=int(round(reads)),
            download_tb=reads * self.granule_mb / 1e6,
            wall_clock_hours=reads * self.seconds_per_granule / max(self.workers, 1) / 3600.0,
        )

    def scaled_to_disk(self, disk_gb: float, min_tiles: int = 20) -> ArchivePlan:
        """Shrink the tile count so the plan fits a disk budget.

        Tiles are cut first because they are the cheapest dimension to lose:
        fewer tiles narrows spatial coverage, whereas cutting years or cadence
        damages the temporal statistics the archive exists to support.
        """
        current = self.cost().disk_gb
        if current <= disk_gb:
            return self
        n = max(min_tiles, int(self.n_tiles * disk_gb / current))
        return replace(self, n_tiles=n, name=f"{self.name} (fitted to {disk_gb:.0f} GB)")


@dataclass(frozen=True, slots=True)
class PlanCost:
    """The three budgets, plus the raw figure for context."""

    plan: ArchivePlan
    disk_gb: float
    raw_gb: float
    granule_reads: int
    download_tb: float
    wall_clock_hours: float

    def as_row(self) -> dict[str, object]:
        p = self.plan
        return {
            "plan": p.name,
            "tiles": p.n_tiles,
            "years": p.years,
            "cadence_min": p.cadence_min,
            "bands": p.n_bands,
            "disk_gb": round(self.disk_gb, 1),
            "download_tb": round(self.download_tb, 2),
            "granule_reads": self.granule_reads,
            "hours": round(self.wall_clock_hours, 1),
        }

    def __str__(self) -> str:  # pragma: no cover - display helper
        p = self.plan
        return (
            f"{p.name:<34} {p.n_tiles:>5} tiles  {p.years:>4.1f} yr  "
            f"{p.cadence_min:>4.0f} min  {p.n_bands:>2} bands  "
            f"{self.disk_gb:>8.1f} GB disk  {self.download_tb:>6.2f} TB wire  "
            f"{self.wall_clock_hours:>6.1f} h"
        )


#: Reference plans. The climatology tier is the expensive one because it is wide.
STANDARD_PLANS: tuple[ArchivePlan, ...] = (
    # FDC is a sparse multiband mask at ~0.3 MB per granule, measured on
    # GOES-18 CONUS. Detection history, persistence and diurnal statistics all
    # come from this, and it costs almost nothing. Take it at full 5 min
    # cadence over as many tiles as you like.
    ArchivePlan(
        "FDC detection history",
        n_tiles=500, years=3, cadence_min=5, n_bands=1,
        granule_mb=0.32, one_file_per_band=False, seconds_per_granule=14.7,
        storage="sparse",
    ),
    ArchivePlan(
        "FDC dense mask (comparison)",
        n_tiles=500, years=3, cadence_min=5, n_bands=1,
        granule_mb=0.32, one_file_per_band=False, seconds_per_granule=14.7,
        storage="dense",
    ),
    ArchivePlan("minimal climatology", n_tiles=100, years=2, cadence_min=15, n_bands=3),
    ArchivePlan("standard climatology", n_tiles=200, years=2, cadence_min=15, n_bands=5),
    ArchivePlan("generous climatology", n_tiles=300, years=3, cadence_min=15, n_bands=5),
    ArchivePlan("high-cadence tier", n_tiles=20, years=0.42, cadence_min=5, n_bands=5),
    ArchivePlan("naive: everything at 5 min", n_tiles=300, years=3, cadence_min=5, n_bands=5),
)


def two_tier_plan(disk_gb: float) -> tuple[ArchivePlan, ArchivePlan]:
    """Recommended (climatology, high-cadence) pair for a disk budget.

    The high-cadence tier is reserved ~15 % of the budget, or 8 GB, whichever is
    larger, it is small in absolute terms and disproportionately useful, since
    it is the only thing that can train an early-detection model.
    """
    hi_budget = max(8.0, 0.15 * disk_gb)
    lo_budget = max(disk_gb - hi_budget, 0.0)
    climatology = ArchivePlan(
        "climatology", n_tiles=300, years=3, cadence_min=15, n_bands=5
    ).scaled_to_disk(lo_budget)
    high_cadence = ArchivePlan(
        "high-cadence", n_tiles=40, years=0.42, cadence_min=5, n_bands=5
    ).scaled_to_disk(hi_budget)
    return climatology, high_cadence


def three_tier_plan(disk_gb: float) -> tuple[ArchivePlan, ArchivePlan, ArchivePlan]:
    """Recommended (detection, climatology, high-cadence) trio for a disk budget.

    The detection tier is new in v0.8 and it changes the shape of Step 2. FDC
    granules measure 0.32 MB against 4.4 MB for a 2 km radiance band, and the
    product is sparse enough to store as detection rows rather than a raster.
    So a wide, long, 5-minute detection history costs well under a gigabyte of
    disk and about a tenth of a terabyte on the wire. It already carries
    everything the persistence and diurnal-detection features need.

    Radiance is still the expensive tier, but v0.9 measured it at 4.4 MB per
    granule rather than the 20 MB assumed, so the whole three-tier backfill
    lands near 3.5 TB on the wire rather than 15 TB. The wall clock is a
    separate question and the earlier "long weekend" figure was wrong: it came
    from the 0.8 s placeholder. The real per-granule time measured on the 7-day
    run is about 14.7 single-worker-equivalent seconds (see
    :data:`DEFAULT_SECONDS_PER_GRANULE`), which put the FDC tier alone near 80
    hours before the navigation cache. Re-measure on the target machine after
    the cache with ``vhagar archive-plan --measure`` before trusting any hours
    figure here.
    """
    detection = ArchivePlan(
        "detection history",
        n_tiles=500, years=3, cadence_min=5, n_bands=1,
        granule_mb=0.32, one_file_per_band=False, seconds_per_granule=14.7,
        storage="sparse",
    )
    radiance_budget = max(disk_gb - detection.cost().disk_gb, 0.0)
    climatology, high_cadence = two_tier_plan(radiance_budget)
    return detection, climatology, high_cadence


def recommend_plan(disk_gb: float) -> str:
    """Human-readable recommendation for a disk budget."""
    detection, clim, hi = three_tier_plan(disk_gb)
    d, c, h = detection.cost(), clim.cost(), hi.cost()
    lines = [
        f"For {disk_gb:.0f} GB of disk:",
        "",
        f"  {d}",
        f"  {c}",
        f"  {h}",
        "",
        f"  total disk      {d.disk_gb + c.disk_gb + h.disk_gb:>8.1f} GB",
        f"  total wire      {d.download_tb + c.download_tb + h.download_tb:>8.2f} TB",
        f"  total wall clock{d.wall_clock_hours + c.wall_clock_hours + h.wall_clock_hours:>8.1f} h "
        f"at {clim.workers} workers",
        "",
        f"  Build the detection tier first. At {d.disk_gb:.2f} GB of disk and "
        f"{d.download_tb:.2f} TB on the",
        "  wire it needs no new decoder, and on its own it unblocks the",
        "  persistence, diurnal and event-history features. Let it tell you which",
        "  tiles actually burn before you spend bandwidth stratifying the",
        "  radiance tiers.",
        "",
        "  Most of the per-granule cost is parse and navigation, not the network:",
        "  a bare S3 read is about 0.12 s against 0.75 s for the full path. So the",
        f"  {clim.workers}-worker figure above matters, but measure it on the real",
        "  operation (vhagar probe-workers) rather than on a bare read, and expect",
        "  it to track your CPU as much as your connection.",
        "",
        "  Read granules in the OUTER loop and tiles in the inner loop: the",
        "  granule-read count above is independent of tile count. Iterating",
        "  per-tile multiplies S3 requests by the tile count.",
    ]
    return "\n".join(lines)


#: The two ABI L2 products VHAGAR draws on, and why they size so differently.
#:
#: * ``FDC`` is the fire product: a mask plus power, area and temperature on a
#:   grid that is almost entirely fill. It compresses to a few hundred KB and
#:   ships every band in one file.
#: * ``CMIP`` is calibrated imagery: dense radiance where every pixel carries
#:   signal, tens of MB, and **one file per channel**.
#:
#: Sizing a radiance archive from an FDC measurement understates it by roughly
#: two orders of magnitude. Measure the product you actually intend to pull.
ABI_PRODUCTS = {
    "FDC": "ABI-L2-FDC",
    "CMIP": "ABI-L2-CMIP",
}


def measure_granule(
    satellite: int = 18,
    domain: str = "C",
    n: int = 3,
    product: str = "FDC",
    channel: str = "C07",
) -> dict[str, float]:
    """Measure real granule size and read time on this connection. Needs network.

    Answers the one thing the arithmetic cannot: whether HDF5 chunked
    range-reads over S3 let you fetch a small bbox without pulling the whole
    file. Pass ``product="CMIP"`` to size the radiance tier; the default only
    tells you about the sparse fire product.
    """
    import time
    from datetime import datetime, timedelta

    import s3fs

    from vhagar.io.goes import GOES_BUCKETS, fdc_key_prefix
    from vhagar.io.goes_reader import list_fdc_granules, open_fdc

    if product not in ABI_PRODUCTS:
        raise ValueError(f"product must be one of {sorted(ABI_PRODUCTS)}, got {product!r}")

    bucket = GOES_BUCKETS[satellite]
    fs = s3fs.S3FileSystem(anon=True)
    end = datetime.now(UTC) - timedelta(hours=2)

    if product == "FDC":
        keys = list_fdc_granules(satellite, end - timedelta(hours=2), end, domain=domain)
    else:
        prefix = fdc_key_prefix(satellite, end, domain).replace(
            ABI_PRODUCTS["FDC"], ABI_PRODUCTS["CMIP"]
        )
        listing = fs.ls(f"{bucket}/{prefix}", detail=False)
        keys = [k.split("/", 1)[1] for k in listing if f"-M6{channel}_" in k]
    if not keys:
        raise RuntimeError(f"no {product} granules found in the sampling window")
    keys = keys[-n:]

    sizes, times = [], []
    for key in keys:
        info = fs.info(f"{bucket}/{key}")
        sizes.append(float(info["size"]))
        t0 = time.perf_counter()
        if product == "FDC":
            open_fdc(key, satellite, bbox=(-124.0, 36.0, -118.0, 42.0))
        else:
            # No decoder for CMIP yet, so time the full fetch a decoder would
            # have to pay. Reading only the first megabyte would report the
            # round-trip latency and call it the granule cost.
            with fs.open(f"{bucket}/{key}", "rb") as fh:
                fh.read()
        times.append(time.perf_counter() - t0)

    mb = sum(sizes) / len(sizes) / 1e6
    secs = sum(times) / len(times)
    return {
        "n_sampled": len(keys),
        "product": product,
        "granule_mb": mb,
        "seconds_per_granule": secs,
        "mb_per_second": mb / secs if secs > 0 else float("nan"),
    }

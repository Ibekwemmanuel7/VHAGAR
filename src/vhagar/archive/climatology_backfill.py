"""Tier B backfill: reduce CMIP stacks into a diurnal climatology, resumably.

This is the radiance-tier counterpart to :mod:`vhagar.archive.backfill`. Where
Tier A stores sparse detection rows, Tier B keeps no frames at all: it folds each
CMIP stack into a per-pixel, per-hour running mean and variance
(:class:`~vhagar.archive.climatology.DiurnalClimatology`) and stores only those
statistics. The climatology lives on the **native ABI 2 km grid**, cropped to the
configured region; no reprojection, consistent with the project rule against
inventing precision.

Resumability without double counting
------------------------------------
Folding a frame into a Welford accumulator is not idempotent, so the Tier A
"replay the granule" trick does not apply. Instead the checkpoint is the single
source of truth: the accumulator is saved to one ``.npz`` that also carries the
watermark of processed timestep ids, and it is written with an atomic replace. A
crash either leaves the previous checkpoint intact (the interrupted batch is
reprocessed cleanly) or the new one complete (the batch is skipped on resume).
There is no window in which a frame is both on disk and not in the watermark.

Concurrency
-----------
Reads are the slow part and they parallelise; the fold is fast and stays on one
thread, so the Welford update needs no lock. Stacks are opened concurrently and
folded as they arrive, so memory is bounded by the worker count rather than by a
day of frames. A manifest and coverage record are kept exactly as in Tier A, so a
loader can tell an observed-but-quiet period from one that was never read.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

import vhagar.io.cmip_reader as cmip
from vhagar.archive.backfill import (
    GranuleRecord,
    _append_manifest,
    coverage_intervals,
    load_manifest,
)
from vhagar.archive.climatology import DiurnalClimatology
from vhagar.io.cmip_reader import CMIP_CHANNELS, CMIP_STACK_TOLERANCE
from vhagar.io.goes import parse_goes_key

log = logging.getLogger(__name__)

__all__ = [
    "ClimatologyBackfillConfig",
    "ClimatologyResult",
    "backfill_climatology",
]

#: Default thermal set for the radiance tier. All 2 km.
DEFAULT_CHANNELS = ("C07", "C11", "C13", "C14", "C15")
#: Checkpoint and bookkeeping filenames inside the output directory.
CHECKPOINT_NAME = "climatology.npz"
MANIFEST_NAME = "_manifest.jsonl"
CONFIG_NAME = "_config.json"


@dataclass(frozen=True, slots=True)
class ClimatologyBackfillConfig:
    """Everything that changes what the climatology contains."""

    out_dir: Path
    start: datetime
    end: datetime
    #: ``(west, south, east, north)`` in degrees. Required: the climatology is a
    #: dense per-pixel raster, so the region must be bounded or memory explodes.
    bbox: tuple[float, float, float, float]
    satellite: int = 18
    domain: str = "C"
    channels: tuple[str, ...] = DEFAULT_CHANNELS
    #: Diurnal sampling. CMIP CONUS lands every 5 minutes; 15 is ample for a
    #: diurnal baseline, so groups are subsampled to this spacing.
    cadence_min: int = 15
    #: Diurnal bins across the day. 24 is hourly; must divide 1440.
    n_bins: int = 24
    workers: int = 8
    tolerance: timedelta = CMIP_STACK_TOLERANCE

    def fingerprint(self) -> dict[str, object]:
        return {
            "satellite": self.satellite,
            "domain": self.domain,
            "channels": list(self.channels),
            "bbox": list(self.bbox),
            "cadence_min": self.cadence_min,
            "n_bins": self.n_bins,
        }


@dataclass(slots=True)
class ClimatologyResult:
    """Summary of one Tier B run."""

    frames_ok: int = 0
    frames_failed: int = 0
    frames_skipped: int = 0
    elapsed_s: float = 0.0
    errors: dict[str, int] = field(default_factory=dict)

    @property
    def frames_per_second(self) -> float:
        return self.frames_ok / self.elapsed_s if self.elapsed_s > 0 else 0.0

    def __str__(self) -> str:
        return ", ".join(
            [
                f"{self.frames_ok} ok",
                f"{self.frames_failed} failed",
                f"{self.frames_skipped} skipped",
                f"{self.elapsed_s / 60:.1f} min",
                f"{self.frames_per_second:.2f} frames/s",
            ]
        )


def _check_config(cfg: ClimatologyBackfillConfig) -> None:
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
            f"{cfg.out_dir} was built with different settings; a climatology cannot "
            f"mix them. Differences (on disk, requested): {diff}. Use a separate directory."
        )


def _cadence_subsample(
    groups: Sequence[dict[str, str]], ref_band: str, satellite: int, cadence_min: int
) -> list[tuple[datetime, str, dict[str, str]]]:
    """Keep one complete stack per cadence bucket, earliest first.

    Returns ``(scan_start, group_id, group)`` triples, where ``group_id`` is the
    reference channel key, unique per timestep.
    """
    triples = sorted(
        (parse_goes_key(g[ref_band], satellite).start, g[ref_band], g) for g in groups
    )
    kept: list[tuple[datetime, str, dict[str, str]]] = []
    seen_buckets: set[tuple] = set()
    step = timedelta(minutes=cadence_min)
    for start, gid, group in triples:
        bucket = int((start - datetime(start.year, 1, 1, tzinfo=UTC)) / step)
        if bucket in seen_buckets:
            continue
        seen_buckets.add(bucket)
        kept.append((start, gid, group))
    return kept


def _open_stack(group: dict[str, str], config: ClimatologyBackfillConfig):
    """Open one timestep's stack. Never raises."""
    gid = group[config.channels[0]]
    try:
        stack = cmip.open_cmip_stack(group, config.satellite, bbox=config.bbox)
        return gid, stack, None
    except Exception as exc:  # noqa: BLE001
        return gid, None, f"{type(exc).__name__}: {exc}"


def _load_checkpoint(
    path: Path, config: ClimatologyBackfillConfig
) -> tuple[DiurnalClimatology | None, set[str]]:
    if not path.exists():
        return None, set()
    clim, meta = DiurnalClimatology.load_with_meta(path)
    processed = {str(k) for k in meta.get("processed", np.array([], dtype=str))}
    return clim, processed


def backfill_climatology(
    config: ClimatologyBackfillConfig, progress=None
) -> ClimatologyResult:
    """Run the Tier B climatology backfill, one day at a time, resumably."""
    for ch in config.channels:
        if ch not in CMIP_CHANNELS:
            raise ValueError(f"unknown channel {ch!r}")

    config.out_dir.mkdir(parents=True, exist_ok=True)
    _check_config(config)
    checkpoint = config.out_dir / CHECKPOINT_NAME
    ref_band = config.channels[0]

    # Discover complete stacks across the window, then thin to the cadence.
    keys_by_channel = {
        ch: cmip.list_cmip_granules(
            config.satellite, config.start, config.end, ch, domain=config.domain
        )
        for ch in config.channels
    }
    groups = cmip.group_cmip_keys_by_timestamp(
        keys_by_channel, config.satellite, tolerance=config.tolerance
    )
    schedule = _cadence_subsample(groups, ref_band, config.satellite, config.cadence_min)

    clim, processed = _load_checkpoint(checkpoint, config)
    result = ClimatologyResult()
    t_start = time.perf_counter()

    # Group the remaining timesteps by day so each day is a checkpoint boundary.
    by_day: dict[datetime, list[tuple[datetime, str, dict[str, str]]]] = {}
    for start, gid, group in schedule:
        if gid in processed:
            result.frames_skipped += 1
            continue
        day = start.replace(hour=0, minute=0, second=0, microsecond=0)
        by_day.setdefault(day, []).append((start, gid, group))

    for day in sorted(by_day):
        records: list[GranuleRecord] = []
        day_ok = 0
        with ThreadPoolExecutor(max_workers=config.workers) as pool:
            futures = [pool.submit(_open_stack, g, config) for _, _, g in by_day[day]]
            times = {gid: start for start, gid, _ in by_day[day]}
            for fut in as_completed(futures):
                gid, stack, err = fut.result()
                if stack is None:
                    result.frames_failed += 1
                    kind = (err or "unknown").split(":", 1)[0]
                    result.errors[kind] = result.errors.get(kind, 0) + 1
                    records.append(GranuleRecord(gid, None, "error", error=err))
                    continue
                if clim is None:
                    clim = DiurnalClimatology(config.channels, stack.shape, config.n_bins)
                try:
                    clim.update(stack)
                except Exception as exc:  # noqa: BLE001
                    result.frames_failed += 1
                    kind = type(exc).__name__
                    result.errors[kind] = result.errors.get(kind, 0) + 1
                    records.append(
                        GranuleRecord(gid, None, "error", error=f"{kind}: {exc}")
                    )
                    continue
                processed.add(gid)
                result.frames_ok += 1
                day_ok += 1
                records.append(GranuleRecord(gid, times[gid].isoformat(), "ok"))

        # Checkpoint first (the source of truth, atomic), then the manifest.
        if clim is not None and day_ok:
            clim.save(checkpoint, meta={"processed": np.array(sorted(processed))})
        _append_manifest(config.out_dir, records)
        if progress:
            progress(day, day_ok)

    result.elapsed_s = time.perf_counter() - t_start
    return result


def climatology_coverage(out_dir: Path | str):
    """Observed intervals for a Tier B directory, from its manifest."""
    return coverage_intervals(load_manifest(out_dir).values())

"""Compact the Tier A detection archive: many small day files into few big ones.

Why
---
The backfill writes one Parquet file per tile per day, which is the right shape
while data is arriving: a five-minute cadence would otherwise leave 288 tiny
files per tile per day. But after a year each tile holds hundreds of small files,
and Parquet's per-file footer and the filesystem's per-file overhead start to
dominate reads. Compaction merges each tile's day files into one file per year,
which is the shape you want for training reads.

Safety
------
Data loss here would be silent and permanent, so the order is strict: read all
of a tile's files, write the merged result to a temporary file, verify the merged
row count equals the sum of the inputs AND that the file on disk reports that many
rows, atomically replace, and only then delete the originals. A crash at any
point leaves either the original day files or a verified compacted file, never a
half-written merge with the sources already gone.

Idempotent and incremental
--------------------------
The compacted file is itself a ``part-*.parquet``, so a later run folds any new
day files into it and rewrites. Running compaction twice with no new data is a
no-op: a tile with a single file (already compacted) is skipped.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from vhagar.archive.backfill import DETECTION_COLUMNS, DETECTIONS_DIR

log = logging.getLogger(__name__)

__all__ = ["CompactionReport", "compact_detections"]


@dataclass(slots=True)
class CompactionReport:
    """Summary of one compaction pass."""

    tiles_compacted: int = 0
    tiles_skipped: int = 0
    files_before: int = 0
    files_after: int = 0
    rows: int = 0

    @property
    def files_removed(self) -> int:
        return self.files_before - self.files_after

    def __str__(self) -> str:
        return ", ".join(
            [
                f"{self.tiles_compacted} tiles compacted",
                f"{self.tiles_skipped} skipped",
                f"{self.files_before} -> {self.files_after} files",
                f"{self.files_removed} removed",
                f"{self.rows:,} rows",
            ]
        )


def compact_detections(
    out_dir: Path | str, min_files: int = 2, dry_run: bool = False
) -> CompactionReport:
    """Merge each tile's per-day Parquet files into one file per year.

    ``min_files`` is the smallest count worth compacting; a tile with fewer files
    is left alone. ``dry_run`` reports what would happen without touching disk.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(out_dir) / DETECTIONS_DIR
    report = CompactionReport()
    if not root.exists():
        return report

    for tile_dir in sorted(p for p in root.glob("year=*/tile=*") if p.is_dir()):
        files = sorted(tile_dir.glob("part-*.parquet"))
        report.files_before += len(files)
        if len(files) < min_files:
            report.files_after += len(files)
            report.tiles_skipped += 1
            continue

        year = tile_dir.parent.name.split("=", 1)[1]
        tables = [pq.read_table(f).select(list(DETECTION_COLUMNS)) for f in files]
        total = sum(t.num_rows for t in tables)

        if dry_run:
            report.tiles_compacted += 1
            report.files_after += 1
            report.rows += total
            continue

        merged = pa.concat_tables(tables)
        if merged.num_rows != total:
            raise RuntimeError(
                f"{tile_dir}: merged {merged.num_rows} rows but inputs summed to {total}; "
                "refusing to delete originals"
            )

        target = tile_dir / f"part-{year}-compacted.parquet"
        tmp = tile_dir / (target.name + ".tmp")
        pq.write_table(merged, tmp, compression="zstd")
        # Verify the bytes on disk before removing anything.
        on_disk = pq.read_metadata(tmp).num_rows
        if on_disk != total:
            os.remove(tmp)
            raise RuntimeError(
                f"{tile_dir}: compacted file has {on_disk} rows, expected {total}; "
                "originals left untouched"
            )
        os.replace(tmp, target)

        for f in files:
            if f != target:
                f.unlink()

        report.tiles_compacted += 1
        report.files_after += 1
        report.rows += total

    return report

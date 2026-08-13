"""Detection-archive compaction: correctness, safety, idempotency.

The property that matters most is that no row is ever lost: the compacted file
holds exactly the rows of the day files it replaces, and originals are removed
only after that is verified.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from _fixtures import _synthetic_fdc

from vhagar.archive.backfill import BackfillConfig, _write_day, detection_table
from vhagar.archive.compaction import compact_detections
from vhagar.io.goes_reader import decode_fdc

T0 = datetime(2026, 8, 1, tzinfo=UTC)


def _cfg(tmp_path):
    return BackfillConfig(out_dir=tmp_path, start=T0, end=T0 + timedelta(days=3))


def _granule():
    return decode_fdc(_synthetic_fdc(n=40), satellite=19)


def _write_days(tmp_path, n_days: int) -> int:
    """Write one granule's detections into each of n_days day files. Returns the
    total row count across all files."""
    total = 0
    for d in range(n_days):
        day = T0 + timedelta(days=d)
        table = detection_table(_granule(), "conus", f"g{d}", _cfg(tmp_path))
        total += _write_day(tmp_path, day, [table])
    return total


def _all_parquet(tmp_path):
    return sorted((tmp_path / "detections").rglob("*.parquet"))


def _row_total(files):
    import pyarrow.parquet as pq

    return sum(pq.read_table(f).num_rows for f in files)


def test_compaction_preserves_every_row(tmp_path):
    pytest.importorskip("pyarrow")
    total = _write_days(tmp_path, 3)
    before = _all_parquet(tmp_path)
    assert len(before) > 0

    report = compact_detections(tmp_path)
    after = _all_parquet(tmp_path)

    assert _row_total(after) == total
    assert report.rows == total
    assert len(after) < len(before)


def test_each_tile_collapses_to_one_file(tmp_path):
    pytest.importorskip("pyarrow")
    _write_days(tmp_path, 3)
    compact_detections(tmp_path)
    # every tile directory now holds exactly one parquet file
    for tile_dir in (tmp_path / "detections").glob("year=*/tile=*"):
        files = list(tile_dir.glob("*.parquet"))
        assert len(files) == 1
        assert files[0].name.endswith("-compacted.parquet")


def test_compaction_is_idempotent(tmp_path):
    pytest.importorskip("pyarrow")
    total = _write_days(tmp_path, 3)
    compact_detections(tmp_path)
    first = _all_parquet(tmp_path)

    second_report = compact_detections(tmp_path)
    second = _all_parquet(tmp_path)
    # nothing left to do: single files per tile are skipped, rows unchanged
    assert second_report.tiles_compacted == 0
    assert [f.name for f in first] == [f.name for f in second]
    assert _row_total(second) == total


def test_incremental_compaction_folds_in_new_days(tmp_path):
    pytest.importorskip("pyarrow")
    _write_days(tmp_path, 2)
    compact_detections(tmp_path)
    # a new day arrives after the first compaction
    late = T0 + timedelta(days=5)
    extra = detection_table(_granule(), "conus", "glate", _cfg(tmp_path))
    added = _write_day(tmp_path, late, [extra])

    total_now = _row_total(_all_parquet(tmp_path))
    report = compact_detections(tmp_path)
    after = _all_parquet(tmp_path)
    assert _row_total(after) == total_now
    assert report.tiles_compacted >= 1
    assert added > 0


def test_dry_run_changes_nothing(tmp_path):
    pytest.importorskip("pyarrow")
    _write_days(tmp_path, 3)
    before = _all_parquet(tmp_path)
    report = compact_detections(tmp_path, dry_run=True)
    after = _all_parquet(tmp_path)
    assert [f.name for f in before] == [f.name for f in after]
    assert report.tiles_compacted > 0  # it reports what it would do


def test_missing_archive_is_a_no_op(tmp_path):
    report = compact_detections(tmp_path / "nothing")
    assert report.tiles_compacted == 0
    assert report.files_before == 0

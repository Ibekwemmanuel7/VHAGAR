"""Tier A backfill. Resumability, coverage accounting and tile assignment.

The properties worth testing here are not "does it download". They are:
a restart must not re-read what it already has, a crash must not record
coverage for rows that were never written, and absence must stay
distinguishable from "never looked". Everything else is plumbing.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from _fixtures import _synthetic_fdc

from vhagar.archive.backfill import (
    DETECTION_COLUMNS,
    MANIFEST_NAME,
    BackfillConfig,
    GranuleRecord,
    _write_day,
    coverage_gaps,
    coverage_intervals,
    detection_table,
    failed_records,
    load_manifest,
)
from vhagar.io.goes_reader import decode_fdc

T0 = datetime(2026, 8, 1, tzinfo=UTC)


def _cfg(tmp_path: Path, **kw) -> BackfillConfig:
    base = dict(out_dir=tmp_path, start=T0, end=T0 + timedelta(days=1))
    base.update(kw)
    return BackfillConfig(**base)


def _granule():
    return decode_fdc(_synthetic_fdc(n=40), satellite=19)


# ------------------------------------------------------- tabulation -------


def test_table_has_every_declared_column_and_nothing_else(tmp_path):
    t = detection_table(_granule(), "conus", "key.nc", _cfg(tmp_path))
    assert set(t) == set(DETECTION_COLUMNS)


def test_table_keeps_both_mask_series(tmp_path):
    """10-15 carries the early detections, 30-35 the confirmed ones."""
    t = detection_table(_granule(), "conus", "k", _cfg(tmp_path, include_filtered=True))
    codes = set(t["mask_code"].tolist())
    assert codes & {10, 13, 11, 15}
    assert 30 in codes


def test_excluding_filtered_drops_only_the_thirties(tmp_path):
    t = detection_table(_granule(), "conus", "k", _cfg(tmp_path, include_filtered=False))
    assert all(c < 30 for c in t["mask_code"])


def test_unreliable_frp_is_nan_not_a_number(tmp_path):
    """Codes 11/12/31/32 are real fires with meaningless power retrievals."""
    t = detection_table(_granule(), "conus", "k", _cfg(tmp_path))
    unreliable = np.isin(t["mask_code"], (11, 12, 31, 32))
    assert unreliable.any(), "fixture should plant at least one saturated pixel"
    assert np.all(np.isnan(t["frp_mw"][unreliable]))
    assert np.isfinite(t["frp_mw"][~unreliable]).any()


def test_confidence_threshold_drops_low_probability(tmp_path):
    loose = detection_table(_granule(), "conus", "k", _cfg(tmp_path))
    strict = detection_table(_granule(), "conus", "k", _cfg(tmp_path, min_confidence=0.9))
    assert len(strict["lat"]) < len(loose["lat"])
    assert np.all(strict["confidence"] >= 0.9)


def test_every_row_gets_a_tile_id(tmp_path):
    t = detection_table(_granule(), "conus", "k", _cfg(tmp_path))
    assert len(t["tile_id"]) == len(t["lat"])
    assert all(tid.startswith("conus/x") for tid in t["tile_id"])


def test_out_of_grid_detections_are_kept_with_an_empty_tile(tmp_path):
    """Losing them silently would bias the archive at the region edge."""
    t = detection_table(_granule(), "europe", "k", _cfg(tmp_path, region="europe"))
    assert len(t["lat"]) > 0
    assert all(tid == "" for tid in t["tile_id"])


def test_empty_granule_returns_typed_empty_columns(tmp_path):
    granule = decode_fdc(_synthetic_fdc(n=40, fire_codes=()), satellite=19)
    t = detection_table(granule, "conus", "k", _cfg(tmp_path))
    assert set(t) == set(DETECTION_COLUMNS)
    assert all(len(v) == 0 for v in t.values())


# ---------------------------------------------------------- manifest ------


def test_manifest_round_trips(tmp_path):
    from vhagar.archive.backfill import _append_manifest

    recs = [
        GranuleRecord("a.nc", T0.isoformat(), "ok", n_detections=3),
        GranuleRecord("b.nc", None, "error", error="OSError: nope"),
    ]
    _append_manifest(tmp_path, recs)
    back = load_manifest(tmp_path)
    assert back["a.nc"].ok and back["a.nc"].n_detections == 3
    assert not back["b.nc"].ok


def test_last_line_wins_so_a_retry_supersedes_a_failure(tmp_path):
    from vhagar.archive.backfill import _append_manifest

    _append_manifest(tmp_path, [GranuleRecord("a.nc", None, "error", error="x")])
    _append_manifest(tmp_path, [GranuleRecord("a.nc", T0.isoformat(), "ok")])
    assert load_manifest(tmp_path)["a.nc"].ok


def test_truncated_final_line_is_survivable(tmp_path):
    """What a kill during a flush leaves behind. That granule is re-attempted."""
    path = tmp_path / MANIFEST_NAME
    path.write_text(
        json.dumps({"key": "a.nc", "scan_start": None, "status": "ok"}) + "\n{\"key\": \"b",
        encoding="utf-8",
    )
    back = load_manifest(tmp_path)
    assert set(back) == {"a.nc"}


def test_missing_manifest_is_an_empty_dict_not_an_error(tmp_path):
    assert load_manifest(tmp_path / "nothing") == {}


def test_incompatible_config_is_refused(tmp_path):
    """A different bbox in the same directory would forge coverage."""
    from vhagar.archive.backfill import _check_config

    _check_config(_cfg(tmp_path, bbox=(-124.0, 36.0, -118.0, 42.0)))
    with pytest.raises(ValueError, match="different settings"):
        _check_config(_cfg(tmp_path, bbox=(-100.0, 30.0, -90.0, 40.0)))


def test_identical_config_is_allowed_to_resume(tmp_path):
    from vhagar.archive.backfill import _check_config

    _check_config(_cfg(tmp_path, bbox=(-124.0, 36.0, -118.0, 42.0)))
    _check_config(_cfg(tmp_path, bbox=(-124.0, 36.0, -118.0, 42.0)))


def test_config_fingerprint_ignores_time_window_and_workers(tmp_path):
    """Extending the window or changing concurrency must not block a resume."""
    a = _cfg(tmp_path, workers=4)
    b = _cfg(tmp_path, workers=32, end=T0 + timedelta(days=400))
    assert a.fingerprint() == b.fingerprint()


# ---------------------------------------------------------- coverage ------


def test_coverage_merges_a_continuous_run_into_one_interval():
    recs = [
        GranuleRecord(f"k{i}", (T0 + timedelta(minutes=5 * i)).isoformat(), "ok")
        for i in range(12)
    ]
    assert len(coverage_intervals(recs)) == 1


def test_coverage_splits_on_a_real_outage():
    early = [
        GranuleRecord(f"a{i}", (T0 + timedelta(minutes=5 * i)).isoformat(), "ok")
        for i in range(6)
    ]
    late = [
        GranuleRecord(f"b{i}", (T0 + timedelta(hours=6, minutes=5 * i)).isoformat(), "ok")
        for i in range(6)
    ]
    assert len(coverage_intervals(early + late)) == 2


def test_coverage_tolerates_a_few_dropped_granules():
    """A hole should mean an outage, not one flaky read."""
    times = [0, 5, 15, 20]  # a single missing granule at minute 10
    recs = [
        GranuleRecord(f"k{m}", (T0 + timedelta(minutes=m)).isoformat(), "ok")
        for m in times
    ]
    assert len(coverage_intervals(recs)) == 1


def test_failed_granules_do_not_count_as_coverage():
    """The whole point. A failed read is 'never looked', not 'nothing burning'."""
    recs = [
        GranuleRecord("a", T0.isoformat(), "ok"),
        GranuleRecord("b", (T0 + timedelta(minutes=5)).isoformat(), "error", error="x"),
    ]
    intervals = coverage_intervals(recs)
    assert intervals == [(T0, T0)]


def test_coverage_of_nothing_is_empty_not_an_exception():
    assert coverage_intervals([]) == []


def test_coverage_gaps_names_the_hole_between_two_intervals():
    """coverage reported '2 intervals' with no explanation; this is the tool
    that says where the hole is and how long it lasted."""
    early = [
        GranuleRecord(f"a{i}", (T0 + timedelta(minutes=5 * i)).isoformat(), "ok")
        for i in range(6)
    ]
    late = [
        GranuleRecord(f"b{i}", (T0 + timedelta(hours=6, minutes=5 * i)).isoformat(), "ok")
        for i in range(6)
    ]
    gaps = coverage_gaps(early + late)
    assert len(gaps) == 1
    start, end, duration = gaps[0]
    assert start < end
    assert duration > timedelta(minutes=20)


def test_a_continuous_run_has_no_gaps():
    recs = [
        GranuleRecord(f"k{i}", (T0 + timedelta(minutes=5 * i)).isoformat(), "ok")
        for i in range(12)
    ]
    assert coverage_gaps(recs) == []


def test_a_single_dropped_granule_is_not_a_gap():
    """The whole point of the tolerance: one flaky read is not a hole."""
    times = [0, 5, 15, 20]  # a single missing granule at minute 10
    recs = [
        GranuleRecord(f"k{m}", (T0 + timedelta(minutes=m)).isoformat(), "ok")
        for m in times
    ]
    assert coverage_gaps(recs) == []


def test_failed_records_lists_only_failures():
    recs = [
        GranuleRecord("a", T0.isoformat(), "ok"),
        GranuleRecord("b", None, "error", error="OSError: nope"),
        GranuleRecord("c", (T0 + timedelta(minutes=5)).isoformat(), "ok"),
    ]
    assert [r.key for r in failed_records(recs)] == ["b"]


# ----------------------------------------------------------- writing ------


def test_write_day_partitions_by_year_and_tile(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")

    table = detection_table(_granule(), "conus", "k", _cfg(tmp_path))
    written = _write_day(tmp_path, T0, [table])
    assert written == len(table["lat"])

    files = sorted((tmp_path / "detections").rglob("*.parquet"))
    assert files, "expected at least one partition"
    assert all("year=2026" in str(f) for f in files)
    assert all("tile=conus_x" in str(f) for f in files)

    back = pq.read_table(files[0])
    assert set(back.column_names) == set(DETECTION_COLUMNS)


def test_write_day_of_nothing_writes_nothing(tmp_path):
    pytest.importorskip("pyarrow")
    granule = decode_fdc(_synthetic_fdc(n=40, fire_codes=()), satellite=19)
    table = detection_table(granule, "conus", "k", _cfg(tmp_path))
    assert _write_day(tmp_path, T0, [table]) == 0
    assert not list((tmp_path / "detections").rglob("*.parquet"))


def test_rewriting_the_same_day_is_idempotent(tmp_path):
    """Rows are written before the manifest, so a crash between them replays."""
    pq = pytest.importorskip("pyarrow.parquet")
    table = detection_table(_granule(), "conus", "k", _cfg(tmp_path))
    _write_day(tmp_path, T0, [table])
    _write_day(tmp_path, T0, [table])
    files = sorted((tmp_path / "detections").rglob("*.parquet"))
    total = sum(pq.read_table(f).num_rows for f in files)
    assert total == len(table["lat"])


# -------------------------------------------------------- the run loop ----
# The resumability logic is the reason this module exists, so it is tested
# against a stubbed reader rather than left to the first real overnight run.


def _stub_reader(monkeypatch, keys, fail: set[str] = frozenset()):
    """Patch out S3. Returns a list that records every key actually fetched."""
    import vhagar.io.goes_reader as reader

    fetched: list[str] = []

    def fake_list(satellite, start, end, domain="C", anon=True):
        return [k for k in keys if k.startswith(f"{start:%Y%m%d}")]

    def fake_open(key, satellite, bbox=None, anon=True):
        fetched.append(key)
        if key in fail:
            raise OSError("simulated S3 hiccup")
        granule = decode_fdc(_synthetic_fdc(n=40), satellite=19)
        minute = int(key.split("-")[-1])
        granule.scan_start = T0 + timedelta(minutes=5 * minute)
        return granule

    monkeypatch.setattr(reader, "list_fdc_granules", fake_list)
    monkeypatch.setattr(reader, "open_fdc", fake_open)
    return fetched


def _keys(n: int) -> list[str]:
    return [f"{T0:%Y%m%d}/g-{i}" for i in range(n)]


def test_a_clean_run_reads_every_granule_once(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    from vhagar.archive.backfill import backfill

    fetched = _stub_reader(monkeypatch, _keys(5))
    result = backfill(_cfg(tmp_path, end=T0 + timedelta(hours=1)))
    assert result.granules_ok == 5
    assert sorted(fetched) == sorted(_keys(5))
    assert result.detections > 0


def test_a_restart_re_reads_nothing(tmp_path, monkeypatch):
    """The property that turns a 20-hour job into a resumable one."""
    pytest.importorskip("pyarrow")
    from vhagar.archive.backfill import backfill

    cfg = _cfg(tmp_path, end=T0 + timedelta(hours=1))
    _stub_reader(monkeypatch, _keys(5))
    backfill(cfg)

    fetched_again = _stub_reader(monkeypatch, _keys(5))
    second = backfill(cfg)
    assert fetched_again == []
    assert second.granules_skipped == 5
    assert second.granules_attempted == 0


def test_a_failed_granule_is_retried_on_the_next_run(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    from vhagar.archive.backfill import backfill

    cfg = _cfg(tmp_path, end=T0 + timedelta(hours=1))
    _stub_reader(monkeypatch, _keys(5), fail={"20260801/g-2"})
    first = backfill(cfg)
    assert first.granules_ok == 4 and first.granules_failed == 1

    retried = _stub_reader(monkeypatch, _keys(5))
    second = backfill(cfg)
    assert retried == ["20260801/g-2"]
    assert second.granules_ok == 1


def test_retry_can_be_switched_off(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    from vhagar.archive.backfill import backfill

    _stub_reader(monkeypatch, _keys(3), fail={"20260801/g-1"})
    backfill(_cfg(tmp_path, end=T0 + timedelta(hours=1)))

    retried = _stub_reader(monkeypatch, _keys(3))
    backfill(_cfg(tmp_path, end=T0 + timedelta(hours=1), retry_failed=False))
    assert retried == []


def test_a_failed_granule_leaves_a_hole_in_coverage_not_a_silent_gap(tmp_path, monkeypatch):
    """A missing row must stay distinguishable from an unread granule."""
    pytest.importorskip("pyarrow")
    from vhagar.archive.backfill import backfill

    _stub_reader(monkeypatch, _keys(3), fail={"20260801/g-1"})
    backfill(_cfg(tmp_path, end=T0 + timedelta(hours=1), retry_failed=False))

    manifest = load_manifest(tmp_path)
    assert manifest["20260801/g-1"].status == "error"
    assert "OSError" in manifest["20260801/g-1"].error
    covered = coverage_intervals(manifest.values())
    assert all(start <= end for start, end in covered)


def test_errors_are_counted_by_type_not_swallowed(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    from vhagar.archive.backfill import backfill

    _stub_reader(monkeypatch, _keys(4), fail=set(_keys(4)))
    result = backfill(_cfg(tmp_path, end=T0 + timedelta(hours=1)))
    assert result.granules_ok == 0
    assert result.errors == {"OSError": 4}


def test_each_day_reports_its_own_elapsed_time(tmp_path, monkeypatch):
    """The per-day progress line read 0.0 min for every day because elapsed_s
    was only ever set on the overall result. Each day must carry its own."""
    pytest.importorskip("pyarrow")
    from vhagar.archive.backfill import backfill

    _stub_reader(monkeypatch, _keys(4))
    seen: list[float] = []

    def progress(_day, day_result):
        if day_result is not None:
            seen.append(day_result.elapsed_s)

    backfill(_cfg(tmp_path, end=T0 + timedelta(hours=1)), progress=progress)
    assert seen, "progress should fire for the worked day"
    assert all(e > 0.0 for e in seen), "per-day elapsed must be set, not left at 0.0"


def test_worker_count_does_not_change_the_result(tmp_path, monkeypatch):
    """Concurrency is a throughput lever, and must not be a correctness one."""
    pytest.importorskip("pyarrow")
    from vhagar.archive.backfill import backfill

    _stub_reader(monkeypatch, _keys(6))
    serial = backfill(_cfg(tmp_path / "a", end=T0 + timedelta(hours=1), workers=1))
    _stub_reader(monkeypatch, _keys(6))
    parallel = backfill(_cfg(tmp_path / "b", end=T0 + timedelta(hours=1), workers=8))
    assert serial.granules_ok == parallel.granules_ok
    assert serial.detections == parallel.detections


# ------------------------------------------------------- worker probe -----
# The first probe measured bare S3 reads and I sized a decode-bound workload
# from it. These tests pin the guards that stop that reading as a result.


def test_probe_rejects_an_unknown_mode():
    from vhagar.archive.backfill import probe_workers

    with pytest.raises(ValueError, match="mode"):
        probe_workers(mode="quick")


def _row(workers, gps, seconds=10.0, spread=0.05):
    from vhagar.archive.backfill import MIN_PROBE_SECONDS

    return {
        "workers": float(workers),
        "granules_per_second": gps,
        "seconds_per_granule": 1.0 / gps,
        "seconds": seconds,
        "spread": spread,
        "too_fast": float(seconds < MIN_PROBE_SECONDS),
    }


def test_a_clean_curve_names_the_smallest_setting_within_ten_percent():
    from vhagar.archive.backfill import recommend_workers

    rows = [_row(1, 5.0), _row(4, 15.0), _row(8, 19.0), _row(16, 20.0), _row(32, 20.1)]
    v = recommend_workers(rows)
    assert v["workers"] == 8
    assert v["best"] == 32


def test_runs_that_were_too_short_refuse_to_name_a_knee():
    """Chidi's first probe finished in 0.5 s and ranked 8 > 32 > 16 > 64."""
    from vhagar.archive.backfill import recommend_workers

    rows = [
        _row(1, 8.69, seconds=1.8), _row(4, 22.68, seconds=0.7),
        _row(8, 35.24, seconds=0.5), _row(16, 29.23, seconds=0.5),
        _row(32, 34.50, seconds=0.5), _row(64, 20.72, seconds=0.8),
    ]
    v = recommend_workers(rows)
    assert v["workers"] is None
    assert "re-probe" in v["reason"]


def test_high_variance_between_passes_also_refuses():
    from vhagar.archive.backfill import recommend_workers

    rows = [_row(n, 10.0 + n, spread=0.6) for n in (1, 4, 8, 16)]
    assert recommend_workers(rows)["workers"] is None


def test_no_measurements_is_reported_not_crashed():
    from vhagar.archive.backfill import recommend_workers

    assert recommend_workers([])["workers"] is None


def test_a_single_slow_noisy_row_does_not_veto_a_clean_curve():
    """One flaky setting should not throw away four good ones."""
    from vhagar.archive.backfill import recommend_workers

    rows = [_row(1, 5.0), _row(4, 15.0), _row(8, 19.5), _row(16, 20.0, spread=0.6)]
    assert recommend_workers(rows)["workers"] == 8

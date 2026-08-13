"""Tier B climatology backfill: reduction, checkpointing, resume, coverage.

The properties that matter mirror Tier A: a resume must not re-fold what the
checkpoint already contains, the cadence subsample is honoured, a failed stack is
recorded not fatal, and coverage is written. Network reads are stubbed, so this
is offline; the real grouping, subsampling and Welford reduction run for real.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

pytest.importorskip("xarray")

from _fixtures import _synthetic_cmip  # noqa: E402

import vhagar.io.cmip_reader as cmip  # noqa: E402
from vhagar.archive.climatology import DiurnalClimatology  # noqa: E402
from vhagar.archive.climatology_backfill import (  # noqa: E402
    CHECKPOINT_NAME,
    ClimatologyBackfillConfig,
    _cadence_subsample,
    backfill_climatology,
    climatology_coverage,
)
from vhagar.io.cmip_reader import decode_cmip, stack_channels  # noqa: E402
from vhagar.io.goes import parse_goes_key  # noqa: E402

T0 = datetime(2026, 8, 3, 12, 0, 0, tzinfo=UTC)
BANDS = ("C07", "C14")
BBOX = (-124.0, 36.0, -118.0, 42.0)


def _cmip_key(channel: str, when: datetime) -> str:
    doy = when.timetuple().tm_yday
    stamp = f"{when.year:04d}{doy:03d}{when.hour:02d}{when.minute:02d}{when.second:02d}0"
    name = f"OR_ABI-L2-CMIPC-M6{channel}_G18_s{stamp}_e{stamp}_c{stamp}.nc"
    return f"ABI-L2-CMIPC/{when.year}/{doy:03d}/{when.hour:02d}/{name}"


def _install_stubs(monkeypatch, n_steps: int, step_min: int = 5, fail_gid: str | None = None):
    """Fabricate 5-minute keys for each band and stub the two network calls."""
    all_keys = {
        b: [_cmip_key(b, T0 + timedelta(minutes=step_min * i)) for i in range(n_steps)]
        for b in BANDS
    }

    def fake_list(satellite, start, end, channel, domain="C", anon=True):
        return [
            k for k in all_keys[channel] if start <= parse_goes_key(k, satellite).start <= end
        ]

    def fake_open_stack(group, satellite, bbox=None, anon=True):
        gid = group["C07"]
        if fail_gid is not None and gid == fail_gid:
            raise OSError("simulated S3 hiccup")
        chans = [
            decode_cmip(_synthetic_cmip(n=30, channel=b), satellite=satellite, channel=b)
            for b in group
        ]
        stack = stack_channels(chans)
        stack.scan_start = parse_goes_key(gid, satellite).start
        return stack

    monkeypatch.setattr(cmip, "list_cmip_granules", fake_list)
    monkeypatch.setattr(cmip, "open_cmip_stack", fake_open_stack)
    return all_keys


def _cfg(tmp_path, **kw):
    base = dict(
        out_dir=tmp_path, start=T0, end=T0 + timedelta(hours=3), bbox=BBOX, channels=BANDS
    )
    base.update(kw)
    return ClimatologyBackfillConfig(**base)


# ------------------------------------------------------- subsampling ------


def test_cadence_subsample_keeps_one_stack_per_bucket():
    groups = [
        {"C07": _cmip_key("C07", T0 + timedelta(minutes=5 * i)),
         "C14": _cmip_key("C14", T0 + timedelta(minutes=5 * i))}
        for i in range(12)  # one hour at 5-minute spacing
    ]
    kept = _cadence_subsample(groups, "C07", satellite=18, cadence_min=15)
    assert len(kept) == 4  # 60 minutes / 15


# ----------------------------------------------------------- reduce -------


def test_backfill_reduces_and_writes_a_checkpoint(tmp_path, monkeypatch):
    _install_stubs(monkeypatch, n_steps=24)  # two hours at 5 min
    result = backfill_climatology(_cfg(tmp_path))
    # two hours at 15-minute cadence is 8 timesteps
    assert result.frames_ok == 8
    assert result.frames_failed == 0

    checkpoint = tmp_path / CHECKPOINT_NAME
    assert checkpoint.exists()
    clim = DiurnalClimatology.load(checkpoint)
    # the 12:00-14:00 window folds into bins 12 and 13
    assert clim.count("C07")[12][0, 0] > 0
    assert clim.count("C07")[13][0, 0] > 0
    assert np.all(clim.count("C07")[3] == 0)


def test_resume_folds_nothing_new(tmp_path, monkeypatch):
    _install_stubs(monkeypatch, n_steps=24)
    first = backfill_climatology(_cfg(tmp_path))
    assert first.frames_ok == 8

    second = backfill_climatology(_cfg(tmp_path))
    assert second.frames_ok == 0
    assert second.frames_skipped == 8


def test_resume_is_numerically_identical_to_one_pass(tmp_path, monkeypatch):
    """Splitting the window across two runs must give the same statistics as one,
    the whole point of the atomic checkpoint plus watermark."""
    _install_stubs(monkeypatch, n_steps=24)

    whole = backfill_climatology(_cfg(tmp_path / "whole"))
    ref = DiurnalClimatology.load(tmp_path / "whole" / CHECKPOINT_NAME)
    assert whole.frames_ok == 8

    part = tmp_path / "part"
    backfill_climatology(_cfg(part, end=T0 + timedelta(minutes=59)))   # first hour
    backfill_climatology(_cfg(part, end=T0 + timedelta(hours=3)))      # resume the rest
    got = DiurnalClimatology.load(part / CHECKPOINT_NAME)

    for b in BANDS:
        assert np.allclose(got.count(b), ref.count(b))
        assert np.allclose(got.mean(b), ref.mean(b), equal_nan=True)
        assert np.allclose(got.std(b), ref.std(b), equal_nan=True)


def test_a_failed_stack_is_recorded_not_fatal(tmp_path, monkeypatch):
    keys = _install_stubs(monkeypatch, n_steps=24)
    # fail the first kept timestep
    fail_gid = keys["C07"][0]
    _install_stubs(monkeypatch, n_steps=24, fail_gid=fail_gid)
    result = backfill_climatology(_cfg(tmp_path))
    assert result.frames_failed == 1
    assert result.frames_ok == 7
    assert result.errors.get("OSError") == 1


def test_incompatible_config_is_refused(tmp_path, monkeypatch):
    _install_stubs(monkeypatch, n_steps=6)
    backfill_climatology(_cfg(tmp_path))
    with pytest.raises(ValueError, match="different settings"):
        backfill_climatology(_cfg(tmp_path, bbox=(-100.0, 30.0, -90.0, 40.0)))


def test_coverage_is_recorded(tmp_path, monkeypatch):
    _install_stubs(monkeypatch, n_steps=24)
    backfill_climatology(_cfg(tmp_path))
    intervals = climatology_coverage(tmp_path)
    assert len(intervals) == 1
    start, end = intervals[0]
    assert start.hour == 12 and end.hour == 13

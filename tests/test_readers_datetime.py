"""GOES/CMIP S3 key prefixes must treat a naive datetime as UTC (not local) and
convert an aware one to UTC, so the day-of-year and hour folders are correct."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from vhagar.io.cmip_reader import cmip_key_prefix
from vhagar.io.goes import fdc_key_prefix

_SAT = 19


def test_naive_datetime_is_treated_as_utc():
    naive = datetime(2026, 8, 26, 18, 30)                       # no tz
    aware = datetime(2026, 8, 26, 18, 30, tzinfo=timezone.utc)  # explicit UTC
    assert fdc_key_prefix(_SAT, naive) == fdc_key_prefix(_SAT, aware)
    assert cmip_key_prefix(_SAT, naive) == cmip_key_prefix(_SAT, aware)
    # 2026-08-26 is day-of-year 238; hour 18
    assert fdc_key_prefix(_SAT, naive).endswith("/2026/238/18/")
    assert cmip_key_prefix(_SAT, naive).endswith("/2026/238/18/")


def test_aware_datetime_is_converted_to_utc():
    # 20:30 at +05:00 is 15:30 UTC on the same day
    local = datetime(2026, 8, 26, 20, 30, tzinfo=timezone(timedelta(hours=5)))
    assert fdc_key_prefix(_SAT, local).endswith("/238/15/")
    assert cmip_key_prefix(_SAT, local).endswith("/238/15/")

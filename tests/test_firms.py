"""FIRMS CSV parsing: dropped-row accounting (bad coordinates or times are
counted and reported, not silently swallowed)."""
from __future__ import annotations

import logging

from vhagar.io.firms import parse_firms_csv

_HEADER = ("latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,"
           "satellite,instrument,confidence,version,bright_ti5,frp,daynight")
_GOOD = "34.0,-118.0,320,0.5,0.5,2026-08-26,1830,N,VIIRS,n,2.0,300,12.5,D"


def test_parse_firms_csv_counts_and_reports_dropped_rows(caplog):
    bad_coord = "x,-118.0,320,0.5,0.5,2026-08-26,1830,N,VIIRS,n,2.0,300,12.5,D"
    bad_time = "34.0,-118.0,320,0.5,0.5,2026-08-26,badtime,N,VIIRS,n,2.0,300,12.5,D"
    text = "\n".join([_HEADER, _GOOD, bad_coord, bad_time])
    with caplog.at_level(logging.WARNING):
        recs = parse_firms_csv(text)
    assert len(recs) == 1                       # only the good row survives
    assert any("dropped 2 of 3" in m for m in caplog.messages)


def test_parse_firms_csv_silent_when_all_rows_parse(caplog):
    text = "\n".join([_HEADER, _GOOD, _GOOD])
    with caplog.at_level(logging.WARNING):
        recs = parse_firms_csv(text)
    assert len(recs) == 2
    assert not caplog.messages                  # no warning when nothing is dropped

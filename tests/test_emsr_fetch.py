"""EMSR batch discovery/ingest: centroid parsing, Koppen labels, manifest build.

All offline. The network path (list_wildfire_candidates) is exercised only through
its pure helpers; the HTTP call itself is not unit-tested here.
"""

from __future__ import annotations

import pytest

from vhagar.labels.emsr_fetch import (
    ingest_delineations,
    koppen_name,
    parse_centroid,
    write_manifest_csv,
)


def test_parse_centroid_reads_lon_lat_in_order():
    assert parse_centroid("POINT (23.44 40.02)") == (23.44, 40.02)
    assert parse_centroid("POINT (-6.69 37.48)") == (-6.69, 37.48)


def test_parse_centroid_rejects_garbage():
    with pytest.raises(ValueError, match="cannot parse"):
        parse_centroid("not a point")


def test_koppen_name_maps_known_and_unknown():
    assert koppen_name(8) == "Csa"      # Mediterranean, matches Greece + California
    assert koppen_name(26) == "Dfb"
    assert koppen_name(0) == "?"        # nodata / ocean
    assert koppen_name(None) == "?"


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")


def test_ingest_prefers_latest_monitoring_step(tmp_path):
    # one AOI mapped twice: MONIT01 then MONIT02; the later one must win
    _touch(tmp_path / "EMSR900_AOI01_DEL_MONIT01_r1/EMSR900_AOI01_DEL_MONIT01_observedEventA_r1.shp")
    _touch(tmp_path / "EMSR900_AOI01_DEL_MONIT02_r1/EMSR900_AOI01_DEL_MONIT02_observedEventA_r1.shp")
    # a second AOI, and a non-burnt layer that must be ignored
    _touch(tmp_path / "EMSR900_AOI02_DEL_MONIT01_r1/EMSR900_AOI02_DEL_MONIT01_observedEventA_r1.shp")
    _touch(tmp_path / "EMSR900_AOI01_DEL_MONIT02_r1/EMSR900_AOI01_DEL_MONIT02_builtUpA_r1.shp")

    rows = ingest_delineations(tmp_path, dates={"EMSR900": "2024-07-01"})
    ids = {r["activation_id"]: r for r in rows}
    assert set(ids) == {"EMSR900_AOI01", "EMSR900_AOI02"}
    assert "MONIT02" in ids["EMSR900_AOI01"]["delineation_path"]   # latest wins
    assert "builtUp" not in ids["EMSR900_AOI01"]["delineation_path"]
    assert ids["EMSR900_AOI01"]["event_date"] == "2024-07-01"


def test_ingest_marks_missing_dates_empty(tmp_path):
    _touch(tmp_path / "EMSR901_AOI01_DEL_MONIT01_r1/EMSR901_AOI01_DEL_MONIT01_observedEventA_r1.shp")
    rows = ingest_delineations(tmp_path)          # no dates map
    assert rows[0]["event_date"] == ""


def test_write_manifest_round_trips(tmp_path):
    rows = [{"activation_id": "EMSR900_AOI01", "delineation_path": "/x/a.shp",
             "event_date": "2024-07-01"}]
    out = write_manifest_csv(rows, tmp_path / "emsr.csv")
    text = out.read_text()
    assert "activation_id,delineation_path,event_date" in text
    assert "EMSR900_AOI01,/x/a.shp,2024-07-01" in text

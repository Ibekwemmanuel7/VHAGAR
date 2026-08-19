"""Label spine: MTBS normalisation, tile assignment, registry persistence, splits.

The normalisation and tile logic are pure and tested here on synthetic rows; the
real file reads (pyogrio) are exercised by the user pointing the CLI at a
downloaded MTBS extract.
"""

from __future__ import annotations

from datetime import date

import pytest

pytest.importorskip("pyproj")  # geo CRS stack (geo extra); skip in a minimal env

from vhagar.grid import AnalysisGrid  # noqa: E402
from vhagar.labels.ingest import normalize_mtbs  # noqa: E402
from vhagar.labels.registry import EventRegistry, LabelQuality, LabelSource  # noqa: E402
from vhagar.labels.tiles import assign_tiles  # noqa: E402


def _mtbs_rows():
    return [
        {
            "Event_ID": "CA3980612050820200815",
            "BurnBndLon": -120.5,
            "BurnBndLat": 39.8,
            "BurnBndAc": 2470.0,
            "Ig_Date": "2020-08-15",
            "Incid_Type": "Wildfire",
        },
        {
            "Event_ID": "TX3110009850820190310",
            "BurnBndLon": -98.2,
            "BurnBndLat": 31.1,
            "BurnBndAc": 500.0,
            "Ig_Date": "2019/03/10",
            "Incid_Type": "Prescribed Fire",
        },
    ]


# ------------------------------------------------------ MTBS ingest -------


def test_mtbs_normalisation_maps_fields_and_quality():
    recs = normalize_mtbs(_mtbs_rows(), severity_dir="s3://sev", geometry_dir="s3://geom")
    assert len(recs) == 2
    a = recs[0]
    assert a.event_id == "mtbs:CA3980612050820200815"
    assert a.source is LabelSource.MTBS
    assert a.quality is LabelQuality.ANALYST_QC
    assert a.region == "conus"
    assert a.ignition_date == date(2020, 8, 15)
    assert a.area_ha == pytest.approx(2470.0 * 0.404686, rel=1e-6)
    assert a.fire_type == "wildland"
    assert recs[1].fire_type == "prescribed"
    assert recs[1].ignition_date == date(2019, 3, 10)  # slash format parsed too


def test_mtbs_with_severity_is_trainable():
    """MTBS carries the dNBR severity raster, which IS the interior mask, so a
    pixel model may train on it."""
    rec = normalize_mtbs(_mtbs_rows(), severity_dir="s3://sev", geometry_dir="s3://geom")[0]
    assert rec.has_interior_mask
    assert rec.severity_path.endswith("_dnbr.tif")
    rec.assert_trainable()  # must not raise


def test_mtbs_is_not_evaluation_only():
    rec = normalize_mtbs(_mtbs_rows())[0]
    assert not rec.is_evaluation_only


def test_rows_missing_id_or_point_are_skipped():
    rows = [{"BurnBndLon": -120.0, "BurnBndLat": 39.0}, {"Event_ID": "x"}]
    assert normalize_mtbs(rows) == []


# ------------------------------------------------ tile assignment ---------


def test_assign_tiles_places_a_conus_point_on_the_grid():
    rec = normalize_mtbs(_mtbs_rows())[0]
    tiles = assign_tiles(rec)
    assert len(tiles) == 1
    assert tiles[0].startswith("conus/x")


def test_a_point_outside_the_region_gets_no_tiles():
    rec = normalize_mtbs(_mtbs_rows())[0]
    rec.lon, rec.lat = 10.0, 5.0  # off West Africa, nowhere near CONUS
    assert assign_tiles(rec) == []


def test_bbox_covers_more_tiles_than_a_point():
    rec = normalize_mtbs(_mtbs_rows())[0]
    point = assign_tiles(rec)
    box = assign_tiles(rec, bbox_4326=(-121.5, 38.8, -119.5, 40.8))  # ~2 degrees
    assert len(box) > len(point)
    assert all(t.startswith("conus/x") for t in box)


# ------------------------------------------------ registry round-trip -----


def test_registry_parquet_round_trip(tmp_path):
    pytest.importorskip("pyarrow")
    recs = normalize_mtbs(_mtbs_rows(), severity_dir="s3://sev", geometry_dir="s3://geom")
    grid = AnalysisGrid("conus")
    for r in recs:
        r.tile_ids = assign_tiles(r, grid)
    reg = EventRegistry(recs)
    path = reg.to_parquet(tmp_path / "registry.parquet")

    back = EventRegistry.from_parquet(path)
    assert len(back) == len(reg)
    by_id = {r.event_id: r for r in back}
    a = by_id["mtbs:CA3980612050820200815"]
    assert a.source is LabelSource.MTBS
    assert a.ignition_date == date(2020, 8, 15)
    assert a.area_ha == pytest.approx(2470.0 * 0.404686, rel=1e-6)
    assert a.tile_ids and a.tile_ids[0].startswith("conus/x")
    assert a.severity_path.endswith("_dnbr.tif")


def test_registry_summary_counts_by_region_and_source():
    reg = EventRegistry(normalize_mtbs(_mtbs_rows()))
    assert reg.summary() == {"conus/mtbs": 2}


# ------------------------------------------------ registry -> splits -------


def test_registry_feeds_leakage_proof_splits():
    from vhagar.eval.splits import leave_year_out, spatial_block_split, verify_no_overlap

    rows = []
    for i in range(12):
        rows.append(
            {
                "Event_ID": f"F{i}",
                "BurnBndLon": -120.0 + i * 0.5,
                "BurnBndLat": 39.0 + (i % 3) * 0.5,
                "BurnBndAc": 1000.0,
                "Ig_Date": f"20{15 + i % 4}-07-0{1 + i % 8}",
                "Incid_Type": "Wildfire",
            }
        )
    reg = EventRegistry(normalize_mtbs(rows))
    units = reg.to_split_units()
    assert len(units) == 12

    verify_no_overlap(spatial_block_split(units, n_folds=3, block_degrees=5.0))
    verify_no_overlap(leave_year_out(units))

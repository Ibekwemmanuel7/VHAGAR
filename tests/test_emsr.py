"""Copernicus EMS ingest and rasterised reference, for leave-one-continent-out.

The geometry maths (record from polygons, rasterise onto a grid) is tested for
real with synthetic shapely polygons; the shapefile read stays at the edge.
"""

from __future__ import annotations

from datetime import date

import pytest

shapely = pytest.importorskip("shapely")
from shapely.geometry import box  # noqa: E402

from vhagar.datasets.t2_optical import rasterize_burned_on_grid  # noqa: E402
from vhagar.io.optical import TargetGrid  # noqa: E402
from vhagar.labels.ingest import build_emsr_record  # noqa: E402
from vhagar.labels.registry import LabelSource  # noqa: E402


def test_build_emsr_record_from_polygons_in_3035():
    # a 2 km x 2 km burned box in EPSG:3035 metres, near central Europe
    poly = box(4_300_000, 2_900_000, 4_302_000, 2_902_000)
    rec = build_emsr_record("EMSR999", "2021-07-20", [poly], "EPSG:3035", "d.shp")
    assert rec.event_id == "emsr:EMSR999"
    assert rec.source is LabelSource.COPERNICUS_EMS
    assert rec.region == "europe"
    assert rec.is_evaluation_only          # held out, never trained on
    assert rec.ignition_date == date(2021, 7, 20)
    assert rec.area_ha == pytest.approx(4 * 100, rel=1e-3)   # 2km x 2km = 400 ha
    assert rec.attributes["delineation_path"] == "d.shp"
    assert 40 < rec.lat < 60 and -10 < rec.lon < 30          # plausibly in Europe


def test_build_emsr_record_rejects_empty():
    with pytest.raises(ValueError, match="no burnt-area"):
        build_emsr_record("EMSR000", "2021-01-01", [], "EPSG:3035", "d.shp")


def test_rasterize_burned_marks_inside_the_polygon():
    # a target grid: 10 x 10 cells of 100 m in EPSG:3035, origin at a round point
    x0, y1 = 4_300_000.0, 2_901_000.0
    grid = TargetGrid(crs="EPSG:3035", transform=(100.0, 0, x0, 0, -100.0, y1),
                      width=10, height=10)
    # burn a 300 m box in the top-left of the window (same CRS, no reprojection)
    poly = box(x0, y1 - 300, x0 + 300, y1)
    burned, valid = rasterize_burned_on_grid([poly], "EPSG:3035", grid)
    assert burned.shape == (10, 10)
    assert valid.all()                        # a delineation labels every pixel
    assert burned[:3, :3].all()               # the 3x3 top-left cells are burned
    assert not burned[5:, 5:].any()           # far corner is unburned
    assert 0 < burned.mean() < 1


def test_rasterize_reprojects_before_burning():
    # polygon given in WGS84 must land in the right place on a 3035 grid
    grid = TargetGrid(crs="EPSG:3035", transform=(100.0, 0, 4_300_000.0, 0, -100.0, 2_901_000.0),
                      width=50, height=50)
    # a lon/lat box roughly over the grid centre
    from pyproj import Transformer
    inv = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lon, lat = inv.transform(4_302_500, 2_898_500)
    poly = box(lon - 0.02, lat - 0.02, lon + 0.02, lat + 0.02)
    burned, _ = rasterize_burned_on_grid([poly], "EPSG:4326", grid)
    assert burned.any()                        # it burned somewhere, not empty
    assert burned.sum() < burned.size          # and not everything

"""Tests for the real-data parcel study code (src/vhagar/parcel/study).

Each test pins a property with a known answer (a tilted plane's slope, a constant score
tracing the diagonal, a facility never straddling a spatial split), on small synthetic
inputs. Skipped when the geo/ML extras are not installed.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")
rasterio = pytest.importorskip("rasterio")

from shapely.geometry import box  # noqa: E402

from vhagar.parcel.study import features as F  # noqa: E402
from vhagar.parcel.study import modeling as M  # noqa: E402
from vhagar.parcel.study.population import label_overlaps  # noqa: E402
from vhagar.parcel.study.sampling import dem_tile_name, draw_sample, fetch_plan  # noqa: E402

# ---------------------------------------------------------------- metrics


def test_constant_score_traces_the_diagonal():
    rng = np.random.default_rng(0)
    n = 500
    y = (rng.random(n) < 0.1).astype(int)
    w = np.where(y == 1, 1.0, 20.0)
    area = rng.uniform(1, 100, n)
    fr = np.array([0.0, 0.1, 0.5, 1.0])
    assert np.allclose(M.recall_curve(y, np.zeros(n), w, area, fr), fr)
    assert M.recall_at_area(y, np.zeros(n), w, area)["recall_top10pct_area"] == \
        pytest.approx(0.10)


def test_perfect_score_captures_cases_first():
    y = np.array([1, 1, 0, 0, 0, 0])
    area = np.array([1, 1, 1, 1, 1, 1.0])
    w = np.ones(6)
    r = M.recall_curve(y, y.astype(float), w, area, np.array([2 / 6, 0.5]))
    assert r[0] == pytest.approx(1.0) and r[1] == pytest.approx(1.0)


def test_missing_scores_rank_last():
    y = np.array([1, 0, 0, 0])
    s = np.array([np.nan, 1.0, 0.5, 0.2])
    r = M.recall_curve(y, s, np.ones(4), np.ones(4), np.array([0.75, 1.0]))
    assert r[0] == pytest.approx(0.0) and r[1] == pytest.approx(1.0)


def test_control_weights_lower_precision_to_population_scale():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.2, 0.5, 0.1])
    unweighted = M.weighted_metrics(y, s, np.ones(4), np.ones(4))
    weighted = M.weighted_metrics(y, s, np.array([1, 1, 100, 100.0]), np.ones(4))
    assert weighted["population_prevalence"] < unweighted["population_prevalence"]
    assert weighted["average_precision"] < unweighted["average_precision"]
    assert weighted["roc_auc"] == pytest.approx(unweighted["roc_auc"])


def test_prior_correction_identity_and_direction():
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(M.prior_correct(p, 0.2, 0.2), p)
    assert (M.prior_correct(p, 0.2, 0.001) < p).all()


def test_reliability_of_a_calibrated_forecast_is_good():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 20000)
    y = (rng.random(20000) < p).astype(int)
    r = M.reliability(y, p, np.ones_like(p))
    assert r["ece"] < 0.02 and r["brier_skill"] > 0.2


# ---------------------------------------------------------------- splits


def _study_frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.15).astype(int)
    fac = np.where(y == 1, rng.integers(0, 20, n), -1)
    fac_lon = -106 + (fac % 5) * 2.5
    fac_lat = 27 + (fac // 5) * 2.0
    lon = rng.uniform(-106, -94, n)
    lat = rng.uniform(26, 36, n)
    return pd.DataFrame({
        "y": y, "case_facility": fac, "case_year": np.where(y == 1, rng.integers(2017, 2026, n), -1),
        "control_half": np.where(y == 0, rng.integers(0, 2, n), -1),
        "block_lon": np.where(y == 1, fac_lon, lon), "block_lat": np.where(y == 1, fac_lat, lat),
        "lon": lon, "lat": lat})


def test_temporal_split_is_prospective_with_disjoint_control_pools():
    df = _study_frame()
    sp = M.temporal_split(df, 2020, noncase_population=10_000)
    tr, te = df.iloc[sp.train], df.iloc[sp.test]
    assert set(sp.train).isdisjoint(sp.test)
    assert (tr.loc[tr.y == 1, "case_year"] <= 2020).all()
    assert (te.loc[te.y == 1, "case_year"] > 2020).all()
    assert set(tr.loc[tr.y == 0, "control_half"]) == {0}
    assert set(te.loc[te.y == 0, "control_half"]) == {1}
    w = sp.test_weight
    assert np.allclose(w[te.y.to_numpy() == 1], 1.0)
    assert w[te.y.to_numpy() == 0].sum() == pytest.approx(10_000)


def test_spatial_folds_never_split_a_facility():
    df = _study_frame()
    folds = M.spatial_splits(df, n_folds=4, block_degrees=2.0, seed=0, noncase_population=5000)
    seen_test = np.zeros(len(df), int)
    for f in folds:
        seen_test[f.test] += 1
        tr_fac = set(df.iloc[f.train].query("y == 1")["case_facility"])
        te_fac = set(df.iloc[f.test].query("y == 1")["case_facility"])
        assert tr_fac.isdisjoint(te_fac)
    assert (seen_test == 1).all()  # every row is held out exactly once


def test_cluster_bootstrap_interval_brackets_point_estimate():
    rng = np.random.default_rng(2)
    n = 800
    y = (rng.random(n) < 0.1).astype(int)
    s = y + rng.normal(0, 0.8, n)
    groups = np.where(y == 1, rng.integers(0, 30, n), -1)
    w, area = np.ones(n), np.ones(n)
    point = M.weighted_metrics(y, s, w, area)["average_precision"]
    ci = M.cluster_bootstrap(y, groups, w, area, {"m": s, "null": np.zeros(n)},
                             "average_precision", n_boot=60, seed=0, reference="null")
    assert ci["m"]["lo"] <= point <= ci["m"]["hi"]
    assert ci["m"]["diff_vs_null"]["lo"] > 0


def test_rules_engine_adapter_handles_exclusion_and_missing():
    base = dict(slope_mean_pct=2.0, ghi_kwh_m2_day=6.0, dist_tx_robust_km=3.0,
                dist_road_km=1.0, whp_pct=20.0, floodplain_frac=0.0,
                protected_overlap_frac=0.0, area_ha=50.0)
    df = pd.DataFrame([base, dict(base, protected_overlap_frac=0.9),
                       dict(base, floodplain_frac=np.nan), dict(base, area_ha=1.0)])
    s, st = M.rules_engine_scores(df)
    assert st.tolist() == ["scored", "ineligible", "insufficient_evidence", "ineligible"]
    assert s[0] > 50 and s[1] == 0.0 and np.isnan(s[2]) and s[3] == 0.0


# ---------------------------------------------------------------- labels + sampling


def test_label_overlaps_split_by_period():
    parcels = gpd.GeoSeries([box(0, 0, 1000, 1000), box(2000, 0, 3000, 1000),
                             box(5000, 0, 6000, 1000)], crs="EPSG:5070")
    fac = gpd.GeoDataFrame({"case_id": [1, 2, 3], "p_year": [2015, 2022, 2019]},
                           geometry=[box(0, 0, 500, 1000), box(2000, 0, 2800, 1000),
                                     box(2900, 0, 3000, 1000)], crs="EPSG:5070")
    lab = label_overlaps(parcels, fac)
    assert lab.loc[0, "overlap_pre_ha"] == pytest.approx(50.0)
    assert lab.loc[1, "overlap_case_ha"] == pytest.approx(90.0)
    assert lab.loc[1, "case_year"] == 2022 and lab.loc[1, "case_facility"] == 2  # largest
    assert lab.loc[2, "overlap_case_ha"] == 0 and lab.loc[2, "case_year"] == -1


def test_draw_sample_is_reproducible_and_weighted():
    rng = np.random.default_rng(3)
    n = 5000
    pop = pd.DataFrame({"pid": [f"p{i}" for i in range(n)], "is_case": rng.random(n) < 0.02,
                        "overlap_case_ha": 0.0, "case_facility": -1})
    pop.loc[pop.is_case, "overlap_case_ha"] = 5.0
    pop.loc[pop.is_case, "case_facility"] = rng.integers(0, 10, int(pop.is_case.sum()))
    pop.loc[10:19, "overlap_case_ha"] = 0.5  # ambiguous slivers (non-case)
    pop.loc[10:19, "is_case"] = False
    a, rec = draw_sample(pop, n_controls=400, seed=7)
    b, _ = draw_sample(pop, n_controls=400, seed=7)
    assert a["pid"].tolist() == b["pid"].tolist()
    assert rec["ambiguous_excluded"] == 10
    ctrl = a[a.role == "control"]
    assert len(ctrl) == 400 and not ctrl["pid"].isin(pop.loc[10:19, "pid"]).any()
    assert ctrl["weight"].sum() == pytest.approx(rec["eligible_noncase"])
    assert set(ctrl["control_half"]) == {0, 1}


def test_dem_tile_name_and_fetch_plan():
    assert dem_tile_name(-102.5, 31.5) == "n32w103"
    assert dem_tile_name(-97.01, 30.99) == "n31w098"
    g = gpd.GeoDataFrame(geometry=gpd.GeoSeries([box(-102.6, 31.6, -102.4, 31.8)],
                                                crs="EPSG:4326").to_crs("EPSG:5070"),
                         crs="EPSG:5070")
    plan = fetch_plan(g, cell_deg=0.25)
    assert plan["dem_tiles"] == ["n32w103"]
    assert [-102.75, 31.5, -102.5, 31.75] in plan["nfhl_cells"]


# ---------------------------------------------------------------- features


def _write_tif(path, arr, transform, crs, nodata=None):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype=arr.dtype, crs=crs, transform=transform,
                       nodata=nodata) as dst:
        dst.write(arr, 1)


def test_slope_of_a_tilted_plane(tmp_path):
    res = 1 / 3600
    lat0 = 31.0
    dx = res * 111_320.0 * math.cos(math.radians(lat0 + 0.05))
    cols, rows = 400, 400
    x = np.arange(cols) * dx
    arr = np.tile(0.05 * x, (rows, 1)).astype("float32")  # 5% rise eastward
    tr = rasterio.transform.from_origin(-102.0, lat0 + 0.1, res, res)
    p = tmp_path / "n32w103.tif"
    _write_tif(p, arr, tr, "EPSG:4326")
    geom = box(-101.98, lat0 + 0.03, -101.95, lat0 + 0.07)
    with rasterio.open(p) as src:
        st = F.slope_stats(geom, ["n32w103"], lambda t: src)
    assert st["slope_mean_pct"] == pytest.approx(5.0, rel=0.05)


def test_zonal_classes_and_mean(tmp_path):
    arr = np.array([[21, 21, 82, 82], [21, 21, 82, 82], [11, 11, 52, 52],
                    [11, 11, 52, 52]], dtype="uint8")
    tr = rasterio.transform.from_origin(0, 120, 30, 30)
    p = tmp_path / "lc.tif"
    _write_tif(p, arr, tr, "EPSG:5070", nodata=250)
    with rasterio.open(p) as src:
        fr = F.zonal_classes(src, box(0, 0, 120, 120), F.NLCD_GROUPS)
        mean = F.zonal_mean(src, box(0, 60, 60, 120))
    assert fr["nlcd_developed_frac"] == pytest.approx(0.25)
    assert fr["nlcd_cropland_frac"] == pytest.approx(0.25)
    assert fr["nlcd_wetland_water_frac"] == pytest.approx(0.25)
    assert fr["nlcd_shrub_grass_frac"] == pytest.approx(0.25)
    assert mean == pytest.approx(21.0)


def test_overlap_fraction_does_not_double_count_overlapping_polygons():
    g = np.array([box(0, 0, 100, 100)])
    polys = np.array([box(0, 0, 50, 100), box(0, 0, 50, 100), box(25, 0, 75, 100)])
    assert F.overlap_fraction(g, polys)[0] == pytest.approx(0.75)


def test_nearest_distance_km():
    g = np.array([box(0, 0, 10, 10), box(5000, 0, 5010, 10)])
    lines = np.array([shapely.LineString([(1010, -100), (1010, 100)])])
    d = F.nearest_distance_km(g, lines)
    assert d[0] == pytest.approx(1.0) and d[1] == pytest.approx(3.99)


def test_interp_grid_is_bilinear():
    pts = [{"lon": lo, "lat": la, "v": lo + 2 * la} for lo in (0.0, 1.0) for la in (0.0, 1.0)]
    out = F.interp_grid(pts, "v", np.array([0.5, 2.0]), np.array([0.5, 0.5]))
    assert out[0] == pytest.approx(1.5) and math.isnan(out[1])


def test_every_model_feature_has_a_spec_with_leakage_rating():
    for f in M.FEATURES + ["dist_tx_km"]:
        assert f in F.FEATURE_SPECS, f
        assert F.FEATURE_SPECS[f].leakage in {"none", "low", "moderate", "high"}

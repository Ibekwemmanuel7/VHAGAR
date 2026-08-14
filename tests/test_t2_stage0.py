"""T2 Stage-0: burned-area sample building and the calibrated-threshold driver.

Synthetic fires with a clean dNBR separation, so a calibrated threshold should
recover the burned mask and the Olofsson-adjusted area should land near truth.
Everything offline.
"""

from __future__ import annotations

import numpy as np
import pytest

from vhagar.datasets.burned_area import T2Sample, make_sample, mtbs_burned_mask
from vhagar.eval.splits import SplitManifest
from vhagar.eval.t2_stage0 import evaluate_fold, run_stage0, summarise_stage0

PIXEL_HA = 0.09


# --------------------------------------------------- dataset builder ------


def test_mtbs_burned_mask_maps_classes():
    sev = np.array([[0, 1, 2], [3, 4, 5], [6, 2, 1]])
    burned, valid = mtbs_burned_mask(sev)
    assert burned.tolist() == [[False, False, True], [True, True, False], [False, True, False]]
    # 0 and 6 are not mapped; 1..5 are valid
    assert valid.tolist() == [[False, True, True], [True, True, True], [False, True, True]]


def test_make_sample_propagates_nodata_into_valid():
    predictor = np.array([[100.0, np.nan], [200.0, 300.0]])
    reference = np.array([[True, True], [False, True]])
    ref_valid = np.array([[True, True], [False, True]])  # bottom-left is unmapped
    s = make_sample("f", predictor, reference, reference_valid=ref_valid)
    # NaN predictor (top-right) and unmapped reference (bottom-left) are excluded
    assert s.valid.tolist() == [[True, False], [False, True]]
    assert s.n_valid == 2


def test_burned_fraction_is_over_valid_only():
    predictor = np.array([[1.0, 1.0], [1.0, np.nan]])
    reference = np.array([[True, False], [True, True]])
    s = make_sample("f", predictor, reference)
    # 3 valid pixels (NaN excluded), 2 of them burned
    assert s.n_valid == 3
    assert s.burned_fraction == pytest.approx(2 / 3)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="does not match"):
        make_sample("f", np.zeros((2, 2)), np.zeros((2, 3), dtype=bool))


def test_is_usable_flags_degenerate_samples():
    good = make_sample("g", np.array([[1.0, 2.0]]), np.array([[True, False]]))
    all_burned = make_sample("b", np.array([[1.0, 2.0]]), np.array([[True, True]]))
    all_cloud = make_sample("c", np.array([[np.nan, np.nan]]), np.array([[True, False]]))
    assert good.is_usable
    assert not all_burned.is_usable      # single class, no threshold to fit
    assert not all_cloud.is_usable       # no valid predictor pixels


def test_sample_save_load_round_trip(tmp_path):
    s = make_sample(
        "mtbs:CA1", np.array([[1.0, np.nan], [3.0, 4.0]]),
        np.array([[True, False], [True, False]]), tile_id="conus/x0001_y0002",
    )
    path = s.save(tmp_path / "s.npz")
    back = T2Sample.load(path)
    assert back.event_id == "mtbs:CA1"
    assert back.tile_id == "conus/x0001_y0002"
    assert np.array_equal(back.valid, s.valid)
    assert np.array_equal(back.reference, s.reference)
    assert back.n_valid == s.n_valid


# ----------------------------------------------------------- driver -------


def _fire(event_id, burned_frac=0.3, n=100, seed=0):
    """A synthetic fire: burned pixels have high dNBR, unburned low, with a
    slight overlap so classification is strong but not perfect (a realistic,
    nonzero-uncertainty case)."""
    rng = np.random.default_rng(seed)
    truth = rng.random((n, n)) < burned_frac
    dnbr = np.where(truth, rng.normal(320, 60, (n, n)), rng.normal(140, 60, (n, n)))
    return make_sample(event_id, dnbr, truth, tile_id="conus/x0001_y0001")


def _manifest(train_ids, test_ids, held="A"):
    return SplitManifest(
        scheme="test",
        folds=[{"train": list(train_ids), "test": list(test_ids), "held_out": held}],
    )


def test_calibrated_threshold_recovers_a_separable_fire():
    train = [_fire(f"tr{i}", seed=i) for i in range(3)]
    test = [_fire(f"te{i}", seed=100 + i) for i in range(2)]
    r = evaluate_fold(train, test, held_out="year", seed=0)
    assert r.f1 > 0.85            # strong but not perfect
    assert r.iou > 0.75
    assert 0.0 < r.threshold < 500.0
    # a real map has nonzero sampling uncertainty, and the error-adjusted area
    # stays close to the mapped area for a good classifier
    assert r.adjusted_burned_ha is not None
    assert r.ci95_ha is not None and r.ci95_ha > 0
    assert r.adjusted_burned_ha == pytest.approx(r.mapped_burned_ha, rel=0.2)


def test_adjusted_area_is_near_the_true_burned_area():
    test = [_fire("te", burned_frac=0.25, n=120, seed=7)]
    train = [_fire("tr", burned_frac=0.25, n=120, seed=8)]
    r = evaluate_fold(train, test, seed=0)
    true_burned_ha = int(np.count_nonzero(test[0].reference & test[0].valid)) * PIXEL_HA
    assert r.adjusted_burned_ha == pytest.approx(true_burned_ha, rel=0.2)


def test_run_stage0_reports_every_fold_and_is_deterministic():
    samples = {f"f{i}": _fire(f"f{i}", seed=i) for i in range(6)}
    manifest = SplitManifest(
        scheme="test",
        folds=[
            {"train": ["f0", "f1", "f2"], "test": ["f3", "f4", "f5"], "held_out": "A"},
            {"train": ["f3", "f4", "f5"], "test": ["f0", "f1", "f2"], "held_out": "B"},
        ],
    )
    a = run_stage0(samples, manifest, seed=0)
    b = run_stage0(samples, manifest, seed=0)
    assert len(a) == 2
    assert [r.held_out for r in a] == ["A", "B"]
    # seeded, so the adjusted areas are reproducible
    assert [r.adjusted_burned_ha for r in a] == [r.adjusted_burned_ha for r in b]

    s = summarise_stage0(a)
    assert s["folds"] == 2
    assert s["f1_mean"] > 0.85
    assert "f1_std" in s


def test_all_unburned_map_reports_no_adjustment_without_crashing():
    # predictor far below any burned dNBR, so nothing is mapped burned
    flat = make_sample("te", np.full((50, 50), 10.0), np.zeros((50, 50), dtype=bool))
    train = [_fire("tr", seed=1)]
    r = evaluate_fold(train, [flat], seed=0)
    assert r.mapped_burned_ha == 0.0
    assert r.adjusted_burned_ha is None
    assert "burned" in r.note
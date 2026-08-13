from __future__ import annotations

import numpy as np
import pytest

from vhagar.eval import metrics as M


def test_confusion_counts_basic():
    t = np.array([1, 1, 0, 0])
    p = np.array([1, 0, 1, 0])
    c = M.confusion_counts(t, p)
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 1)
    assert c.precision == pytest.approx(0.5)
    assert c.recall == pytest.approx(0.5)
    assert c.f1 == pytest.approx(0.5)
    assert c.iou == pytest.approx(1 / 3)


def test_perfect_prediction():
    t = np.array([[1, 0], [0, 1]])
    assert M.iou(t, t) == pytest.approx(1.0)
    assert M.dice(t, t) == pytest.approx(1.0)
    assert M.burned_area_ratio(t, t) == pytest.approx(1.0)


def test_burned_area_ratio_exposes_bias_that_iou_hides():
    truth = np.zeros((20, 20), dtype=int)
    truth[5:15, 5:15] = 1                      # 100 burned cells
    over = np.zeros_like(truth)
    over[3:17, 3:17] = 1                       # 196 cells, superset
    assert M.burned_area_ratio(truth, over) == pytest.approx(1.96)
    assert M.iou(truth, over) < 1.0


def test_average_precision_matches_sklearn_semantics():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    # sklearn.metrics.average_precision_score gives 0.8333...
    assert M.average_precision(y, s) == pytest.approx(0.8333333, abs=1e-6)


def test_average_precision_nan_without_positives():
    rng = np.random.default_rng(0)
    assert np.isnan(M.average_precision(np.zeros(5), rng.random(5)))


def test_average_precision_perfect_ranking():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    assert M.average_precision(y, s) == pytest.approx(1.0)


def test_tolerance_rescues_a_one_cell_offset():
    t = np.zeros((16, 16), dtype=int)
    t[8, 8] = 1
    p = np.zeros_like(t)
    p[8, 9] = 1
    assert M.f1_with_tolerance(t, p, 0) == pytest.approx(0.0)
    assert M.f1_with_tolerance(t, p, 1) == pytest.approx(1.0)


def test_brier_decomposition_identity():
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)
    d = M.brier_decomposition(y, p, n_bins=20)
    # Murphy: brier == reliability - resolution + uncertainty
    assert d["brier"] == pytest.approx(
        d["reliability"] - d["resolution"] + d["uncertainty"], abs=1e-3
    )


def test_calibrated_model_has_low_ece():
    rng = np.random.default_rng(1)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert M.expected_calibration_error(y, p, n_bins=10) < 0.02


def test_overconfident_model_has_high_ece():
    rng = np.random.default_rng(2)
    y = (rng.random(20000) < 0.05).astype(int)
    p = np.full(20000, 0.9)
    assert M.expected_calibration_error(y, p, n_bins=10) > 0.5


def test_log_loss_and_brier_are_proper():
    """A proper score is minimised by reporting the true probability."""
    rng = np.random.default_rng(3)
    true_p = 0.3
    y = (rng.random(200000) < true_p).astype(int)
    honest = M.log_loss(y, np.full_like(y, true_p, dtype=float))
    for lie in (0.1, 0.2, 0.4, 0.6):
        assert M.log_loss(y, np.full_like(y, lie, dtype=float)) > honest
        assert M.brier_score(y, np.full_like(y, lie, dtype=float)) > M.brier_score(
            y, np.full_like(y, true_p, dtype=float)
        )


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        M.confusion_counts(np.zeros(4), np.zeros(5))

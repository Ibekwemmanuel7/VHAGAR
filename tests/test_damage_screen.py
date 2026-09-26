"""Tests for the post-fire damage-screen scoring core (pure numpy, CI-safe)."""
from __future__ import annotations

import numpy as np

from vhagar.eval.damage_screen import (
    binary_screen,
    confusion,
    ordinal_metrics,
    rbr_to_class_index,
    severity_to_class_index,
)


def test_severity_to_class_index_mtbs():
    assert list(severity_to_class_index(np.array([0, 1, 2, 3, 4, 5]))) == [0, 0, 1, 2, 3, 0]


def test_rbr_to_class_index_vhagar():
    import numpy as _np
    rbr = _np.array([50, 100, 300, 440, 500, 700, _np.nan])
    # <100 No Damage, 100-440 Minor, 440-660 Major, >660 Destroyed, NaN -> 0
    assert list(rbr_to_class_index(rbr)) == [0, 1, 1, 2, 2, 3, 0]


def test_binary_perfect_screen():
    truth = np.array([1, 1, 0, 0, 1], dtype=bool)
    s = binary_screen(truth, truth)
    assert s["pod"] == 1.0 and s["far"] == 0.0 and s["f1"] == 1.0
    assert s["tp"] == 3 and s["tn"] == 2 and s["fp"] == 0 and s["fn"] == 0


def test_binary_baseline_and_skill():
    # 3 destroyed of 5 -> predict-all-destroyed F1 = 2*3/(2*3+2) = 0.75
    truth = np.array([1, 1, 1, 0, 0], dtype=bool)
    allpos = binary_screen(np.ones(5, bool), truth)
    assert allpos["predict_all_destroyed_f1"] == 0.75
    assert allpos["f1"] == 0.75                       # predicting all == the baseline
    assert allpos["skill_f1_over_baseline"] == 0.0
    # a screen that also gets the two survivors right beats the baseline
    good = binary_screen(truth, truth)
    assert good["skill_f1_over_baseline"] > 0.0


def test_binary_shape_guard():
    try:
        binary_screen(np.ones(3, bool), np.ones(4, bool))
    except ValueError:
        return
    raise AssertionError("expected ValueError on shape mismatch")


def test_confusion_and_ordinal():
    true_idx = [0, 1, 2, 3, 3, 3]
    pred_idx = [0, 1, 3, 3, 3, 2]     # class 2 truth predicted 3 (off by one); one 3 predicted 2
    cm = confusion(true_idx, pred_idx, 4)
    assert cm.sum() == 6
    assert cm[0, 0] == 1 and cm[1, 1] == 1
    m = ordinal_metrics(cm)
    assert 0.0 <= m["accuracy"] <= 1.0
    # off-by-one errors keep kappa high, well above chance
    assert m["quadratic_weighted_kappa"] > 0.7
    assert m["support"] == [1, 1, 1, 3]


def test_qwk_perfect_is_one():
    cm = np.diag([5, 3, 2, 4])
    assert ordinal_metrics(cm)["quadratic_weighted_kappa"] == 1.0

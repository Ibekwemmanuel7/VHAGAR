"""T1 temporal-anomaly early detection: forecaster, matched-FAR, lead time."""

from __future__ import annotations

import numpy as np
import pytest

from vhagar.eval.t1_temporal import (
    DiurnalForecaster,
    calibrate_threshold_to_far,
    early_detection_experiment,
    synthetic_bt_series,
)


def test_synthetic_series_injects_a_fire_ramp():
    hours, bt, fp, onset = synthetic_bt_series(n_days=2, seed=0)
    assert bt.shape[0] > 1 and bt.shape[1] == len(hours)
    # the fire pixel ends far hotter than it starts; a clean pixel does not
    assert bt[fp, -1] - bt[fp, onset] > 50
    assert abs(bt[0, -1] - bt[0, onset]) < 20


def test_forecaster_removes_the_diurnal_cycle():
    hours, bt, fp, onset = synthetic_bt_series(n_days=3, seed=1)
    fc = DiurnalForecaster.fit(hours[:onset], bt[:, :onset], n_harmonics=3)
    resid = fc.residual(hours[:onset], bt[:, :onset])
    # on clear-sky history the residual is small and centred near zero (diurnal removed)
    assert abs(float(resid[:-1].mean())) < 0.5
    assert float(resid[:-1].std()) < 4.0


def test_calibrate_threshold_hits_the_target_far():
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 1, 100_000)
    thr = calibrate_threshold_to_far(scores, 0.01)
    assert abs((scores > thr).mean() - 0.01) < 0.002


def test_residual_detector_leads_the_absolute_threshold_at_equal_far():
    hours, bt, fp, onset = synthetic_bt_series(n_days=4, fire_ramp_k_per_h=20, seed=2)
    fc = DiurnalForecaster.fit(hours[:onset], bt[:, :onset], n_harmonics=3)
    r = early_detection_experiment(hours, bt, fp, onset, fc, target_far=0.01)
    # residual anomaly catches the night fire well before the absolute cut
    assert r.residual_detect_min_after_onset < r.absolute_detect_min_after_onset
    assert r.lead_minutes > 20.0


def test_temporal_net_forecasts_next_bt_frame():
    pytest.importorskip("torch")
    from vhagar.eval.t1_temporal import train_temporal_net

    rng = np.random.default_rng(0)
    cube = rng.normal(285, 3, size=(20, 8, 8)).astype("float32")   # tiny BT cube
    model = train_temporal_net(cube, window=4, epochs=1)
    import torch
    with torch.no_grad():
        out = model(torch.zeros(1, 4, 1, 8, 8))
    assert tuple(out.shape) == (1, 1, 8, 8)

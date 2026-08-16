"""T1 temporal-anomaly early detection: forecaster, matched-FAR, lead time."""

from __future__ import annotations

import numpy as np
import pytest

from vhagar.eval.t1_temporal import (
    DiurnalForecaster,
    calibrate_threshold_to_far,
    climatology_diurnal_amplitude,
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


def test_climatology_amplitude_recovers_injected_diurnal(tmp_path):
    # build a tiny per-pixel per-UTC-hour climatology with a known 20 K diurnal swing
    hours = np.arange(24)
    amp = 20.0
    mean = 285.0 + (amp / 2.0) * np.cos(2 * np.pi * (hours - 14) / 24.0)  # [24]
    mean = np.repeat(mean[:, None, None], 4, axis=1).repeat(5, axis=2)     # [24,4,5]
    cnt = np.full_like(mean, 4.0)
    m2 = np.full_like(mean, 3.0)   # var ~ 1 K^2 -> sigma ~ 1 K with count 4
    p = tmp_path / "clim.npz"
    np.savez(p, **{"C07::mean": mean, "C07::m2": m2, "C07::count": cnt})
    out = climatology_diurnal_amplitude(p, channel="C07", min_bins=8)
    assert out["n_pixels"] == 20
    assert abs(out["amplitude_k_median"] - amp) < 0.5
    assert out["sigma_k_median"] > 0


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

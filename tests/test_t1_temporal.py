"""T1 temporal-anomaly early detection: forecaster, matched-FAR, lead time."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from vhagar.eval.t1_temporal import (
    DiurnalForecaster,
    HourlyBaselineForecaster,
    calibrate_threshold_to_far,
    climatology_diurnal_amplitude,
    early_detection_experiment,
    hourly_baseline_residual,
    real_lead_experiment,
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


def test_hourly_baseline_is_nan_safe_and_removes_diurnal():
    # two pixels, 3 days at 1h cadence, a 10 K diurnal swing, with holes punched in
    hours = np.tile(np.arange(24), 3).astype(float)
    diurnal = 285.0 + 5.0 * np.cos(2 * np.pi * (hours - 14) / 24.0)
    bt = np.stack([diurnal, diurnal + 2.0])            # [2, 72]
    bt[0, ::7] = np.nan                                # cloud holes
    fc = HourlyBaselineForecaster.fit(hours, bt, n_bins=24)
    resid = fc.residual(hours, bt)
    # residual is ~0 where valid (diurnal removed), NaN where the pixel was NaN
    assert np.isnan(resid[0, 0]) == np.isnan(bt[0, 0])
    assert abs(float(np.nanmean(resid))) < 0.5
    assert np.isfinite(fc.baseline).all()


def test_real_lead_experiment_residual_leads_fdc():
    # 50 pixels x 200 frames @5min; diurnal baseline + noise. Inject a ramp on 5 fire
    # pixels starting at frame 100; let "FDC" only flag them 8 frames later (a late
    # absolute threshold). The residual should fire earlier -> positive lead.
    rng = np.random.default_rng(0)
    T, P = 200, 50
    hours = (np.arange(T) * 5 / 60.0) % 24
    diurnal = 285.0 + 6.0 * np.cos(2 * np.pi * (hours - 14) / 24.0)
    bt = diurnal[None, :] + rng.normal(0, 0.5, size=(P, T))
    fire = np.arange(5)
    onset = 100
    ramp = np.maximum(0.0, np.arange(T) - onset) * 1.5
    bt[fire] += ramp[None, :]
    resid = hourly_baseline_residual(hours, bt, n_bins=24, clear_mask=np.arange(T) < onset)
    first_idx = np.full(P, -1, dtype=np.int64)
    first_idx[fire] = onset + 8                         # FDC is 8 frames (40 min) late
    r = real_lead_experiment(resid, first_idx, target_far=0.01, cadence_min=5)
    assert r.n_fire_pixels == 5
    assert r.median_lead_min > 0
    assert r.frac_residual_led >= 0.8


def test_assemble_cube_drops_mismatched_grid():
    from vhagar.archive.temporal_cube import assemble_cube
    from vhagar.io.cmip_reader import CMIPChannel

    def _frame(minute, shape=(4, 5), corner=36.0):
        H, W = shape
        lat = np.full(shape, corner) + np.arange(H)[:, None] * 0.1
        lon = np.full(shape, -120.0) + np.arange(W)[None, :] * 0.1
        z = np.zeros(shape)
        return CMIPChannel(
            satellite=18, band="C07", wavelength_um=3.9,
            scan_start=datetime(2026, 8, 1, 0, minute, tzinfo=UTC),
            bt_k=np.full(shape, 285.0), dqf=z.astype("int16"),
            saturated=z.astype(bool), lat=lat, lon=lon,
            view_zenith_deg=z, true_pixel_area_m2=z, projection=None,
        )

    good = [_frame(0), _frame(5), _frame(10)]
    bad = _frame(7, shape=(4, 6))                       # different width
    cube = assemble_cube([good[0], good[2], bad, good[1]], "C07")
    assert cube.shape == (3, 4, 5)                      # bad dropped, order restored
    assert cube.times == [f.scan_start for f in good]


def test_solar_zenith_cube_is_small_at_local_noon():
    from vhagar.archive.temporal_cube import solar_zenith_cube

    lat = np.array([[40.0, 40.0]])
    lon = np.array([[-100.0, -100.0]])
    # ~19 UTC is near local solar noon at 100 W on the summer solstice (doy 172)
    times = [datetime(2026, 6, 21, 19, 0, tzinfo=UTC)]
    z = solar_zenith_cube(lat, lon, times)
    assert z.shape == (1, 1, 2)
    assert float(z[0, 0, 0]) < 25.0


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

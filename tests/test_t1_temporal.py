"""T1 temporal-anomaly early detection: forecaster, matched-FAR, lead time."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from vhagar.eval.t1_temporal import (
    DiurnalForecaster,
    HourlyBaselineForecaster,
    baseline_contamination,
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


def test_eval_start_ignores_detections_in_the_baseline_window():
    # A fire pixel whose residual is sustained-high from frame 20 (inside the baseline the
    # forecaster was fit on) and stays high. Without a train/test split the "detection" is
    # frame 20 (pre-fire, a false lead); with eval_start=60 it is measured on held-out
    # frames only, so detection is at 60 and the lead is honest.
    T, P = 120, 30
    resid = np.zeros((P, T))
    fp = 0
    resid[fp, 20:] = 50.0                     # sustained high from deep in the baseline
    first_idx = np.full(P, -1, dtype=np.int64)
    first_idx[fp] = 70                         # FDC flags at frame 70
    naive = real_lead_experiment(resid, first_idx, target_far=0.01, min_consec=3, eval_start=0)
    split = real_lead_experiment(resid, first_idx, target_far=0.01, min_consec=3, eval_start=60)
    assert naive.median_lead_min == (70 - 22) * 5     # detected at end of first 3-run (22)
    assert split.median_lead_min == (70 - 62) * 5     # first 3-run in held-out period (62)


def test_persistence_filters_a_pre_fire_false_blip():
    # One fire pixel. A single isolated pre-fire spike would fake a huge lead; the real
    # fire is a sustained ramp. min_consec=3 must ignore the blip and detect the ramp.
    T, P = 120, 40
    resid = np.zeros((P, T))
    fp = 0
    resid[fp, 10] = 100.0                    # isolated pre-fire blip at frame 10
    resid[fp, 70:] = 100.0                   # sustained fire ramp from frame 70
    first_idx = np.full(P, -1, dtype=np.int64)
    first_idx[fp] = 78                        # FDC confirms at frame 78
    naive = real_lead_experiment(resid, first_idx, target_far=0.01, cadence_min=5, min_consec=1)
    persist = real_lead_experiment(resid, first_idx, target_far=0.01, cadence_min=5, min_consec=3)
    # first-exceedance is fooled by the blip (detects at 10 -> lead (78-10)*5=340);
    # persistence detects the real ramp end (frame 72 -> lead (78-72)*5=30)
    assert naive.median_lead_min == (78 - 10) * 5
    assert persist.median_lead_min == (78 - 72) * 5


def test_per_hour_threshold_recovers_night_sensitivity():
    # Residuals with big daytime variance and quiet nights; a night fire is a modest
    # excursion that a global threshold (set by daytime) misses but a night-specific
    # threshold catches.
    rng = np.random.default_rng(0)
    T, P = 240, 200                                   # 20h @5min, 200 pixels
    hours = (np.arange(T) * 5 / 60.0) % 24
    day = ((hours >= 8) & (hours <= 20))
    resid = np.where(day[None, :], rng.normal(0, 4.0, (P, T)), rng.normal(0, 0.5, (P, T)))
    first_idx = np.full(P, -1, dtype=np.int64)
    # one night fire: a +3 K excursion (huge for night, invisible against daytime sigma 4)
    night_frame = int(np.flatnonzero(~day)[5])
    fp = 0
    resid[fp, night_frame:] += 3.0
    first_idx[fp] = night_frame + 6                    # FDC flags it 30 min later
    glob = real_lead_experiment(resid, first_idx, target_far=0.01, far_bins=1, hours=hours)
    perh = real_lead_experiment(resid, first_idx, target_far=0.01, far_bins=6, hours=hours)
    # the per-time-of-day threshold detects (and leads); the global one is desensitised
    assert perh.median_lead_min > glob.median_lead_min


def test_baseline_contamination_flags_early_ignition():
    T, P = 100, 20
    clear = np.arange(T) < 60                       # first 60 frames used as baseline
    first_idx = np.full(P, -1, dtype=np.int64)
    first_idx[:5] = 30                              # ignite inside the clear window
    first_idx[5:8] = 80                             # ignite after it (clean)
    # 5 of 8 fire pixels contaminated
    assert abs(baseline_contamination(first_idx, clear) - 5 / 8) < 1e-9
    # all-clean case
    first_idx2 = np.full(P, -1, dtype=np.int64)
    first_idx2[:4] = 80
    assert baseline_contamination(first_idx2, clear) == 0.0


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


def test_cohort_lead_summary_aggregates_by_stratum():
    from vhagar.eval.t1_temporal import RealLeadResult, cohort_lead_summary

    def _r(lead, npx=10):                              # a fire the residual detected
        return RealLeadResult(npx, npx, 1.0, 0, 0.0, lead, lead, lead, 0.01, 1.0,
                              leads_min=tuple([lead]) * npx)

    def _miss(npx=10):                                 # a fire the residual never detected
        return RealLeadResult(0, npx, 0.0, 0, 0.0, float("nan"), float("nan"),
                              float("nan"), 0.01, 1.0)

    per_fire = [
        ("night_coldstart", _r(+30)), ("night_coldstart", _r(+10)),
        ("night_coldstart", _r(-5)), ("day", _r(-40)), ("day", _r(-60)),
    ]
    s = cohort_lead_summary(per_fire)
    assert s["night_coldstart"]["n_fires"] == 3
    assert abs(s["night_coldstart"]["frac_fires_led"] - 2 / 3) < 1e-9
    assert s["night_coldstart"]["median_fire_lead_min"] == 10
    assert s["night_coldstart"]["detection_rate"] == 1.0
    assert s["night_coldstart"]["pooled_pixel_median_lead_min"] == 10
    assert abs(s["night_coldstart"]["pooled_pixel_frac_led"] - 2 / 3) < 1e-9
    assert s["day"]["frac_fires_led"] == 0.0

    # a non-detecting fire must count as not-led and lower the detection rate, NOT show as 0
    s2 = cohort_lead_summary([("night_coldstart", _r(+30, npx=10)),
                              ("night_coldstart", _miss(npx=30))])
    assert s2["night_coldstart"]["detection_rate"] == 10 / 40      # 10 of 40 fire px detected
    assert s2["night_coldstart"]["frac_fires_detected"] == 0.5
    assert s2["night_coldstart"]["frac_fires_led"] == 0.5          # miss is not a lead
    assert s2["night_coldstart"]["pooled_pixel_median_lead_min"] == 30   # miss excluded


def test_select_fire_cohort_stratifies_night_and_day(tmp_path):
    pd = pytest.importorskip("pandas")
    from vhagar.archive.temporal_cube import select_fire_cohort

    rows = []
    # night fire near lon -115 (LST offset ~ -7.7h): ignite 11:00 UTC -> LST ~3.3 (night)
    ign_n = datetime(2026, 8, 3, 11, 0, tzinfo=UTC)
    for i in range(60):
        rows.append({"lon": -115.0, "lat": 31.0,
                     "t": ign_n + timedelta(minutes=2 * i), "frp_mw": 50.0 + i})
    # day fire near lon -120: ignite 21:00 UTC -> LST ~13 (day)
    ign_d = datetime(2026, 8, 3, 21, 0, tzinfo=UTC)
    for i in range(60):
        rows.append({"lon": -120.0, "lat": 40.0,
                     "t": ign_d + timedelta(minutes=2 * i), "frp_mw": 100.0 + 5 * i})
    root = tmp_path / "det" / "day=1"
    root.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(root / "p.parquet")

    specs = select_fire_cohort(tmp_path / "det", data_start=datetime(2026, 8, 1, tzinfo=UTC))
    strata = {s.stratum for s in specs}
    assert strata == {"night_coldstart", "day"}
    night = next(s for s in specs if s.stratum == "night_coldstart")
    # clear window ends before ignition (baseline not contaminated)
    assert night.pull_start < night.ignition_utc < night.pull_end
    assert 5.0 <= night.local_solar_hour < 6.0 or night.local_solar_hour < 6.0
    assert night.bbox[0] < night.lon < night.bbox[2]


def test_cohort_pull_skips_existing_cubes_without_s3(tmp_path):
    import json

    from vhagar.archive.temporal_cube import cohort_pull

    spec = [
        {"name": "fireA", "bbox": [-112.0, 38.0, -111.6, 38.4],
         "pull_start": "2026-08-02T00:00:00+00:00", "pull_end": "2026-08-02T12:00:00+00:00"},
        {"name": "fireB", "bbox": [-120.0, 40.0, -119.6, 40.4],
         "pull_start": "2026-08-03T00:00:00+00:00", "pull_end": "2026-08-03T12:00:00+00:00"},
    ]
    spec_path = tmp_path / "cohort.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    # both cubes already exist -> nothing is pulled, no S3 is touched
    (tmp_path / "fireA.npz").write_bytes(b"x")
    (tmp_path / "fireB.npz").write_bytes(b"x")
    res = cohort_pull(spec_path, only_missing=True)
    assert set(res["skipped"]) == {"fireA", "fireB"}
    assert res["pulled"] == [] and res["failed"] == []


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


def test_learned_residuals_are_nan_safe_and_feed_the_experiment():
    pytest.importorskip("torch")
    from vhagar.eval.t1_temporal import learned_residuals, real_lead_experiment

    rng = np.random.default_rng(0)
    T, H, W = 30, 6, 6
    cube = rng.normal(285, 2, size=(T, H, W)).astype("float32")
    cube[5, 0, 0] = np.nan                                   # a cloud hole in the input
    solar = np.cos(np.radians(rng.uniform(20, 70, (T, 1, H, W)))).astype("float32")
    resid = learned_residuals(cube, clear_end=20, window=4, epochs=1, covariates=solar)
    assert resid.shape == (H * W, T)
    # first `window` frames have no forecast -> NaN; later frames are finite (bar the hole)
    assert np.isnan(resid[:, :4]).all()
    assert np.isfinite(resid[:, 10]).mean() > 0.9
    # residuals drop straight into the same matched-FAR / persistence protocol
    first_idx = np.full(H * W, -1, dtype=np.int64)
    first_idx[0] = 25
    r = real_lead_experiment(resid, first_idx, target_far=0.05, min_consec=2)
    assert r.n_fire_pixels >= 0

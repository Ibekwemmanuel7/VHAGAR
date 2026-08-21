"""Physics-informed next-day spread forecaster tests (synthetic; numpy core)."""
from __future__ import annotations

import numpy as np
import pytest

from vhagar.datasets.wildfirespread import CHANNELS, synthetic_wfs_fire


def test_synthetic_sample_is_well_formed():
    s, horizon = synthetic_wfs_fire(np.random.default_rng(0))
    assert s.features.shape == (len(CHANNELS), 48, 48)
    assert s.fire_t.dtype == bool and s.fire_t1.dtype == bool
    assert s.new_burn.any()                       # there is genuinely new burn to predict
    assert (s.fire_t1 | ~s.fire_t).all()          # already-burning cells stay burning
    assert s.is_usable and horizon > 0.0


def test_physics_prior_beats_persistence_buffer_on_new_burn():
    pytest.importorskip("scipy")
    from vhagar.eval.wildfirespread import (
        persistence_buffer_forecast,
        physics_prior,
        score_forecast,
    )
    phys, pers = [], []
    for i in range(10):
        s, h = synthetic_wfs_fire(np.random.default_rng(100 + i))
        phys.append(score_forecast(physics_prior(s, horizon=h * 2), s)["ap"])
        pers.append(score_forecast(persistence_buffer_forecast(s), s)["ap"])
    # the fast-marching prior should discriminate new burn better than a blind buffer
    assert np.nanmean(phys) > np.nanmean(pers)


def test_temperature_calibration_does_not_worsen_brier():
    pytest.importorskip("scipy")
    from vhagar.eval.metrics import brier_score
    from vhagar.eval.wildfirespread import (
        apply_temperature,
        fit_temperature,
        physics_prior,
    )
    s, h = synthetic_wfs_fire(np.random.default_rng(3))
    incr = (~s.fire_t) & s.valid
    p = physics_prior(s, horizon=h * 2)[incr]
    y = s.fire_t1[incr].astype(float)
    t = fit_temperature(p, y)
    b0 = brier_score(y, np.clip(p, 1e-6, 1 - 1e-6))
    b1 = brier_score(y, apply_temperature(p, t))
    assert b1 <= b0 + 1e-6                          # T=1 is in the search space, so never worse
    assert t > 0.0


def test_evaluate_nextday_runs_without_torch():
    pytest.importorskip("scipy")
    from vhagar.eval.wildfirespread import evaluate_nextday
    samples = {}
    for i in range(6):
        s, _ = synthetic_wfs_fire(np.random.default_rng(200 + i), fire_id=f"f{i}")
        samples[s.fire_id] = s
    rep = evaluate_nextday(samples, k=3, with_corrector=False, calibrate=True, seed=0)
    assert "physics" in rep["summary"] and "persistence_buffer" in rep["summary"]
    assert rep["summary"]["physics"]["fires"] == len(samples)
    # calibrated physics should beat the buffer baseline on AP across fires
    assert rep["summary"]["physics"]["ap_mean"] > rep["summary"]["persistence_buffer"]["ap_mean"]


def test_corrector_trains_and_predicts_if_torch():
    pytest.importorskip("torch")
    pytest.importorskip("scipy")
    from vhagar.eval.wildfirespread import predict_corrector, train_corrector
    train = [synthetic_wfs_fire(np.random.default_rng(i), fire_id=f"f{i}")[0] for i in range(3)]
    model, stats = train_corrector(train, epochs=1, widths=(8, 16), seed=0)
    p = predict_corrector(model, stats, train[0])
    assert p.shape == train[0].fire_t.shape and p.min() >= 0.0 and p.max() <= 1.0

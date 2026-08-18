"""T3 expected-burned-area (E[BA]) tests: proper scores + heavy-tailed head."""
from __future__ import annotations

import numpy as np
import pytest

from vhagar.eval.metrics import crps_from_quantiles, pinball_loss


def test_pinball_loss_known_value():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 1.0, 1.0])
    # errors 0,1,2 at q=0.5 -> mean(0, 0.5, 1.0) = 0.5
    assert abs(pinball_loss(y, pred, 0.5) - 0.5) < 1e-9
    # asymmetry: under-prediction penalised more at high quantile
    assert pinball_loss(y, pred, 0.9) > pinball_loss(y, pred, 0.1)


def test_crps_rewards_sharper_calibrated_quantiles():
    rng = np.random.default_rng(0)
    y = rng.normal(10.0, 2.0, 500)
    taus = np.array([0.1, 0.5, 0.9])
    from scipy.stats import norm
    sharp = np.column_stack([norm.ppf(t, 10.0, 2.0) * np.ones_like(y) for t in taus])
    wide = np.column_stack([norm.ppf(t, 10.0, 6.0) * np.ones_like(y) for t in taus])
    assert crps_from_quantiles(y, sharp, taus) < crps_from_quantiles(y, wide, taus)
    with pytest.raises(ValueError):
        crps_from_quantiles(y, sharp[:, :2], taus)   # shape mismatch


def test_scenario_is_heavy_tailed():
    from vhagar.eval.burned_area import synthetic_burned_area_scenario
    X, area, yr, lon, lat, fn = synthetic_burned_area_scenario(np.random.default_rng(1), n=3000)
    assert area.min() >= 1.0 and area.max() <= 300_000.0
    # a long right tail: p99 is many times the median
    assert np.quantile(area, 0.99) > 5 * np.median(area)
    assert fn == ["dryness", "fuel", "wind", "slope"]


def test_model_quantiles_are_monotone_and_tail_fits():
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    from vhagar.eval.burned_area import BurnedAreaModel, synthetic_burned_area_scenario
    X, area, *_ = synthetic_burned_area_scenario(np.random.default_rng(2), n=2000)
    m = BurnedAreaModel(seed=0, use_tail=True).fit(X, area)
    q = m.predict_quantiles(X[:20])
    assert np.all(np.diff(q, axis=1) >= -1e-6)      # monotone across quantile levels
    assert m._gpd is not None                        # GPD tail was fit


def test_expected_ba_scales_with_ignition_probability():
    pytest.importorskip("sklearn")
    from vhagar.eval.burned_area import (
        BurnedAreaModel,
        expected_burned_area,
        synthetic_burned_area_scenario,
    )
    X, area, *_ = synthetic_burned_area_scenario(np.random.default_rng(3), n=1500)
    m = BurnedAreaModel(seed=0).fit(X, area)
    q = m.predict_quantiles(X)
    eba_lo = expected_burned_area(np.full(X.shape[0], 0.01), q)
    eba_hi = expected_burned_area(np.full(X.shape[0], 0.04), q)
    assert np.allclose(eba_hi, 4.0 * eba_lo, rtol=1e-6)
    assert np.all(eba_lo >= 0)


@pytest.mark.slow
def test_model_beats_climatology_on_crps():
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    from vhagar.eval.burned_area import evaluate_expected_ba, synthetic_burned_area_scenario
    X, area, yr, lon, lat, fn = synthetic_burned_area_scenario(np.random.default_rng(0), n=4000)
    r = evaluate_expected_ba(X, area, lon, lat, n_folds=4, seed=0, use_tail=True)
    assert r["crps_skill_vs_climatology"] > 0.03          # covariates carry real signal
    assert r["crps"] < r["crps_climatology"]
    # RMSE is tail-unstable: its fold std is a large fraction of its mean
    assert r["rmse_std"] / max(r["rmse_mean"], 1.0) > 0.3

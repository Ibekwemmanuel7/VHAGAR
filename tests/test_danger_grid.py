"""T3 Layer-3 deep challenger: FSS + gridded shadow-mode harness tests."""
from __future__ import annotations

import numpy as np
import pytest

from vhagar.eval.metrics import fractions_skill_score as fss


def test_fss_perfect_and_neighborhood_growth():
    obs = np.zeros((20, 20))
    obs[10, 10] = 1
    assert abs(fss(obs, obs.astype(float), 3, threshold=0.5) - 1.0) < 1e-9
    off = np.zeros((20, 20))
    off[10, 12] = 1.0                       # two cells off
    assert fss(obs, off, 5, threshold=0.5) > fss(obs, off, 1, threshold=0.5)


def test_fss_probabilistic_variant_rewards_sharpness():
    from scipy.ndimage import uniform_filter
    obs = np.zeros((24, 24))
    obs[12, 12] = 1
    near = uniform_filter(obs, 5)           # probability mass on the right place
    far = np.full_like(obs, obs.mean())     # flat climatology
    assert fss(obs, near, 5, threshold=None) > fss(obs, far, 5, threshold=None)


def test_grid_scenario_shapes_and_signal():
    from vhagar.eval.danger_grid import synthetic_ignition_grid
    X, ev, fn, ckm = synthetic_ignition_grid(np.random.default_rng(0), T=20, H=32, W=32)
    assert X.shape == (20, 3, 32, 32) and ev.shape == (20, 32, 32)
    assert 0.005 < ev.mean() < 0.3 and fn == ["dryness", "fuel", "wind"]


def test_neighborhood_pool_smooths():
    from vhagar.eval.danger_grid import neighborhood_pool
    X = np.random.default_rng(0).random((4, 2, 16, 16)).astype(np.float32)
    pooled = neighborhood_pool(X, 5)
    assert pooled.shape == X.shape
    assert pooled.std() < X.std()           # pooling reduces variance


@pytest.mark.slow
def test_shadow_evaluate_signal_and_gate_consistency():
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    from vhagar.eval.danger_grid import shadow_evaluate, synthetic_ignition_grid
    X, ev, fn, ckm = synthetic_ignition_grid(np.random.default_rng(0), intercept=-9.0, obs_noise=0.15)
    r = shadow_evaluate(X, ev, cell_km=ckm, n_folds=4, seed=0)
    b, c = r["baseline"], r["challenger"]
    # real learnable signal: both clearly beat the base rate on AUPRC
    assert b["auprc"] > r["base_rate"] and c["auprc"] > r["base_rate"]
    # the spatial challenger denoises: better FSS at the largest scale
    assert c["fss"][120] >= b["fss"][120]
    # the promotion gate is exactly "beat baseline on AUPRC AND Brier"
    assert r["promote"] == (c["auprc"] > b["auprc"] and c["brier"] < b["brier"])


def test_torch_challenger_shapes():
    torch = pytest.importorskip("torch")
    from vhagar.models.ignition_conv import predict_spatial, soft_fss_loss, train_spatial
    rng = np.random.default_rng(0)
    X = rng.random((6, 3, 16, 16)).astype(np.float32)
    ev = (rng.random((6, 16, 16)) < 0.1).astype(np.int8)
    net = train_spatial(X, ev, epochs=2, seed=0)
    pred = predict_spatial(net, X)
    assert pred.shape == (6, 16, 16) and pred.min() >= 0 and pred.max() <= 1
    logits = torch.zeros((2, 1, 8, 8))
    target = torch.zeros((2, 1, 8, 8))
    assert float(soft_fss_loss(logits, target, 3)) >= 0.0

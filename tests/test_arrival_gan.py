"""T4 conditional arrival-time GAN: pure-contract tests + torch shapes."""
from __future__ import annotations

import numpy as np
import pytest

from vhagar.models.arrival_gan import (
    build_conditioning,
    denormalize_arrival,
    make_training_pair,
    normalize_arrival,
)


def test_build_conditioning_channels():
    H = W = 16
    observed = np.zeros((H, W))
    observed[8, 8] = 1
    det = np.zeros((H, W))
    cov = np.zeros((3, H, W))
    cond = build_conditioning(observed, det, cov)
    assert cond.shape == (5, H, W)              # observed + det-time + 3 covariates
    assert cond[0, 8, 8] == 1.0
    assert cond.dtype == np.float32


def test_normalize_arrival_roundtrip_and_inf():
    T = np.array([[0.0, 5.0], [10.0, np.inf]])
    n = normalize_arrival(T, tmax=10.0)
    assert n.max() <= 1.0 and n.min() >= 0.0
    assert n[1, 1] == 1.0                        # unreachable -> 1
    assert abs(denormalize_arrival(n[0, 1], 10.0) - 5.0) < 1e-6


def test_make_training_pair_shapes():
    cond, target, ros = make_training_pair(np.random.default_rng(0))
    assert cond.shape[0] == 5 and cond.shape[1:] == target.shape == ros.shape
    assert target.min() >= 0.0 and target.max() <= 1.0
    assert (cond[0] >= 0).all() and (cond[0] <= 1).all()   # observed mask


def test_torch_generator_and_losses():
    torch = pytest.importorskip("torch")
    from vhagar.models.arrival_gan import (
        ArrivalGenerator,
        eikonal_residual_loss,
        predict_arrival,
        train_arrival_gan,
    )
    cond, target, ros = make_training_pair(np.random.default_rng(0))
    gen = ArrivalGenerator(cond.shape[0])
    pred = predict_arrival(gen, cond)
    assert pred.shape == target.shape and pred.min() >= 0.0 and pred.max() <= 1.0
    arr = torch.rand(1, ros.shape[0], ros.shape[1])
    loss = eikonal_residual_loss(arr, torch.as_tensor(ros)[None], tmax=1.0)
    assert float(loss) >= 0.0
    g = train_arrival_gan([make_training_pair(np.random.default_rng(i)) for i in range(2)], epochs=1)
    assert predict_arrival(g, cond).shape == target.shape

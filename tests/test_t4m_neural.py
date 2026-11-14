"""Smoke tests for the optional torch T4-M physics-guided U-Net."""
from __future__ import annotations

import numpy as np
import pytest

from vhagar.datasets.t4m_tensor import T4MTensorCase


def test_masked_physics_guided_unet_trains_one_sparse_case():
    torch = pytest.importorskip("torch")
    from vhagar.models.t4m_neural import (
        masked_arrival_l1_loss,
        predict_arrival,
        train_physics_guided_unet,
    )

    target = np.full((8, 8), np.nan, dtype=np.float32)
    target[3, 3], target[5, 5] = 0.3, 0.6
    valid = np.isfinite(target)
    case = T4MTensorCase(
        conditioning=np.zeros((6, 8, 8), dtype=np.float32), target_arrival=target,
        target_valid=valid, prior_ros=np.ones((8, 8), dtype=np.float32), tmax_hours=4.0,
        fire_id="test", cutoff_utc="2026-01-01T00:00:00Z", n_input_observations=1,
        n_target_observations=2,
    )
    loss = masked_arrival_l1_loss(torch.tensor([[0.2, 0.7]]), torch.tensor([[0.3, 0.6]]), torch.tensor([[True, True]]))
    assert float(loss) == pytest.approx(0.1)
    model = train_physics_guided_unet([case], epochs=1, device="cpu")
    pred = predict_arrival(model, case.conditioning)
    assert pred.shape == (8, 8)
    assert np.isfinite(pred).all() and pred.min() >= 0 and pred.max() <= 1

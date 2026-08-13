"""Model and loss tests. Skipped cleanly when the torch extra is absent."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vhagar.models.segmentation import (  # noqa: E402
    SiameseChangeNet,
    TemporalAnomalyNet,
    UNet,
)
from vhagar.train.losses import (  # noqa: E402
    ComboLoss,
    DiceLoss,
    FocalLoss,
    TverskyLoss,
    suggest_pos_weight,
)


@pytest.mark.parametrize("size", [64, 96])
def test_unet_preserves_spatial_shape(size):
    m = UNet(in_channels=6)
    out = m(torch.zeros(2, 6, size, size))
    assert out.shape == (2, 1, size, size)


def test_siamese_shares_encoder_weights():
    m = SiameseChangeNet(in_channels=6)
    # One encoder, used twice -- not two encoders.
    encoders = [n for n, _ in m.named_children() if n == "encoder"]
    assert len(encoders) == 1
    out = m(torch.zeros(1, 6, 64, 64), torch.zeros(1, 6, 64, 64))
    assert out.shape == (1, 1, 64, 64)


def test_siamese_is_sensitive_to_change_not_to_absolute_level():
    torch.manual_seed(0)
    m = SiameseChangeNet(in_channels=4).eval()
    a = torch.rand(1, 4, 32, 32)
    b = torch.rand(1, 4, 32, 32)
    with torch.no_grad():
        no_change = m(a, a)
        change = m(a, b)
    assert not torch.allclose(no_change, change, atol=1e-4)


def test_temporal_anomaly_shapes_and_window_guard():
    m = TemporalAnomalyNet(in_channels=3, window=6)
    assert m(torch.zeros(2, 6, 3, 16, 16)).shape == (2, 1, 16, 16)
    with pytest.raises(ValueError, match="expected T=6"):
        m(torch.zeros(2, 5, 3, 16, 16))


def test_temporal_anomaly_residual_sign():
    pred = torch.full((1, 1, 4, 4), 300.0)
    obs = torch.full((1, 1, 4, 4), 340.0)
    assert torch.all(TemporalAnomalyNet.anomaly(pred, obs) > 0)


def test_gradients_flow_end_to_end():
    m = UNet(in_channels=4)
    loss_fn = ComboLoss(pos_weight=10.0)
    x = torch.rand(2, 4, 32, 32)
    y = torch.zeros(2, 1, 32, 32)
    y[:, :, 10:20, 10:20] = 1.0
    loss = loss_fn(m(x), y)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("loss_cls", [DiceLoss, TverskyLoss, FocalLoss])
def test_losses_are_finite_and_non_negative(loss_cls):
    loss_fn = loss_cls()
    logits = torch.randn(3, 1, 16, 16)
    target = (torch.rand(3, 1, 16, 16) > 0.95).float()
    v = loss_fn(logits, target)
    assert torch.isfinite(v) and v >= 0


def test_dice_loss_is_minimised_by_a_correct_prediction():
    loss_fn = DiceLoss()
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 4:12, 4:12] = 1.0
    good = torch.where(target > 0, 8.0, -8.0)
    bad = -good
    assert loss_fn(good, target) < loss_fn(bad, target)


def test_tversky_penalises_false_negatives_more_than_false_positives():
    """beta > alpha is the right asymmetry for a detection product."""
    loss_fn = TverskyLoss(alpha=0.3, beta=0.7)
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 4:12, 4:12] = 1.0

    misses = torch.full_like(target, -8.0)                    # predicts nothing
    over = torch.full_like(target, 8.0)                       # predicts everything
    assert loss_fn(misses, target) > loss_fn(over, target)


def test_suggest_pos_weight_is_capped():
    assert suggest_pos_weight(0.5) == pytest.approx(1.0)
    assert suggest_pos_weight(0.001, cap=50.0) == 50.0
    with pytest.raises(ValueError):
        suggest_pos_weight(0.0)


def test_determinism_same_seed_same_output():
    def run():
        torch.manual_seed(42)
        m = UNet(in_channels=3)
        torch.manual_seed(7)
        return m(torch.rand(1, 3, 32, 32))

    assert torch.equal(run(), run())

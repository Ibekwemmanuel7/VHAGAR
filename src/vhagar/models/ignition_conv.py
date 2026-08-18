"""The T3 Layer-3 deep challenger: a spatial ignition net with a soft-FSS loss.

The architecture (`docs/00` 5.4) proposes a ConvLSTM / U-Net-3+ trained with a
Fractions Skill Score loss as a *challenger* to the gradient boosting, promoted
only if it wins on blocked AUPRC and Brier. This is that model: a compact U-Net
over the ``[C, H, W]`` danger field per day, trained with a differentiable
soft-FSS term plus BCE. It is torch-guarded, the numerics run on a GPU box; the
shadow-mode scoring and promotion gate live in ``eval/danger_grid.py`` and score
whatever probability field this produces.

Keeping it small is deliberate: the honest expectation is that the deep model
earns its place at seasonal lead times, not daily, so the challenger is cheap
and only promoted on evidence.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SpatialIgnitionNet", "soft_fss_loss", "train_spatial", "predict_spatial"]


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("the deep ignition challenger needs pytorch") from exc
    return torch


def soft_fss_loss(logits, target, neighborhood: int, eps: float = 1e-6):
    """Differentiable (1 - FSS): neighborhood fractions via average pooling.

    Minimising this maximises the Fractions Skill Score, so the network is
    trained on the same spatial, neighborhood-scale objective it is judged on,
    rather than on a pixel-exact loss that a rare, point-like target defeats.
    """
    torch = _torch()
    import torch.nn.functional as F

    p = torch.sigmoid(logits)
    pad = neighborhood // 2
    of = F.avg_pool2d(target, neighborhood, stride=1, padding=pad, count_include_pad=False)
    ff = F.avg_pool2d(p, neighborhood, stride=1, padding=pad, count_include_pad=False)
    fbs = ((of - ff) ** 2).mean()
    ref = (of ** 2).mean() + (ff ** 2).mean() + eps
    return fbs / ref


def _build_net(in_ch: int):
    torch = _torch()
    import torch.nn as nn

    def block(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(4, b), nn.ReLU(),
                             nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(4, b), nn.ReLU())

    class SpatialIgnitionNet(nn.Module):
        """A small U-Net: two downsamples, two upsamples, 1-channel logit map."""

        def __init__(self, c):
            super().__init__()
            self.e1 = block(c, 16)
            self.e2 = block(16, 32)
            self.e3 = block(32, 64)
            self.pool = nn.MaxPool2d(2)
            self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.d2 = block(64, 32)
            self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.d1 = block(32, 16)
            self.head = nn.Conv2d(16, 1, 1)

        def forward(self, x):
            s1 = self.e1(x)
            s2 = self.e2(self.pool(s1))
            b = self.e3(self.pool(s2))
            d2 = self.d2(torch.cat([self.up2(b), s2], 1))
            d1 = self.d1(torch.cat([self.up1(d2), s1], 1))
            return self.head(d1)

    return SpatialIgnitionNet(in_ch)


def SpatialIgnitionNet(in_ch: int):  # noqa: N802 (factory, keeps torch import lazy)
    """Construct the spatial ignition U-Net for ``in_ch`` input channels."""
    return _build_net(in_ch)


def train_spatial(X, events, neighborhood: int = 5, epochs: int = 40,
                  lr: float = 3e-3, fss_weight: float = 0.5, seed: int = 0):
    """Train on ``X`` ``[T, C, H, W]`` with binary ``events`` ``[T, H, W]``.

    Loss is BCE + ``fss_weight`` x soft-FSS. Returns the fitted model (on CPU).
    """
    torch = _torch()
    import torch.nn.functional as F

    torch.manual_seed(seed)
    x = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(events)[:, None, :, :], dtype=torch.float32)
    net = SpatialIgnitionNet(x.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    pos_weight = torch.tensor([(y.numel() - y.sum()) / (y.sum() + 1.0)])
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        logits = net(x)
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss = bce + fss_weight * soft_fss_loss(logits, y, neighborhood)
        loss.backward()
        opt.step()
    return net


def predict_spatial(net, X):
    """Probability field ``[T, H, W]`` for ``X`` ``[T, C, H, W]``."""
    torch = _torch()
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(torch.as_tensor(np.asarray(X), dtype=torch.float32)))
    return p[:, 0].cpu().numpy()

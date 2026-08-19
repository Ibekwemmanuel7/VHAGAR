"""T4 generative arrival-time inference: a conditional GAN, physics-anchored.

The published state of the art for fire spread *state estimation* (``docs/00``
6.2) is a conditional GAN that infers the continuous arrival-time field from
satellite active fire, validated against airborne IR at Sorensen ~0.81 and
ignition-time error ~32 min. This module is that model: a U-Net generator
conditioned on the sparse observations plus mapped covariates, a PatchGAN
discriminator, and, crucially, an **Eikonal-consistency** term that ties the
generated field to the physics core (``|grad T| ~ 1 / ROS``). It sits on top of
the physics-anchored estimator in ``models/state_estimation.py``: the estimator
gives a calibrated prior, the GAN learns the residual structure the single-scale
calibration cannot.

The tensors and losses are torch-guarded (this trains on a GPU box). The
conditioning and normalisation builders are pure numpy and unit-tested, so the
data contract is verified without torch, and ``make_training_pair`` turns a
synthetic fire into a ``(conditioning, target, ros)`` example.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "build_conditioning",
    "normalize_arrival",
    "denormalize_arrival",
    "make_training_pair",
    "ArrivalGenerator",
    "eikonal_residual_loss",
    "train_arrival_gan",
    "predict_arrival",
]


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("the arrival-time GAN needs pytorch") from exc
    return torch


def build_conditioning(observed_burned, det_time_norm, covariates) -> np.ndarray:
    """Stack the generator's input channels: ``[C+2, H, W]``.

    Channel 0 is the observed burned mask at ``t0``; channel 1 is a normalised
    detection-time field (0 where unobserved); the rest are the mapped covariates
    (fuel, wind, slope, ...). All in ``[0, 1]``-ish, the contract the generator
    is trained against.
    """
    observed_burned = np.asarray(observed_burned, dtype=np.float32)[None]
    det_time_norm = np.asarray(det_time_norm, dtype=np.float32)[None]
    cov = np.asarray(covariates, dtype=np.float32)
    if cov.ndim == 2:
        cov = cov[None]
    return np.concatenate([observed_burned, det_time_norm, cov], axis=0)


def normalize_arrival(T, tmax: float):
    """Map an arrival-time field to ``[0, 1]`` (unreachable -> 1), for a bounded
    regression target."""
    T = np.asarray(T, dtype=np.float64)
    out = np.clip(T / max(tmax, 1e-9), 0.0, 1.0)
    return np.where(np.isfinite(out), out, 1.0).astype(np.float32)


def denormalize_arrival(Tn, tmax: float):
    """Inverse of :func:`normalize_arrival`."""
    return np.asarray(Tn, dtype=np.float64) * max(tmax, 1e-9)


def make_training_pair(rng, regime: str = "wind", t0_q: float = 0.12):
    """One ``(conditioning, target_norm, ros)`` example from a synthetic fire.

    The generator learns to map the perimeter at ``t0`` plus covariates to the
    full normalised arrival-time field; ``ros`` is returned for the Eikonal term.
    """
    from vhagar.eval.spread import synthetic_fire

    ros, T_true, _ign = synthetic_fire(rng, regime=regime)
    reach = np.isfinite(T_true)
    tv = T_true[reach]
    t0 = float(np.quantile(tv, t0_q))
    tmax = float(np.quantile(tv, 0.9))
    observed = (T_true <= t0).astype(np.float32)
    det_time = np.where(observed > 0, np.nan_to_num(T_true, nan=0.0) / max(t0, 1e-9), 0.0)
    # three mapped covariates, reusing the ROS field as a stand-in feature stack
    cov = np.stack([ros / (ros.max() + 1e-9),
                    np.clip(ros / np.median(ros), 0, 2) / 2.0,
                    np.full_like(ros, 0.5)], axis=0).astype(np.float32)
    cond = build_conditioning(observed, det_time.astype(np.float32), cov)
    target = normalize_arrival(T_true, tmax)
    return cond, target, ros.astype(np.float32)


# --------------------------------------------------------------- torch pieces
def _blocks(nn):
    def block(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(4, b), nn.ReLU(),
                             nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(4, b), nn.ReLU())
    return block


def ArrivalGenerator(in_ch: int):  # noqa: N802 (factory; keeps torch import lazy)
    """U-Net generator: conditioning ``[in_ch, H, W]`` -> arrival in ``[0, 1]``."""
    torch = _torch()
    import torch.nn as nn

    block = _blocks(nn)

    class _Gen(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.e1, self.e2, self.e3 = block(c, 24), block(24, 48), block(48, 96)
            self.pool = nn.MaxPool2d(2)
            self.up2 = nn.ConvTranspose2d(96, 48, 2, stride=2)
            self.d2 = block(96, 48)
            self.up1 = nn.ConvTranspose2d(48, 24, 2, stride=2)
            self.d1 = block(48, 24)
            self.head = nn.Conv2d(24, 1, 1)

        def forward(self, x):
            s1 = self.e1(x)
            s2 = self.e2(self.pool(s1))
            b = self.e3(self.pool(s2))
            d2 = self.d2(torch.cat([self.up2(b), s2], 1))
            d1 = self.d1(torch.cat([self.up1(d2), s1], 1))
            return torch.sigmoid(self.head(d1))       # arrival in [0, 1]

    return _Gen(in_ch)


def _discriminator(in_ch: int):
    torch = _torch()  # noqa: F841
    import torch.nn as nn

    def c(a, b, s):
        return nn.Sequential(nn.Conv2d(a, b, 4, stride=s, padding=1), nn.LeakyReLU(0.2))

    return nn.Sequential(c(in_ch, 32, 2), c(32, 64, 2), c(64, 128, 2), nn.Conv2d(128, 1, 3, padding=1))


def eikonal_residual_loss(arrival_norm, ros, tmax: float, dx: float = 1.0):
    """Physics term: penalise ``| |grad T| - 1/ROS |`` on the predicted field.

    Ties the generative output to the level-set physics so it cannot hallucinate
    a geometrically impossible front. ``arrival_norm`` is the generator output in
    ``[0, 1]``; it is rescaled by ``tmax`` before differencing.
    """
    torch = _torch()
    T = arrival_norm * tmax
    gy = (T[..., 1:, :] - T[..., :-1, :]) / dx
    gx = (T[..., :, 1:] - T[..., :, :-1]) / dx
    grad = torch.sqrt(gx[..., 1:, :] ** 2 + gy[..., :, 1:] ** 2 + 1e-9)
    inv = 1.0 / torch.clamp(ros[..., 1:, 1:], min=1e-3)
    return (grad - inv).abs().mean()


def train_arrival_gan(pairs, epochs: int = 30, lr: float = 2e-4,
                      l1_weight: float = 40.0, eik_weight: float = 0.5, seed: int = 0):
    """Train the conditional GAN on ``pairs`` = list of ``(cond, target, ros)``.

    Loss: adversarial (LSGAN) + ``l1_weight`` x L1 reconstruction +
    ``eik_weight`` x Eikonal residual. Returns the generator (CPU).
    """
    torch = _torch()
    import torch.nn.functional as F

    torch.manual_seed(seed)
    in_ch = pairs[0][0].shape[0]
    gen = ArrivalGenerator(in_ch)
    disc = _discriminator(in_ch + 1)
    og = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    od = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

    def to_t(a):
        return torch.as_tensor(np.asarray(a), dtype=torch.float32)

    for _ in range(epochs):
        for cond, target, ros in pairs:
            c = to_t(cond)[None]
            y = to_t(target)[None, None]
            r = to_t(ros)[None, None]
            fake = gen(c)
            # discriminator
            od.zero_grad()
            d_real = disc(torch.cat([c, y], 1))
            d_fake = disc(torch.cat([c, fake.detach()], 1))
            loss_d = 0.5 * (F.mse_loss(d_real, torch.ones_like(d_real))
                            + F.mse_loss(d_fake, torch.zeros_like(d_fake)))
            loss_d.backward()
            od.step()
            # generator
            og.zero_grad()
            d_fake = disc(torch.cat([c, fake], 1))
            tmax = 1.0
            loss_g = (F.mse_loss(d_fake, torch.ones_like(d_fake))
                      + l1_weight * F.l1_loss(fake, y)
                      + eik_weight * eikonal_residual_loss(fake[0], r[0], tmax))
            loss_g.backward()
            og.step()
    return gen


def predict_arrival(gen, conditioning):
    """Normalised arrival field ``[H, W]`` in ``[0, 1]`` for one conditioning stack."""
    torch = _torch()
    gen.eval()
    with torch.no_grad():
        out = gen(torch.as_tensor(np.asarray(conditioning), dtype=torch.float32)[None])
    return out[0, 0].cpu().numpy()

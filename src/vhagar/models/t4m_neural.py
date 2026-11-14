"""Sparse-label, physics-guided neural arrival model for T4-M.

The U-Net learns residual structure on top of a physics-prior arrival channel.
Unlike the synthetic-only GAN training path, this trainer supports sparse thermal
front labels through a masked arrival loss.  The Eikonal term supplies a physical
constraint across unlabelled cells.  It is GPU-ready but deliberately does not
claim a trained operational model until real incident replay results exist.
"""

from __future__ import annotations

import numpy as np

__all__ = ["masked_arrival_l1_loss", "train_physics_guided_unet", "predict_arrival"]


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("the T4-M neural model needs pytorch") from exc
    return torch


def masked_arrival_l1_loss(prediction, target, valid_mask):
    """L1 arrival loss over observed future front cells only."""
    torch = _torch()
    valid = valid_mask.bool()
    if not bool(valid.any()):
        raise ValueError("masked arrival loss needs at least one observed target cell")
    return torch.abs(prediction[valid] - target[valid]).mean()


def train_physics_guided_unet(cases, *, epochs: int = 20, lr: float = 2e-4,
                              data_weight: float = 40.0, eikonal_weight: float = 0.5,
                              device: str | None = None, seed: int = 0):
    """Fit a U-Net to sparse future labels plus Eikonal consistency.

    ``cases`` are :class:`~vhagar.datasets.t4m_tensor.T4MTensorCase` instances
    drawn only from training incidents.  The caller owns incident/time splits and
    must select a GPU device explicitly for operational-scale training.
    """
    if not cases:
        raise ValueError("need at least one T4-M tensor case")
    torch = _torch()
    from vhagar.models.arrival_gan import ArrivalGenerator, eikonal_residual_loss

    torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ArrivalGenerator(cases[0].conditioning.shape[0]).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for case in cases:
            if case.conditioning.shape[0] != cases[0].conditioning.shape[0]:
                raise ValueError("all tensor cases need the same conditioning channels")
            x = torch.as_tensor(case.conditioning, dtype=torch.float32, device=dev)[None]
            target = torch.as_tensor(np.nan_to_num(case.target_arrival, nan=0.0), dtype=torch.float32, device=dev)[None, None]
            valid = torch.as_tensor(case.target_valid, device=dev)[None, None]
            ros = torch.as_tensor(case.prior_ros, dtype=torch.float32, device=dev)[None]
            prediction = model(x)
            optimizer.zero_grad()
            loss = (data_weight * masked_arrival_l1_loss(prediction, target, valid)
                    + eikonal_weight * eikonal_residual_loss(prediction[0], ros, case.tmax_hours))
            loss.backward()
            optimizer.step()
    return model


def predict_arrival(model, conditioning) -> np.ndarray:
    """Return normalised arrival-time field from a fitted T4-M U-Net."""
    torch = _torch()
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        out = model(torch.as_tensor(conditioning, dtype=torch.float32, device=dev)[None])
    return out[0, 0].detach().cpu().numpy()

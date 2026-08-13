"""Losses for extreme class imbalance.

Burned pixels are typically 0.1-5% of a chip. Two findings from the literature
are encoded here as defaults:

* **Combo loss (weighted BCE + Dice) is the reliable default.** Dice supplies
  a region-overlap gradient that survives imbalance; BCE keeps per-pixel
  calibration from collapsing.
* **Focal loss is not a free win.** It helps for isolated small targets and
  measurably *reduces recall for clustered fires*. Do not reach for it
  reflexively; A/B it per task and report which you used.

Also note: a loss that optimises Dice/IoU produces **miscalibrated
probabilities**. If the output feeds a probabilistic product (danger, burn
probability), recalibrate on a held-out, base-rate-preserving set --
:mod:`vhagar.eval.metrics` has the diagnostics.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False
    nn = object  # type: ignore[assignment]


__all__ = ["ComboLoss", "DiceLoss", "FocalLoss", "TverskyLoss"]


if _TORCH:

    def _flatten(logits, target):
        return logits.reshape(logits.shape[0], -1), target.reshape(target.shape[0], -1).float()

    class DiceLoss(nn.Module):
        """Soft Dice, computed per-sample then averaged.

        Per-sample (not per-batch) matters: a batch-level Dice lets chips with
        large fires dominate and effectively ignores the small ones, which are
        exactly the omission cases the product cares about.
        """

        def __init__(self, smooth: float = 1.0) -> None:
            super().__init__()
            self.smooth = smooth

        def forward(self, logits, target):
            p, t = _flatten(logits, target)
            p = torch.sigmoid(p)
            inter = (p * t).sum(dim=1)
            denom = p.sum(dim=1) + t.sum(dim=1)
            dice = (2 * inter + self.smooth) / (denom + self.smooth)
            return 1.0 - dice.mean()

    class TverskyLoss(nn.Module):
        """Tversky loss -- Dice with asymmetric FP/FN weighting.

        ``beta > alpha`` penalises false negatives more, which is the right
        asymmetry for a *detection* product where a missed fire costs far more
        than a false alarm an analyst dismisses. Say out loud which asymmetry
        you chose and why; it is a policy decision, not a hyperparameter.
        """

        def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0) -> None:
            super().__init__()
            self.alpha, self.beta, self.smooth = alpha, beta, smooth

        def forward(self, logits, target):
            p, t = _flatten(logits, target)
            p = torch.sigmoid(p)
            tp = (p * t).sum(dim=1)
            fp = (p * (1 - t)).sum(dim=1)
            fn = ((1 - p) * t).sum(dim=1)
            tv = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            return 1.0 - tv.mean()

    class FocalLoss(nn.Module):
        """Focal loss. Use with care -- see module docstring."""

        def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
            super().__init__()
            self.gamma, self.alpha = gamma, alpha

        def forward(self, logits, target):
            p, t = _flatten(logits, target)
            bce = F.binary_cross_entropy_with_logits(p, t, reduction="none")
            prob = torch.sigmoid(p)
            p_t = prob * t + (1 - prob) * (1 - t)
            alpha_t = self.alpha * t + (1 - self.alpha) * (1 - t)
            return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()

    class ComboLoss(nn.Module):
        """Weighted BCE + Dice. The VHAGAR default for T2 and T4.

        ``pos_weight`` should be estimated from the *training folds only*.
        A reasonable starting point is ``(1 - base_rate) / base_rate``, capped
        at ~50 -- an uncapped weight at 0.1% base rate gives 999x and makes the
        model predict fire everywhere.

        >>> import torch
        >>> loss = ComboLoss(pos_weight=10.0)
        >>> float(loss(torch.zeros(2, 1, 8, 8), torch.zeros(2, 1, 8, 8))) > 0
        True
        """

        def __init__(
            self,
            bce_weight: float = 0.5,
            dice_weight: float = 0.5,
            pos_weight: float | None = None,
        ) -> None:
            super().__init__()
            self.bce_weight = bce_weight
            self.dice_weight = dice_weight
            self.dice = DiceLoss()
            self.register_buffer(
                "pos_weight",
                torch.tensor(float(pos_weight)) if pos_weight is not None else None,
                persistent=False,
            )

        def forward(self, logits, target):
            p, t = _flatten(logits, target)
            bce = F.binary_cross_entropy_with_logits(
                p, t, pos_weight=self.pos_weight if self.pos_weight is not None else None
            )
            return self.bce_weight * bce + self.dice_weight * self.dice(logits, target)

    def suggest_pos_weight(base_rate: float, cap: float = 50.0) -> float:
        """``(1 - r) / r``, capped. Estimate ``base_rate`` on training folds only."""
        if not 0.0 < base_rate < 1.0:
            raise ValueError("base_rate must be in (0, 1)")
        return float(min((1.0 - base_rate) / base_rate, cap))

else:  # pragma: no cover

    def _missing(*_a, **_k):
        raise ImportError("vhagar.train.losses requires torch: pip install 'vhagar[torch]'")

    DiceLoss = TverskyLoss = FocalLoss = ComboLoss = _missing  # type: ignore[assignment]
    suggest_pos_weight = _missing  # type: ignore[assignment]

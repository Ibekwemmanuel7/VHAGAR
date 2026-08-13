"""Segmentation backbones for T2 (burned area) and T4 (spread).

Two models live here:

``UNet``
    The permanent baseline. Not a strawman -- a well-tuned U-Net beats a
    large fraction of published transformer results on burned-area
    segmentation, and if your foundation-model fine-tune does not beat it you
    have learned something important.

``SiameseChangeNet``
    The intended production T2 model. Burned area is intrinsically bi-temporal
    (pre-fire and post-fire composites), and a single-date segmenter has to
    infer "was this already bare ground?" from texture alone. A shared-weight
    encoder with multi-scale feature differencing gives the decoder the change
    signal directly.

The encoder is pluggable so the same decoder can sit on a randomly initialised
CNN, a ``timm`` backbone, or a geospatial foundation model encoder
(Prithvi-EO-2.0, TerraMind -- both Apache-2.0).
"""

from __future__ import annotations

from collections.abc import Sequence

try:  # torch is an optional extra
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH = True
except ImportError:  # pragma: no cover - minimal environment
    _TORCH = False
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


def _require_torch() -> None:
    if not _TORCH:
        raise ImportError(
            "vhagar.models.segmentation requires torch: pip install 'vhagar[torch]'"
        )


if _TORCH:

    class ConvBlock(nn.Module):
        """Two 3x3 convolutions with GroupNorm.

        GroupNorm rather than BatchNorm: fire chips are extremely imbalanced
        and batches are small at 320x320x(many bands), so batch statistics are
        noisy and non-stationary across tiles.
        """

        def __init__(self, in_ch: int, out_ch: int, groups: int = 8) -> None:
            super().__init__()
            g = min(groups, out_ch)
            while out_ch % g:
                g -= 1
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.GroupNorm(g, out_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.GroupNorm(g, out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class Encoder(nn.Module):
        """Plain convolutional encoder returning multi-scale features."""

        def __init__(self, in_channels: int, widths: Sequence[int] = (32, 64, 128, 256)) -> None:
            super().__init__()
            self.widths = tuple(widths)
            self.stages = nn.ModuleList()
            prev = in_channels
            for w in self.widths:
                self.stages.append(ConvBlock(prev, w))
                prev = w
            self.pool = nn.MaxPool2d(2)

        def forward(self, x) -> list:
            feats = []
            for i, stage in enumerate(self.stages):
                if i:
                    x = self.pool(x)
                x = stage(x)
                feats.append(x)
            return feats

    class Decoder(nn.Module):
        """U-Net decoder with skip connections."""

        def __init__(self, widths: Sequence[int], out_channels: int = 1) -> None:
            super().__init__()
            widths = list(widths)
            self.ups = nn.ModuleList()
            self.blocks = nn.ModuleList()
            for i in range(len(widths) - 1, 0, -1):
                self.ups.append(nn.ConvTranspose2d(widths[i], widths[i - 1], 2, stride=2))
                self.blocks.append(ConvBlock(widths[i - 1] * 2, widths[i - 1]))
            self.head = nn.Conv2d(widths[0], out_channels, 1)

        def forward(self, feats: list):
            x = feats[-1]
            for i, (up, block) in enumerate(zip(self.ups, self.blocks, strict=True)):
                skip = feats[-2 - i]
                x = up(x)
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
                x = block(torch.cat([x, skip], dim=1))
            return self.head(x)

    class UNet(nn.Module):
        """Baseline U-Net. Logits out; apply the loss's own sigmoid.

        >>> import torch
        >>> m = UNet(in_channels=6)
        >>> m(torch.zeros(1, 6, 64, 64)).shape
        torch.Size([1, 1, 64, 64])
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int = 1,
            widths: Sequence[int] = (32, 64, 128, 256),
        ) -> None:
            super().__init__()
            self.encoder = Encoder(in_channels, widths)
            self.decoder = Decoder(widths, out_channels)

        def forward(self, x):
            return self.decoder(self.encoder(x))

    class SiameseChangeNet(nn.Module):
        """Bi-temporal siamese segmenter for burned area.

        Takes pre-fire and post-fire composites, encodes both with a
        **shared-weight** encoder, and fuses per-scale as
        ``[|f_post - f_pre|, f_post]``. The absolute difference gives the
        decoder the change signal explicitly; keeping ``f_post`` retains
        post-fire context (e.g. distinguishing a burn scar from a harvested
        clearcut, which have similar dNBR but different texture).

        >>> import torch
        >>> m = SiameseChangeNet(in_channels=6)
        >>> pre = torch.zeros(1, 6, 64, 64); post = torch.zeros(1, 6, 64, 64)
        >>> m(pre, post).shape
        torch.Size([1, 1, 64, 64])
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int = 1,
            widths: Sequence[int] = (32, 64, 128, 256),
        ) -> None:
            super().__init__()
            self.encoder = Encoder(in_channels, widths)
            self.fuse = nn.ModuleList([ConvBlock(w * 2, w) for w in widths])
            self.decoder = Decoder(widths, out_channels)

        def forward(self, pre, post):
            f_pre = self.encoder(pre)
            f_post = self.encoder(post)
            fused = [
                fuse(torch.cat([torch.abs(b - a), b], dim=1))
                for fuse, a, b in zip(self.fuse, f_pre, f_post, strict=True)
            ]
            return self.decoder(fused)

    class TemporalAnomalyNet(nn.Module):
        """Per-pixel temporal anomaly detector for geostationary 3.9 um series.

        This is the one learned component on the T1 detection critical path.
        It consumes a window of past brightness temperatures plus exogenous
        covariates (solar zenith, view zenith, day-of-year encoding) and
        predicts the *expected* current brightness temperature. Large positive
        residuals are candidate fires.

        Framing it as forecasting-then-residual rather than direct
        classification matters: it needs no fire labels to train (train on
        clear-sky history), and it degrades gracefully in novel regions.

        Input  : ``(B, T, C, H, W)``
        Output : ``(B, 1, H, W)`` predicted BT for the next step

        >>> import torch
        >>> m = TemporalAnomalyNet(in_channels=3, window=6)
        >>> m(torch.zeros(2, 6, 3, 16, 16)).shape
        torch.Size([2, 1, 16, 16])
        """

        def __init__(self, in_channels: int, window: int, hidden: int = 64) -> None:
            super().__init__()
            self.window = window
            self.temporal = nn.Conv3d(
                in_channels, hidden, kernel_size=(window, 3, 3), padding=(0, 1, 1)
            )
            self.norm = nn.GroupNorm(min(8, hidden), hidden)
            self.spatial = nn.Sequential(
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, 1, 1),
            )

        def forward(self, x):
            if x.shape[1] != self.window:
                raise ValueError(f"expected T={self.window}, got {x.shape[1]}")
            # (B, T, C, H, W) -> (B, C, T, H, W)
            h = self.temporal(x.permute(0, 2, 1, 3, 4)).squeeze(2)
            return self.spatial(self.norm(h))

        @staticmethod
        def anomaly(predicted, observed):
            """Residual excursion in kelvin. Positive == hotter than expected."""
            return observed - predicted

else:  # pragma: no cover - torch-free environment

    def UNet(*_a, **_k):  # type: ignore[misc]
        _require_torch()

    def SiameseChangeNet(*_a, **_k):  # type: ignore[misc]
        _require_torch()

    def TemporalAnomalyNet(*_a, **_k):  # type: ignore[misc]
        _require_torch()


__all__ = ["SiameseChangeNet", "TemporalAnomalyNet", "UNet"]

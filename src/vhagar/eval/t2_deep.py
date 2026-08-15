"""T2 deep models on the multi-channel stack: a multi-channel U-Net and a Siamese
change model, both measured against the same RBR threshold on identical folds.

Where ``t2_unet`` feeds a segmenter the single RBR channel to ask "does spatial
context beat a pointwise cut on the same input?", this asks the next question: "do
richer inputs, pre and post NBR as separate channels, help?" Two models share the
protocol:

``unet``
    A U-Net over the full stack ``[pre_nbr, post_nbr, dnbr]``. More channels than
    the RBR U-Net, same architecture.
``siamese``
    ``SiameseChangeNet``: a shared-weight encoder sees the pre-fire and post-fire
    NBR separately and the decoder works on their multi-scale difference. Burned
    area is intrinsically bi-temporal, so giving the model the change signal
    directly, rather than a pre-differenced RBR, is the architecture's intended T2
    model.

Same discipline as everywhere: leakage-proof grouped folds, per-channel
standardisation and ``pos_weight`` fit on train fires only, skill over the predict-
all-burned baseline, and the RBR threshold measured on the identical fold. The stack
lives in ``T2Sample.stack`` (populated by ``build_optical_sample(..., with_stack=True)``);
a sample without a stack falls back to its single predictor channel. Numpy pieces are
importable without torch; the train loop needs the torch extra.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from vhagar.eval.t2_unet import UNetFoldResult, grouped_folds

__all__ = [
    "ChannelStandardizer",
    "channel_standardizer_from",
    "random_feature_crops",
    "train_deep",
    "evaluate_deep",
    "run_deep_cv",
    "summarise_deep_cv",
]


@dataclass(frozen=True, slots=True)
class ChannelStandardizer:
    """Per-channel robust standardisation (median/MAD), fit on train valid pixels."""

    center: np.ndarray  # [C]
    scale: np.ndarray   # [C]

    def apply(self, feats: np.ndarray, valid: np.ndarray) -> np.ndarray:
        c = self.center[:, None, None]
        s = self.scale[:, None, None]
        out = (feats.astype(np.float32) - c) / s
        out = np.clip(out, -5.0, 5.0)
        out[:, ~valid] = 0.0
        return out


def channel_standardizer_from(samples: Sequence) -> ChannelStandardizer:
    """Fit a per-channel standardiser over the valid pixels of the training samples."""
    per_channel: list[list[np.ndarray]] = None  # type: ignore[assignment]
    for s in samples:
        if s.n_valid == 0:
            continue
        f = s.features  # [C, H, W]
        if per_channel is None:
            per_channel = [[] for _ in range(f.shape[0])]
        for c in range(f.shape[0]):
            per_channel[c].append(f[c][s.valid])
    if not per_channel:
        raise ValueError("no valid training pixels to fit standardiser")
    centers, scales = [], []
    for vals in per_channel:
        v = np.concatenate(vals)
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        centers.append(med)
        scales.append(1.4826 * mad if mad > 0 else (float(np.std(v)) or 1.0))
    return ChannelStandardizer(center=np.array(centers, np.float32), scale=np.array(scales, np.float32))


def random_feature_crops(
    sample, crop: int, n: int, rng: np.random.Generator,
    min_valid_frac: float = 0.25, burned_bias: float = 0.5,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Sample ``n`` ``[C, crop, crop]`` feature tiles from one fire, burned-biased.

    Mirrors ``t2_unet.random_crops`` but crops the whole channel stack. Returns
    ``(feats, burned, valid)`` with ``feats`` shaped ``[C, crop, crop]``.
    """
    feats = sample.features  # [C, H, W]
    burned, valid = sample.reference, sample.valid
    C, H, W = feats.shape
    if crop > H or crop > W:
        padc = ((0, 0), (0, max(0, crop - H)), (0, max(0, crop - W)))
        feats = np.pad(feats, padc, mode="reflect")
        pad2 = ((0, max(0, crop - H)), (0, max(0, crop - W)))
        burned = np.pad(burned, pad2, mode="reflect")
        valid = np.pad(valid, pad2, mode="constant", constant_values=False)
        _, H, W = feats.shape

    burned_valid = np.argwhere(burned & valid)
    out, tries = [], 0
    while len(out) < n and tries < n * 20:
        tries += 1
        if burned_valid.size and rng.random() < burned_bias:
            cy, cx = burned_valid[rng.integers(len(burned_valid))]
            y0 = int(np.clip(cy - crop // 2, 0, H - crop))
            x0 = int(np.clip(cx - crop // 2, 0, W - crop))
        else:
            y0 = int(rng.integers(0, H - crop + 1))
            x0 = int(rng.integers(0, W - crop + 1))
        v = valid[y0:y0 + crop, x0:x0 + crop]
        if v.mean() < min_valid_frac:
            continue
        out.append((
            feats[:, y0:y0 + crop, x0:x0 + crop].copy(),
            burned[y0:y0 + crop, x0:x0 + crop].copy(),
            v.copy(),
        ))
    return out


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "t2-deep needs torch; install the torch extra (pip install vhagar with torch)"
        ) from exc
    return torch


def _build_model(model_kind: str, in_channels: int, widths, torch):
    from vhagar.models.segmentation import SiameseChangeNet, UNet

    if model_kind == "unet":
        return UNet(in_channels=in_channels, out_channels=1, widths=tuple(widths))
    if model_kind == "siamese":
        # Pre and post NBR are one channel each; the shared encoder sees them apart.
        return SiameseChangeNet(in_channels=1, out_channels=1, widths=tuple(widths))
    raise ValueError(f"model_kind must be 'unet' or 'siamese', got {model_kind!r}")


def _forward(model, model_kind: str, x):
    # x: [B, C, H, W] standardised features. Siamese splits channel 0 (pre) and 1 (post).
    if model_kind == "siamese":
        return model(x[:, 0:1], x[:, 1:2])
    return model(x)


def train_deep(
    train_samples: Sequence,
    model_kind: str = "siamese",
    crop: int = 128,
    crops_per_fire: int = 32,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    widths: Sequence[int] = (16, 32, 64, 128),
    seed: int = 0,
    device: str | None = None,
):
    """Train a multi-channel U-Net or Siamese change model. Deterministic.

    Returns ``(model, standardizer)``. Masked weighted-BCE + Dice, ``pos_weight`` and
    per-channel standardisation from train fires only.
    """
    torch = _require_torch()
    import torch.nn.functional as F  # noqa: N812

    from vhagar.train.losses import suggest_pos_weight
    from vhagar.train.train import set_seeds

    set_seeds(seed, deterministic=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    std = channel_standardizer_from(train_samples)
    in_channels = int(train_samples[0].features.shape[0])

    nb = sum(int(np.count_nonzero(s.reference & s.valid)) for s in train_samples)
    nv = sum(int(s.n_valid) for s in train_samples)
    base_rate = max(nb / nv, 1e-4) if nv else 0.1
    pos_weight = torch.tensor(suggest_pos_weight(base_rate), device=dev)

    def masked_combo(logits, target, valid):
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight, reduction="none"
        )
        bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
        p = torch.sigmoid(logits) * valid
        t = target * valid
        inter = (p * t).flatten(1).sum(1)
        denom = p.flatten(1).sum(1) + t.flatten(1).sum(1)
        dice = 1.0 - ((2 * inter + 1.0) / (denom + 1.0)).mean()
        return 0.5 * bce + 0.5 * dice

    model = _build_model(model_kind, in_channels, widths, torch).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    model.train()
    for _ in range(epochs):
        tiles = []
        for s in train_samples:
            tiles.extend(random_feature_crops(s, crop, crops_per_fire, rng))
        if not tiles:
            continue
        rng.shuffle(tiles)
        for i in range(0, len(tiles), batch_size):
            batch = tiles[i:i + batch_size]
            feats = np.stack([std.apply(f, v) for f, _, v in batch])       # [B,C,h,w]
            tgts = np.stack([t.astype(np.float32) for _, t, _ in batch])
            valids = np.stack([v.astype(np.float32) for _, _, v in batch])
            x = torch.from_numpy(feats).to(dev)
            y = torch.from_numpy(tgts[:, None]).to(dev)
            m = torch.from_numpy(valids[:, None]).to(dev)
            loss = masked_combo(_forward(model, model_kind, x), y, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model, std


def evaluate_deep(model, std: ChannelStandardizer, sample, model_kind: str, device=None):
    """Run the model over a fire's full window; return ``(f1, iou, naive_f1)``."""
    torch = _require_torch()
    from vhagar.eval.metrics import confusion_counts

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    feats = std.apply(sample.features, sample.valid)  # [C,H,W]
    C, H, W = feats.shape
    ph, pw = (-H) % 8, (-W) % 8
    if ph or pw:
        feats = np.pad(feats, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    with torch.no_grad():
        x = torch.from_numpy(feats[None]).to(dev)
        prob = torch.sigmoid(_forward(model, model_kind, x))[0, 0].cpu().numpy()[:H, :W]
    v = sample.valid
    pred = prob[v] > 0.5
    truth = sample.reference[v]
    cc = confusion_counts(truth.astype(np.uint8), pred.astype(np.uint8))
    naive = confusion_counts(truth.astype(np.uint8), np.ones(truth.shape, np.uint8))
    return float(cc.f1), float(cc.iou), float(naive.f1)


def run_deep_cv(
    samples_by_id: dict,
    model_kind: str = "siamese",
    k: int = 5,
    method: str = "global",
    objective: str = "youden",
    seed: int = 0,
    **train_kw,
) -> list[UNetFoldResult]:
    """Grouped k-fold: train the deep model per fold, score each held-out fire, and
    measure the RBR threshold on the identical fold. Needs torch."""
    from vhagar.eval.t2_stage0 import evaluate_fold

    usable = {i: s for i, s in samples_by_id.items() if s.is_usable}
    folds = grouped_folds(list(usable), k, seed=seed)
    results: list[UNetFoldResult] = []
    for fold in folds:
        train = [usable[i] for i in fold["train"] if i in usable]
        test = [usable[i] for i in fold["test"] if i in usable]
        if not train or not test:
            continue
        model, std = train_deep(train, model_kind=model_kind, seed=seed, **train_kw)
        for s in test:
            if not (0.0 < s.burned_fraction < 1.0):
                continue
            f1, iou, naive = evaluate_deep(model, std, s, model_kind)
            tr = evaluate_fold(train, [s], method=method, objective=objective, seed=seed)
            results.append(UNetFoldResult(
                held_out=s.event_id, f1=f1, iou=iou, naive_f1=naive,
                thr_f1=tr.f1, thr_naive_f1=tr.naive_f1,
            ))
    return results


def summarise_deep_cv(results: Sequence[UNetFoldResult]) -> dict:
    """Mean skill over naive for the deep model and the threshold, on the same fires."""
    if not results:
        return {"fires": 0}
    md = np.array([r.skill_f1 for r in results])
    th = np.array([r.thr_skill_f1 for r in results])
    return {
        "fires": len(results),
        "deep_f1_mean": float(np.mean([r.f1 for r in results])),
        "deep_skill_mean": float(md.mean()),
        "thr_skill_mean": float(th.mean()),
        "deep_minus_thr": float((md - th).mean()),
        "deep_beats_thr": int(np.sum(md > th)),
    }

"""T2 companion baseline: a plain U-Net over the RBR field, vs the RBR threshold.

The permanent-baselines rule (docs/11) asks for a learned segmenter next to the
pointwise threshold. This is the honest comparison: the U-Net sees exactly the same
input the threshold does, the single-channel RBR window, so the question it answers
is narrow and fair, does a spatial model beat a pointwise cut on identical inputs?
If the U-Net cannot beat a tuned threshold on the same folds, that is worth knowing
before reaching for anything fancier.

Discipline carried over from the threshold baseline:

* **Leakage-proof.** Folds are grouped by fire; no fire is in both train and test.
* **Skill over naive.** Every fold reports F1 minus the predict-all-burned baseline,
  not raw F1, because the windows are class-imbalanced.
* **Stats from train only.** Input standardisation and the loss ``pos_weight`` are
  computed on the training fires of each fold, never the test fire.

The numpy pieces (cropping, standardisation, fold construction) are importable
without torch so they can be unit-tested in any environment; only the train/eval
loop needs ``vhagar[torch]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Standardizer",
    "grouped_folds",
    "random_crops",
    "standardizer_from",
    "UNetFoldResult",
    "train_unet",
    "evaluate_unet",
    "run_unet_cv",
    "summarise_unet_cv",
]


# --------------------------------------------------------------- numpy core ---


@dataclass(frozen=True, slots=True)
class Standardizer:
    """Robust per-channel standardisation fit on training valid pixels.

    Uses median and MAD (scaled to a std-equivalent) rather than mean/std so a few
    extreme RBR pixels do not set the scale. Invalid pixels are mapped to 0 after
    standardisation, i.e. the channel mean, so they carry no signal.
    """

    center: float
    scale: float

    def apply(self, img: np.ndarray, valid: np.ndarray) -> np.ndarray:
        out = (img.astype(np.float32) - self.center) / self.scale
        out = np.clip(out, -5.0, 5.0)
        out[~valid] = 0.0
        return out


def standardizer_from(samples: Sequence) -> Standardizer:
    """Fit a :class:`Standardizer` on the valid pixels of the training samples."""
    vals = np.concatenate([s.predictor[s.valid] for s in samples if s.n_valid > 0])
    if vals.size == 0:
        raise ValueError("no valid training pixels to fit standardiser")
    center = float(np.median(vals))
    mad = float(np.median(np.abs(vals - center)))
    scale = 1.4826 * mad if mad > 0 else (float(np.std(vals)) or 1.0)
    return Standardizer(center=center, scale=scale)


def random_crops(
    sample,
    crop: int,
    n: int,
    rng: np.random.Generator,
    min_valid_frac: float = 0.25,
    burned_bias: float = 0.5,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Sample ``n`` fixed ``crop`` x ``crop`` tiles from one fire's window.

    A fraction ``burned_bias`` of the tiles are centred on a burned pixel so the
    model sees the rare positive class during training; the rest are uniform. A
    tile with too few valid pixels (< ``min_valid_frac``) is rejected and resampled.
    Returns ``(img, burned, valid)`` numpy tiles. Windows smaller than ``crop`` are
    reflect-padded up to ``crop``.
    """
    H, W = sample.predictor.shape
    pred, burned, valid = sample.predictor, sample.reference, sample.valid
    if crop > H or crop > W:
        pad = ((0, max(0, crop - H)), (0, max(0, crop - W)))
        pred = np.pad(pred, pad, mode="reflect")
        burned = np.pad(burned, pad, mode="reflect")
        valid = np.pad(valid, pad, mode="constant", constant_values=False)
        H, W = pred.shape

    burned_valid = np.argwhere(burned & valid)
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    tries = 0
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
            pred[y0:y0 + crop, x0:x0 + crop].copy(),
            burned[y0:y0 + crop, x0:x0 + crop].copy(),
            v.copy(),
        ))
    return out


def grouped_folds(event_ids: Sequence[str], k: int, seed: int = 0) -> list[dict]:
    """Split fire ids into ``k`` leakage-proof folds (each fire tests in one fold).

    Returns a list of ``{"train": [...], "test": [...]}``. Fires are shuffled with a
    fixed seed then round-robin assigned, so folds are balanced in count and
    reproducible. A fire never appears in both sides of a fold.
    """
    ids = list(event_ids)
    if k < 2 or k > len(ids):
        raise ValueError(f"k must be in [2, n_fires]; got k={k}, n={len(ids)}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    buckets: list[list[str]] = [[] for _ in range(k)]
    for j, i in enumerate(order):
        buckets[j % k].append(ids[int(i)])
    folds = []
    for f in range(k):
        test = buckets[f]
        train = [x for b in buckets if b is not buckets[f] for x in b]
        folds.append({"train": train, "test": test})
    return folds


# ----------------------------------------------------------------- torch ------


@dataclass(slots=True)
class UNetFoldResult:
    """One fold's U-Net outcome, per held-out fire plus the threshold it is measured
    against on the identical fold."""

    held_out: str
    f1: float
    iou: float
    naive_f1: float
    thr_f1: float
    thr_naive_f1: float

    @property
    def skill_f1(self) -> float:
        return self.f1 - self.naive_f1

    @property
    def thr_skill_f1(self) -> float:
        return self.thr_f1 - self.thr_naive_f1


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "t2-unet needs torch; install the torch extra (pip install vhagar with torch)"
        ) from exc
    return torch


def train_unet(
    train_samples: Sequence,
    crop: int = 128,
    crops_per_fire: int = 32,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    widths: Sequence[int] = (16, 32, 64, 128),
    seed: int = 0,
    device: str | None = None,
):
    """Train a single-channel U-Net on the training fires. Deterministic.

    Returns ``(model, standardizer)``. Loss is ComboLoss (weighted BCE + Dice) with
    ``pos_weight`` from the training base rate, computed over valid pixels only.
    """
    torch = _require_torch()
    import torch.nn.functional as F  # noqa: N812

    from vhagar.models.segmentation import UNet
    from vhagar.train.losses import suggest_pos_weight
    from vhagar.train.train import set_seeds

    set_seeds(seed, deterministic=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    std = standardizer_from(train_samples)

    # Base rate over valid training pixels -> pos_weight (train folds only).
    nb = sum(int(np.count_nonzero(s.reference & s.valid)) for s in train_samples)
    nv = sum(int(s.n_valid) for s in train_samples)
    base_rate = max(nb / nv, 1e-4) if nv else 0.1
    pos_weight = torch.tensor(suggest_pos_weight(base_rate), device=dev)

    def masked_combo(logits, target, valid):
        # Weighted BCE + soft Dice, both restricted to valid pixels so cloud/nodata
        # contribute nothing. Dice keeps spatial structure (per-image set overlap),
        # which is why the loss is computed on the masked maps, not on a flat vector.
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

    model = UNet(in_channels=1, out_channels=1, widths=tuple(widths)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    model.train()
    for _ in range(epochs):
        tiles = []
        for s in train_samples:
            tiles.extend(random_crops(s, crop, crops_per_fire, rng))
        if not tiles:
            continue
        rng.shuffle(tiles)
        for i in range(0, len(tiles), batch_size):
            batch = tiles[i:i + batch_size]
            imgs = np.stack([std.apply(img, val) for img, _, val in batch])
            tgts = np.stack([tgt.astype(np.float32) for _, tgt, _ in batch])
            valids = np.stack([val.astype(np.float32) for _, _, val in batch])
            x = torch.from_numpy(imgs[:, None]).to(dev)
            y = torch.from_numpy(tgts[:, None]).to(dev)
            m = torch.from_numpy(valids[:, None]).to(dev)
            loss = masked_combo(model(x), y, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model, std


def _confusion(truth: np.ndarray, pred: np.ndarray):
    from vhagar.eval.metrics import confusion_counts

    return confusion_counts(truth.astype(np.uint8), pred.astype(np.uint8))


def evaluate_unet(model, std: Standardizer, sample, device=None):
    """Run the U-Net over one fire's full window and score it over valid pixels.

    Returns ``(f1, iou, naive_f1)``. The window is reflect-padded up to a multiple
    of 8 (the encoder downsamples three times) so any odd size runs in a single
    forward pass, then cropped back. These windows are small enough for one pass.
    """
    torch = _require_torch()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    H, W = sample.predictor.shape
    x = std.apply(sample.predictor, sample.valid)
    ph, pw = (-H) % 8, (-W) % 8
    if ph or pw:
        x = np.pad(x, ((0, ph), (0, pw)), mode="reflect")
    with torch.no_grad():
        t = torch.from_numpy(x[None, None]).to(dev)
        prob = torch.sigmoid(model(t))[0, 0].cpu().numpy()[:H, :W]
    v = sample.valid
    pred = prob[v] > 0.5
    truth = sample.reference[v]
    cc = _confusion(truth, pred)
    naive = _confusion(truth, np.ones_like(truth, dtype=bool))
    return float(cc.f1), float(cc.iou), float(naive.f1)


def run_unet_cv(
    samples_by_id: dict,
    k: int = 5,
    method: str = "global",
    objective: str = "youden",
    strata=None,
    seed: int = 0,
    **train_kw,
) -> list[UNetFoldResult]:
    """Grouped k-fold: train a U-Net per fold, score each held-out fire, and measure
    the RBR threshold on the identical fold for a head-to-head. Needs torch."""
    from vhagar.eval.t2_stage0 import evaluate_fold

    usable = {k_: s for k_, s in samples_by_id.items() if s.is_usable}
    folds = grouped_folds(list(usable), k, seed=seed)
    results: list[UNetFoldResult] = []
    for fold in folds:
        train = [usable[i] for i in fold["train"] if i in usable]
        test = [usable[i] for i in fold["test"] if i in usable]
        if not train or not test:
            continue
        model, std = train_unet(train, seed=seed, **train_kw)
        for s in test:
            if not (0.0 < s.burned_fraction < 1.0):
                continue
            f1, iou, naive = evaluate_unet(model, std, s)
            # Threshold baseline on the same fold, same held-out fire.
            tr = evaluate_fold(train, [s], method=method, objective=objective,
                               strata=strata, seed=seed)
            results.append(UNetFoldResult(
                held_out=s.event_id, f1=f1, iou=iou, naive_f1=naive,
                thr_f1=tr.f1, thr_naive_f1=tr.naive_f1,
            ))
    return results


def summarise_unet_cv(results: Sequence[UNetFoldResult]) -> dict:
    """Mean skill over naive for the U-Net and the threshold, on the same fires."""
    if not results:
        return {"folds": 0}
    us = np.array([r.skill_f1 for r in results])
    th = np.array([r.thr_skill_f1 for r in results])
    return {
        "fires": len(results),
        "unet_f1_mean": float(np.mean([r.f1 for r in results])),
        "unet_skill_mean": float(us.mean()),
        "thr_skill_mean": float(th.mean()),
        "unet_minus_thr": float((us - th).mean()),
        "unet_beats_thr": int(np.sum(us > th)),
    }

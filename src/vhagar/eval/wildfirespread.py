"""Physics-informed, calibrated next-day spread forecasting + honest evaluation.

The contribution over a plain segmentation net: the fast-marching front is used
as a *physics prior*, the learned model refines it, and the output is
temperature-calibrated so its probabilities mean what they say. Everything is
scored on the incremental new-burn region (predicting already-burning cells is
trivial persistence), with both discrimination (AP, F1, IoU) and calibration
(Brier, ECE) proper scores, under leave-fire-out cross-validation.

Three forecasters are compared on identical held-out fires:

* ``persistence_buffer`` - dilate today's fire by a fixed radius (the standard
  no-skill spatial baseline);
* ``physics`` - the fast-marching prior, temperature-calibrated on train fires;
* ``corrector`` - a U-Net that takes the features **plus the physics prior** and
  learns the residual the prior misses (needs torch; skipped if absent).

The numpy pieces (prior, baseline, calibration, scoring, CV plumbing) are pure
and unit-tested; only the corrector needs torch.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from vhagar.eval.metrics import (
    average_precision,
    brier_score,
    confusion_counts,
    expected_calibration_error,
)

__all__ = [
    "physics_prior",
    "persistence_buffer_forecast",
    "fit_temperature",
    "apply_temperature",
    "score_forecast",
    "train_corrector",
    "predict_corrector",
    "evaluate_nextday",
]

_EPS = 1e-6


def physics_prior(sample, *, horizon: float = 12.0) -> np.ndarray:
    """Physics prior: the fast-marching front's burn probability a ``horizon``
    ahead of today's fire, from an isotropic ROS built from fuel/wind/slope."""
    from vhagar.models.spread import rate_of_spread, spread_forecast

    ros = rate_of_spread(sample.fuel, sample.wind, sample.slope)
    _, prob, _ = spread_forecast(sample.fire_t, ros, horizon=horizon)
    return np.clip(prob, 0.0, 1.0)


def persistence_buffer_forecast(sample, *, radius_cells: float = 2.0) -> np.ndarray:
    """No-skill spatial baseline: today's fire dilated by ``radius_cells``."""
    from vhagar.models.spread import persistence_buffer

    _, prob = persistence_buffer(sample.fire_t, radius_cells)
    return np.clip(prob, 0.0, 1.0)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def fit_temperature(prob, y_true, *, bounds=(0.05, 20.0)) -> float:
    """Temperature that best calibrates ``prob`` against ``y_true`` (minimises
    Brier on the pooled pixels). T>1 softens over-confident priors, T<1 sharpens."""
    from scipy.optimize import minimize_scalar

    z = _logit(prob)
    y = np.asarray(y_true, dtype=np.float64)
    if y.size == 0 or len(np.unique(y)) < 2:
        return 1.0

    def obj(t):
        p = 1.0 / (1.0 + np.exp(-z / max(t, 1e-3)))
        return float(np.mean((p - y) ** 2))

    res = minimize_scalar(obj, bounds=bounds, method="bounded")
    return float(res.x) if res.success else 1.0


def apply_temperature(prob, temperature: float) -> np.ndarray:
    """Rescale probabilities through a fitted temperature."""
    return 1.0 / (1.0 + np.exp(-_logit(prob) / max(temperature, 1e-3)))


def score_forecast(prob, sample, *, thresh: float = 0.5) -> dict:
    """Score a probability field on the incremental new-burn region.

    Returns discrimination (AP, F1, IoU) and calibration (Brier, ECE) proper
    scores over the cells that were not already burning on day t."""
    incr = (~sample.fire_t) & sample.valid
    y = sample.fire_t1[incr].astype(np.uint8)
    p = np.clip(np.asarray(prob)[incr], 0.0, 1.0)
    if y.size == 0 or y.sum() == 0:
        return {"ap": float("nan"), "f1": 0.0, "iou": 0.0, "brier": float("nan"),
                "ece": float("nan"), "base_rate": 0.0, "n": int(y.size)}
    pred = p > thresh
    cc = confusion_counts(y.astype(bool), pred)
    return {"ap": float(average_precision(y, p)),
            "f1": float(cc.f1), "iou": float(cc.iou),
            "brier": float(brier_score(y, p)),
            "ece": float(expected_calibration_error(y, p, n_bins=10, equal_mass=True)),
            "base_rate": float(y.mean()), "n": int(y.size)}


# ------------------------------------------------------------------ torch corrector
def _corrector_input(sample, horizon: float) -> np.ndarray:
    """Stack features + the physics prior into the corrector's input tensor."""
    prior = physics_prior(sample, horizon=horizon)[None]
    return np.concatenate([sample.features, prior], axis=0)


def train_corrector(
    train_samples: Sequence, *, horizon: float = 12.0, epochs: int = 25,
    crop: int | None = None, lr: float = 1e-3, widths=(16, 32, 64), seed: int = 0,
    device: str | None = None,
):
    """Train the physics-informed U-Net corrector. Returns ``(model, stats)`` where
    ``stats`` are the per-channel train mean/std for standardisation. Needs torch."""
    try:
        import torch
        import torch.nn.functional as F  # noqa: N812
    except ImportError as exc:  # pragma: no cover - exercised only where torch absent
        raise ImportError("the spread corrector needs pytorch") from exc

    from vhagar.models.segmentation import UNet

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    X = [_corrector_input(s, horizon) for s in train_samples]
    C = X[0].shape[0]
    allpix = np.concatenate([x.reshape(C, -1) for x in X], axis=1)
    mean = allpix.mean(axis=1)
    std = allpix.std(axis=1) + 1e-6
    # class imbalance on the new-burn target
    pos = np.mean([s.fire_t1[(~s.fire_t) & s.valid].mean() if ((~s.fire_t) & s.valid).any()
                   else 0.0 for s in train_samples])
    pos_weight = torch.tensor(float(np.clip((1 - pos) / max(pos, 1e-3), 1.0, 50.0)), device=dev)

    model = UNet(in_channels=C, widths=tuple(widths)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(train_samples))
        for i in order:
            s = train_samples[i]
            x = (X[i] - mean[:, None, None]) / std[:, None, None]
            y = s.fire_t1.astype(np.float32)
            v = s.valid.astype(np.float32)
            xt = torch.as_tensor(x[None], dtype=torch.float32, device=dev)
            yt = torch.as_tensor(y[None, None], device=dev)
            vt = torch.as_tensor(v[None, None], device=dev)
            logit = model(xt)
            loss_map = F.binary_cross_entropy_with_logits(
                logit, yt, pos_weight=pos_weight, reduction="none")
            loss = (loss_map * vt).sum() / (vt.sum() + 1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, {"mean": mean, "std": std, "horizon": horizon}


def predict_corrector(model, stats: Mapping, sample) -> np.ndarray:
    """Corrector burn probability for one fire (reflect-padded to a multiple of 8)."""
    import torch

    dev = next(model.parameters()).device
    x = (_corrector_input(sample, stats["horizon"]) - stats["mean"][:, None, None]) / stats["std"][:, None, None]
    C, H, W = x.shape
    ph, pw = (-H) % 8, (-W) % 8
    if ph or pw:
        x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    model.eval()
    with torch.no_grad():
        t = torch.as_tensor(x[None], dtype=torch.float32, device=dev)
        prob = torch.sigmoid(model(t))[0, 0].cpu().numpy()[:H, :W]
    return np.clip(prob, 0.0, 1.0)


# ------------------------------------------------------------------ leave-fire-out CV
def evaluate_nextday(
    samples_by_id: Mapping, *, k: int = 5, horizon: float = 12.0, radius_cells: float = 2.0,
    calibrate: bool = True, with_corrector: bool = True, seed: int = 0, corrector_kw: Mapping | None = None,
) -> dict:
    """Leave-fire-out comparison of persistence-buffer, the calibrated physics
    prior, and (if torch) the physics-informed corrector.

    Returns per-fire scores per method and their means. The physics temperature is
    fit on each fold's TRAIN fires only (no test leakage); the corrector likewise
    trains on train fires only."""
    from vhagar.eval.t2_unet import grouped_folds

    ids = [i for i, s in samples_by_id.items() if getattr(s, "is_usable", True)]
    if len(ids) < 2:
        raise ValueError("need at least 2 usable fires")
    kk = int(min(k, len(ids)))
    folds = grouped_folds(ids, k=kk, seed=seed)

    per_fire: dict[str, list[dict]] = {"persistence_buffer": [], "physics": [], "corrector": []}
    notes: list[str] = []
    ckw = dict(corrector_kw) if corrector_kw else {}

    for fold in folds:
        train = [samples_by_id[i] for i in fold["train"]]
        test = [samples_by_id[i] for i in fold["test"]]

        # calibrate the physics prior on train fires' pooled new-burn pixels
        temp = 1.0
        if calibrate:
            zp, zy = [], []
            for s in train:
                incr = (~s.fire_t) & s.valid
                zp.append(physics_prior(s, horizon=horizon)[incr])
                zy.append(s.fire_t1[incr].astype(np.float64))
            if zp:
                temp = fit_temperature(np.concatenate(zp), np.concatenate(zy))

        model = stats = None
        if with_corrector:
            try:
                model, stats = train_corrector(train, horizon=horizon, seed=seed, **ckw)
            except ImportError:
                if "corrector skipped: torch not installed" not in notes:
                    notes.append("corrector skipped: torch not installed")

        for s in test:
            per_fire["persistence_buffer"].append(
                {"fire": s.fire_id, **score_forecast(persistence_buffer_forecast(s, radius_cells=radius_cells), s)})
            prob = apply_temperature(physics_prior(s, horizon=horizon), temp)
            per_fire["physics"].append({"fire": s.fire_id, **score_forecast(prob, s)})
            if model is not None:
                per_fire["corrector"].append(
                    {"fire": s.fire_id, **score_forecast(predict_corrector(model, stats, s), s)})

    def _summ(rows):
        if not rows:
            return {}
        keys = ("ap", "f1", "iou", "brier", "ece")
        return {f"{k}_mean": float(np.nanmean([r[k] for r in rows])) for k in keys} | {"fires": len(rows)}

    summary = {m: _summ(rows) for m, rows in per_fire.items() if rows}
    return {"per_fire": {m: r for m, r in per_fire.items() if r}, "summary": summary,
            "temperature_last": temp, "notes": notes, "n_fires": len(ids), "folds": kk}

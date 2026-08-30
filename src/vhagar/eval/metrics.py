"""Metrics for imbalanced, spatially structured fire prediction.

Pure NumPy so it runs anywhere, including CI without torch.

Guidance encoded here
---------------------
* **Accuracy and ROC-AUC are near-useless** at fire base rates (10^-5 to 10^-7
  per km^2-day for ignition; 0.1-1% of pixels for next-day spread). Use
  average precision, and remember AP is base-rate dependent -- it is only
  comparable between models evaluated on *identical* splits.
* **Tune only on proper scoring rules** (log loss, Brier, CRPS). F1, CSI and
  IoU are improper: their optimum depends on the threshold you happen to pick.
  Report them at operational thresholds as decision diagnostics, never as the
  objective for model selection.
* **Burned-area ratio** (predicted/observed) is mandatory alongside IoU. IoU
  hides systematic area bias; the ratio exposes it immediately.
* Spatial tolerance changes everything. The same frozen model can score ~0.05%
  F1 under exact pixel matching and 7-30% under an 8-cell tolerance. Always
  state the tolerance. :func:`f1_with_tolerance` makes it explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ConfusionCounts",
    "average_precision",
    "brier_decomposition",
    "brier_score",
    "burned_area_ratio",
    "confusion_counts",
    "critical_success_index",
    "crps_from_quantiles",
    "dice",
    "expected_calibration_error",
    "f1_with_tolerance",
    "fractions_skill_score",
    "iou",
    "log_loss",
    "pinball_loss",
    "pod_far",
    "reliability_curve",
    "skill_score",
    "sorensen",
]


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return float(self.tp / d) if d else float("nan")

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return float(self.tp / d) if d else float("nan")

    @property
    def f1(self) -> float:
        # 2*tp / (2*tp + fp + fn) -- equivalent to the harmonic mean of
        # precision and recall, but well defined when one of them is 0/0.
        # NaN is reserved for "nothing to score" (no positives predicted or
        # present); a wrong prediction must score 0.0, not NaN.
        denom = 2 * self.tp + self.fp + self.fn
        return float(2 * self.tp / denom) if denom else float("nan")

    @property
    def iou(self) -> float:
        d = self.tp + self.fp + self.fn
        return float(self.tp / d) if d else float("nan")

    def as_dict(self) -> dict[str, float]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "iou": self.iou,
        }


def _binary(y) -> np.ndarray:
    a = np.asarray(y)
    return (a > 0).astype(bool)


def confusion_counts(y_true, y_pred) -> ConfusionCounts:
    t, p = _binary(y_true).ravel(), _binary(y_pred).ravel()
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch {t.shape} vs {p.shape}")
    tp = int(np.sum(t & p))
    fp = int(np.sum(~t & p))
    fn = int(np.sum(t & ~p))
    tn = int(np.sum(~t & ~p))
    return ConfusionCounts(tp, fp, fn, tn)


def iou(y_true, y_pred) -> float:
    """Jaccard index (intersection over union)."""
    return confusion_counts(y_true, y_pred).iou


def dice(y_true, y_pred) -> float:
    """Dice coefficient == F1 on binary masks."""
    return confusion_counts(y_true, y_pred).f1


#: The fire-simulation literature calls the Dice coefficient the Sorensen index.
sorensen = dice


def pod_far(y_true, y_pred) -> tuple[float, float]:
    """Probability of detection and false alarm ratio -- the operational pair."""
    c = confusion_counts(y_true, y_pred)
    pod = c.recall
    far = float(c.fp / (c.tp + c.fp)) if (c.tp + c.fp) else float("nan")
    return pod, far


def critical_success_index(y_true, y_pred) -> float:
    """CSI / threat score. Improper -- diagnostic use only."""
    return confusion_counts(y_true, y_pred).iou


def burned_area_ratio(y_true, y_pred) -> float:
    """Predicted burned area / observed burned area.

    1.0 is unbiased; <1 is under-prediction. Report this next to IoU always --
    a model can hold IoU steady while systematically under- or over-burning.
    """
    t, p = _binary(y_true).sum(), _binary(y_pred).sum()
    return float(p / t) if t else float("nan")


def average_precision(y_true, y_score) -> float:
    """Area under the precision-recall curve, computed as the step-wise sum.

    Equivalent to ``sklearn.metrics.average_precision_score`` without the
    dependency. Returns NaN when there are no positives.
    """
    t = _binary(y_true).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    if t.shape != s.shape:
        raise ValueError(f"shape mismatch {t.shape} vs {s.shape}")
    n_pos = int(t.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-s, kind="mergesort")
    t = t[order]
    s = s[order]
    tp = np.cumsum(t)
    fp = np.cumsum(~t)
    # Group tied scores: evaluate precision/recall only at distinct thresholds,
    # counting every sample at that score together (sklearn's convention). Summing
    # per sample lets an arbitrary tie order between positives and negatives distort
    # AP; taking the last index of each equal-score run removes that dependence.
    distinct = np.concatenate([np.diff(s) != 0, [True]])   # last index of each tie run
    tp = tp[distinct]
    fp = fp[distinct]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    # Step-wise: sum precision at each threshold over the recall increments.
    d_recall = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(precision * d_recall))


def f1_with_tolerance(y_true: np.ndarray, y_pred: np.ndarray, tolerance_cells: int = 0) -> float:
    """F1 where a prediction counts as a hit if truth exists within N cells.

    Exact pixel matching is unreasonably strict for fire fronts whose label
    geolocation error is itself 1-2 cells. Reporting F1 without stating the
    tolerance makes numbers incomparable across papers -- so this function
    forces you to name it.

    Uses a square dilation window (Chebyshev distance), 2-D inputs only.
    """
    t = _binary(y_true)
    p = _binary(y_pred)
    if t.ndim != 2:
        raise ValueError("f1_with_tolerance expects 2-D masks")
    if tolerance_cells <= 0:
        return confusion_counts(t, p).f1

    def dilate(mask: np.ndarray, r: int) -> np.ndarray:
        out = np.zeros_like(mask, dtype=bool)
        rows, cols = mask.shape
        for dy in range(-r, r + 1):
            ys = slice(max(0, dy), rows + min(0, dy))
            yt = slice(max(0, -dy), rows + min(0, -dy))
            for dx in range(-r, r + 1):
                xs = slice(max(0, dx), cols + min(0, dx))
                xt = slice(max(0, -dx), cols + min(0, -dx))
                out[yt, xt] |= mask[ys, xs]
        return out

    t_dil = dilate(t, tolerance_cells)
    p_dil = dilate(p, tolerance_cells)
    tp_p = int(np.sum(p & t_dil))          # predictions near some truth
    tp_r = int(np.sum(t & p_dil))          # truths near some prediction
    precision = tp_p / max(int(p.sum()), 1)
    recall = tp_r / max(int(t.sum()), 1)
    return float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


# --------------------------------------------------------------------------
# Probabilistic scores -- these are what you tune on.
# --------------------------------------------------------------------------


def brier_score(y_true, y_prob) -> float:
    t = _binary(y_true).ravel().astype(np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64).ravel(), 0.0, 1.0)
    return float(np.mean((p - t) ** 2))


def log_loss(y_true, y_prob, eps: float = 1e-15) -> float:
    t = _binary(y_true).ravel().astype(np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64).ravel(), eps, 1 - eps)
    return float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))


def brier_decomposition(y_true, y_prob, n_bins: int = 10) -> dict[str, float]:
    """Murphy decomposition: Brier = reliability - resolution + uncertainty.

    Reliability (lower is better) is the calibration term; resolution (higher
    is better) is the discrimination term; uncertainty is the irreducible base
    rate variance.
    """
    t = _binary(y_true).ravel().astype(np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64).ravel(), 0.0, 1.0)
    n = t.size
    base = float(t.mean())
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    reliability = 0.0
    resolution = 0.0
    for k in range(n_bins):
        m = idx == k
        nk = int(m.sum())
        if nk == 0:
            continue
        pk = float(p[m].mean())
        ok = float(t[m].mean())
        reliability += nk * (pk - ok) ** 2
        resolution += nk * (ok - base) ** 2
    reliability /= n
    resolution /= n
    uncertainty = base * (1 - base)
    return {
        "brier": brier_score(t, p),
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "base_rate": base,
    }


def reliability_curve(y_true, y_prob, n_bins: int = 10, equal_mass: bool = True):
    """Return ``(mean_predicted, observed_frequency, count)`` per bin.

    ``equal_mass=True`` uses quantile bins, which is the right default at fire
    base rates -- equal-width bins leave almost every bin empty and make ECE
    look artificially small.
    """
    t = _binary(y_true).ravel().astype(np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64).ravel(), 0.0, 1.0)
    if equal_mass:
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(p, qs))
        if edges.size < 2:
            # Degenerate: every prediction is the same value (e.g. a constant
            # forecast). One bin is the correct answer, not zero bins.
            return (
                np.array([float(p.mean())]),
                np.array([float(t.mean())]),
                np.array([p.size]),
            )
        idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
        n_eff = len(edges) - 1
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
        n_eff = n_bins

    mean_pred, obs_freq, counts = [], [], []
    for k in range(n_eff):
        m = idx == k
        nk = int(m.sum())
        counts.append(nk)
        mean_pred.append(float(p[m].mean()) if nk else np.nan)
        obs_freq.append(float(t[m].mean()) if nk else np.nan)
    return np.array(mean_pred), np.array(obs_freq), np.array(counts)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10, equal_mass: bool = True) -> float:
    """ECE with quantile bins by default.

    ECE is biased by binning choice -- always report the binning scheme, and
    prefer :func:`brier_decomposition`'s reliability term for model selection.
    """
    mean_pred, obs_freq, counts = reliability_curve(y_true, y_prob, n_bins, equal_mass)
    total = counts.sum()
    if total == 0:
        return float("nan")
    valid = counts > 0
    return float(np.sum(counts[valid] * np.abs(mean_pred[valid] - obs_freq[valid])) / total)


def skill_score(score: float, reference: float, perfect: float = 0.0) -> float:
    """Generic skill score, e.g. Brier Skill Score against a climatology.

    ``(reference - score) / (reference - perfect)``. Positive means better than
    the reference. Always use a *climatological* reference, never a constant.
    """
    denom = reference - perfect
    return float((reference - score) / denom) if denom else float("nan")


def pinball_loss(y_true, y_pred, quantile: float) -> float:
    """Quantile (pinball) loss at level ``quantile``.

    Proper scoring rule for a single predictive quantile: it is minimised by the
    true conditional quantile, so it is the right target for heavy-tailed
    burned-area regression where squared error is dominated by a few extremes.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    q = np.asarray(y_pred, dtype=np.float64).ravel()
    d = y - q
    return float(np.mean(np.maximum(quantile * d, (quantile - 1.0) * d)))


def crps_from_quantiles(y_true, quantile_preds, taus) -> float:
    """Continuous Ranked Probability Score from a set of predictive quantiles.

    Uses the quantile decomposition CRPS = 2 * integral_0^1 pinball_tau d tau,
    approximated as twice the mean pinball loss over the provided ``taus``
    (Gneiting & Raftery). ``quantile_preds`` is ``(n, K)`` at levels ``taus``
    ``(K,)``. Lower is better; unlike RMSE it is stable under heavy tails.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    q = np.asarray(quantile_preds, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64).ravel()
    if q.ndim != 2 or q.shape[1] != taus.size:
        raise ValueError("quantile_preds must be (n, len(taus))")
    pl = [pinball_loss(y, q[:, k], float(taus[k])) for k in range(taus.size)]
    return float(2.0 * np.mean(pl))


def fractions_skill_score(obs, pred, neighborhood: int, threshold: float | None = None) -> float:
    """Fractions Skill Score for a 2D field: spatial verification for rare,
    point-like events.

    Pixel-exact scores punish a forecast that is right about *where fire is
    likely* but off by a cell. FSS instead compares the fraction of exceedances
    in a moving ``neighborhood`` x ``neighborhood`` window between the binary
    observation and the forecast:

        FSS = 1 - mean((O_frac - F_frac)^2) / (mean(O_frac^2) + mean(F_frac^2))

    1 is perfect, 0 is no skill. It rises with neighborhood size, which is the
    point: report it at several scales (e.g. 40 / 80 / 120 km) rather than at one
    pixel. ``obs`` is binary. With ``threshold`` the forecast is binarised at
    that probability (classic FSS); with ``threshold=None`` the probability field
    is used directly as the forecast fraction (the probabilistic variant, fairer
    when comparing calibrated fields of different smoothness).
    """
    from scipy.ndimage import uniform_filter

    o = np.asarray(obs, dtype=np.float64)
    if threshold is None:
        f = np.clip(np.asarray(pred, dtype=np.float64), 0.0, 1.0)
    else:
        f = (np.asarray(pred, dtype=np.float64) >= threshold).astype(np.float64)
    of = uniform_filter(o, size=neighborhood, mode="constant")
    ff = uniform_filter(f, size=neighborhood, mode="constant")
    fbs = float(np.mean((of - ff) ** 2))
    ref = float(np.mean(of ** 2) + np.mean(ff ** 2))
    return 1.0 - fbs / ref if ref > 0 else float("nan")

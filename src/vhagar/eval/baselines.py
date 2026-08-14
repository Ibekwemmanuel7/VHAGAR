"""Mandatory baselines.

Every VHAGAR experiment reports these alongside the model. They are not
optional and they are not "for the appendix" -- a large fraction of published
wildfire deep learning never beats a well-tuned trivial baseline, and several
papers simply do not report one.

    T1 detection    : operational contextual product (the sensor's own mask)
    T2 burned area  : calibrated spectral-index threshold
    T3 danger       : pixel x day-of-year climatology; FWI threshold
    T4 spread       : persistence; persistence + calibrated isotropic buffer
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "climatology_baseline",
    "isotropic_buffer",
    "otsu_threshold",
    "persistence",
    "persistence_with_buffer",
    "threshold_baseline",
    "tune_threshold",
]


def otsu_threshold(index: np.ndarray, nbins: int = 256, clip_percentile: float = 1.0) -> float:
    """Otsu's threshold: the cut that maximises between-class variance.

    An **adaptive, calibration-free** alternative to a globally tuned threshold.
    Each fire's own burn-severity distribution is roughly bimodal (unburned low,
    burned high), and Otsu finds the split between the two modes from that
    distribution alone, so it can transfer across fuel types where a single fixed
    cutoff does not.

    ``clip_percentile`` trims the histogram range to the ``[p, 100-p]`` percentiles
    before binning. This matters for RBR, whose heavy tails would otherwise dump
    almost every pixel into one bin and place the threshold on an outlier.
    Non-finite values are ignored.

    >>> import numpy as np
    >>> x = np.concatenate([np.full(100, 0.0), np.full(100, 1.0)])
    >>> 0.0 < otsu_threshold(x) < 1.0
    True
    """
    a = np.asarray(index, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        raise ValueError("index contains no finite values")
    lo = float(np.percentile(a, clip_percentile))
    hi = float(np.percentile(a, 100.0 - clip_percentile))
    if lo == hi:
        lo, hi = float(a.min()), float(a.max())
    if lo == hi:
        return lo

    hist, edges = np.histogram(a, bins=nbins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    w = hist.astype(np.float64) / hist.sum()
    cum_w = np.cumsum(w)
    cum_mean = np.cumsum(w * centers)
    global_mean = cum_mean[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (global_mean * cum_w - cum_mean) ** 2 / (cum_w * (1.0 - cum_w))
    between[~np.isfinite(between)] = 0.0
    return float(centers[int(np.argmax(between))])


def persistence(previous_mask: np.ndarray) -> np.ndarray:
    """Tomorrow's burned mask == today's.

    On the standard 375 m next-day spread benchmark this scores AP ~0.19,
    against ~0.37-0.40 for the best published models. That is a real ~2x gain,
    but note how much of what a model learns is simply the previous mask.
    """
    return (np.asarray(previous_mask) > 0).astype(np.uint8)


def isotropic_buffer(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Square (Chebyshev) dilation of a binary mask."""
    m = np.asarray(mask) > 0
    if radius_cells <= 0:
        return m.astype(np.uint8)
    out = np.zeros_like(m, dtype=bool)
    rows, cols = m.shape
    for dy in range(-radius_cells, radius_cells + 1):
        ys = slice(max(0, dy), rows + min(0, dy))
        yt = slice(max(0, -dy), rows + min(0, -dy))
        for dx in range(-radius_cells, radius_cells + 1):
            xs = slice(max(0, dx), cols + min(0, dx))
            xt = slice(max(0, -dx), cols + min(0, -dx))
            out[yt, xt] |= m[ys, xs]
    return out.astype(np.uint8)


def persistence_with_buffer(previous_mask: np.ndarray, radius_cells: int = 2) -> np.ndarray:
    """Persistence plus a calibrated isotropic growth ring.

    This is the baseline the literature almost never reports and which closes
    much of the apparent gap to deep models. Calibrate ``radius_cells`` on the
    training folds against median observed daily growth -- do not guess it.
    """
    return isotropic_buffer(previous_mask, radius_cells)


def climatology_baseline(
    history: np.ndarray,
    doy: int,
    window_days: int = 15,
    doy_axis: int = 0,
) -> np.ndarray:
    """Pixel x day-of-year climatological event frequency.

    ``history`` is a boolean/0-1 array with a day-of-year axis of length 366.
    Returns the mean occurrence rate in a +/- ``window_days`` window, which is
    the correct reference for a Brier Skill Score. A constant base rate is
    *not* an acceptable reference -- it makes seasonality look like skill.
    """
    h = np.moveaxis(np.asarray(history, dtype=np.float64), doy_axis, 0)
    n_doy = h.shape[0]
    idx = [(doy - 1 + d) % n_doy for d in range(-window_days, window_days + 1)]
    return h[idx].mean(axis=0)


def threshold_baseline(index: np.ndarray, threshold: float, greater: bool = True) -> np.ndarray:
    """Binary mask from a continuous index (e.g. RBR, dNBR, FWI)."""
    a = np.asarray(index, dtype=np.float64)
    return ((a > threshold) if greater else (a < threshold)).astype(np.uint8)


def tune_threshold(
    index: np.ndarray,
    truth: np.ndarray,
    objective="f1",
    n_steps: int = 200,
) -> tuple[float, float]:
    """Grid-search the index threshold that maximises an objective on *training* folds.

    ``objective`` is ``"f1"``, ``"iou"``, ``"youden"`` (equivalently
    ``"balanced"``), or a callable ``(truth, pred) -> score``. Use ``"youden"``
    when the training pool is class-imbalanced (for example burn-heavy analysis
    windows): F1 and IoU reward predicting the majority class, so on such a pool
    they drive the threshold to a degenerate extreme that does not transfer to a
    balanced test window; Youden's J does not.

    Returns ``(threshold, score)``. Tuning on the test fold is leakage; this
    function does not know which fold you passed it, so discipline is yours.
    """
    from vhagar.eval.metrics import confusion_counts

    a = np.asarray(index, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        raise ValueError("index contains no finite values")
    lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if lo == hi:
        return lo, 0.0

    def _youden(t, p):
        # Youden's J = TPR - FPR = sensitivity + specificity - 1. Unlike F1 and
        # IoU it does not reward predicting the majority class, so on a burn-heavy
        # window pool it does not collapse the threshold to "everything burned".
        # That makes the cut transfer to a class-balanced test window instead of
        # to only-mostly-burned ones. See docs/11 for the measured difference.
        c = confusion_counts(t, p)
        tpr = c.tp / (c.tp + c.fn) if (c.tp + c.fn) else 0.0
        fpr = c.fp / (c.fp + c.tn) if (c.fp + c.tn) else 0.0
        return tpr - fpr

    if callable(objective):
        score_fn = objective
    elif objective == "f1":
        score_fn = lambda t, p: confusion_counts(t, p).f1  # noqa: E731
    elif objective == "iou":
        score_fn = lambda t, p: confusion_counts(t, p).iou  # noqa: E731
    elif objective in ("youden", "balanced"):
        score_fn = _youden
    else:
        raise ValueError(f"unknown objective {objective!r}")

    best_t, best_s = lo, -np.inf
    for t in np.linspace(lo, hi, n_steps):
        s = score_fn(truth, threshold_baseline(a, float(t)))
        if np.isfinite(s) and s > best_s:
            best_t, best_s = float(t), float(s)
    return best_t, best_s

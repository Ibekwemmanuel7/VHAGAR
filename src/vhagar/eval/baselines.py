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
    "persistence",
    "persistence_with_buffer",
    "threshold_baseline",
    "tune_threshold",
]


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

    if callable(objective):
        score_fn = objective
    elif objective == "f1":
        score_fn = lambda t, p: confusion_counts(t, p).f1  # noqa: E731
    elif objective == "iou":
        score_fn = lambda t, p: confusion_counts(t, p).iou  # noqa: E731
    else:
        raise ValueError(f"unknown objective {objective!r}")

    best_t, best_s = lo, -np.inf
    for t in np.linspace(lo, hi, n_steps):
        s = score_fn(truth, threshold_baseline(a, float(t)))
        if np.isfinite(s) and s > best_s:
            best_t, best_s = float(t), float(s)
    return best_t, best_s

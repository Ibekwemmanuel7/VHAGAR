"""Post-fire structure-damage screening metrics (the rapid-damage-assessment tier).

The industry-standard post-fire task is: given post-fire evidence (here, a mapped
burn-severity raster), classify each structure's damage, and validate against ground
truth. The accepted US ground truth is CAL FIRE DINS; the ML community's damage
taxonomy is the xView2/xBD four-level scale (No Damage, Minor, Major, Destroyed).

This module is the scoring core, pure numpy so it runs in the core CI env with no
raster or pandas dependency. It provides:
  * ``binary_screen`` : destroyed-vs-not metrics WITH the mandatory predict-all
    baseline and the skill over it, because a DINS sample is destroyed-heavy and a
    high F1 can be pure base rate (the recurring honesty trap in this project);
  * ``confusion``     : a k-by-k confusion matrix for the ordinal damage classes;
  * ``ordinal_metrics``: overall accuracy, per-class recall/precision, and the
    quadratic-weighted kappa, the standard metric for an ordered-category task,
    which rewards being off by one class less than being off by three.

Nothing here samples a raster or reads DINS; the driver script does that and calls
these. Keeping the metrics pure means they are unit-tested without a geospatial stack.
"""
from __future__ import annotations

import numpy as np

__all__ = ["binary_screen", "confusion", "ordinal_metrics",
           "SEVERITY_CLASS_NAMES", "severity_to_class_index",
           "DAMAGE_RBR_BREAKPOINTS", "rbr_to_class_index"]

#: xView2/xBD-aligned ordinal damage classes, index 0..3.
SEVERITY_CLASS_NAMES = ["No Damage", "Minor", "Major", "Destroyed"]

#: Scaled RBR / dNBR (x1000) breakpoints for the four damage classes, derived from
#: VHAGAR's Key & Benson (2006) severity thresholds (100, 270, 440, 660): the low and
#: moderate-low severity bins are merged into Minor, moderate-high is Major, and high is
#: Destroyed. This is what lets the pack run on VHAGAR's OWN T2 RBR product, not MTBS.
DAMAGE_RBR_BREAKPOINTS = (100.0, 440.0, 660.0)


def severity_to_class_index(sev):
    """Map MTBS thematic burn-severity codes to the four-class damage grade index.

    MTBS: 1 unburned-to-low, 2 low, 3 moderate, 4 high, 5 greenness, 6 mask (0 = no
    data). The screen maps unburned/greenness/mask to No Damage (0), low to Minor (1),
    moderate to Major (2), high to Destroyed (3). A screening heuristic, not calibrated:
    vegetation severity is not structure severity."""
    sev = np.asarray(sev)
    idx = np.zeros(sev.shape, dtype=int)
    idx[sev == 2] = 1
    idx[sev == 3] = 2
    idx[sev == 4] = 3
    return idx


def rbr_to_class_index(rbr, breakpoints=DAMAGE_RBR_BREAKPOINTS):
    """Map a scaled RBR / dNBR severity (x1000, VHAGAR's continuous T2 severity metric)
    to the four-class damage grade index using VHAGAR's Key-Benson-derived thresholds.

    Bins with ``np.digitize`` (right=False): below 100 -> No Damage (0), 100-440 ->
    Minor (1), 440-660 -> Major (2), above 660 -> Destroyed (3). NaN maps to 0. Same
    screening caveat as the MTBS path: vegetation burn severity is not structure damage."""
    idx = np.asarray(rbr, dtype=float)
    out = np.digitize(idx, np.asarray(breakpoints, dtype=float), right=False)
    return np.where(np.isnan(idx), 0, out).astype(int)


def binary_screen(pred_destroyed, truth_destroyed) -> dict:
    """Confusion and skill of a binary destroyed screen against observed destroyed.

    Returns POD (recall), FAR (false-alarm rate = FP/(TP+FP)), precision, F1,
    accuracy, the raw cell counts, the observed base rate, the predict-all-destroyed
    F1 baseline, and the model's F1 skill over that baseline. A positive skill is the
    only honest evidence the screen beats "call everything destroyed"."""
    pred = np.asarray(pred_destroyed, dtype=bool)
    truth = np.asarray(truth_destroyed, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {truth.shape}")
    n = truth.size
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    tn = int(np.sum(~pred & ~truth))
    pod = tp / (tp + fn) if (tp + fn) else float("nan")
    far = fp / (tp + fp) if (tp + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    accuracy = (tp + tn) / n if n else float("nan")
    base = truth.mean() if n else float("nan")
    # predict-all-destroyed: TP = positives, FP = negatives, FN = 0 -> F1 = 2P/(2P+N)
    p, neg = int(truth.sum()), int((~truth).sum())
    naive_f1 = (2 * p) / (2 * p + neg) if (2 * p + neg) else float("nan")
    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "pod": _r(pod), "far": _r(far), "precision": _r(precision),
        "f1": _r(f1), "accuracy": _r(accuracy),
        "base_rate_destroyed": _r(base),
        "predict_all_destroyed_f1": _r(naive_f1),
        "skill_f1_over_baseline": _r((f1 - naive_f1) if np.isfinite(f1) and np.isfinite(naive_f1) else float("nan")),
    }


def confusion(true_idx, pred_idx, k: int) -> np.ndarray:
    """k-by-k confusion matrix, rows = true class, cols = predicted class. Indices
    must be integers in [0, k)."""
    t = np.asarray(true_idx, dtype=int)
    p = np.asarray(pred_idx, dtype=int)
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (t, p), 1)
    return cm


def _quadratic_weighted_kappa(cm: np.ndarray) -> float:
    """Cohen's kappa with quadratic weights, the standard agreement metric for an
    ordered-category confusion matrix. 1.0 is perfect, 0.0 is chance, negative is
    worse than chance."""
    k = cm.shape[0]
    total = cm.sum()
    if total == 0:
        return float("nan")
    obs = cm / total
    row = obs.sum(axis=1)
    col = obs.sum(axis=0)
    exp = np.outer(row, col)
    idx = np.arange(k)
    w = (idx[:, None] - idx[None, :]) ** 2 / ((k - 1) ** 2 if k > 1 else 1)
    denom = float(np.sum(w * exp))
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum(w * obs) / denom)


def ordinal_metrics(cm: np.ndarray) -> dict:
    """Overall accuracy, per-class recall and precision, and the quadratic-weighted
    kappa for an ordinal damage confusion matrix (rows true, cols predicted)."""
    cm = np.asarray(cm, dtype=float)
    total = cm.sum()
    acc = float(np.trace(cm) / total) if total else float("nan")
    diag = np.diag(cm)
    recall = np.divide(diag, cm.sum(axis=1), out=np.full(cm.shape[0], np.nan), where=cm.sum(axis=1) > 0)
    precision = np.divide(diag, cm.sum(axis=0), out=np.full(cm.shape[0], np.nan), where=cm.sum(axis=0) > 0)
    return {
        "accuracy": _r(acc),
        "per_class_recall": [_r(x) for x in recall],
        "per_class_precision": [_r(x) for x in precision],
        "quadratic_weighted_kappa": _r(_quadratic_weighted_kappa(cm)),
        "support": [int(x) for x in cm.sum(axis=1)],
    }


def _r(x, nd: int = 4):
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return x

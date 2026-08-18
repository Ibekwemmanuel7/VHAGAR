"""T3 Layer 3: the deep challenger, in shadow mode.

The architecture (`docs/00` section 5.4) puts a spatial deep model (ConvLSTM /
U-Net-3+) in as a *challenger*, trained with a Fractions Skill Score loss and
evaluated at neighborhood scales, and **promoted only if it beats gradient
boosting on blocked, base-rate-preserving AUPRC and Brier**. It expects to earn
its place at seasonal lead times, not daily. This module is the honest harness
for that verdict.

Ignition danger is gridded (cell x day), so verification is spatial. The harness:

- builds a gridded ignition world with spatially autocorrelated weather/fuel;
- fits a **pointwise** gradient-boosting baseline (no spatial context) and a
  **spatial** challenger (the same booster on neighborhood-pooled features, a
  runnable stand-in for the ConvLSTM), under leave-time-block-out CV;
- scores both with Fractions Skill Score at several neighborhoods plus
  base-rate-preserving AUPRC and Brier;
- applies the promotion gate: the challenger is promoted only if it beats the
  baseline on AUPRC **and** Brier (FSS improvement alone is not enough).

The torch ConvLSTM/U-Net challenger itself lives in ``models/ignition_conv.py``
(torch-guarded, runs on a GPU box); this harness scores whatever gridded
probability field a model produces.
"""

from __future__ import annotations

import numpy as np

from vhagar.eval.metrics import average_precision, brier_score, fractions_skill_score

__all__ = [
    "synthetic_ignition_grid",
    "neighborhood_pool",
    "shadow_evaluate",
    "to_odd",
]


def to_odd(k: float) -> int:
    """Nearest odd integer >= 1 (uniform_filter wants an odd window)."""
    k = max(1, int(round(k)))
    return k if k % 2 == 1 else k + 1


def _smooth_field(rng, shape, passes: int = 4) -> np.ndarray:
    """A spatially autocorrelated field in [0, 1] (smoothed white noise)."""
    from scipy.ndimage import uniform_filter

    a = rng.standard_normal(shape)
    for _ in range(passes):
        a = uniform_filter(a, size=5, mode="wrap")
    a = (a - a.min()) / (np.ptp(a) + 1e-9)
    return a


def synthetic_ignition_grid(rng, T: int = 36, H: int = 40, W: int = 40,
                            cell_km: float = 20.0, obs_noise: float = 0.15,
                            intercept: float = -8.0):
    """A gridded ignition world with smooth, moving weather and fuel.

    Returns ``(X, events, feature_names, cell_km)`` where ``X`` is
    ``[T, C, H, W]`` and ``events`` is ``[T, H, W]`` binary. Ignition is driven
    by the *clean* spatially coherent fields, but the model only sees per-cell
    *noisy observations* (``obs_noise``). Spatial pooling denoises them, so a
    model with neighborhood context can recover signal a pointwise model cannot,
    which is exactly what neighborhood verification is meant to reward.
    """
    fn = ["dryness", "fuel", "wind"]
    fuel = _smooth_field(rng, (H, W))                    # static-ish fuel bed
    X = np.zeros((T, 3, H, W), dtype=np.float32)
    events = np.zeros((T, H, W), dtype=np.int8)
    dry = _smooth_field(rng, (H, W))
    for t in range(T):
        # weather drifts day to day (a smooth random walk of fields)
        dry = np.clip(0.85 * dry + 0.15 * _smooth_field(rng, (H, W)), 0, 1)
        wind = _smooth_field(rng, (H, W))
        # Strong, high-contrast danger: a few cells are very likely to ignite,
        # most are near zero. Contrast (not just base rate) is what makes the
        # field learnable; the intercept sets the overall rate.
        lin = 5.0 * dry + 4.0 * fuel + 2.5 * wind + intercept
        p = 1.0 / (1.0 + np.exp(-lin))
        events[t] = (rng.random((H, W)) < p).astype(np.int8)
        noise = rng.normal(0.0, obs_noise, (3, H, W)).astype(np.float32)
        X[t, 0] = np.clip(dry + noise[0], 0, 1)
        X[t, 1] = np.clip(fuel + noise[1], 0, 1)
        X[t, 2] = np.clip(wind + noise[2], 0, 1)
    return X, events, fn, cell_km


def neighborhood_pool(X, radius_cells: int):
    """Mean-pool each channel over a square neighborhood, per time slice.

    ``X`` is ``[T, C, H, W]``; returns the same shape. This is the spatial
    context the pointwise baseline lacks and the challenger gets.
    """
    from scipy.ndimage import uniform_filter

    n = to_odd(radius_cells)
    out = np.empty_like(X)
    for t in range(X.shape[0]):
        for c in range(X.shape[1]):
            out[t, c] = uniform_filter(X[t, c], size=n, mode="wrap")
    return out


def _time_blocks(T: int, n_folds: int) -> np.ndarray:
    """Contiguous time-block fold ids (leave-time-block-out)."""
    return np.minimum((np.arange(T) * n_folds) // T, n_folds - 1)


def _oof_grid(design, y, groups, T, H, W, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    oof = np.full(y.shape[0], np.nan, dtype=np.float64)
    folds = int(min(len(np.unique(groups)), max(2, len(np.unique(groups)))))
    for tr, te in GroupKFold(n_splits=folds).split(design, y, groups):
        if len(np.unique(y[tr])) < 2:
            oof[te] = float(y[tr].mean())
            continue
        m = HistGradientBoostingClassifier(max_depth=4, max_iter=150,
                                           learning_rate=0.08, random_state=seed)
        m.fit(design[tr], y[tr])
        oof[te] = m.predict_proba(design[te])[:, 1]
    return oof.reshape(T, H, W)


def _fss_multiscale(events, prob, cell_km, neighborhoods_km):
    """Probabilistic FSS averaged over time, at each neighborhood scale.

    Uses the probability field directly as the forecast fraction
    (``threshold=None``), which fairly credits a better-calibrated, smoother
    field rather than rewarding a noisy one that a hard threshold would scatter.
    """
    T = events.shape[0]
    out = {}
    for km in neighborhoods_km:
        n = to_odd(km / cell_km)
        vals = [fractions_skill_score(events[t], prob[t], n, None) for t in range(T)]
        out[int(km)] = float(np.nanmean(vals))
    return out


def shadow_evaluate(X, events, cell_km: float = 20.0,
                    neighborhoods_km=(40, 80, 120), n_folds: int = 4,
                    seed: int = 0, pool_km: float = 40.0) -> dict:
    """Pointwise baseline vs spatial challenger, blocked in time, with the
    promotion gate. Returns FSS-by-scale, AUPRC and Brier for both, and the
    promotion decision."""
    T, C, H, W = X.shape
    y = events.reshape(-1).astype(int)
    groups = np.repeat(_time_blocks(T, n_folds), H * W)

    point = X.reshape(T, C, H * W).transpose(0, 2, 1).reshape(-1, C)     # [N, C]
    pooled = neighborhood_pool(X, to_odd(pool_km / cell_km))
    pooled = pooled.reshape(T, C, H * W).transpose(0, 2, 1).reshape(-1, C)
    chall_design = np.column_stack([point, pooled])                      # [N, 2C]

    base_prob = _oof_grid(point, y, groups, T, H, W, seed)
    chall_prob = _oof_grid(chall_design, y, groups, T, H, W, seed)

    def scores(prob):
        p = prob.reshape(-1)
        ok = ~np.isnan(p)
        return {"auprc": average_precision(y[ok], p[ok]),
                "brier": brier_score(y[ok], p[ok]),
                "fss": _fss_multiscale(events, np.nan_to_num(prob, nan=float(np.nanmean(prob))),
                                       cell_km, neighborhoods_km)}

    base_s, chall_s = scores(base_prob), scores(chall_prob)
    promote = (chall_s["auprc"] > base_s["auprc"]) and (chall_s["brier"] < base_s["brier"])
    return {
        "baseline": base_s, "challenger": chall_s,
        "promote": bool(promote),
        "base_rate": float(events.mean()),
        "neighborhoods_km": list(neighborhoods_km),
        "verdict": ("PROMOTE: challenger beats the baseline on AUPRC and Brier"
                    if promote else
                    "SHADOW: challenger does not beat the baseline on both AUPRC and Brier"),
    }

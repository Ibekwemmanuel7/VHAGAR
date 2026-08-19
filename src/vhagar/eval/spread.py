"""T4 spread evaluation: honest next-day skill, incremental, with the baselines.

The architecture (`docs/00` section 6.1) states the achievable ceiling plainly:
next-day burned-mask **average precision in the 0.35-0.45 band**, with IoU of
roughly 0.6-0.8 on wind-driven fires. The binding constraints are label quality,
fuel-map error and wind downscaling, not model architecture, and *any claim of
much above 0.5 AP is almost certainly a leaky split or cumulative-rather-than-
incremental burned area.* This module encodes that discipline:

- a synthetic fire is grown to truth with the Fast Marching solver;
- the forecaster sees the perimeter at ``t0`` and a **noisy** ROS field (the
  fuel/wind error that is the real ceiling) and propagates it forward;
- scoring is on the **incremental** new-burn region only (cells not already
  burned at ``t0``), so cumulative area cannot inflate the number;
- the mandatory **persistence + buffer** baseline is scored the same way.

Metrics: AP, IoU and Dice at the mask level, burned-area ratio (predicted /
observed, which exposes bias IoU hides), and arrival-time MAE. Stratified by a
wind-driven vs plume-like regime.
"""

from __future__ import annotations

import numpy as np

from vhagar.eval.metrics import average_precision, burned_area_ratio, dice, iou
from vhagar.models.spread import (
    fast_marching_arrival,
    persistence_buffer,
    rate_of_spread,
    spread_forecast,
)

__all__ = ["synthetic_fire", "run_case", "evaluate_spread"]


def _smooth(rng, shape, passes=4):
    from scipy.ndimage import uniform_filter

    a = rng.standard_normal(shape)
    for _ in range(passes):
        a = uniform_filter(a, size=5, mode="reflect")
    return (a - a.min()) / (np.ptp(a) + 1e-9)


def synthetic_fire(rng, H: int = 64, W: int = 64, regime: str = "wind"):
    """Grow one synthetic fire to truth. Returns ``(ros_nominal, T_true, ignition)``.

    ``ros_nominal`` is the rate of spread implied by the *mapped* covariates,
    the best a forecaster could estimate. The *actual* fire also feels two
    effects the forecaster cannot see, which is where real skill is lost:

    - **suppression / a fuel break**: a forward half-plane where the fire is held
      to a crawl (crews, a road, a river);
    - **spotting**: a few embers that ignite ahead of the front and burn back.

    ``regime="wind"`` is a fast, well-fuelled fire; ``regime="plume"`` is slower
    and fuel-limited. The isotropic solver does not render wind-driven
    *elongation* (anisotropy is the noted next step), so the regimes differ in
    rate and coherence, not shape.
    """
    fuel = _smooth(rng, (H, W))
    slope = 0.5 * _smooth(rng, (H, W))
    if regime == "wind":
        wind = np.clip(0.7 + 0.3 * _smooth(rng, (H, W)), 0, 1)
    else:
        wind = 0.25 * _smooth(rng, (H, W))
        fuel = fuel * (0.4 + 0.6 * _smooth(rng, (H, W)))
    ros_nominal = rate_of_spread(fuel, wind, slope)

    cy, cx = H // 2 + int(rng.integers(-4, 5)), W // 2 + int(rng.integers(-4, 5))
    ign = np.zeros((H, W), dtype=bool)
    ign[cy, cx] = True

    # hidden suppression: a forward half-plane held to a crawl
    ang = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:H, 0:W]
    proj = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
    ros_actual = np.where(proj > rng.uniform(3, 10), ros_nominal * 0.15, ros_nominal)
    # fine-scale fuel heterogeneity: real, sub-map-resolution, and unforecastable
    # from the smooth mapped covariates. This is a genuine ceiling on skill.
    ros_actual = ros_actual * np.exp(rng.normal(0.0, 0.5, (H, W)))

    T_true = fast_marching_arrival(ros_actual, ign)
    # hidden spotting: a few embers land ahead and burn back
    reach = np.isfinite(T_true)
    tv = T_true[reach]
    if tv.size and rng.random() < 0.8:
        lo, hi = np.quantile(tv, 0.3), np.quantile(tv, 0.6)
        cand = np.argwhere(reach & (T_true >= lo) & (T_true <= hi))
        for _ in range(int(rng.integers(1, 4))):
            if not len(cand):
                break
            sy, sx = cand[rng.integers(len(cand))]
            spot = np.zeros((H, W), dtype=bool)
            spot[sy, sx] = True
            t_spot = float(T_true[sy, sx]) * float(rng.uniform(0.55, 0.8))
            T_true = np.minimum(T_true, t_spot + fast_marching_arrival(ros_actual, spot))
    return ros_nominal, T_true, ign


def _label_noise(mask, rng, frac):
    """Perturb the observed perimeter: toggle a fraction of boundary cells,
    standing in for the 0.71-0.93 F1 of satellite-derived perimeters."""
    from scipy.ndimage import binary_dilation

    if frac <= 0:
        return mask
    band = binary_dilation(mask) ^ mask
    flip = band & (rng.random(mask.shape) < frac)
    return mask ^ flip


def run_case(rng, H=64, W=64, regime="wind", ros_err=0.7, label_noise=0.08,
             t0_q=0.10, tH_q=0.20):
    """One fire: physics forecast vs persistence+buffer, scored on the new-burn
    region. Returns ``{model: {metrics}}``."""
    ros_nominal, T_true, _ign = synthetic_fire(rng, H, W, regime)
    reach = np.isfinite(T_true)
    tv = T_true[reach]
    t0 = float(np.quantile(tv, t0_q))
    tH = float(np.quantile(tv, tH_q))
    horizon = max(tH - t0, 1e-3)

    burned0 = T_true <= t0
    burned_obs = _label_noise(burned0, rng, label_noise)
    future_true = T_true <= tH
    incr = ~burned_obs                                   # score only new ground
    y = (future_true & incr)[incr]

    # forecaster estimates ROS from the mapped covariates with a spatially
    # CORRELATED error (fuel-map + wind-downscaling bias, structured, not iid),
    # and cannot see the suppression or spotting. That gap is the real ceiling.
    bias = (_smooth(rng, ros_nominal.shape) - 0.5) * 2.0    # smooth field in [-1, 1]
    ros_est = ros_nominal * np.exp(ros_err * bias)
    _mF, pF, aF = spread_forecast(burned_obs, ros_est, horizon)

    mean_ros = float(np.median(ros_est[burned_obs])) if burned_obs.any() else float(ros_est.mean())
    radius = mean_ros * horizon
    _mB, pB = persistence_buffer(burned_obs, radius)

    def score(prob):
        p = prob[incr]
        pred = (p >= 0.5)
        out = {"ap": average_precision(y, p), "iou": iou(y, pred),
               "dice": dice(y, pred), "ba_ratio": burned_area_ratio(y, pred)}
        return out

    res = {"physics": score(pF), "persistence_buffer": score(pB),
           "persistence": {"ap": average_precision(y, np.zeros_like(y, dtype=float)),
                           "iou": 0.0, "dice": 0.0, "ba_ratio": 0.0}}
    # arrival-time MAE on cells that truly burn in the window (physics only)
    burn_win = y & (np.isfinite(aF[incr]))
    if burn_win.any():
        true_dt = (T_true - t0)[incr][burn_win]
        res["physics"]["arrival_mae"] = float(np.mean(np.abs(aF[incr][burn_win] - true_dt)))
    else:
        res["physics"]["arrival_mae"] = float("nan")
    res["_meta"] = {"regime": regime, "new_burn_rate": float(y.mean())}
    return res


def evaluate_spread(n_fires: int = 12, regimes=("wind", "plume"), seed: int = 0,
                    ros_err: float = 0.7, label_noise: float = 0.08) -> dict:
    """Aggregate many synthetic fires per regime. Returns per-regime means for
    the physics forecast, persistence+buffer and persistence."""
    rng = np.random.default_rng(seed)
    out: dict = {}
    for regime in regimes:
        cases = [run_case(rng, regime=regime, ros_err=ros_err, label_noise=label_noise)
                 for _ in range(n_fires)]
        agg: dict = {}
        for model in ("physics", "persistence_buffer", "persistence"):
            keys = ["ap", "iou", "dice", "ba_ratio"] + (["arrival_mae"] if model == "physics" else [])
            agg[model] = {k: float(np.nanmean([c[model][k] for c in cases])) for k in keys}
        agg["new_burn_rate"] = float(np.mean([c["_meta"]["new_burn_rate"] for c in cases]))
        out[regime] = agg
    return out

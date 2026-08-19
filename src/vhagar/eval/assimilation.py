"""T4 sequential assimilation: does the arrival-time analysis tighten with passes?

Satellites observe a fire on sparse, timed passes. After each pass the analysis
re-calibrates the per-fire ROS to *all* detections so far and re-forecasts to the
next pass. This harness measures, honestly, whether that assimilation loop beats
the two things it must beat:

- **naive persistence**: "the fire is where it was last seen" (no growth), the
  operational default between passes;
- **the uncalibrated prior**: the mapped ROS run forward with no correction, so
  the value of the per-fire calibration is isolated.

Scored with Sorensen (Dice) perimeter agreement and false-alarm ratio at the next
pass, plus the calibrated scale ``k`` (which should converge as more passes are
assimilated). Truth is a Fast-Marching fire; the prior ROS carries a deliberate
per-fire bias for the calibration to remove.
"""

from __future__ import annotations

import numpy as np

from vhagar.eval.metrics import dice, pod_far
from vhagar.eval.spread import synthetic_fire
from vhagar.models.spread import fast_marching_arrival
from vhagar.models.state_estimation import estimate_arrival_field

__all__ = ["assimilation_experiment"]


def _one_fire(rng, regime, n_passes, prior_bias, det_noise):
    ros_nominal, T_true, ign = synthetic_fire(rng, regime=regime)
    prior_ros = ros_nominal * prior_bias           # biased prior the loop must correct
    prior_arrival = fast_marching_arrival(prior_ros, ign)   # uncalibrated (k=1)

    reach = np.isfinite(T_true)
    tv = T_true[reach]
    # equal spacing in TIME so each forecast horizon is comparable; the gain
    # across passes then reflects better calibration, not an easier horizon.
    t_lo, t_hi = float(np.quantile(tv, 0.12)), float(np.quantile(tv, 0.55))
    pass_times = np.linspace(t_lo, t_hi, n_passes)

    rows = []
    for j in range(n_passes - 1):
        tj, tnext = float(pass_times[j]), float(pass_times[j + 1])
        det = T_true <= tj
        det_rc = np.argwhere(det)
        det_times = T_true[det] * np.exp(rng.normal(0, det_noise, det.sum()))
        state = estimate_arrival_field(prior_ros, ign, det_rc, det_times)

        truth_next = T_true <= tnext
        analysis_next = state.burned_by(tnext)
        prior_next = prior_arrival <= tnext        # uncalibrated prior forecast
        # score the INCREMENTAL new burn between passes: where naive persistence
        # (fire stays put) has no skill and the analysis must actually forecast.
        incr = ~det
        y = (truth_next & incr)[incr]
        a = (analysis_next & incr)[incr]
        p = (prior_next & incr)[incr]
        rows.append({
            "step": j,
            "k": state.k,
            # incremental Sorensen: the forecast skill on new ground
            "soren_analysis": dice(y, a),
            "soren_prior": dice(y, p),
            "soren_naive": 0.0,                     # persistence predicts no new burn
            "far_analysis": pod_far(y, a)[1],
            # full-perimeter Sorensen of the analysis (the operational number)
            "soren_full": dice(truth_next, analysis_next),
        })
    return rows


def assimilation_experiment(n_fires: int = 12, n_passes: int = 5,
                            regime: str = "wind", prior_bias: float = 0.6,
                            det_noise: float = 0.05, seed: int = 0) -> dict:
    """Aggregate the assimilation loop over many fires.

    Returns per-step means (Sorensen for analysis / uncalibrated prior / naive
    persistence, false-alarm ratio, and the calibrated scale ``k``) plus overall
    means. ``prior_bias`` is the true multiplicative error in the prior ROS that
    calibration should recover (ideal ``k`` ~ ``1 / prior_bias``).
    """
    rng = np.random.default_rng(seed)
    by_step: dict[int, list] = {}
    for _ in range(n_fires):
        for r in _one_fire(rng, regime, n_passes, prior_bias, det_noise):
            by_step.setdefault(r["step"], []).append(r)

    steps = []
    for j in sorted(by_step):
        rs = by_step[j]
        steps.append({
            "step": j,
            "k": float(np.median([r["k"] for r in rs])),
            "soren_analysis": float(np.mean([r["soren_analysis"] for r in rs])),
            "soren_prior": float(np.mean([r["soren_prior"] for r in rs])),
            "soren_naive": float(np.mean([r["soren_naive"] for r in rs])),
            "far_analysis": float(np.mean([r["far_analysis"] for r in rs])),
            "soren_full": float(np.mean([r["soren_full"] for r in rs])),
        })
    overall = {
        "soren_analysis": float(np.mean([s["soren_analysis"] for s in steps])),
        "soren_prior": float(np.mean([s["soren_prior"] for s in steps])),
        "soren_naive": float(np.mean([s["soren_naive"] for s in steps])),
        "far_analysis": float(np.mean([s["far_analysis"] for s in steps])),
        "soren_full": float(np.mean([s["soren_full"] for s in steps])),
        "k_final": steps[-1]["k"] if steps else float("nan"),
        "k_ideal": 1.0 / prior_bias,
    }
    return {"steps": steps, "overall": overall, "prior_bias": prior_bias}

"""T4 state estimation: a continuous fire arrival-time analysis from detections.

The architecture (`docs/00` section 6.2) calls state estimation the highest
return on investment in spread: fuse the sparse, timed satellite active-fire
detections into a *continuous* arrival-time field, and re-calibrate the per-fire
rate-of-spread online, which is where hybrid ML demonstrably earns its keep. The
published state of the art is a conditional GAN that infers arrival time from
active fire (Sorensen ~0.81, ignition-time error ~32 min); that generative model
is torch work for a GPU box (the U-Net in ``models/ignition_conv.py`` is the
machinery). The *physics-anchored* estimator here needs no GPU and is the honest
core it would sit on.

The idea: the perimeter is the level set of an arrival-time field satisfying the
Eikonal equation with some ROS field. A prior ROS (from mapped fuel/wind/slope)
fixes the *spatial pattern* of spread; a single per-fire scale then aligns the
*rate* to the observed detection times. Because scaling ROS by ``k`` scales all
arrival times by ``1/k``, the calibration is a one-parameter robust fit, exactly
the "per-fire ROS adjustment factor calibrated online" the architecture asks for.
As more passes arrive the fit tightens (assimilation).
"""

from __future__ import annotations

import numpy as np

from vhagar.models.spread import fast_marching_arrival

__all__ = ["calibrate_ros_scale", "estimate_arrival_field", "AnalysisState"]


def calibrate_ros_scale(prior_arrival, det_rc, det_times) -> float:
    """Per-fire ROS scale ``k`` aligning the prior arrival field to detections.

    ``prior_arrival`` is the FMM arrival field for the prior ROS (scale 1);
    ``det_rc`` is an ``(n, 2)`` array of detected cell row/cols and ``det_times``
    their observed first-detection times. Since arrival ~ 1/ROS, the analysed
    arrival is ``prior_arrival / k`` with a robust ``k = median(prior/observed)``.
    Returns 1.0 when there is nothing to fit.
    """
    det_rc = np.asarray(det_rc)
    det_times = np.asarray(det_times, dtype=np.float64)
    if det_rc.size == 0:
        return 1.0
    t0 = prior_arrival[det_rc[:, 0], det_rc[:, 1]]
    good = np.isfinite(t0) & (t0 > 0) & (det_times > 0)
    if not good.any():
        return 1.0
    return float(np.median(t0[good] / det_times[good]))


class AnalysisState:
    """A fire arrival-time analysis: the calibrated field plus its ROS scale."""

    __slots__ = ("arrival", "k", "prior_ros", "ignition")

    def __init__(self, arrival, k, prior_ros, ignition):
        self.arrival = arrival          # continuous arrival-time field (analysis)
        self.k = k                      # calibrated per-fire ROS scale
        self.prior_ros = prior_ros
        self.ignition = ignition

    def burned_by(self, t) -> np.ndarray:
        """Analysed burned mask at time ``t``."""
        return self.arrival <= t


def estimate_arrival_field(prior_ros, ignition, det_rc, det_times,
                           dx: float = 1.0) -> AnalysisState:
    """Infer the continuous arrival-time analysis from timed detections.

    Runs the Fast Marching solver on the prior ROS from the ignition, calibrates
    the per-fire scale to the detections, and returns the analysed arrival field
    ``prior_arrival / k``. With no detections it returns the uncalibrated prior.
    """
    prior_arrival = fast_marching_arrival(prior_ros, ignition, dx=dx)
    k = calibrate_ros_scale(prior_arrival, det_rc, det_times)
    return AnalysisState(prior_arrival / max(k, 1e-6), k, prior_ros, ignition)

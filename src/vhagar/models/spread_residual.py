"""T4-M: transparent residual correction for a physics spread prior.

This module deliberately begins with a small regularised model rather than a
neural network.  A physical spread model provides a positive prior ROS field;
the learned component estimates a *bounded multiplicative correction* from
covariates.  It is therefore possible to inspect the correction, fall back to
the prior when the model is not fitted, and require a neural model to beat the
same leakage-safe evaluation.

The training target is the log ratio between a physics arrival time and a timed
thermal observation.  It is a correction target, not a claim that first-seen
thermal detections are exact ignition times: caller code must retain sensor
latency, geolocation uncertainty, and observation-quality fields in its event
record.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "observed_log_ros_ratio",
    "RidgeResidualCorrector",
    "leave_one_fire_out",
    "leave_one_region_out",
    "OODGate",
]


def observed_log_ros_ratio(prior_arrival, det_rc, det_times) -> np.ndarray:
    """Return observed ``log(ROS_observed / ROS_prior)`` at timed detections.

    If a prior reaches a cell at ``t_prior`` and the thermal record first sees it
    at ``t_observed``, the locally implied multiplier is
    ``t_prior / t_observed`` because arrival time is inverse to ROS.  Invalid,
    seed, unreachable, or non-positive-time observations are returned as NaN so
    callers cannot silently train on them.
    """
    arrival = np.asarray(prior_arrival, dtype=np.float64)
    rc = np.asarray(det_rc, dtype=np.int64)
    times = np.asarray(det_times, dtype=np.float64)
    if rc.ndim != 2 or rc.shape[1] != 2 or len(rc) != len(times):
        raise ValueError("det_rc must be (n, 2) and match det_times")
    prior = arrival[rc[:, 0], rc[:, 1]]
    out = np.full(len(times), np.nan, dtype=np.float64)
    good = np.isfinite(prior) & (prior > 0.0) & np.isfinite(times) & (times > 0.0)
    out[good] = np.log(prior[good] / times[good])
    return out


@dataclass
class RidgeResidualCorrector:
    """Regularised log-ROS residual model with a bounded operational output.

    ``fit`` uses only training fires supplied by the caller.  ``predict_log_ratio``
    returns a log multiplier and ``correct_ros`` applies it to a positive prior.
    The bounds prevent an extrapolating model from turning a modest physics error
    into an implausibly fast or slow forecast; they are a guardrail, not calibrated
    uncertainty bounds.
    """

    alpha: float = 1.0
    max_log_multiplier: float = float(np.log(3.0))
    coef_: np.ndarray | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    residual_quantiles_: tuple[float, float] | None = None

    def fit(self, features, log_ratio, sample_weight=None) -> RidgeResidualCorrector:
        """Fit ridge regression after z-scoring features (intercept unpenalised)."""
        X = np.asarray(features, dtype=np.float64)
        y = np.asarray(log_ratio, dtype=np.float64)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("features must be (n, p) and match log_ratio")
        good = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float64)
            if len(w) != len(y):
                raise ValueError("sample_weight must match log_ratio")
            good &= np.isfinite(w) & (w > 0.0)
            w = w[good]
        else:
            w = np.ones(int(good.sum()), dtype=np.float64)
        X, y = X[good], y[good]
        if len(y) < 2:
            raise ValueError("need at least two finite residual observations")

        self.mean_ = np.average(X, axis=0, weights=w)
        self.scale_ = np.sqrt(np.average((X - self.mean_) ** 2, axis=0, weights=w))
        self.scale_ = np.where(self.scale_ > 1e-9, self.scale_, 1.0)
        Z = (X - self.mean_) / self.scale_
        A = np.column_stack([np.ones(len(Z)), Z])
        root_w = np.sqrt(w)[:, None]
        penalty = np.diag(np.r_[0.0, np.full(Z.shape[1], max(self.alpha, 0.0))])
        self.coef_ = np.linalg.solve((A * root_w).T @ (A * root_w) + penalty, (A * root_w).T @ (y * root_w[:, 0]))
        residual = y - A @ self.coef_
        self.residual_quantiles_ = tuple(float(v) for v in np.quantile(residual, [0.1, 0.9]))
        return self

    def predict_log_ratio(self, features) -> np.ndarray:
        """Predict bounded log ROS multipliers; fail closed before fitting."""
        if self.coef_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit the residual corrector before prediction")
        X = np.asarray(features, dtype=np.float64)
        one = X.ndim == 1
        X = np.atleast_2d(X)
        if X.shape[1] != len(self.mean_):
            raise ValueError("feature count differs from fitted model")
        pred = self.coef_[0] + ((X - self.mean_) / self.scale_) @ self.coef_[1:]
        pred = np.clip(pred, -self.max_log_multiplier, self.max_log_multiplier)
        return pred[0] if one else pred

    def predict_log_ratio_interval(self, features) -> tuple[np.ndarray, np.ndarray]:
        """Return empirical 10--90% residual intervals around the correction.

        These are training-residual intervals, not a calibrated guarantee.  The
        time-forward replay scores their coverage explicitly; if it is poor, the
        interval must not be presented as operational uncertainty.
        """
        if self.residual_quantiles_ is None:
            raise RuntimeError("fit the residual corrector before prediction")
        pred = self.predict_log_ratio(features)
        lo, hi = self.residual_quantiles_
        return pred + lo, pred + hi

    def correct_ros(self, prior_ros, features) -> np.ndarray:
        """Apply the learned multiplicative correction to a positive physics ROS."""
        prior = np.asarray(prior_ros, dtype=np.float64)
        correction = np.exp(self.predict_log_ratio(features))
        return np.maximum(prior * correction, 0.0)


@dataclass
class OODGate:
    """Diagonal Mahalanobis gate for out-of-training-distribution covariates."""
    threshold: float = 3.0
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, features) -> OODGate:
        X = np.asarray(features, float)
        if X.ndim != 2 or len(X) < 2:
            raise ValueError("need at least two training feature rows")
        self.mean_ = X.mean(0)
        self.scale_ = np.maximum(X.std(0), 1e-6)
        return self

    def trust(self, features) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit the OOD gate before use")
        X = np.atleast_2d(np.asarray(features, float))
        distance = np.sqrt(np.mean(((X - self.mean_) / self.scale_) ** 2, axis=1))
        return np.clip(1.0 - distance / self.threshold, 0.0, 1.0)

    def blend_log_ratio(self, prediction, features) -> np.ndarray:
        """Shrink learned correction to zero (physics-only) outside support."""
        return np.asarray(prediction, float) * self.trust(features)


def leave_one_fire_out(features, log_ratio, fire_id, *, alpha: float = 1.0) -> list[dict]:
    """Evaluate correction residuals by withholding whole incidents.

    Returns one record per fire with the no-correction baseline (zero log ratio)
    and residual model RMSE.  This is intentionally incident-level, never a
    random point split: adjacent observations from one fire are not independent.
    """
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(log_ratio, dtype=np.float64)
    groups = np.asarray(fire_id)
    if len(X) != len(y) or len(y) != len(groups):
        raise ValueError("features, log_ratio, and fire_id must have equal length")
    out: list[dict] = []
    for held_out in np.unique(groups):
        train, test = groups != held_out, groups == held_out
        model = RidgeResidualCorrector(alpha=alpha).fit(X[train], y[train])
        pred = model.predict_log_ratio(X[test])
        good = np.isfinite(y[test]) & np.isfinite(pred)
        if not good.any():
            continue
        baseline = float(np.sqrt(np.mean(y[test][good] ** 2)))
        corrected = float(np.sqrt(np.mean((y[test][good] - pred[good]) ** 2)))
        out.append({"fire_id": held_out.item() if hasattr(held_out, "item") else held_out,
                    "n_test": int(good.sum()), "baseline_rmse": baseline,
                    "residual_rmse": corrected})
    return out


def leave_one_region_out(features, log_ratio, region_id, *, alpha: float = 1.0) -> list[dict]:
    """Geographic-transfer evaluation: withhold every fire in one region.

    Regions should be ecological/fire-weather strata chosen before fitting (for
    example ecoregion × broad climate regime), never latitude/longitude bins
    tuned after looking at scores.  This asks whether residual learning transfers
    beyond the geography represented in its training fires.
    """
    X, y, groups = np.asarray(features, float), np.asarray(log_ratio, float), np.asarray(region_id)
    if len(X) != len(y) or len(y) != len(groups):
        raise ValueError("features, log_ratio, and region_id must have equal length")
    out = []
    for held_out in np.unique(groups):
        train, test = groups != held_out, groups == held_out
        model = RidgeResidualCorrector(alpha=alpha).fit(X[train], y[train])
        pred = model.predict_log_ratio(X[test])
        good = np.isfinite(y[test]) & np.isfinite(pred)
        if good.any():
            out.append({"region_id": held_out.item() if hasattr(held_out, "item") else held_out,
                        "n_test": int(good.sum()),
                        "baseline_rmse": float(np.sqrt(np.mean(y[test][good] ** 2))),
                        "residual_rmse": float(np.sqrt(np.mean((y[test][good] - pred[good]) ** 2)))})
    return out

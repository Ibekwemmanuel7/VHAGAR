"""Time-forward T4-M replay: physics prior vs learned residual correction.

This is deliberately an *observation-space* evaluation.  Until an event has
validated time-resolved perimeter/front labels, the honest question is whether a
forecasted arrival time agrees with later independent thermal observations, not
whether it recreates a final incident polygon.  The runner keeps three facts
separate: physics-only skill, cross-incident learned correction, and lawful
within-incident calibration using only observations before the issue time.
"""

from __future__ import annotations

import numpy as np

from vhagar.datasets.t4m import (
    T4M_FEATURE_COLUMNS,
    time_forward_split,
    training_rows,
    validate_t4m_observations,
)
from vhagar.models.spread_residual import OODGate, RidgeResidualCorrector

__all__ = ["run_t4m_time_forward_replay"]


def _weighted_rmse(y, pred, weight) -> float:
    return float(np.sqrt(np.average((y - pred) ** 2, weights=weight)))


def _weighted_mae(y, pred, weight) -> float:
    return float(np.average(np.abs(y - pred), weights=weight))


def run_t4m_time_forward_replay(
    historical_observations,
    target_observations,
    cutoff_utc,
    *,
    alpha: float = 1.0,
    min_quality: float = 0.5,
) -> dict:
    """Replay one target fire at ``cutoff_utc`` with no future-label leakage.

    ``historical_observations`` must contain other fires only.  The learned
    residual model is fit to those incidents; a robust event offset is then
    estimated only from the target's pre-cutoff observations.  Future target
    observations are held out for arrival-error and empirical 10--90% interval
    coverage.  Returned values are diagnostics, not an operational forecast
    certificate: thermal first-seen time has sensor cadence and detection latency.
    """
    hist = validate_t4m_observations(historical_observations)
    target = validate_t4m_observations(target_observations)
    target_ids = set(target["fire_id"])
    if len(target_ids) != 1:
        raise ValueError("target_observations must contain exactly one fire")
    target_id = next(iter(target_ids))
    if target_id in set(hist["fire_id"]):
        raise ValueError("historical_observations must exclude the target fire")

    train = training_rows(hist, min_quality=min_quality)
    if train["fire_id"].nunique() < 2:
        raise ValueError("need training observations from at least two historical fires")
    X_train = train.loc[:, T4M_FEATURE_COLUMNS].to_numpy()
    model = RidgeResidualCorrector(alpha=alpha).fit(
        X_train, train["log_ros_ratio"].to_numpy(), train["sample_weight"].to_numpy()
    )
    ood_gate = OODGate().fit(X_train)

    early, future = time_forward_split(target, cutoff_utc)
    early_train = training_rows(early, min_quality=min_quality) if len(early) else early.iloc[0:0]
    future_train = training_rows(future, min_quality=min_quality) if len(future) else future.iloc[0:0]
    if future_train.empty:
        raise ValueError("no valid held-out target observations after cutoff")

    # Legal online adaptation: calculate a robust per-fire residual only from
    # pre-cutoff observations.  Shrink it toward zero when only a few points are
    # available, avoiding a single uncertain thermal pixel dominating the update.
    event_offset = 0.0
    if not early_train.empty:
        p_early = model.predict_log_ratio(early_train.loc[:, T4M_FEATURE_COLUMNS].to_numpy())
        raw_offset = float(np.median(early_train["log_ros_ratio"].to_numpy() - p_early))
        event_offset = float(np.clip(raw_offset * len(early_train) / (len(early_train) + 5.0),
                                     -model.max_log_multiplier, model.max_log_multiplier))

    X_future = future_train.loc[:, T4M_FEATURE_COLUMNS].to_numpy()
    y = future_train["log_ros_ratio"].to_numpy()
    weight = future_train["sample_weight"].to_numpy()
    raw_learned = np.clip(model.predict_log_ratio(X_future) + event_offset,
                          -model.max_log_multiplier, model.max_log_multiplier)
    # A learned residual is not allowed to extrapolate with full strength into
    # covariate space absent from the other incidents.  Trust=0 is a clean
    # physics-only fallback; the per-fire correction is likewise shrunk so a
    # handful of early thermal points cannot override an OOD guardrail.
    ood_trust = ood_gate.trust(X_future)
    learned = raw_learned * ood_trust
    residual_lo, residual_hi = model.residual_quantiles_  # fitted above
    lo, hi = learned + residual_lo, learned + residual_hi
    coverage = float(np.average((y >= lo) & (y <= hi), weights=weight))

    # Arrival time is the decision-facing quantity: T_corrected = T_prior / k.
    # Use the same key ordering as training_rows rather than positional dataframe
    # coincidence.  This remains correct if quality filtering removes observations.
    lookup = future.set_index("observation_id")
    kept_ids = future_train["observation_id"]
    prior_h = lookup.loc[kept_ids, "prior_arrival_hours"].to_numpy(dtype=float)
    observed_h = prior_h / np.exp(y)
    corrected_h = prior_h / np.exp(learned)
    # Disconnected fuel components can produce a numerical "unreachable"
    # arrival far beyond the observed incident window.  Reporting its raw
    # difference as a mean number of hours (millions in practice) says less
    # than a bounded decision-horizon error and lets one cell dominate a fire.
    # Cap both forecasts at the observed evaluation window plus two nominal
    # timing-sigma intervals; retain the cap count so this is not hidden.
    time_sigma_min = lookup.loc[kept_ids, "time_sigma_min"].to_numpy(dtype=float)
    horizon_h = float(np.max(observed_h) + 2.0 * np.median(time_sigma_min) / 60.0)
    prior_capped = np.minimum(prior_h, horizon_h)
    corrected_capped = np.minimum(corrected_h, horizon_h)
    n_arrival_capped = int(((prior_h > horizon_h) | (corrected_h > horizon_h)).sum())

    return {
        "fire_id": target_id,
        "cutoff_utc": str(cutoff_utc),
        "n_historical_fires": int(train["fire_id"].nunique()),
        "n_train": int(len(train)),
        "n_calibration": int(len(early_train)),
        "n_evaluation": int(len(future_train)),
        "event_log_ros_offset": event_offset,
        "mean_ood_trust": float(np.average(ood_trust, weights=weight)),
        "min_ood_trust": float(np.min(ood_trust)),
        "physics_log_ros_rmse": _weighted_rmse(y, np.zeros_like(y), weight),
        "residual_log_ros_rmse": _weighted_rmse(y, learned, weight),
        "arrival_error_cap_hours": horizon_h,
        "n_arrival_capped": n_arrival_capped,
        "physics_arrival_mae_hours": _weighted_mae(observed_h, prior_capped, weight),
        "residual_arrival_mae_hours": _weighted_mae(observed_h, corrected_capped, weight),
        "interval_10_90_coverage": coverage,
        "scoring": "quality-filtered, post-cutoff thermal first-seen observations only",
        "scope": "arrival-time observation replay; not a mapped-perimeter validation",
    }

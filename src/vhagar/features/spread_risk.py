"""Operational spread-risk triage score (weather-driven, transparent).

A 0-100 spread-risk index per fire event from current weather. Wind dominates
fire spread; low humidity and heat amplify it. Deliberately simple and readable,
an operator can agree with the formula at a glance. This is an operational triage
signal for the live feed, NOT a calibrated fire-danger index, that is what the
T3 danger stack (FWI + cause-stratified ignition + E[BA]) is for.
"""

from __future__ import annotations

__all__ = ["RISK_BANDS", "spread_risk_score", "risk_class", "risk_color"]

# (threshold, label, colour) ascending.
RISK_BANDS = [
    (0.0, "Low", "#2E9E5B"),
    (25.0, "Moderate", "#E0B43A"),
    (50.0, "High", "#E8552B"),
    (75.0, "Extreme", "#B71C1C"),
]
_UNKNOWN = ("Unknown", "#8A8F98")


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def spread_risk_score(temp_c, rh_pct, wind_speed_ms):
    """0-100 spread-risk score, or None if no weather is available.

    Weights: wind 55%, dryness 30%, heat 15%, each normalised over an
    operationally meaningful range.
    """
    if temp_c is None and rh_pct is None and wind_speed_ms is None:
        return None
    wind = wind_speed_ms if wind_speed_ms is not None else 0.0
    rh = rh_pct if rh_pct is not None else 50.0
    temp = temp_c if temp_c is not None else 20.0
    wind_score = _clamp(wind / 15.0)              # ~15 m/s -> max
    dryness_score = _clamp((45.0 - rh) / 45.0)    # RH < 45% dry
    heat_score = _clamp((temp - 15.0) / 25.0)     # 15..40 C
    return round(100.0 * (0.55 * wind_score + 0.30 * dryness_score + 0.15 * heat_score), 1)


def risk_class(score) -> str:
    """Map a score to a band label ('Low'..'Extreme', or 'Unknown')."""
    if score is None:
        return _UNKNOWN[0]
    label = RISK_BANDS[0][1]
    for thr, lbl, _clr in RISK_BANDS:
        if score >= thr:
            label = lbl
    return label


def risk_color(score) -> str:
    """Map a score to its band colour."""
    if score is None:
        return _UNKNOWN[1]
    color = RISK_BANDS[0][2]
    for thr, _lbl, clr in RISK_BANDS:
        if score >= thr:
            color = clr
    return color

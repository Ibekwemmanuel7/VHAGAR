"""Versioned, explainable scoring configuration for parcel suitability.

Weights, thresholds, and the direction of every feature live here as data, not as
unexplained constants buried in code. Each use (solar, wind, conservation, residential,
data center) maps the four components (physical, access, hazard, market) to features, each
feature to a weight and a transparent scoring curve. The engine reads this config; a
caller can pass a different versioned config to A/B a ruleset.

Scoring curve kinds (all map a raw value to a 0..100 suitability sub-score):
  ramp_up     value <= lo -> 0,  value >= hi -> 100, linear between (higher is better)
  ramp_down   value <= lo -> 100, value >= hi -> 0  (lower is better)
  band        full score inside [opt_lo, opt_hi], ramps to 0 at [zero_lo, zero_hi]
  frac_up     a 0..1 fraction -> 0..100 (higher better)
  frac_down   a 0..1 fraction -> 100..0 (lower better)
  category    map a category string to a score; value None means "unknown -> no score"
"""
from __future__ import annotations

__all__ = ["SCORING_CONFIG_VERSION", "SCORING_CONFIG", "score_feature", "min_area_ha_for"]

SCORING_CONFIG_VERSION = "parcel-scoring-0.1.0"

_MARKET = {"kind": "category",
           "map": {"supportive": 90.0, "mixed": 55.0, "opposed": 15.0, "unknown": None}}


def _comp(weight, min_inputs, features):
    return {"weight": weight, "min_inputs": min_inputs, "features": features}


SCORING_CONFIG = {
    "version": SCORING_CONFIG_VERSION,
    # minimum parcel size for a use to be scorable at all (hard gate, not a soft penalty)
    "min_area_ha": {"solar": 2.0, "wind": 5.0, "data_center": 1.0,
                    "residential": 0.1, "conservation": 0.0},
    "uses": {
        "solar": {
            "physical": _comp(0.40, 2, [
                ("slope_pct", 0.5, {"kind": "ramp_down", "lo": 3, "hi": 15}),
                ("irradiance_kwh_m2_day", 0.5, {"kind": "ramp_up", "lo": 3.5, "hi": 6.5})]),
            "access": _comp(0.30, 1, [
                ("transmission_distance_km", 0.7, {"kind": "ramp_down", "lo": 1, "hi": 25}),
                ("road_distance_km", 0.3, {"kind": "ramp_down", "lo": 0.2, "hi": 10})]),
            "hazard": _comp(0.20, 1, [
                ("wildfire_exposure_0_100", 0.6, {"kind": "ramp_down", "lo": 10, "hi": 80}),
                ("floodplain_frac", 0.2, {"kind": "frac_down"}),
                ("protected_overlap_frac", 0.2, {"kind": "frac_down"})]),
            "market": _comp(0.10, 0, [("market_support_status", 1.0, _MARKET)]),
        },
        "wind": {
            "physical": _comp(0.45, 1, [
                ("mean_wind_ms_100m", 0.8, {"kind": "ramp_up", "lo": 5.0, "hi": 9.0}),
                ("slope_pct", 0.2, {"kind": "band", "opt_lo": 0, "opt_hi": 20, "zero_lo": -1, "zero_hi": 40})]),
            "access": _comp(0.25, 1, [
                ("transmission_distance_km", 0.8, {"kind": "ramp_down", "lo": 1, "hi": 30}),
                ("road_distance_km", 0.2, {"kind": "ramp_down", "lo": 0.5, "hi": 15})]),
            "hazard": _comp(0.20, 1, [
                ("wildfire_exposure_0_100", 0.5, {"kind": "ramp_down", "lo": 10, "hi": 85}),
                ("protected_overlap_frac", 0.5, {"kind": "frac_down"})]),
            "market": _comp(0.10, 0, [("market_support_status", 1.0, _MARKET)]),
        },
        "conservation": {
            # for conservation the ecological signals are POSITIVE, not constraints
            "physical": _comp(0.45, 1, [
                ("habitat_sensitivity_0_100", 0.6, {"kind": "ramp_up", "lo": 20, "hi": 90}),
                ("protected_overlap_frac", 0.4, {"kind": "frac_up"})]),
            "access": _comp(0.05, 0, [
                ("road_distance_km", 1.0, {"kind": "band", "opt_lo": 0.2, "opt_hi": 5, "zero_lo": 0, "zero_hi": 25})]),
            "hazard": _comp(0.25, 1, [
                ("developed_cover_frac", 0.7, {"kind": "frac_down"}),
                ("wildfire_exposure_0_100", 0.3, {"kind": "ramp_down", "lo": 20, "hi": 95})]),
            "market": _comp(0.25, 0, [("market_support_status", 1.0, _MARKET)]),
        },
        "residential": {
            "physical": _comp(0.35, 2, [
                ("slope_pct", 0.6, {"kind": "ramp_down", "lo": 5, "hi": 25}),
                ("developed_cover_frac", 0.4, {"kind": "frac_up"})]),
            "access": _comp(0.30, 1, [
                ("road_distance_km", 1.0, {"kind": "ramp_down", "lo": 0.1, "hi": 5})]),
            "hazard": _comp(0.25, 1, [
                ("wildfire_exposure_0_100", 0.5, {"kind": "ramp_down", "lo": 10, "hi": 75}),
                ("floodplain_frac", 0.3, {"kind": "frac_down"}),
                ("protected_overlap_frac", 0.2, {"kind": "frac_down"})]),
            "market": _comp(0.10, 0, [("market_support_status", 1.0, _MARKET)]),
        },
        "data_center": {
            "physical": _comp(0.30, 1, [
                ("slope_pct", 1.0, {"kind": "ramp_down", "lo": 2, "hi": 12})]),
            "access": _comp(0.40, 1, [
                ("transmission_distance_km", 0.8, {"kind": "ramp_down", "lo": 0.5, "hi": 15}),
                ("road_distance_km", 0.2, {"kind": "ramp_down", "lo": 0.2, "hi": 8})]),
            "hazard": _comp(0.20, 1, [
                ("wildfire_exposure_0_100", 0.5, {"kind": "ramp_down", "lo": 10, "hi": 70}),
                ("floodplain_frac", 0.5, {"kind": "frac_down"})]),
            "market": _comp(0.10, 0, [("market_support_status", 1.0, _MARKET)]),
        },
    },
}


def min_area_ha_for(use: str, config: dict = SCORING_CONFIG) -> float:
    return float(config.get("min_area_ha", {}).get(use, 0.0))


def _clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def score_feature(value, spec: dict):
    """Map a raw feature value to a 0..100 sub-score using a transparent curve, or None
    when the value cannot be scored (missing, or an 'unknown' category)."""
    if value is None:
        return None
    kind = spec["kind"]
    if kind == "ramp_up":
        lo, hi = spec["lo"], spec["hi"]
        return round(_clamp((value - lo) / (hi - lo) * 100.0), 1)
    if kind == "ramp_down":
        lo, hi = spec["lo"], spec["hi"]
        return round(_clamp((hi - value) / (hi - lo) * 100.0), 1)
    if kind == "frac_up":
        return round(_clamp(float(value) * 100.0), 1)
    if kind == "frac_down":
        return round(_clamp((1.0 - float(value)) * 100.0), 1)
    if kind == "band":
        opt_lo, opt_hi = spec["opt_lo"], spec["opt_hi"]
        zlo, zhi = spec["zero_lo"], spec["zero_hi"]
        if opt_lo <= value <= opt_hi:
            return 100.0
        if value < opt_lo:
            return round(_clamp((value - zlo) / (opt_lo - zlo) * 100.0), 1) if opt_lo > zlo else 0.0
        return round(_clamp((zhi - value) / (zhi - opt_hi) * 100.0), 1) if zhi > opt_hi else 0.0
    if kind == "category":
        mapped = spec["map"].get(str(value))
        return None if mapped is None else float(mapped)
    raise ValueError(f"unknown scoring kind: {kind}")

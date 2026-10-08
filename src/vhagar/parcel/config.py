"""Versioned, explainable scoring configuration for parcel suitability.

Weights, thresholds, required inputs, and hard exclusions live here as data, not as
constants buried in the engine. A caller can pass a different versioned config to A/B a
ruleset, and every result records the config version that produced it.

Contract (enforced by ``vhagar.parcel.engine``):

* ``required_components``: every listed component must be scored before an overall score
  is produced. There is no renormalisation around a missing component, so a component that
  cannot be assessed can never raise the overall score by disappearing.
* Every feature inside a component is REQUIRED. A component is scored only when all of its
  features are present. Feature weights are fixed; they are never re-weighted over whatever
  happens to be available. This is what stops, for example, a wind score from being
  computed out of terrain slope alone.
* ``exclusions``: hard, per-use eligibility gates. A triggered exclusion returns
  ``ineligible`` with the exclusion reason and no overall score. An exclusion whose input is
  missing cannot be cleared, so the parcel is ``insufficient_evidence``, not eligible by
  default. Thresholds are screening choices recorded here for audit; they are not legal or
  regulatory determinations.
* ``market`` is reported as an evidence status and is NOT blended into the overall score
  (``in_overall: False``). Community and market evidence is contested, LLM-assisted, and
  human-reviewed; mixing it into a physical suitability number would hide it.

Scoring curve kinds (all map a raw value to a 0..100 sub-score):
  ramp_up     value <= lo -> 0,  value >= hi -> 100, linear between (higher is better)
  ramp_down   value <= lo -> 100, value >= hi -> 0  (lower is better)
  band        full score inside [opt_lo, opt_hi], ramps to 0 at [zero_lo, zero_hi]
  frac_up     a 0..1 fraction -> 0..100 (higher better)
  frac_down   a 0..1 fraction -> 100..0 (lower better)
  category    map a category string to a score; value None or "unknown" means no score
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "SCORING_CONFIG_VERSION",
    "SCORING_CONFIG",
    "score_feature",
    "min_area_ha_for",
    "required_features_for",
    "exclusion_triggered",
]

SCORING_CONFIG_VERSION = "parcel-scoring-0.2.0"

_MARKET_CURVE = {"kind": "category",
                 "map": {"supportive": 90.0, "mixed": 55.0, "opposed": 15.0, "unknown": None}}


def _comp(weight: float, features: list, *, in_overall: bool = True) -> dict[str, Any]:
    return {"weight": weight, "in_overall": in_overall, "features": features}


def _market() -> dict[str, Any]:
    return _comp(0.0, [("market_support_status", 1.0, _MARKET_CURVE)], in_overall=False)


def _excl(feature: str, threshold: float, code: str) -> dict[str, Any]:
    """Exclusion: ineligible when ``feature`` > ``threshold``."""
    return {"feature": feature, "op": ">", "threshold": threshold, "code": code}


SCORING_CONFIG: dict[str, Any] = {
    "version": SCORING_CONFIG_VERSION,
    # Minimum parcel size for a use (hard gate: below it the parcel is ineligible for that use).
    "min_area_ha": {"solar": 2.0, "wind": 5.0, "data_center": 1.0,
                    "residential": 0.1, "conservation": 0.0},
    "uses": {
        "solar": {
            "required_components": ["physical", "access", "hazard"],
            "exclusions": [_excl("protected_overlap_frac", 0.10, "protected_area_overlap"),
                           _excl("floodplain_frac", 0.25, "floodplain_overlap")],
            "components": {
                "physical": _comp(0.45, [
                    ("slope_pct", 0.4, {"kind": "ramp_down", "lo": 3, "hi": 15}),
                    ("irradiance_kwh_m2_day", 0.6, {"kind": "ramp_up", "lo": 3.5, "hi": 6.5})]),
                "access": _comp(0.30, [
                    ("transmission_distance_km", 0.7, {"kind": "ramp_down", "lo": 1, "hi": 25}),
                    ("road_distance_km", 0.3, {"kind": "ramp_down", "lo": 0.2, "hi": 10})]),
                "hazard": _comp(0.25, [
                    ("wildfire_exposure_0_100", 0.6, {"kind": "ramp_down", "lo": 10, "hi": 80}),
                    ("floodplain_frac", 0.2, {"kind": "frac_down"}),
                    ("protected_overlap_frac", 0.2, {"kind": "frac_down"})]),
                "market": _market(),
            },
        },
        "wind": {
            "required_components": ["physical", "access", "hazard"],
            "exclusions": [_excl("protected_overlap_frac", 0.10, "protected_area_overlap")],
            "components": {
                "physical": _comp(0.50, [
                    ("mean_wind_ms_100m", 0.8, {"kind": "ramp_up", "lo": 5.0, "hi": 9.0}),
                    ("slope_pct", 0.2, {"kind": "band", "opt_lo": 0, "opt_hi": 20,
                                        "zero_lo": -1, "zero_hi": 40})]),
                "access": _comp(0.25, [
                    ("transmission_distance_km", 0.8, {"kind": "ramp_down", "lo": 1, "hi": 30}),
                    ("road_distance_km", 0.2, {"kind": "ramp_down", "lo": 0.5, "hi": 15})]),
                "hazard": _comp(0.25, [
                    ("wildfire_exposure_0_100", 0.5, {"kind": "ramp_down", "lo": 10, "hi": 85}),
                    ("protected_overlap_frac", 0.5, {"kind": "frac_down"})]),
                "market": _market(),
            },
        },
        "conservation": {
            # For conservation, ecological signals are POSITIVE, not constraints.
            "required_components": ["physical", "access", "hazard"],
            "exclusions": [_excl("developed_cover_frac", 0.60, "predominantly_developed")],
            "components": {
                "physical": _comp(0.55, [
                    ("habitat_sensitivity_0_100", 0.6, {"kind": "ramp_up", "lo": 20, "hi": 90}),
                    ("protected_overlap_frac", 0.4, {"kind": "frac_up"})]),
                "access": _comp(0.05, [
                    ("road_distance_km", 1.0, {"kind": "band", "opt_lo": 0.2, "opt_hi": 5,
                                               "zero_lo": 0, "zero_hi": 25})]),
                "hazard": _comp(0.40, [
                    ("developed_cover_frac", 0.7, {"kind": "frac_down"}),
                    ("wildfire_exposure_0_100", 0.3, {"kind": "ramp_down", "lo": 20, "hi": 95})]),
                "market": _market(),
            },
        },
        "residential": {
            "required_components": ["physical", "access", "hazard"],
            "exclusions": [_excl("floodplain_frac", 0.50, "floodplain_overlap"),
                           _excl("protected_overlap_frac", 0.10, "protected_area_overlap")],
            "components": {
                "physical": _comp(0.40, [
                    ("slope_pct", 0.6, {"kind": "ramp_down", "lo": 5, "hi": 25}),
                    ("developed_cover_frac", 0.4, {"kind": "frac_up"})]),
                "access": _comp(0.30, [
                    ("road_distance_km", 1.0, {"kind": "ramp_down", "lo": 0.1, "hi": 5})]),
                "hazard": _comp(0.30, [
                    ("wildfire_exposure_0_100", 0.5, {"kind": "ramp_down", "lo": 10, "hi": 75}),
                    ("floodplain_frac", 0.3, {"kind": "frac_down"}),
                    ("protected_overlap_frac", 0.2, {"kind": "frac_down"})]),
                "market": _market(),
            },
        },
        "data_center": {
            "required_components": ["physical", "access", "hazard"],
            "exclusions": [_excl("floodplain_frac", 0.10, "floodplain_overlap"),
                           _excl("protected_overlap_frac", 0.10, "protected_area_overlap")],
            "components": {
                "physical": _comp(0.30, [
                    ("slope_pct", 1.0, {"kind": "ramp_down", "lo": 2, "hi": 12})]),
                "access": _comp(0.45, [
                    ("transmission_distance_km", 0.8, {"kind": "ramp_down", "lo": 0.5, "hi": 15}),
                    ("road_distance_km", 0.2, {"kind": "ramp_down", "lo": 0.2, "hi": 8})]),
                "hazard": _comp(0.25, [
                    ("wildfire_exposure_0_100", 0.5, {"kind": "ramp_down", "lo": 10, "hi": 70}),
                    ("floodplain_frac", 0.5, {"kind": "frac_down"})]),
                "market": _market(),
            },
        },
    },
}


def min_area_ha_for(use: str, config: dict = SCORING_CONFIG) -> float:
    """Minimum parcel area (ha) for ``use`` under ``config``."""
    return float(config.get("min_area_ha", {}).get(use, 0.0))


def required_features_for(use: str, config: dict = SCORING_CONFIG) -> list[str]:
    """Every feature that must be present for ``use`` to receive an overall score: all
    features of the required components plus every exclusion input. Order is stable."""
    use_cfg = config["uses"][use]
    out: list[str] = []
    for name in use_cfg["required_components"]:
        out += [f for (f, _w, _s) in use_cfg["components"][name]["features"]]
    out += [e["feature"] for e in use_cfg.get("exclusions", [])]
    return list(dict.fromkeys(out))


def exclusion_triggered(value: float, rule: dict) -> bool:
    """True when ``value`` violates the exclusion ``rule``."""
    if rule["op"] == ">":
        return float(value) > float(rule["threshold"])
    if rule["op"] == "<":
        return float(value) < float(rule["threshold"])
    raise ValueError(f"unknown exclusion op: {rule['op']}")


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def score_feature(value: Any, spec: dict) -> float | None:
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
